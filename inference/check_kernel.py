"""Test T2 (driver-plan.md, section 4): several events through one compiled sampler, in one process.

Builds `init` and `step` once (kernel.make_sampler), then samples each event to termination. Passes if
`ns_init` and `ns_step` are each traced and XLA-compiled exactly once over all events, and every run
gives a finite ln Z. Counts come from jax_log_compiles records (pjit trace, MLIR lowering, XLA compile)
and from the jitted functions' cache sizes. Other small compiles (prior draws, key splits, finalise,
whose shapes vary with the number of dead points) are counted and reported, but do not fail the test.

    python check_kernel.py --catalogue catalogue.csv --events 0 1 2 --n-live 256

If a second event recompiles, rerun with --explain-cache-misses for JAX's reason (usually a dtype,
weak type or shape that differs between events' EventData).

T7 later: the same with --all-params (12 parameters) and --no-log-compiles; then only cache sizes and
the first-step timings are available.

Test T4: the same script, run as separate processes that share a persistent compilation cache
(--cache-dir). Process 1 (--expect-cache miss, empty dir) must miss and write `ns_init`/`ns_step`;
process 2 (--expect-cache hit) must load both from disk: one cache hit each, no miss, and no
"Not writing persistent cache entry" record. Tracing and lowering still run in every process; with
log_compiles, JAX's "Finished XLA compilation" record also appears on a hit (it times the disk load).
--summary-out writes ln Z, iteration/evaluation counts and a hash of the posterior samples per event;
--compare-to checks them for exact equality against an earlier run's summary.

    python check_kernel.py --catalogue catalogue.csv --events 0 --cache-dir jax_cache_t4 \
        --expect-cache miss --summary-out t4_run1.json
    python check_kernel.py --catalogue catalogue.csv --events 0 --cache-dir jax_cache_t4 \
        --expect-cache hit --summary-out t4_run2.json --compare-to t4_run1.json

Test T5: the sampler saved with jax.export (export_kernel.py) and loaded in later processes, which must
not trace `ns_init`/`ns_step` at all. --export-out DIR runs as T2/T4, then exports the traced functions.
--load-export DIR replaces make_sampler with export_kernel.load_sampler; the jitted wrappers around the
loaded programs are `ns_init_exported`/`ns_step_exported`, and the T2/T4 checks (one trace, lower and
compile each; --expect-cache) apply to those instead. The wrapper's module differs from the one T4
cached, so the first load misses the persistent cache and later loads hit it. Results must match the
T4 run exactly (--compare-to).

    python check_kernel.py --catalogue catalogue.csv --events 0 --cache-dir jax_cache_t5 \
        --export-out export_t5 --summary-out t5_run1.json --compare-to t4_run1.json
    python check_kernel.py --catalogue catalogue.csv --events 0 --cache-dir jax_cache_t5 \
        --load-export export_t5 --expect-cache miss --summary-out t5_run2.json --compare-to t4_run1.json
    python check_kernel.py --catalogue catalogue.csv --events 0 --cache-dir jax_cache_t5 \
        --load-export export_t5 --expect-cache hit --summary-out t5_run3.json --compare-to t4_run1.json
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import warnings
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from check_event_model import jimgw_prior
from event_model import FIXED_PARAMS, EventModel, event_data_from_likelihood, gibbs_blocks
from heterodyne import DEFAULT_N_BINS
from injection import inject, make_likelihood, make_waveform, read_catalogue
from export_kernel import ExportMismatch, export_sampler, load_sampler
from kernel import (SamplerSettings, enable_persistent_cache, make_sampler, rebuild_required_by_block,
                    run_event)

KERNEL_FUNS = ("ns_init", "ns_step")
EXPORTED_FUNS = ("ns_init_exported", "ns_step_exported")   # jitted wrappers of the loaded export
# Traced only when exporting; a process that loads the export must trace none of these.
TRACED_FUNS = KERNEL_FUNS + ("ns_init_flat", "ns_step_flat")
STAGES = ("trace", "lower", "compile", "cache_hit")
_PATTERNS = [
    ("trace", re.compile(r"^Finished tracing \+ transforming (?:jit[_(])?([\w.<>-]+)\)? for pjit in ([\d.e-]+) sec")),
    ("lower", re.compile(r"^Finished jaxpr to MLIR module conversion (?:jit[_(])?([\w.<>-]+)\)? in ([\d.e-]+) sec")),
    ("compile", re.compile(r"^Finished XLA compilation of (?:jit[_(])?([\w.<>-]+)\)? in ([\d.e-]+) sec")),
    ("cache_hit", re.compile(r"^Persistent compilation cache hit for '(?:jit[_(])?([\w.<>-]+)'()")),
    ("cache_miss", re.compile(r"^PERSISTENT COMPILATION CACHE MISS for '(?:jit[_(])?([\w.<>-]+)'()")),
    ("not_written", re.compile(r"^Not writing persistent cache entry for '(?:jit[_(])?([\w.<>-]+)'()")),
]
_QUIET = re.compile(r"^(Finished |Compiling |Persistent compilation cache|Not writing persistent cache)")


class CompileLog(logging.Handler):
    """Counts jax_log_compiles records per (function, stage); echoes only the kernel's own records.

    The bulk of jax_log_compiles output (eager primitives, nested jits inside the trace) is swallowed;
    any other JAX warning is passed through to stderr.
    """

    def __init__(self, echo=KERNEL_FUNS):
        super().__init__(logging.DEBUG)
        self.echo = set(echo)
        self.counts = defaultdict(int)       # (fun, stage) -> n
        self.seconds = defaultdict(float)    # (fun, stage) -> total s

    def emit(self, record):
        msg = record.getMessage()
        for stage, pat in _PATTERNS:
            m = pat.match(msg)
            if m:
                fun = m.group(1)
                self.counts[fun, stage] += 1
                if m.group(2):
                    self.seconds[fun, stage] += float(m.group(2))
                if fun in self.echo:
                    if stage == "not_written":       # JAX's reason is the point of this record
                        print(f"    [jax] {stage:9s} {msg}")
                    else:
                        print(f"    [jax] {stage:9s} {fun}"
                              + (f" {float(m.group(2)):.2f} s" if m.group(2) else ""))
                return
        if record.levelno < logging.WARNING:         # debug chatter from jax._src.compiler (T4 mode)
            return
        if not _QUIET.match(msg):
            print(f"[{record.name}] {msg}", file=sys.stderr)

    def snapshot(self):
        return dict(self.counts)


def signature(ev):
    """Abstract signature of an EventData: what decides whether the kernel recompiles."""
    return jax.tree.map(lambda a: (a.shape, str(a.dtype), bool(getattr(a, "weak_type", False))), ev)


def event_summary(res):
    """What must be bit-identical between two runs of the same event and seed on one machine.

    `log_Z_error` is left out: anesthetic draws its ln Z samples from numpy's unseeded global RNG.
    """
    post = np.ascontiguousarray(res["posterior_samples"], dtype=np.float64)
    return {"log_Z": float(res["log_Z"]),                    # JSON keeps the full float64 repr
            "n_iterations": int(res["n_iterations"]),
            "n_likelihood_evaluations": int(res["n_likelihood_evaluations"]),
            "n_posterior": int(post.shape[0]),
            "posterior_sha256": hashlib.sha256(post.tobytes()).hexdigest()}


def compare_summaries(summary, path):
    """Exact comparison against an earlier run's --summary-out, over the events both ran."""
    with open(path) as f:
        ref = json.load(f)["events"]
    common = sorted(set(summary) & set(ref), key=int)
    print(f"\n=== exact comparison with {path} (events {common})")
    if not common:
        print("  FAIL: no events in common")
        return False
    ok = True
    for eid in common:
        diffs = {k: (summary[eid][k], ref[eid].get(k)) for k in summary[eid] if summary[eid][k] != ref[eid].get(k)}
        ok &= not diffs
        print(f"  event {eid}: " + ("identical" if not diffs else
                                   "FAIL: " + "; ".join(f"{k} {a!r} vs {b!r}" for k, (a, b) in diffs.items())))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--events", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-live", type=int, default=256)
    ap.add_argument("--steps-per-dim", type=int, default=5)
    ap.add_argument("--delete-frac", type=float, default=0.5)
    ap.add_argument("--termination-dlogz", type=float, default=0.5)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--all-params", action="store_true", help="12-parameter layout (T7); default intrinsic only")
    ap.add_argument("--no-log-compiles", action="store_true", help="skip jax_log_compiles (T7 timing runs)")
    ap.add_argument("--explain-cache-misses", action="store_true")
    ap.add_argument("--cache-dir", help="persistent compilation cache directory (T4)")
    ap.add_argument("--expect-cache", choices=["miss", "hit"],
                    help="T4 pass condition for ns_init/ns_step: 'miss' = compiled and written (empty dir), "
                         "'hit' = loaded from --cache-dir")
    ap.add_argument("--summary-out", help="write per-event ln Z, counts and posterior hash to this JSON")
    ap.add_argument("--compare-to", help="earlier --summary-out JSON; results must match it exactly")
    ap.add_argument("--export-out", help="after sampling, export init/step to this new directory (T5)")
    ap.add_argument("--load-export", help="load init/step from this export instead of tracing them (T5)")
    ap.add_argument("--no-donate", action="store_true",
                    help="with --load-export: don't donate the state buffers to the loaded step")
    ap.add_argument("--git-hash", help="recorded in the export manifest (informational)")
    args = ap.parse_args()
    if args.export_out and args.load_export:
        ap.error("--export-out and --load-export are exclusive")
    if args.expect_cache and not args.cache_dir:
        ap.error("--expect-cache needs --cache-dir")
    if args.expect_cache and args.no_log_compiles:
        ap.error("--expect-cache needs the jax_log_compiles records; drop --no-log-compiles")

    cache_dir = enable_persistent_cache(args.cache_dir) if args.cache_dir else None
    test = "T5" if (args.export_out or args.load_export) else "T4" if args.expect_cache else "T2"
    kernel_funs = EXPORTED_FUNS if args.load_export else KERNEL_FUNS

    intrinsic_only = not args.all_params
    log = None
    if not args.no_log_compiles:
        jax.config.update("jax_log_compiles", True)
        log = CompileLog(echo=TRACED_FUNS + EXPORTED_FUNS)
        jax_logger = logging.getLogger("jax")
        jax_logger.addHandler(log)
        jax_logger.setLevel(logging.WARNING)
        jax_logger.propagate = False
        if cache_dir is not None:
            # Cache misses and "Not writing ..." are DEBUG unless jax_explain_cache_misses is on (which
            # also floods tracing-cache explanations). Route this one module's records, DEBUG included,
            # only to our handler, which counts the cache records and drops the rest.
            compiler_logger = logging.getLogger("jax._src.compiler")
            compiler_logger.addHandler(log)
            compiler_logger.setLevel(logging.DEBUG)
            compiler_logger.propagate = False
    if args.explain_cache_misses:
        jax.config.update("jax_explain_cache_misses", True)
        if log is not None:
            logging.getLogger("jax").propagate = True      # let the explanations reach stderr
    # CPU has no buffer donation; the step still works, JAX just warns on every call.
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

    settings = SamplerSettings(n_live=args.n_live, delete_frac=args.delete_frac,
                               steps_per_dim=args.steps_per_dim, termination_dlogz=args.termination_dlogz)
    rows = read_catalogue(args.catalogue)
    waveform = make_waveform()
    model = EventModel(waveform, intrinsic_only)
    blocks = gibbs_blocks(intrinsic_only)
    layout = "4p" if intrinsic_only else "12p"
    print(f"{test}: {layout}, events {args.events}, {settings}, n_bins {args.n_bins}, "
          f"backend {jax.default_backend()}, devices {jax.devices()}, jax {jax.__version__}")
    if cache_dir is not None:
        n_entries = sum(1 for p in cache_dir.iterdir() if p.is_file())
        print(f"persistent cache: {cache_dir} ({n_entries} files at start), expect {args.expect_cache or '-'}")
        if args.expect_cache == "miss" and n_entries and not args.load_export:
            print("  WARN: cache dir is not empty; ns_init/ns_step may hit instead of miss")

    init = step = rebuild = sig0 = None
    results, ok = [], True
    for eid in args.events:
        row = rows[eid]
        t0 = time.perf_counter()
        inj = inject(row, waveform)
        lik = make_likelihood(inj, waveform, n_bins=args.n_bins,
                              fixed_names=FIXED_PARAMS if intrinsic_only else ())
        ev = event_data_from_likelihood(lik, inj.mc_lo, inj.mc_hi, inj.params)
        key = jax.random.key(row["sampler_seed"])
        prior_key, sampler_key = jax.random.split(key)
        draws = jimgw_prior(inj.mc_lo, inj.mc_hi, intrinsic_only).sample(prior_key, settings.n_live)
        positions = jnp.stack([draws[k] for k in model.names]).T
        t_setup = time.perf_counter() - t0
        snr = float(np.sqrt(sum(float(ifo.optimal_snr) ** 2 for ifo in inj.ifos)))
        print(f"\n=== event {eid}: d_L={row['d_L']:.0f} Mpc duration={inj.duration:.0f} s "
              f"f_max_lik={inj.f_max_likelihood:.0f} Hz SNR={snr:.1f} [host setup {t_setup:.1f} s]")

        if init is None:
            rebuild = rebuild_required_by_block(model, lik, blocks)
            print(f"  rebuild_required_by_block: {rebuild}")
            if intrinsic_only and rebuild != {(0, 1, 2, 3): True}:
                print("  FAIL: unexpected rebuild map for the intrinsic-only block")
                ok = False
            if args.load_export:
                t0 = time.perf_counter()
                try:
                    init, step = load_sampler(args.load_export, model, settings, rebuild, ev,
                                              donate=not args.no_donate)
                except ExportMismatch as e:
                    print(f"  FAIL: {e}")
                    raise SystemExit(1)
                info = init.manifest["info"]
                print(f"  loaded export {args.load_export} (donate {not args.no_donate}) in {time.perf_counter() - t0:.2f} s "
                      f"(made {info['created']} on {info['host']}, git {info['git_hash']})")
            else:
                init, step = make_sampler(model, settings, rebuild)
            sig0 = signature(ev)
        else:
            # rebuild map is event-independent; check it rather than trust it
            r = rebuild_required_by_block(model, lik, blocks)
            if r != rebuild:
                print(f"  FAIL: rebuild map differs from the first event's: {r}")
                ok = False
            if signature(ev) != sig0:
                print(f"  WARN: EventData signature differs from the first event's (expect a recompile):\n"
                      f"    {signature(ev)}\n    vs {sig0}")

        before = log.snapshot() if log else {}
        t0 = time.perf_counter()
        res = run_event(init, step, ev, positions, sampler_key, settings)
        t_run = time.perf_counter() - t0
        after = log.snapshot() if log else {}
        new = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
        kernel_new = {f"{f}.{s}": n for (f, s), n in new.items() if f in kernel_funs}
        other_new = sum(n for (f, s), n in new.items() if f not in kernel_funs and s == "compile")

        finite = np.isfinite(res["log_Z"]) and np.isfinite(res["log_Z_error"])
        ok &= bool(finite)
        print(f"  ln Z = {res['log_Z']:.3f} ± {res['log_Z_error']:.3f}  N_iter={res['n_iterations']}  "
              f"evals={res['n_likelihood_evaluations']:,}  posterior samples={len(res['posterior_samples'])}"
              f"{'' if finite else '  FAIL: non-finite'}")
        print(f"  wall {t_run:.1f} s = init {res['t_init']:.1f} + first step {res['t_first_step']:.1f} "
              f"+ rest {res['t_steps']:.1f} + finalise {res['t_finalise']:.1f}")
        if log:
            print(f"  new kernel trace/lower/compile records: {kernel_new or 'none'}; "
                  f"other XLA compiles this event: {other_new}")
        results.append((eid, res, t_setup, t_run))
        last_ev, last_positions = ev, positions

    print("\n=== summary")
    print(f"  {'event':>5} {'setup s':>8} {'wall s':>8} {'init s':>7} {'1st step s':>10} "
          f"{'s/iter after':>12} {'N_iter':>6} {'ln Z':>9} {'σ':>6}")
    for eid, res, t_setup, t_run in results:
        per_iter = res["t_steps"] / max(res["n_iterations"] - 1, 1)
        print(f"  {eid:>5} {t_setup:8.1f} {t_run:8.1f} {res['t_init']:7.2f} {res['t_first_step']:10.2f} "
              f"{per_iter:12.3f} {res['n_iterations']:6d} {res['log_Z']:9.3f} {res['log_Z_error']:6.3f}")

    print("\n=== compile counts over all events (each must be 1)")
    for name, fn in zip(kernel_funs, (init, step)):
        n_cache = getattr(fn, "jitted", fn)._cache_size()
        line = f"  {name}: jit cache size {n_cache}"
        ok &= n_cache == 1
        if log:
            # with a persistent-cache hit, the "compile" record times the disk load instead
            for stage in ("trace", "lower", "compile"):
                n = log.counts.get((name, stage), 0)
                ok &= n == 1
                line += f", {stage} {n} ({log.seconds.get((name, stage), 0.0):.1f} s)"
        print(line)
    if log:
        n_other = sum(n for (f, s), n in log.counts.items() if f not in kernel_funs and s == "compile")
        print(f"  other XLA compiles (eager ops, finalise, prior draws; informational): {n_other}")
        if args.load_export:
            traced = {f: log.counts.get((f, "trace"), 0) for f in TRACED_FUNS}
            bad = any(traced.values())
            ok &= not bad
            print(f"  traces of the exported functions (must all be 0): {traced}" + ("  FAIL" if bad else ""))

    if cache_dir is not None and log:
        want = {"miss": {"cache_hit": 0, "cache_miss": 1, "not_written": 0},
                "hit": {"cache_hit": 1, "cache_miss": 0, "not_written": 0}}.get(args.expect_cache)
        print(f"\n=== persistent cache records (expect {args.expect_cache or '-'})")
        for name in kernel_funs:
            got = {s: log.counts.get((name, s), 0) for s in ("cache_hit", "cache_miss", "not_written")}
            bad = want is not None and got != want
            ok &= not bad
            print(f"  {name}: hit {got['cache_hit']}, miss {got['cache_miss']}, not written "
                  f"{got['not_written']}" + (f"  FAIL: expected {want}" if bad else ""))
        hits = sum(n for (f, s), n in log.counts.items() if f not in kernel_funs and s == "cache_hit")
        misses = sum(n for (f, s), n in log.counts.items() if f not in kernel_funs and s == "cache_miss")
        print(f"  other functions (informational): {hits} hits, {misses} misses")
        n_entries = sum(1 for p in cache_dir.iterdir() if p.is_file())
        print(f"  {cache_dir}: {n_entries} files at end")

    if args.export_out:
        print(f"\n=== export to {args.export_out}")
        before = log.snapshot() if log else {}
        try:
            t = export_sampler(args.export_out, init, step, model, settings, rebuild, last_ev,
                               jnp.asarray(last_positions), git_hash=args.git_hash)
            print(f"  exported in {t['t_export_init']:.1f} s (init) + {t['t_export_step']:.1f} s (step), "
                  f"{t['bytes'] / 1e6:.1f} MB")
        except Exception as e:      # report and fail the test, but still write the summary below
            print(f"  FAIL: {type(e).__name__}: {e}")
            ok = False
        if log:
            after = log.snapshot()
            new = {f"{f}.{s}": after[f, s] - before.get((f, s), 0) for f, s in after
                   if f in TRACED_FUNS and after[f, s] != before.get((f, s), 0)}
            print(f"  trace/lower records during export (informational): {new or 'none'}")

    summary = {str(eid): event_summary(res) for eid, res, _, _ in results}
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump({"events": summary, "settings": repr(settings), "n_bins": args.n_bins,
                       "layout": layout, "jax": jax.__version__,
                       "mode": "load-export" if args.load_export else "export" if args.export_out else "trace"}, f, indent=1)
        print(f"\nwrote {args.summary_out}")
    if args.compare_to:
        ok &= compare_summaries(summary, args.compare_to)

    print(f"\n{test} {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
