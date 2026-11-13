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
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
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
from kernel import SamplerSettings, make_sampler, rebuild_required_by_block, run_event

KERNEL_FUNS = ("ns_init", "ns_step")
STAGES = ("trace", "lower", "compile", "cache_hit")
_PATTERNS = [
    ("trace", re.compile(r"^Finished tracing \+ transforming (?:jit[_(])?([\w.<>-]+)\)? for pjit in ([\d.e-]+) sec")),
    ("lower", re.compile(r"^Finished jaxpr to MLIR module conversion (?:jit[_(])?([\w.<>-]+)\)? in ([\d.e-]+) sec")),
    ("compile", re.compile(r"^Finished XLA compilation of (?:jit[_(])?([\w.<>-]+)\)? in ([\d.e-]+) sec")),
    ("cache_hit", re.compile(r"^Persistent compilation cache hit for '(?:jit[_(])?([\w.<>-]+)'()")),
]
_QUIET = re.compile(r"^(Finished |Compiling |Persistent compilation cache|Not writing persistent cache)")


class CompileLog(logging.Handler):
    """Counts jax_log_compiles records per (function, stage); echoes only the kernel's own records.

    The bulk of jax_log_compiles output (eager primitives, nested jits inside the trace) is swallowed;
    any other JAX warning is passed through to stderr.
    """

    def __init__(self):
        super().__init__(logging.DEBUG)
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
                if fun in KERNEL_FUNS:
                    print(f"    [jax] {stage:9s} {fun}" + (f" {float(m.group(2)):.2f} s" if m.group(2) else ""))
                return
        if not _QUIET.match(msg):
            print(f"[{record.name}] {msg}", file=sys.stderr)

    def snapshot(self):
        return dict(self.counts)


def signature(ev):
    """Abstract signature of an EventData: what decides whether the kernel recompiles."""
    return jax.tree.map(lambda a: (a.shape, str(a.dtype), bool(getattr(a, "weak_type", False))), ev)


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
    args = ap.parse_args()

    intrinsic_only = not args.all_params
    log = None
    if not args.no_log_compiles:
        jax.config.update("jax_log_compiles", True)
        log = CompileLog()
        jax_logger = logging.getLogger("jax")
        jax_logger.addHandler(log)
        jax_logger.setLevel(logging.WARNING)
        jax_logger.propagate = False
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
    print(f"T2: {layout}, events {args.events}, {settings}, n_bins {args.n_bins}, "
          f"backend {jax.default_backend()}, devices {jax.devices()}")

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
        kernel_new = {f"{f}.{s}": n for (f, s), n in new.items() if f in KERNEL_FUNS}
        other_new = sum(n for (f, s), n in new.items() if f not in KERNEL_FUNS and s == "compile")

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

    print("\n=== summary")
    print(f"  {'event':>5} {'setup s':>8} {'wall s':>8} {'init s':>7} {'1st step s':>10} "
          f"{'s/iter after':>12} {'N_iter':>6} {'ln Z':>9} {'σ':>6}")
    for eid, res, t_setup, t_run in results:
        per_iter = res["t_steps"] / max(res["n_iterations"] - 1, 1)
        print(f"  {eid:>5} {t_setup:8.1f} {t_run:8.1f} {res['t_init']:7.2f} {res['t_first_step']:10.2f} "
              f"{per_iter:12.3f} {res['n_iterations']:6d} {res['log_Z']:9.3f} {res['log_Z_error']:6.3f}")

    print("\n=== compile counts over all events (each must be 1)")
    for name, fn in zip(KERNEL_FUNS, (init, step)):
        n_cache = fn._cache_size()
        line = f"  {name}: jit cache size {n_cache}"
        ok &= n_cache == 1
        if log:
            for stage in ("trace", "lower", "compile"):
                n = log.counts.get((name, stage), 0)
                ok &= n == 1
                line += f", {stage} {n} ({log.seconds.get((name, stage), 0.0):.1f} s)"
        print(line)
    if log:
        n_other = sum(n for (f, s), n in log.counts.items() if f not in KERNEL_FUNS and s == "compile")
        print(f"  other XLA compiles (eager ops, finalise, prior draws; informational): {n_other}")

    print(f"\nT2 {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
