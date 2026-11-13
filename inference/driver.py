"""Many-event driver: sample catalogue events with one compiled SwiG kernel per process (driver-plan.md, 2.5).

Task `i` of `N` takes catalogue rows `i::N` (within --event-range, if given) and loops over them:
build the injection and EventData on the host, sample with the compile-once `init`/`step`
(kernel.make_sampler, or export_kernel.load_sampler with --load-export), and write one results file
per event. Run one process per GPU.

Restarts: an event is done when `event_NNNNN.npz` exists (or `event_NNNNN.failed`, unless
--retry-failed). Files are written to `<name>.tmp` and renamed, so a killed process never leaves a
results file that looks finished; at start-up the task removes its own events' stale `.tmp` files.
There is no checkpoint inside an event: an interrupted event reruns from scratch, with the same seeds,
so it gives the same result.

Deadline: before each event the task estimates the event's cost from the events it has done so far
(--first-event-estimate before the first one) and exits cleanly, with status 0, if it would not
finish before the deadline minus --margin. The deadline is --deadline (ISO time, or +SECONDS from
start), else the Slurm job's end time from squeue, else none.

Exact-likelihood check: on events selected by a hash of event_id (fraction --check-frac), the
heterodyned logL of the max-likelihood and N_CHECK random posterior samples is compared with jimgw's
TransientLikelihoodFD (full frequency grid). That program closes over the event's data, so it compiles
afresh for every checked event.

Results file `event_NNNNN.npz` (np.load(..., allow_pickle=False)):
    meta                      JSON string: row, truths, ln Z, counts, SNRs, timings, check, settings,
                              versions, environment, compiled_fresh, ...  json.loads(str(f["meta"]))
    names                     sampled parameter names (column order of the sample arrays)
    truths                    injected values in sampling space
    posterior_samples         [n, d] equally weighted posterior (anesthetic posterior_points)
    posterior_log_likelihood  [n]   heterodyned logL
    posterior_log_prior       [n]   PE prior density (sampling space), for dividing it out
    weighted_samples          [m, d] dead points carrying all but WEIGHT_TAIL of the posterior mass
    weighted_log_weight       [m]   normalised log posterior weights (logsumexp = 0)
    weighted_log_likelihood   [m]
    weighted_log_prior        [m]

Failed event: `event_NNNNN.failed` (JSON: error, traceback, timings). The task continues with the next
event; it stops with status 1 after --max-consecutive-failures failures in a row.

Example (local, 4 intrinsic parameters, first 5 catalogue rows, one task):
    python driver.py --catalogue catalogue.csv --out-dir runs_t6 --layout 4p --n-live 256 \
        --event-range 0 5 --cache-dir jax_cache_t6

Test-only: DRIVER_TEST_PAUSE_BEFORE_RENAME=<s> sleeps between writing a `.tmp` and renaming it (T6).
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform

from event_model import FIXED_PARAMS, EventModel, gibbs_blocks
from export_kernel import ExportMismatch, load_sampler
from heterodyne import DEFAULT_N_BINS
from injection import F_MIN, make_waveform, read_catalogue
from kernel import SamplerSettings, enable_persistent_cache, make_sampler, rebuild_required_by_block, run_event
from prepare import prepare_event, sampling_truths
from run_io import package_versions

DRIVER_FORMAT = 1           # bump when the results file layout changes
N_CHECK, CHECK_TOL = 50, 0.1   # as swig.py
WEIGHT_TAIL = 1e-6          # posterior mass dropped from weighted_samples


# ---------------------------------------------------------------- files

def event_path(out_dir, eid, suffix):
    return Path(out_dir) / f"event_{eid:05d}.{suffix}"


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path, write):
    """Call `write(f)` on `<path>.tmp`, fsync it, rename it to `path`. `path` is complete or absent."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    pause = float(os.environ.get("DRIVER_TEST_PAUSE_BEFORE_RENAME", 0) or 0)
    if pause > 0:
        print(f"  [test] wrote {tmp.name}; sleeping {pause:.0f} s before the rename", flush=True)
        time.sleep(pause)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def write_json_atomic(path, obj):
    write_atomic(path, lambda f: f.write(json.dumps(obj, indent=1, default=_json_default).encode()))


def _json_default(x):
    if isinstance(x, (np.generic, np.ndarray)):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def check_run_config(out_dir, config):
    """All tasks writing to one out_dir must use the same sampler/program configuration."""
    path = Path(out_dir) / "driver_config.json"
    if path.exists():
        old = json.loads(path.read_text())
        diffs = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        if diffs:
            raise SystemExit(f"{path} has a different configuration; use another --out-dir.\n"
                             + "\n".join(f"  {k}: {a!r} (dir) vs {b!r} (this run)" for k, (a, b) in diffs.items()))
    else:
        # pid-unique tmp: tasks starting together may race here; they write identical content
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(config, indent=1))
        os.replace(tmp, path)


# ---------------------------------------------------------------- deadline

def parse_deadline(arg, t_start):
    """--deadline: '+SECONDS' (from start) or an ISO time (local time if no zone). Returns epoch seconds."""
    if arg.startswith("+"):
        return t_start + float(arg[1:])
    return datetime.fromisoformat(arg).timestamp()


def slurm_end_time():
    """The Slurm job's end time (epoch s) from squeue, or None outside Slurm / if squeue fails."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True,
                             timeout=30, check=True).stdout.strip()
        return datetime.fromisoformat(out).timestamp()      # squeue prints local time
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"WARN: could not read the Slurm end time ({type(e).__name__}: {e}); no deadline")
        return None


# ---------------------------------------------------------------- per-event pieces

def check_selected(eid, frac):
    """Deterministic ~frac selection of events for the exact-likelihood check."""
    h = int.from_bytes(hashlib.sha256(f"event_id={eid}".encode()).digest()[:8], "big")
    return h < frac * 2 ** 64


def exact_check(prep, model, waveform, samples, het_logl):
    """max |heterodyned - exact logL| at the max-logL sample and N_CHECK random posterior samples."""
    inj = prep.inj
    exact = TransientLikelihoodFD(
        inj.ifos,
        waveform=waveform,
        fixed_parameters={k: inj.params[k] for k in (FIXED_PARAMS if model.intrinsic_only else ())},
        trigger_time=inj.gps,
        f_min=F_MIN,
        f_max=inj.f_max_likelihood,
        phase_marginalization=True,
    )
    names = model.names
    exact_logl = jax.jit(lambda x: exact.evaluate(MassRatioToSymmetricMassRatioTransform.forward(
        {k: x[i] for i, k in enumerate(names)})))
    n = len(het_logl)
    rng = np.random.default_rng(prep.row["event_id"])
    idx = np.r_[np.argmax(het_logl), rng.choice(n, min(N_CHECK, n), replace=False)]
    ex = np.array([float(exact_logl(jnp.asarray(samples[i]))) for i in idx])
    d = het_logl[idx] - ex
    max_abs = float(np.max(np.abs(d)))
    return {"n": int(len(idx)), "max_abs_dlogl": max_abs, "mean_dlogl": float(np.mean(d)),
            "passed": bool(max_abs <= CHECK_TOL), "tol": CHECK_TOL}


def weighted_posterior(ns, n_dims):
    """Dead points carrying all but WEIGHT_TAIL of the posterior mass, with normalised log weights."""
    w = np.asarray(ns.get_weights(), dtype=np.float64)
    w = w / w.sum()
    order = np.argsort(w)[::-1]
    k = int(np.searchsorted(np.cumsum(w[order]), 1.0 - WEIGHT_TAIL)) + 1
    keep = np.sort(order[:min(k, len(w))])
    keep = keep[w[keep] > 0]
    w_keep = w[keep]
    with np.errstate(divide="ignore"):
        logw = np.log(w_keep) - np.log(w_keep.sum())
    return (np.asarray(ns.iloc[keep, :n_dims], dtype=np.float64), logw,
            np.asarray(ns["logL"].iloc[keep], dtype=np.float64))


def cache_sizes(*fns):
    return tuple(getattr(f, "jitted", f)._cache_size() for f in fns)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layout", choices=["4p", "12p"], required=True,
                    help="4p: intrinsic only, extrinsics fixed at injected values (local tests); 12p: production")
    ap.add_argument("--n-live", type=int, required=True)
    ap.add_argument("--steps-per-dim", type=int, default=5)
    ap.add_argument("--delete-frac", type=float, default=0.5)
    ap.add_argument("--termination-dlogz", type=float, default=0.5)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                    help="default: $SLURM_ARRAY_TASK_ID, else 0")
    ap.add_argument("--task-count", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
                    help="default: $SLURM_ARRAY_TASK_COUNT, else 1")
    ap.add_argument("--event-range", type=int, nargs=2, metavar=("START", "STOP"),
                    help="only event_ids in [START, STOP); default: the whole catalogue")
    ap.add_argument("--cache-dir", help="JAX persistent compilation cache (section 3a)")
    ap.add_argument("--load-export", help="load init/step from this jax.export directory (section 3b); "
                                          "falls back to tracing if its manifest does not match")
    ap.add_argument("--check-frac", type=float, default=0.05, help="fraction of events given the exact check")
    ap.add_argument("--deadline", help="ISO time or +SECONDS; default: the Slurm job's end time")
    ap.add_argument("--margin", type=float, default=120.0, help="seconds kept free before the deadline")
    ap.add_argument("--first-event-estimate", type=float, default=900.0,
                    help="seconds assumed for the first event (setup + compile + sampling)")
    ap.add_argument("--check-estimate", type=float, default=600.0,
                    help="seconds assumed for the exact check until one has been timed")
    ap.add_argument("--retry-failed", action="store_true", help="rerun events with a .failed marker")
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--git-hash", help="recorded in each results file (informational)")
    args = ap.parse_args()
    if not 0 <= args.task_index < args.task_count:
        ap.error(f"task index {args.task_index} not in [0, {args.task_count})")

    t_start = time.time()
    deadline = parse_deadline(args.deadline, t_start) if args.deadline else slurm_end_time()
    cache_dir = enable_persistent_cache(args.cache_dir) if args.cache_dir else None
    # CPU has no buffer donation; the step still works, JAX just warns on every call.
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_only = args.layout == "4p"
    settings = SamplerSettings(n_live=args.n_live, delete_frac=args.delete_frac,
                               steps_per_dim=args.steps_per_dim, termination_dlogz=args.termination_dlogz)
    waveform = make_waveform()
    model = EventModel(waveform, intrinsic_only)
    blocks = gibbs_blocks(intrinsic_only)
    run_config = {"driver_format": DRIVER_FORMAT, "layout": args.layout, "names": list(model.names),
                  "settings": asdict(settings), "n_bins": args.n_bins,
                  "blocks": [list(b) for b in blocks], "waveform": type(waveform).__name__,
                  "f_ref": getattr(waveform, "f_ref", None)}
    check_run_config(out_dir, run_config)

    rows = read_catalogue(args.catalogue)
    ids = range(len(rows)) if args.event_range is None else range(max(args.event_range[0], 0),
                                                                    min(args.event_range[1], len(rows)))
    mine = list(ids)[args.task_index::args.task_count]

    # stale temporaries of *this task's* events only (another task may be mid-write on its own)
    for eid in mine:
        for suffix in ("npz", "failed"):
            tmp = event_path(out_dir, eid, suffix + ".tmp")
            if tmp.exists():
                print(f"removing stale {tmp.name}")
                tmp.unlink()

    def is_done(eid):
        return event_path(out_dir, eid, "npz").exists() or (
            not args.retry_failed and event_path(out_dir, eid, "failed").exists())
    todo = [eid for eid in mine if not is_done(eid)]

    env = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                          "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS")}
    env.update(hostname=socket.gethostname(), machine=platform.machine(), backend=jax.default_backend(),
               device_kind=jax.devices()[0].device_kind, pid=os.getpid())
    versions = package_versions()
    print(f"driver: task {args.task_index}/{args.task_count}, {args.layout}, {settings}, n_bins {args.n_bins}")
    print(f"  backend {env['backend']} ({env['device_kind']}), jax {jax.__version__}, host {env['hostname']}, "
          f"pid {env['pid']}")
    print(f"  events: {len(mine)} for this task, {len(mine) - len(todo)} already done, {len(todo)} to run")
    print(f"  out {out_dir.resolve()}, cache {cache_dir or '-'}, export {args.load_export or '-'}")
    print(f"  deadline: " + (datetime.fromtimestamp(deadline).isoformat(timespec='seconds') +
                             f" ({(deadline - t_start) / 60:.1f} min from now), margin {args.margin:.0f} s"
                             if deadline else "none"), flush=True)

    log_prior = jax.jit(jax.vmap(model.log_prior, in_axes=(0, None)))    # per-sample PE prior density
    init = step = rebuild = None
    sampler_mode = None
    t_event_max = None              # largest (setup + sampling + write) so far, excluding the check
    t_check_max = None
    n_ok = n_failed = consecutive_failures = 0
    stopped_for_deadline = False

    for eid in todo:
        will_check = check_selected(eid, args.check_frac)
        if deadline is not None:
            est = args.first_event_estimate if t_event_max is None else 1.2 * t_event_max
            if will_check:
                est += args.check_estimate if t_check_max is None else 1.2 * t_check_max
            left = deadline - time.time() - args.margin
            if est > left:
                print(f"\nstopping before event {eid}: needs ~{est:.0f} s, {max(left, 0):.0f} s left "
                      f"before the deadline minus margin", flush=True)
                stopped_for_deadline = True
                break

        row = rows[eid]
        t_event0 = time.perf_counter()
        timings = {}
        try:
            prep = prepare_event(row, waveform, model, settings.n_live, args.n_bins)
            timings["setup"] = prep.t_setup
            print(f"\n=== event {eid}: d_L={row['d_L']:.0f} Mpc duration={prep.inj.duration:.0f} s "
                  f"f_max_lik={prep.inj.f_max_likelihood:.0f} Hz SNR={prep.snr['network']:.1f} "
                  f"[setup {prep.t_setup:.1f} s]", flush=True)

            if init is None:
                t0 = time.perf_counter()
                rebuild = rebuild_required_by_block(model, prep.likelihood, blocks)
                if args.load_export:
                    try:
                        init, step = load_sampler(args.load_export, model, settings, rebuild, prep.ev)
                        sampler_mode = "export"
                        info = init.manifest["info"]
                        print(f"  loaded export {args.load_export} (made {info['created']} on {info['host']}, "
                              f"git {info['git_hash']})")
                    except ExportMismatch as e:
                        print(f"  WARN: {e}\n  falling back to tracing the sampler")
                if init is None:
                    init, step = make_sampler(model, settings, rebuild)
                    sampler_mode = "trace"
                timings["make_sampler"] = time.perf_counter() - t0
            else:
                r = rebuild_required_by_block(model, prep.likelihood, blocks)
                if r != rebuild:
                    raise RuntimeError(f"rebuild map {r} differs from the first event's {rebuild}")

            before = cache_sizes(init, step)
            res = run_event(init, step, prep.ev, prep.positions, prep.sampler_key, settings)
            compiled_fresh = cache_sizes(init, step) != before
            for k in ("init", "first_step", "steps", "finalise"):
                timings[k] = res[f"t_{k}"]

            t0 = time.perf_counter()
            ns = res["nested_samples"]
            post = np.ascontiguousarray(res["posterior_samples"], dtype=np.float64)
            post_logl = np.asarray(res["posterior_log_likelihood"], dtype=np.float64)
            w_samples, w_logw, w_logl = weighted_posterior(ns, model.n_dims)
            post_logp = np.asarray(log_prior(jnp.asarray(post), prep.ev))
            w_logp = np.asarray(log_prior(jnp.asarray(w_samples), prep.ev))
            timings["postprocess"] = time.perf_counter() - t0

            check = None
            if will_check:
                t0 = time.perf_counter()
                check = exact_check(prep, model, waveform, post, post_logl)
                timings["check"] = time.perf_counter() - t0
                t_check_max = max(t_check_max or 0.0, timings["check"])

            truths = sampling_truths(prep.inj, model.names)
            meta = {
                "driver_format": DRIVER_FORMAT,
                "event_id": eid,
                "row": row,
                "names": list(model.names),
                "truths": dict(zip(model.names, truths.tolist())),
                "log_Z": res["log_Z"],
                "log_Z_error": res["log_Z_error"],
                "n_iterations": res["n_iterations"],
                "n_likelihood_evaluations": res["n_likelihood_evaluations"],
                "n_likelihood_evaluations_stepping_out": res["n_likelihood_evaluations_stepping_out"],
                "n_likelihood_evaluations_shrinking": res["n_likelihood_evaluations_shrinking"],
                "n_posterior": int(post.shape[0]),
                "n_weighted": int(w_samples.shape[0]),
                "n_eff_kish": float(1.0 / np.sum(np.exp(2 * w_logw))),
                "snr": prep.snr,
                "duration": prep.inj.duration,
                "f_max_likelihood": prep.inj.f_max_likelihood,
                "mc_prior": [prep.inj.mc_lo, prep.inj.mc_hi],
                "exact_check": check,
                "compiled_fresh": bool(compiled_fresh),
                "sampler_mode": sampler_mode,
                "timings": timings,
                "run_config": run_config,
                "versions": versions,
                "environment": env,
                "git_hash": args.git_hash,
                "task": [args.task_index, args.task_count],
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            arrays = {
                "names": np.array(model.names),
                "truths": truths,
                "posterior_samples": post,
                "posterior_log_likelihood": post_logl,
                "posterior_log_prior": post_logp,
                "weighted_samples": w_samples,
                "weighted_log_weight": w_logw,
                "weighted_log_likelihood": w_logl,
                "weighted_log_prior": w_logp,
            }
            timings["total_before_write"] = time.perf_counter() - t_event0
            t0 = time.perf_counter()
            write_atomic(event_path(out_dir, eid, "npz"),
                         lambda f: np.savez(f, meta=np.array(json.dumps(meta, default=_json_default)), **arrays))
            t_write = time.perf_counter() - t0
            event_path(out_dir, eid, "failed").unlink(missing_ok=True)     # a successful retry

            t_event_max = max(t_event_max or 0.0,
                              time.perf_counter() - t_event0 - timings.get("check", 0.0))
            n_ok += 1
            consecutive_failures = 0
            chk = "" if check is None else (f"  check max|dlogL|={check['max_abs_dlogl']:.3g} "
                                            + ("ok" if check["passed"] else "FAIL"))
            print(f"  ln Z = {res['log_Z']:.3f} ± {res['log_Z_error']:.3f}  N_iter={res['n_iterations']}  "
                  f"posterior {post.shape[0]} (weighted {w_samples.shape[0]}, Kish n_eff "
                  f"{meta['n_eff_kish']:.0f}){chk}")
            print(f"  total {time.perf_counter() - t_event0:.1f} s = setup {timings['setup']:.1f} + init "
                  f"{timings['init']:.1f} + 1st step {timings['first_step']:.1f} + rest {timings['steps']:.1f}"
                  + (f" + check {timings['check']:.1f}" if check else "")
                  + f" + write {t_write:.1f}; compiled_fresh={compiled_fresh}", flush=True)

        except Exception as e:
            n_failed += 1
            consecutive_failures += 1
            tb = traceback.format_exc()
            print(f"  FAILED event {eid}: {type(e).__name__}: {e}\n{tb}", flush=True)
            write_json_atomic(event_path(out_dir, eid, "failed"), {
                "event_id": eid, "error": f"{type(e).__name__}: {e}", "traceback": tb,
                "timings": timings, "environment": env, "git_hash": args.git_hash,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\n{consecutive_failures} consecutive failures; giving up", flush=True)
                break

    remaining = sum(1 for eid in mine if not is_done(eid))
    print(f"\ndone: {n_ok} ok, {n_failed} failed this run; {remaining} of this task's {len(mine)} events left"
          + ("; stopped for the deadline" if stopped_for_deadline else "")
          + f"; {(time.time() - t_start) / 60:.1f} min", flush=True)
    sys.exit(1 if consecutive_failures >= args.max_consecutive_failures else 0)


if __name__ == "__main__":
    main()
