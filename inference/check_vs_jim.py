"""Test T3 (driver-plan.md, section 4): the compile-once sampler (kernel.py) vs jimgw's `BlackJAXSwiGSampler`.

T1 checked single likelihood/prior values. This checks the *sampler*: that `ns_init`/`ns_step` are the
same algorithm as jimgw's (inner steps, n_delete, block covariances, rebuild map, termination).

Both sides get the same likelihood object, the same initial live points and the same PRNG key
(jimgw is driven through `jim.sampler.sample(key, positions)`, which is what `Jim.sample` calls).
They differ only in that event data are arguments here and compile-time constants in jimgw, so the
runs should track each other until floating-point differences flip a slice decision.

Parts:
- T3a, one iteration: jimgw-built initial state vs `ns_init`, then one jimgw step vs one `ns_step`
  from the same state with the same key (the first step key of the full runs). Pass if every state
  and info leaf agrees to `--step-tol` (relative to max(|x|, 1)).
- Full runs per seed, both sides. Pass if, over the seeds:
  - |mean ln Z difference| < 2 x its standard error (from the quoted σ's),
  - posterior medians differ by < 0.2 posterior sd (seed means),
  - N_iter and k̄ (evals per slice step per deleted point) agree within 10% (seed means).
  Also reported: per-seed values, how far the two runs' sorted dead-point logL sequences agree,
  each side's ln Z scatter across seeds vs its quoted σ (in the 4p setup this should be ~1).

Each run's dead points are saved to --out, so `--compare-only` recomputes the comparison without
sampling.

    python check_vs_jim.py --catalogue catalogue.csv --events 0 --seeds 0 1 2 --n-live 256
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import time
import warnings
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from anesthetic.samples import NestedSamples
from blackjax.ns.adaptive import init as _ns_adaptive_init
from blackjax.ns.base import init_state_strategy as _init_state_strategy
from jimgw.core.jim import Jim
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.config import BlackJAXSwiGConfig

from check_event_model import jimgw_prior
from event_model import FIXED_PARAMS, EventModel, event_data_from_likelihood, gibbs_blocks
from heterodyne import DEFAULT_N_BINS
from injection import inject, make_likelihood, make_waveform, read_catalogue
from kernel import SamplerSettings, make_sampler, rebuild_required_by_block, run_event

SIDES = ("ours", "jimgw")
LNZ_NSIGMA = 2.0
MEDIAN_TOL_SD = 0.2
REL_TOL_ITER = 0.10
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "runs" / "t3"


# ------------------------------------------------------------------ jimgw side

def make_jim(likelihood, prior, settings, blocks):
    """Jim with the same sampler settings as `settings`; no checkpoint dir, so no cache redirect."""
    config = BlackJAXSwiGConfig(
        blocks=blocks,
        n_live=settings.n_live,
        n_delete_frac=settings.delete_frac,
        num_inner_steps_per_dim=settings.steps_per_dim,
        num_gibbs_sweeps=settings.num_gibbs_sweeps,
        termination_dlogz=settings.termination_dlogz,
        n_devices=1,
        checkpoint_dir=None,
        checkpoint_interval=0,
    )
    return Jim(likelihood, prior, likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
               sampler_config=config, seed=0)   # seed unused: keys are passed explicitly


def run_jimgw(jim, positions, key):
    """jimgw's full NSS loop (BlackJAXNSSSampler._sample), including its own per-call compile."""
    t0 = time.perf_counter()
    jim.sampler.sample(key, positions)
    wall = time.perf_counter() - t0
    diag = jim.sampler._get_diagnostics()
    return {"nested_samples": jim.sampler._nested_samples, "n_iterations": jim.sampler._n_iterations,
            "n_likelihood_evaluations": diag["n_likelihood_evaluations"], "wall": wall}


def jimgw_init_and_step(jim, n_delete):
    """jimgw's initial-state builder (initialize_nested_sampler_state, mesh=None) and its step."""
    s = jim.sampler
    init_one = partial(_init_state_strategy, logprior_fn=s._log_prior_fn, loglikelihood_fn=s._log_likelihood_fn)

    def init(positions):
        return _ns_adaptive_init(
            positions,
            init_state_fn=lambda pos: jax.lax.map(init_one, pos, batch_size=n_delete),
            update_inner_kernel_params_fn=s._update_inner_kernel_params_fn,
        )
    return jax.jit(init), jax.jit(s._build_nested_sampler(n_delete, None).step)


# ------------------------------------------------------------------ T3a: one iteration

def _host(x):
    if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
        x = jax.random.key_data(x)
    x = np.asarray(x)
    return x.astype(np.float64) if x.dtype == bool or np.issubdtype(x.dtype, np.integer) else x


def leaf_diffs(a, b):
    """[(path, max abs diff, max rel diff)] for pytrees of the same structure.

    Non-finite entries must match exactly (positions and values); rel is |Δ| / max(|b|, 1).
    """
    if jax.tree.structure(a) != jax.tree.structure(b):
        raise ValueError(f"pytree structures differ:\n  {jax.tree.structure(a)}\n  {jax.tree.structure(b)}")
    out = []
    for (path, x), y in zip(jax.tree_util.tree_flatten_with_path(a)[0], jax.tree.leaves(b)):
        x, y = _host(x), _host(y)
        name = jax.tree_util.keystr(path)
        fx, fy = np.isfinite(x), np.isfinite(y)
        if x.shape != y.shape or not np.array_equal(fx, fy) or not np.array_equal(x[~fx], y[~fy], equal_nan=True):
            out.append((name, np.inf, np.inf))
            continue
        d = np.abs(x[fx] - y[fy])
        rel = d / np.maximum(np.abs(y[fy]), 1.0)
        out.append((name, float(d.max(initial=0.0)), float(rel.max(initial=0.0))))
    return out


def report_diffs(label, diffs, tol):
    worst = max((r for _, _, r in diffs), default=0.0)
    ok = worst < tol
    print(f"  {label}: {len(diffs)} leaves, max rel |Δ| = {worst:.3g}  {'ok' if ok else 'FAIL'}")
    for name, d, r in diffs:
        if r > 0:
            print(f"      {name:60s} abs {d:.3g}  rel {r:.3g}{'  <-- over tol' if r >= tol else ''}")
    return ok


def check_one_iteration(jim, init, step, ev, positions, step_key, n_delete, tol):
    jim_init, jim_step = jimgw_init_and_step(jim, n_delete)
    st_jim = jax.block_until_ready(jim_init(positions))
    st_ours = jax.block_until_ready(init(positions, ev))
    ok = report_diffs("initial state (ns_init vs jimgw)", leaf_diffs(st_ours, st_jim), tol)

    # Both steps start from ns_init's state, so this isolates the step. Ours donates its state: copy it.
    st_copy = jax.tree.map(jnp.copy, st_ours)
    new_jim, info_jim = jax.block_until_ready(jim_step(step_key, st_ours))
    new_ours, info_ours = jax.block_until_ready(step(step_key, st_copy, ev))
    ok &= report_diffs("state after one step", leaf_diffs(new_ours, new_jim), tol)
    ok &= report_diffs("step info (dead points, slice counts)", leaf_diffs(info_ours, info_jim), tol)
    return ok


# ------------------------------------------------------------------ runs: save, load, summarise

def run_path(out, eid, seed, side):
    return Path(out) / f"event{eid}_seed{seed}_{side}.npz"


def save_run(path, res, n_dims, wall, meta):
    ns = res["nested_samples"]
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp,
             position=ns.iloc[:, :n_dims].to_numpy(dtype=np.float64),
             logL=ns["logL"].to_numpy(dtype=np.float64),
             logL_birth=ns["logL_birth"].to_numpy(dtype=np.float64),
             n_iterations=res["n_iterations"],
             n_likelihood_evaluations=res["n_likelihood_evaluations"],
             wall=wall,
             meta=json.dumps(meta))
    tmp.replace(path)


def weighted_quantiles(x, w, qs):
    order = np.argsort(x)
    c = np.cumsum(w[order])
    c = (c - 0.5 * w[order]) / c[-1]
    return np.interp(qs, c, x[order])


def summarise(path):
    """Everything compared, recomputed from a saved run (ln Z error uses a fixed numpy seed)."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    ns = NestedSamples(z["position"], logL=z["logL"], logL_birth=z["logL_birth"],
                       logzero=np.nan, dtype=np.float64)
    np.random.seed(0)
    log_Z = float(np.asarray(ns.logZ()).item())
    log_Z_error = float(np.std(np.asarray(ns.logZ(nsamples=200))))
    w = np.asarray(ns.get_weights(), dtype=np.float64)
    x = ns.iloc[:, :meta["n_dims"]].to_numpy(dtype=np.float64)   # ns rows, matching w
    mean = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    q = np.array([weighted_quantiles(x[:, i], w, [0.05, 0.5, 0.95]) for i in range(x.shape[1])])
    n_iter = int(z["n_iterations"])
    n_evals = int(z["n_likelihood_evaluations"])
    return {
        "log_Z": log_Z, "log_Z_error": log_Z_error,
        "mean": mean, "sd": sd, "q05": q[:, 0], "median": q[:, 1], "q95": q[:, 2],
        "n_iterations": n_iter, "n_evals": n_evals,
        "kbar": n_evals / (n_iter * meta["n_delete"] * meta["slice_steps_per_point"]),
        "ess": float(w.sum() ** 2 / (w ** 2).sum()),
        "sorted_logL": np.sort(z["logL"]),
        "wall": float(z["wall"]), "names": meta["names"],
    }


def leading_agreement(a, b, rtol=1e-6):
    n = min(len(a), len(b))
    agree = np.abs(a[:n] - b[:n]) <= rtol * np.maximum(1.0, np.abs(b[:n]))
    return (int(np.argmin(agree)) if not agree.all() else n), n


def compare_event(eid, seeds, out):
    runs = {side: [summarise(run_path(out, eid, s, side)) for s in seeds] for side in SIDES}
    names = runs["ours"][0]["names"]
    ok = True

    print(f"\n=== event {eid}: per seed")
    print(f"  {'seed':>4} {'side':>6} {'ln Z':>10} {'σ':>6} {'N_iter':>6} {'evals':>9} {'k̄':>5} "
          f"{'ESS':>6} {'wall s':>7}   dead logL agree")
    for i, s in enumerate(seeds):
        for side in SIDES:
            r = runs[side][i]
            agree = ""
            if side == "jimgw":
                k, n = leading_agreement(runs["ours"][i]["sorted_logL"], r["sorted_logL"])
                agree = f"{k}/{n}" + (" (identical)" if k == n and len(runs["ours"][i]["sorted_logL"]) == n
                                      and len(r["sorted_logL"]) == n else "")
            print(f"  {s:>4} {side:>6} {r['log_Z']:10.3f} {r['log_Z_error']:6.3f} {r['n_iterations']:6d} "
                  f"{r['n_evals']:9,d} {r['kbar']:5.2f} {r['ess']:6.0f} {r['wall']:7.1f}   {agree}")
        d = runs["ours"][i]["log_Z"] - runs["jimgw"][i]["log_Z"]
        sig = np.hypot(runs["ours"][i]["log_Z_error"], runs["jimgw"][i]["log_Z_error"])
        print(f"  {s:>4}  ΔlnZ = {d:+.3f} ({d / sig:+.1f} σ)")

    n = len(seeds)
    agg = {}
    for side in SIDES:
        lz = np.array([r["log_Z"] for r in runs[side]])
        sz = np.array([r["log_Z_error"] for r in runs[side]])
        agg[side] = {
            "log_Z": lz.mean(), "se": np.sqrt(np.mean(sz ** 2) / n),
            "scatter": lz.std(ddof=1) if n > 1 else np.nan, "sigma": np.sqrt(np.mean(sz ** 2)),
            "median": np.mean([r["median"] for r in runs[side]], axis=0),
            "sd": np.mean([r["sd"] for r in runs[side]], axis=0),
            "n_iterations": np.mean([r["n_iterations"] for r in runs[side]]),
            "kbar": np.mean([r["kbar"] for r in runs[side]]),
        }
    a, b = agg["ours"], agg["jimgw"]

    print(f"\n=== event {eid}: over seeds {seeds}")
    d, se = a["log_Z"] - b["log_Z"], np.hypot(a["se"], b["se"])
    passed = abs(d) < LNZ_NSIGMA * se
    ok &= passed
    print(f"  ln Z mean: ours {a['log_Z']:.3f} ± {a['se']:.3f}, jimgw {b['log_Z']:.3f} ± {b['se']:.3f}; "
          f"Δ = {d:+.3f} ({d / se:+.1f} σ)  {'ok' if passed else 'FAIL'}")
    for side in SIDES:
        g = agg[side]
        print(f"  ln Z scatter across seeds, {side:5s}: {g['scatter']:.3f} vs quoted σ {g['sigma']:.3f} "
              f"({g['scatter'] / g['sigma']:.1f}×; informational)")

    print(f"  posterior medians (seed means), |Δ| in units of posterior sd (tol {MEDIAN_TOL_SD}):")
    for i, name in enumerate(names):
        dm = (a["median"][i] - b["median"][i]) / b["sd"][i]
        passed = abs(dm) < MEDIAN_TOL_SD
        ok &= passed
        print(f"    {name:9s} ours {a['median'][i]:.6g}  jimgw {b['median'][i]:.6g}  sd {b['sd'][i]:.3g}  "
              f"Δ {dm:+.3f} sd  {'ok' if passed else 'FAIL'}")
    for key in ("n_iterations", "kbar"):
        rel = a[key] / b[key] - 1
        passed = abs(rel) < REL_TOL_ITER
        ok &= passed
        print(f"  {key:12s} ours {a[key]:.2f}  jimgw {b[key]:.2f}  ({rel:+.1%})  {'ok' if passed else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--events", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-live", type=int, default=256)
    ap.add_argument("--steps-per-dim", type=int, default=5)
    ap.add_argument("--delete-frac", type=float, default=0.5)
    ap.add_argument("--termination-dlogz", type=float, default=0.5)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--all-params", action="store_true", help="12-parameter layout; default intrinsic only")
    ap.add_argument("--step-tol", type=float, default=1e-8, help="T3a relative tolerance per leaf")
    ap.add_argument("--skip-one-step", action="store_true", help="skip T3a")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="directory for per-run dead points")
    ap.add_argument("--compare-only", action="store_true", help="no sampling: compare runs saved in --out")
    args = ap.parse_args()

    out = Path(args.out)
    if args.compare_only:
        ok = all([compare_event(eid, args.seeds, out) for eid in args.events])  # noqa: C419 (report every event)
        print(f"\nT3 {'PASSED' if ok else 'FAILED'} (compare-only, from {out})")
        raise SystemExit(0 if ok else 1)
    out.mkdir(parents=True, exist_ok=True)

    # CPU has no buffer donation; the step still works, JAX just warns on every call.
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

    intrinsic_only = not args.all_params
    settings = SamplerSettings(n_live=args.n_live, delete_frac=args.delete_frac,
                               steps_per_dim=args.steps_per_dim, termination_dlogz=args.termination_dlogz)
    rows = read_catalogue(args.catalogue)
    waveform = make_waveform()
    model = EventModel(waveform, intrinsic_only)
    blocks = gibbs_blocks(intrinsic_only)
    layout = "4p" if intrinsic_only else "12p"
    slice_steps_per_point = settings.steps_per_dim * model.n_dims * settings.num_gibbs_sweeps
    print(f"T3: {layout}, events {args.events}, seeds {args.seeds}, {settings}, n_bins {args.n_bins}, "
          f"backend {jax.default_backend()}, devices {jax.devices()}, out {out}")

    init = step = None
    ok = True
    for eid in args.events:
        row = rows[eid]
        t0 = time.perf_counter()
        inj = inject(row, waveform)
        lik = make_likelihood(inj, waveform, n_bins=args.n_bins,
                              fixed_names=FIXED_PARAMS if intrinsic_only else ())
        ev = event_data_from_likelihood(lik, inj.mc_lo, inj.mc_hi, inj.params)
        jim = make_jim(lik, jimgw_prior(inj.mc_lo, inj.mc_hi, intrinsic_only), settings, blocks)
        t_setup = time.perf_counter() - t0
        snr = float(np.sqrt(sum(float(ifo.optimal_snr) ** 2 for ifo in inj.ifos)))
        print(f"\n=== event {eid}: d_L={row['d_L']:.0f} Mpc duration={inj.duration:.0f} s "
              f"f_max_lik={inj.f_max_likelihood:.0f} Hz SNR={snr:.1f} [host setup {t_setup:.1f} s]")

        # structural checks: same parameter order and same rebuild map as jimgw's own sampler
        names_ok = tuple(jim.sampling_parameter_names) == model.names
        rebuild = rebuild_required_by_block(model, lik, blocks)
        rebuild_ok = rebuild == dict(jim._sampler_backend_kwargs["rebuild_required_by_block"])
        ok &= names_ok and rebuild_ok
        print(f"  parameter order matches jimgw: {'ok' if names_ok else 'FAIL'}; "
              f"rebuild map {rebuild} matches jimgw: {'ok' if rebuild_ok else 'FAIL'}")
        if init is None:
            init, step = make_sampler(model, settings, rebuild)

        meta = {"event_id": eid, "names": list(model.names), "n_dims": model.n_dims,
                "n_delete": settings.n_delete, "slice_steps_per_point": slice_steps_per_point,
                "settings": vars(settings) | {"n_bins": args.n_bins, "layout": layout}}
        for i, seed in enumerate(args.seeds):
            init_key, run_key = jax.random.split(jax.random.fold_in(jax.random.key(row["sampler_seed"]), seed))
            positions = jim.sample_initial_positions(settings.n_live, init_key)

            if i == 0 and not args.skip_one_step:
                print(f"  T3a, seed {seed}: one iteration from identical positions and key")
                _, first_step_key = jax.random.split(run_key)   # the first step key of both loops below
                passed = check_one_iteration(jim, init, step, ev, positions, first_step_key,
                                             settings.n_delete, args.step_tol)
                ok &= passed
                print(f"  T3a {'ok' if passed else 'FAIL'}")

            print(f"  seed {seed}: ours ...", end="", flush=True)
            t0 = time.perf_counter()
            res = run_event(init, step, ev, positions, run_key, settings)
            save_run(run_path(out, eid, seed, "ours"), res, model.n_dims, time.perf_counter() - t0, meta)
            print(f" ln Z {res['log_Z']:.3f} ({time.perf_counter() - t0:.0f} s); jimgw ...", end="", flush=True)
            res = run_jimgw(jim, positions, run_key)
            save_run(run_path(out, eid, seed, "jimgw"), res, model.n_dims, res["wall"], meta)
            print(f" ({res['wall']:.0f} s, includes its compile)")

        ok &= compare_event(eid, args.seeds, out)

    print(f"\nT3 {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
