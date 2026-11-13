"""Test T1 (driver-plan.md, section 4): event_model's pure functions vs jimgw, on catalogue events.

For each event and each layout (intrinsic-only, all 12 parameters), compare at N prior draws:
- log prior          vs Jim.evaluate_prior
- log likelihood     vs Jim's likelihood (direct)
- cached likelihood  vs Jim's SwiG build_cache / log_likelihood_from_cache_fn
- log posterior      vs Jim.evaluate_posterior
plus out-of-bounds M_c points (both must give -inf).

Also checks the rule that event data is not closed over: one jitted function is called with every
event's EventData and must compile once (its cache size stays 1).

    python check_event_model.py --catalogue catalogue.csv --events 0 1 2
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.config import BlackJAXSwiGConfig

from event_model import (ALL_PARAMS, FIXED_PARAMS, FIXED_PRIORS, EventModel, event_data_from_likelihood,
                         gibbs_blocks)
from heterodyne import DEFAULT_N_BINS
from injection import inject, make_likelihood, make_waveform, read_catalogue
from prepare import jimgw_prior   # moved to prepare.py (no JAX_PLATFORMS side effect); re-exported

TOL = 1e-9


def make_jim(likelihood, prior, intrinsic_only, ckpt_dir):
    config = BlackJAXSwiGConfig(blocks=gibbs_blocks(intrinsic_only), n_live=64, n_delete_frac=0.5,
                                num_inner_steps_per_dim=1, num_gibbs_sweeps=1, n_devices=1,
                                checkpoint_dir=ckpt_dir)
    return Jim(likelihood, prior, likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
               sampler_config=config, seed=0)


def max_abs_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    both_inf = np.isneginf(a) & np.isneginf(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.all(np.isfinite(a) | both_inf):
        return np.inf, np.nan
    d = np.abs(a - b)[~both_inf]
    rel = d / np.maximum(np.abs(b[~both_inf]), 1.0)
    return float(d.max(initial=0.0)), float(rel.max(initial=0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--events", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    args = ap.parse_args()

    rows = read_catalogue(args.catalogue)
    waveform = make_waveform()
    models = {io: EventModel(waveform, io) for io in (True, False)}

    # One jitted, vmapped function per layout and quantity; EventData is an argument, never a constant.
    def batched(fn):
        return jax.jit(jax.vmap(fn, in_axes=(0, None)))
    ours = {io: {"prior": batched(m.log_prior),
                 "ll": batched(m.log_likelihood),
                 "ll_cache": batched(lambda x, ev, m=m: m.ll_from_cache(x, m.build_cache(x, ev), ev)),
                 "post": batched(m.log_posterior)} for io, m in models.items()}

    results, ok = [], True
    with tempfile.TemporaryDirectory() as ckpt_dir:
        for eid in args.events:
            row = rows[eid]
            t0 = time.perf_counter()
            inj = inject(row, waveform)
            t_inject = time.perf_counter() - t0
            snr = float(np.sqrt(sum(float(ifo.optimal_snr) ** 2 for ifo in inj.ifos)))
            print(f"\n=== event {eid}: z={row['z']:.4f} d_L={row['d_L']:.0f} Mpc "
                  f"m_src=({row['m1_src']:.3f}, {row['m2_src']:.3f}) duration={inj.duration:.0f} s "
                  f"f_max_lik={inj.f_max_likelihood:.0f} Hz network SNR={snr:.1f}  [inject {t_inject:.1f} s]")

            for io in (True, False):
                model = models[io]
                t0 = time.perf_counter()
                lik = make_likelihood(inj, waveform, n_bins=args.n_bins,
                                      fixed_names=FIXED_PARAMS if io else ())
                ev = event_data_from_likelihood(lik, inj.mc_lo, inj.mc_hi, inj.params)
                t_lik = time.perf_counter() - t0

                jim = make_jim(lik, jimgw_prior(inj.mc_lo, inj.mc_hi, io), io, ckpt_dir)
                assert tuple(jim.sampling_parameter_names) == model.names, jim.sampling_parameter_names
                kw = jim._sampler_backend_kwargs
                ref = {
                    "prior": jax.jit(jax.vmap(jim.evaluate_prior)),
                    "ll": jax.jit(jax.vmap(jim._log_likelihood_fn)),
                    "ll_cache": jax.jit(jax.vmap(
                        lambda x, kw=kw: kw["log_likelihood_from_cache_fn"](x, kw["build_cache"](x)))),
                    "post": jax.jit(jax.vmap(jim.evaluate_posterior)),
                }

                draws = jim.prior.sample(jax.random.key(1000 + eid), args.n_draws)
                x = jnp.stack([draws[k] for k in model.names]).T
                # out-of-bounds M_c: just outside each edge, and far outside
                x_oob = x[:4].at[:, 0].set(
                    jnp.array([inj.mc_lo * (1 - 1e-6), inj.mc_hi * (1 + 1e-6), 0.5, 5.0]))

                layout = "4p " if io else "12p"
                for q in ("prior", "ll", "ll_cache", "post"):
                    d_abs, d_rel = max_abs_diff(ours[io][q](x, ev), ref[q](x))
                    passed = d_abs < TOL
                    ok &= passed
                    results.append((eid, layout, q, d_abs, d_rel, passed))
                    print(f"  {layout} {q:9s} max|Δ| = {d_abs:.3g}  (rel {d_rel:.2g})  {'ok' if passed else 'FAIL'}")
                # unjitted: a second batch size would add a compile to the counts checked below
                oob = {"prior": (model.log_prior, jim.evaluate_prior),
                       "post": (model.log_posterior, jim.evaluate_posterior)}
                for q, (f_ours, f_ref) in oob.items():
                    a = np.asarray(jax.vmap(f_ours, in_axes=(0, None))(x_oob, ev))
                    b = np.asarray(jax.vmap(f_ref)(x_oob))
                    passed = bool(np.all(np.isneginf(a)) and np.all(np.isneginf(b)))
                    ok &= passed
                    print(f"  {layout} {q:9s} out-of-bounds M_c -> -inf: {'ok' if passed else f'FAIL {a} {b}'}")
                ll = np.asarray(ref["ll"](x))
                print(f"  {layout} n_bins={lik.n_bins}  logL at draws: [{ll.min():.4g}, {ll.max():.4g}]  "
                      f"[likelihood setup {t_lik:.1f} s]")

    print("\n=== compile counts (must all be 1: one program for every event)")
    for io in (True, False):
        for q, f in ours[io].items():
            n = f._cache_size()
            ok &= n == 1
            print(f"  {'4p ' if io else '12p'} {q:9s} {n}")

    print(f"\nT1 {'PASSED' if ok else 'FAILED'} (tolerance |Δ| < {TOL:g})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
