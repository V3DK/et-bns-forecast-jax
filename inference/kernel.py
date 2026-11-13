"""Compile-once SwiG nested sampler: jitted `init(positions, ev)` and `step(key, state, ev)`.

jimgw builds its sampler from closures over the likelihood and prior, so every event's data becomes
constants in the compiled program (driver-plan.md, section 1). Here the sampler is built *inside* the
traced functions, from `EventModel`'s pure functions with `ev` bound by `partial`. The closures then
capture tracers, not constants, and one compiled program serves every event with the same settings.

`make_sampler` may close over things that are the same for every event (the model, `SamplerSettings`,
the block layout), never over anything from an `EventData`.

Copies of jimgw 0.5.2 (pinned):
- `ns_step`  <- BlackJAXSwiGSampler._build_nested_sampler (samplers/blackjax/ns/swig.py), mesh=None branch
- `ns_init`  <- initialize_nested_sampler_state in BlackJAXNSSSampler._sample (samplers/blackjax/ns/nss.py)
- `update_block_covariances` <- BlackJAXSwiGSampler.__init__
- `run_event` <- the loop, finalise and NestedSamples in BlackJAXNSSSampler._sample / _get_diagnostics,
  without checkpointing or jimgw's cache redirect.
"""
import importlib.metadata
import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from anesthetic.samples import NestedSamples
from blackjax.ns.adaptive import init as _ns_adaptive_init
from blackjax.ns.base import init_state_strategy as _init_state_strategy
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.ns.utils import finalise
from blackjax.smc.tuning.from_particles import particles_covariance_matrix

from jimgw.core.single_event.blocked_likelihood import _build_rebuild_required_by_block
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.blackjax.ns.swig import _build_swig_constrained_step

# The private functions above are copied/imported against these versions.
PINNED = {"jimgw": "0.5.2", "blackjax": "1.7.1"}
for _pkg, _ver in PINNED.items():
    if importlib.metadata.version(_pkg) != _ver:
        raise ImportError(f"kernel.py uses private {_pkg} {_ver} internals; found "
                          f"{importlib.metadata.version(_pkg)}. Re-check the copies before changing the pin.")


@dataclass(frozen=True)
class SamplerSettings:
    """Everything about the sampler that is baked into the compiled program (plus termination)."""
    n_live: int
    delete_frac: float = 0.5
    steps_per_dim: int = 5
    num_gibbs_sweeps: int = 1
    termination_dlogz: float = 0.5      # as swig.py; host-side only, not part of the program

    @property
    def n_delete(self):
        return int(self.n_live * self.delete_frac)


def rebuild_required_by_block(model, likelihood, blocks):
    """jimgw's {block index tuple: needs waveform rebuild}, from any one event's jimgw likelihood.

    It depends only on which parameters the waveform reads, not on the event's data, so it is computed
    once per process. Order follows `blocks`, as in jimgw.
    """
    return _build_rebuild_required_by_block(
        likelihood, blocks,
        parameter_names=model.names,
        sample_transforms=[],
        likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
    )


def make_sampler(model, settings, rebuild):
    """Return jitted `init(positions, ev) -> state` and `step(key, state, ev) -> (state, info)`.

    Call once per process: each returned function compiles on its first call and is then reused for
    every event whose `EventData` has the same shapes and dtypes.
    """
    rebuild = dict(rebuild)
    block_indices = tuple(rebuild)
    assert sorted(i for b in block_indices for i in b) == list(range(model.n_dims)), block_indices
    n_delete = settings.n_delete
    if settings.n_live % n_delete:
        raise ValueError(f"n_live {settings.n_live} must be a multiple of n_delete {n_delete} (lax.map batches)")

    def update_block_covariances(rng_key, state, info, params=None):
        del rng_key, info, params
        cov = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
        return {"block_covariances": tuple(cov[jnp.ix_(jnp.asarray(ix), jnp.asarray(ix))]
                                           for ix in block_indices)}

    def ns_init(positions, ev):
        init_one = partial(_init_state_strategy,
                           logprior_fn=partial(model.log_prior, ev=ev),
                           loglikelihood_fn=partial(model.log_likelihood, ev=ev))
        return _ns_adaptive_init(
            positions,
            init_state_fn=lambda pos: jax.lax.map(init_one, pos, batch_size=n_delete),
            update_inner_kernel_params_fn=update_block_covariances,
        )

    def ns_step(key, state, ev):
        constrained_step = _build_swig_constrained_step(
            log_prior_fn=partial(model.log_prior, ev=ev),
            build_cache=partial(model.build_cache, ev=ev),
            log_likelihood_from_cache_fn=partial(model.ll_from_cache, ev=ev),
            rebuild_required_by_block=rebuild,
            num_gibbs_sweeps=settings.num_gibbs_sweeps,
            num_inner_steps_per_dim=settings.steps_per_dim,
            periodic=None,
            n_dims=model.n_dims,
        )
        kernel = build_from_mcmc_kernel(
            constrained_step,
            num_inner_steps=1,
            update_inner_kernel_params_fn=update_block_covariances,
            num_delete=n_delete,
        )
        return kernel(key, state)

    return jax.jit(ns_init), jax.jit(ns_step, donate_argnums=1)


def should_terminate(state, termination_dlogz):
    """As jimgw: stop when the live points' share of the evidence is below termination_dlogz."""
    dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
    return bool(jnp.isfinite(dlogz) and dlogz < termination_dlogz)


def run_event(init, step, ev, positions, key, settings):
    """Sample one event to termination. Returns a dict of results and timings (host arrays).

    `positions`: [n_live, n_dims] initial live points in sampling space. `key`: a typed PRNG key.
    Timings: `t_init` and `t_first_step` include any trace/compile; `t_steps` is the remaining steps.
    """
    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[0] != settings.n_live:
        raise ValueError(f"positions shape {positions.shape}, expected ({settings.n_live}, n_dims)")
    n_dims = positions.shape[1]

    t0 = time.perf_counter()
    state = jax.block_until_ready(init(positions, ev))
    t_init = time.perf_counter() - t0

    dead, n_iter, t_first_step = [], 0, 0.0
    t0 = time.perf_counter()
    while not should_terminate(state, settings.termination_dlogz):   # host sync each iteration
        key, step_key = jax.random.split(key)
        state, info = step(step_key, state, ev)
        dead.append(info)
        n_iter += 1
        if n_iter == 1:
            jax.block_until_ready(state)
            t_first_step = time.perf_counter() - t0
            t0 = time.perf_counter()
    jax.block_until_ready(state)
    t_steps = time.perf_counter() - t0 if n_iter > 1 else 0.0
    if n_iter == 0:
        raise RuntimeError("terminated before the first step: check the initial positions")

    t0 = time.perf_counter()
    final = jax.device_get(finalise(state, dead))
    logL_birth = np.asarray(final.particles.loglikelihood_birth)
    ns = NestedSamples(np.asarray(final.particles.position),
                       logL=np.asarray(final.particles.loglikelihood),
                       logL_birth=np.where(np.isnan(logL_birth), -np.inf, logL_birth),
                       logzero=np.nan, dtype=np.float64)
    ui = final.update_info
    n_steps_out, n_shrink = int(np.sum(ui.num_expansions)), int(np.sum(ui.num_shrink))
    posterior = ns.posterior_points()
    out = {
        "log_Z": float(np.asarray(ns.logZ()).item()),
        "log_Z_error": float(np.std(np.asarray(ns.logZ(nsamples=100)))),
        "n_iterations": n_iter,
        "n_likelihood_evaluations": n_steps_out + n_shrink,
        "n_likelihood_evaluations_stepping_out": n_steps_out,
        "n_likelihood_evaluations_shrinking": n_shrink,
        "posterior_samples": np.asarray(posterior.iloc[:, :n_dims]),
        "posterior_log_likelihood": np.asarray(posterior["logL"]),
        "t_init": t_init,
        "t_first_step": t_first_step,
        "t_steps": t_steps,
        "nested_samples": ns,
    }
    out["t_finalise"] = time.perf_counter() - t0
    return out
