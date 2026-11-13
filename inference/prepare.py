"""Per-event host setup shared by the driver and the check scripts: catalogue row -> sampler inputs.

`prepare_event` rebuilds the injection, the binned heterodyned likelihood, its `EventData`, and the
initial live points, all deterministically from the row. check_kernel.py (T2-T5) and driver.py use it,
so a driver result can be compared bit for bit with the earlier tests' results for the same event.

No side effects on import (in particular, no JAX_PLATFORMS override): the driver imports this on GPU.
"""
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.core.prior import CombinePrior, CosinePrior, PowerLawPrior, SinePrior, UniformPrior

from event_model import ALL_PARAMS, FIXED_PARAMS, FIXED_PRIORS, event_data_from_likelihood
from injection import inject, make_likelihood


def jimgw_prior(mc_lo, mc_hi, intrinsic_only):
    """The CombinePrior from swig.py, built from event_model.FIXED_PRIORS so the two cannot drift."""
    def one(name):
        if name == "M_c":
            return UniformPrior(mc_lo, mc_hi, parameter_names=["M_c"])
        kind, *args = FIXED_PRIORS[name]
        return {"uniform": lambda: UniformPrior(*args, parameter_names=[name]),
                "sine": lambda: SinePrior(parameter_names=[name]),
                "cosine": lambda: CosinePrior(parameter_names=[name]),
                "powerlaw": lambda: PowerLawPrior(*args, parameter_names=[name])}[kind]()
    names = ALL_PARAMS[:4] if intrinsic_only else ALL_PARAMS
    return CombinePrior([one(n) for n in names])


@dataclass
class PreparedEvent:
    row: dict
    inj: object               # injection.Injection (detectors hold the strain; needed by the exact check)
    likelihood: object        # BinnedHeterodynedLikelihoodFD
    ev: object                # EventData
    positions: jax.Array      # [n_live, n_dims] initial live points, sampling space
    sampler_key: jax.Array    # typed PRNG key for the NS loop
    snr: dict                 # optimal SNR per detector and "network"
    t_setup: float            # host seconds for all of the above


def prepare_event(row, waveform, model, n_live, n_bins):
    """Everything the sampler needs for one catalogue row. Seeds come from the row only."""
    t0 = time.perf_counter()
    inj = inject(row, waveform)
    lik = make_likelihood(inj, waveform, n_bins=n_bins,
                          fixed_names=FIXED_PARAMS if model.intrinsic_only else ())
    ev = event_data_from_likelihood(lik, inj.mc_lo, inj.mc_hi, inj.params)
    key = jax.random.key(row["sampler_seed"])
    prior_key, sampler_key = jax.random.split(key)
    draws = jimgw_prior(inj.mc_lo, inj.mc_hi, model.intrinsic_only).sample(prior_key, n_live)
    positions = jnp.stack([draws[k] for k in model.names]).T
    snr = {ifo.name: float(ifo.optimal_snr) for ifo in inj.ifos}
    snr["network"] = float(np.sqrt(sum(v ** 2 for v in snr.values())))
    jax.block_until_ready((ev, positions))
    return PreparedEvent(row=row, inj=inj, likelihood=lik, ev=ev, positions=positions,
                         sampler_key=sampler_key, snr=snr, t_setup=time.perf_counter() - t0)


def sampling_truths(inj, names):
    """Injected values in sampling space (q = m2/m1 instead of eta), in `names` order."""
    p = dict(inj.params)
    p["q"] = inj.row["m2_src"] / inj.row["m1_src"]    # same in source and detector frame
    return np.array([p[k] for k in names], dtype=np.float64)
