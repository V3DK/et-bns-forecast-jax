""" fast inference on a single injected event: intrinsic-only (M_c, q, Lambda_1, Lambda_2) """
import os

# Expose every CPU core as a JAX device; must happen before jax is imported.
N_DEVICES = 4  # set to your performance-core count (`sysctl -n hw.perflevel0.physicalcpu`)
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_DEVICES}"

import time
from datetime import datetime
from pathlib import Path

import corner
import numpy as np
import jax
import jax.numpy as jnp

import bilby
import bilby.gw.conversion as conv
from bilby.gw.conversion import lambda_from_mass_and_family

import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.detector import get_ET
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.config import BlackJAXNSSConfig

from run_io import describe_prior, prior_space_truths, save_config, save_results

RUN_NAME = f"injection_fast_{datetime.now():%Y%m%d_%H%M%S}"
OUT = Path(__file__).resolve().parent / "runs" / RUN_NAME
OUT.mkdir(parents=True, exist_ok=False)   # fail rather than overwrite


print(f"JAX devices: {len(jax.devices())}")

waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20)
gps = 1126259462.4
SKY_SEED, NOISE_SEED, SAMPLER_SEED = 0, 1, 0
random_samples = jax.random.uniform(jax.random.key(SKY_SEED), (3,), maxval=jnp.pi)
noise_key = jax.random.key(NOISE_SEED)

EOS = "MPA1"
fam = lalsim.CreateSimNeutronStarFamily(lalsim.SimNeutronStarEOSByName(EOS))

M_tot = 2.5
m1_inj = m2_inj = M_tot / 2

injection_parameters = {
    "M_c": conv.component_masses_to_chirp_mass(m1_inj, m2_inj),
    "eta": conv.component_masses_to_symmetric_mass_ratio(m1_inj, m2_inj),
    "s1_z": 0.0,
    "s2_z": 0.0,
    "lambda_1": lambda_from_mass_and_family(m1_inj, fam),
    "lambda_2": lambda_from_mass_and_family(m2_inj, fam),
    "iota": 0.5,
    "d_L": 500.0,
    "ra": random_samples[0] * 2.0,
    "dec": random_samples[1] - jnp.pi / 2,
    "psi": random_samples[2],
    "t_c": 0.0,
    "phase_c": 0.0,
}
injection_parameters = {k: float(v) for k, v in injection_parameters.items()}

# Held at injected values: extrinsics + spins. phase_c stays marginalised.
FIXED = ["s1_z", "s2_z", "iota", "d_L", "ra", "dec", "psi", "t_c"]
fixed_parameters = {k: injection_parameters[k] for k in FIXED}

f_min, f_max = 20.0, 2000.0
duration = 256.0
sampling_frequency = 4096.0

ifos = get_ET()
et_psd = Path(bilby.__file__).parent / "gw/detector/noise_curves/ET_D_psd.txt"
for ifo in ifos:
    ifo.load_and_set_psd(psd_file=str(et_psd))
    ifo.inject_signal(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=gps,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=f_min,
        f_max=f_max,
        zero_noise=False,
        rng_key=noise_key,
    )

prior = CombinePrior(
    [
        UniformPrior(1.08, 1.10, parameter_names=["M_c"]),
        UniformPrior(0.5, 1.0, parameter_names=["q"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
    ]
)

likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

phase_marginalization = True

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    fixed_parameters=fixed_parameters,
    trigger_time=gps,
    f_min=f_min,
    f_max=f_max,
    reference_parameters=injection_parameters,
    phase_marginalization=phase_marginalization,
)

sampler_config = BlackJAXNSSConfig(
    n_live=2 * N_DEVICES * round(1000 / (2 * N_DEVICES)),  # n_live, n_delete divisible by N_DEVICES
    n_delete_frac=0.5,
    num_inner_steps_per_dim=10,
    termination_dlogz=0.1,
    n_devices=N_DEVICES,
    checkpoint_dir=str(OUT / "ckpt"),
    checkpoint_interval=120,
)

jim = Jim(
    likelihood,
    prior,
    likelihood_transforms=likelihood_transforms,
    sampler_config=sampler_config,
    seed=SAMPLER_SEED,
    verbose=True,
)

truth_values = prior_space_truths(injection_parameters, likelihood_transforms, jim.prior.parameter_names)

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$s_{1,z}$",
    "s2_z": r"$s_{2,z}$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
}

save_config(OUT, __file__, {
    "injection": {
        "eos": EOS,
        "M_tot": M_tot,
        "parameters": injection_parameters,   # likelihood space
        "truths": truth_values,               # prior space
    },
    "data": {
        "detectors": [ifo.name for ifo in ifos],
        "psd_file": et_psd,
        "gps": gps,
        "f_min": f_min,
        "f_max": f_max,
        "duration": duration,
        "sampling_frequency": sampling_frequency,
        "zero_noise": False,
        "sky_seed": SKY_SEED,
        "noise_seed": NOISE_SEED,
        "optimal_snr": {ifo.name: float(ifo.optimal_snr) for ifo in ifos},
        "match_filtered_snr": {ifo.name: abs(complex(ifo.match_filtered_snr)) for ifo in ifos},
        "network_optimal_snr": float(np.sqrt(sum(ifo.optimal_snr**2 for ifo in ifos))),
    },
    "waveform": {"name": type(waveform).__name__, "f_ref": waveform.f_ref},
    "likelihood": {
        "type": type(likelihood).__name__,
        "phase_marginalization": phase_marginalization,
        "fixed_parameters": fixed_parameters,
    },
    "prior": describe_prior(prior),
    "parameter_names": list(jim.prior.parameter_names),
    "parameter_labels": parameter_labels,
    "likelihood_transforms": [repr(t) for t in likelihood_transforms],
    "periodic": None,
    "sampler": {
        "seed": SAMPLER_SEED,
        **sampler_config.model_dump(mode="json"),
    },
})

x = jim.prior.sample(jax.random.key(42), 1000)
x = jnp.stack([x[k] for k in jim.prior.parameter_names]).T
f = jax.jit(jax.vmap(jim.evaluate_posterior))
f(x).block_until_ready()                      # compile
t = time.time(); f(x).block_until_ready()
us_per_eval = (time.time() - t) / 1000 * 1e6
print(f"{us_per_eval:.1f} µs per eval")


start_time = time.time()
jim.sample()
sampling_time = time.time() - start_time
print(f"Sampling took {sampling_time / 60:.2f} mins")


chains = save_results(OUT, jim, {
    "us_per_posterior_eval": us_per_eval,
})
print(f"Likelihood evaluations: {jim.get_diagnostics()['n_likelihood_evaluations']:,}")

truths = [truth_values[k] for k in jim.prior.parameter_names]


fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
    # quantiles=[0.05, 0.5, 0.95],
    show_titles=True,
)
fig.savefig(OUT / "corner.png", dpi=150)