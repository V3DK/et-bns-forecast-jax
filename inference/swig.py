""" SwiG inference on a single injected event: intrinsic-only (M_c, q, Lambda_1, Lambda_2) or all parameters

Default: CPU on a Mac, one forced JAX host device per performance core.
  Mac:  python swig.py                       (override core count with N_DEVICES=k)
  GPU:  REQUIRE_GPU=1 python swig.py         (N_DEVICES = number of visible GPUs)
"""
import os
import subprocess

# True: sample only (M_c, q, Lambda_1, Lambda_2), holding spins + extrinsics at injected values.
# False: sample all parameters.
INTRINSIC_ONLY = False

# Backend selection; must happen before jax is imported.
USE_GPU = os.environ.get("REQUIRE_GPU", "0") not in ("", "0")

SAMPLER_SEED  = int(os.environ.get("SAMPLER_SEED", 2))
STEPS_PER_DIM = int(os.environ.get("STEPS_PER_DIM", 5))
DELETE_FRAC   = float(os.environ.get("DELETE_FRAC", 0.5))
PROFILE       = os.environ.get("PROFILE", "0") not in ("", "0")

def _perf_cores():
    """Mac performance-core count; efficiency cores straggle and stall synchronised steps."""
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                                  capture_output=True, text=True, check=True).stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return os.cpu_count() or 4


if USE_GPU:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")   # fail loudly instead of falling back to CPU
else:
    os.environ["JAX_PLATFORMS"] = "cpu"              # also sidesteps jax-metal (no float64)
    _n_host = int(os.environ.get("N_DEVICES") or _perf_cores())
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
        f" --xla_force_host_platform_device_count={_n_host}")


import time
from datetime import datetime
from pathlib import Path

import corner
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import logging, re
CACHE_DIR = os.environ.get("JAX_CACHE_DIR")
if CACHE_DIR:
    jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

class CompileTimer(logging.Handler):
    """Sums tracing + lowering + XLA compile time from jax_log_compiles messages."""
    _pat = re.compile(r"in ([0-9.]+) sec")
    def __init__(self): super().__init__(); self.reset()
    def reset(self): self.total, self.n = 0.0, 0
    def emit(self, rec):
        msg = rec.getMessage()
        m = self._pat.search(msg)
        if m and msg.startswith("Finished"):
            self.total += float(m.group(1)); self.n += 1
compile_timer = CompileTimer()
logging.getLogger("jax").addHandler(compile_timer)
jax.config.update("jax_log_compiles", True)

_expected = "gpu" if USE_GPU else "cpu"
if jax.default_backend() != _expected:
    raise RuntimeError(f"expected {_expected} backend, got {jax.devices()}")
N_DEVICES = jax.local_device_count()   # GPUs on a GPU node, forced host devices on a Mac
print(f"backend={jax.default_backend()}, devices={jax.devices()}")

import bilby
import bilby.gw.conversion as conv
from bilby.gw.conversion import lambda_from_mass_and_family
from bilby.gw.cosmology import get_cosmology

import lal
import lalsimulation as lalsim


from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior
)
from jimgw.core.single_event.detector import get_ET
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.samplers.config import BlackJAXSwiGConfig

from run_io import describe_prior, prior_space_truths, save_config, save_results

# RUN_NAME = f"{'4p' if INTRINSIC_ONLY else '12p'}_{os.environ.get('SLURM_JOB_ID', f'{datetime.now():%Y%m%d_%H%M%S}')}"
RUN_NAME = (f"{'4p' if INTRINSIC_ONLY else '12p'}_s{STEPS_PER_DIM}_f{DELETE_FRAC}_seed{SAMPLER_SEED}_"
            f"{os.environ.get('SLURM_JOB_ID', f'{datetime.now():%Y%m%d_%H%M%S}')}")

RUNS_ROOT = Path(os.environ.get("RUNS_DIR", Path(__file__).resolve().parent / "runs"))
OUT = RUNS_ROOT / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)

print(f"JAX devices: {len(jax.devices())}")


def merger_cutoff_frequency(waveform, params, f_lo, f_hi, threshold, n=20000):
    """First frequency where |h| drops below `threshold` x its inspiral trend (|h| ~ f^-7/6).

    The heterodyne divides by the reference waveform, so it breaks where the reference
    tapers to zero after merger; the likelihood band must stop before that.
    """
    f = jnp.linspace(f_lo, f_hi, n)
    h = waveform(f, {k: jnp.asarray(v) for k, v in params.items()})
    amp = np.sqrt(np.abs(np.asarray(h["p"])) ** 2 + np.abs(np.asarray(h["c"])) ** 2)
    f = np.asarray(f)
    trend = amp * f ** (7 / 6)
    trend /= np.median(trend[f < 3 * f_lo])
    below = np.nonzero(trend < threshold)[0]
    return float(f[below[0]]) if below.size else float(f_hi)

waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20)
gps = 1126259462.4
# SKY_SEED, NOISE_SEED, SAMPLER_SEED = 42, 43, 44 # used to be 0, 1, 2

SKY_SEED, NOISE_SEED = 0, 1          # SAMPLER_SEED now from env

random_samples = jax.random.uniform(jax.random.key(SKY_SEED), (3,))
noise_key = jax.random.key(NOISE_SEED)

EOS = "MPA1"
fam = lalsim.CreateSimNeutronStarFamily(lalsim.SimNeutronStarEOSByName(EOS))

COSMOLOGY = "Planck18"
d_L = 1000.0 # Mpc, used to be 500.0
z = float(conv.luminosity_distance_to_redshift(d_L, cosmology=get_cosmology(COSMOLOGY)))

# m1 must be heavier to respect q setup
m1_src = 1.6 # 1.5
m2_src = 1.2 # 1.25
m1_det, m2_det = (1+z) * m1_src, (1+z) * m2_src

injection_parameters = {
    "M_c": conv.component_masses_to_chirp_mass(m1_det, m2_det),
    "eta": conv.component_masses_to_symmetric_mass_ratio(m1_det, m2_det),
    "s1_z": 0.0,
    "s2_z": 0.0,
    "lambda_1": lambda_from_mass_and_family(m1_src, fam),
    "lambda_2": lambda_from_mass_and_family(m2_src, fam),
    "iota": 0.5,
    "d_L": d_L,
    "ra": random_samples[0] * 2 * jnp.pi,
    "dec": jnp.arcsin(2 * random_samples[1] - 1),  # isotropic: uniform in sin(dec)
    "psi": random_samples[2] * jnp.pi,
    "t_c": 0.0,
    "phase_c": 0.0,
}
injection_parameters = {k: float(v) for k, v in injection_parameters.items()}

# Held at injected values when INTRINSIC_ONLY: extrinsics + spins. phase_c stays marginalised.
FIXED = ["s1_z", "s2_z", "iota", "d_L", "ra", "dec", "psi", "t_c"] if INTRINSIC_ONLY else []
fixed_parameters = {k: injection_parameters[k] for k in FIXED}

# setup for M_c prior -> search in real PE returns a chirp mass estimate
Mc = injection_parameters["M_c"]
half_width = 0.01 * Mc

f_min = 5.0
f_max =  2000.0
sampling_frequency = 4096.0

# duration must hold the longest template in the prior from f_min to merger, else the early
# inspiral wraps around. Longest = lowest M_c, equal masses, max prior spin (conservative bound).
POST_TRIGGER = 2.0   # inject_signal puts 2 s of data after the trigger
SAFETY = 1.1
Mc_lo = Mc - half_width
m_eq = Mc_lo * 2 ** 0.2 * lal.MSUN_SI # approximate as equal mass
chirp_time = lalsim.SimInspiralChirpTimeBound(f_min, m_eq, m_eq, 0.05, 0.05)
duration = float(2 ** np.ceil(np.log2(SAFETY * chirp_time + POST_TRIGGER)))
print(f"chirp time from {f_min:g} Hz: {chirp_time:.0f} s -> duration {duration:.0f} s")

ifos = get_ET()
et_psd = Path(bilby.__file__).parent / "gw/detector/noise_curves/ET_D_psd.txt"
for ifo, key in zip(ifos, jax.random.split(noise_key, len(ifos))):   # independent noise per ifo
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
        rng_key=key,
    )

# likelihood band stops just before the reference waveform's merger taper (see merger_cutoff_frequency);
# 0.2 keeps all tidal information for the MPA1 1.5+1.25 event while the heterodyne stays accurate
CUTOFF_THRESHOLD = 0.2
f_max_likelihood = min(f_max, merger_cutoff_frequency(waveform, injection_parameters, f_min, f_max, CUTOFF_THRESHOLD))
print(f"likelihood f_max: {f_max_likelihood:.0f} Hz")

intrinsic_priors = [
    UniformPrior(Mc - half_width, Mc + half_width, parameter_names=["M_c"]),
    UniformPrior(0.5, 1.0, parameter_names=["q"]),
    UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
    UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
]
other_priors = [
    UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
    UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
    SinePrior(parameter_names=["iota"]),
    PowerLawPrior(10.0, 2000.0, 2.0, parameter_names=["d_L"]),
    UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
    UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
    CosinePrior(parameter_names=["dec"]),
    UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
]
prior = CombinePrior(intrinsic_priors if INTRINSIC_ONLY else intrinsic_priors + other_priors)

# maps from prior space to likelihood space (don't need to be invertible)
likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

# sampling transforms map from prior space to sampling space (must be bijective -> invertible)

phase_marginalization = True

likelihood = HeterodynedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    fixed_parameters=fixed_parameters,
    trigger_time=gps,
    f_min=f_min,
    f_max=f_max_likelihood,
    reference_parameters=injection_parameters,
    phase_marginalization=phase_marginalization,
)

if INTRINSIC_ONLY:
    # only this one block because 4 param
    blocks = [["M_c", "q", "lambda_1", "lambda_2"]]
else:
    blocks = [
        ["M_c", "q", "lambda_1", "lambda_2"],
        ["s1_z", "s2_z"],
        ["iota", "d_L"],
        ["ra", "dec", "psi"],
        ["t_c"],
    ]

# n_live rounded to a multiple of 4*N_DEVICES so n_live and n_delete are divisible by N_DEVICES.
# Check the printed value matches across machines: it depends on N_DEVICES.
N_LIVE = int(os.environ.get("N_LIVE", 1024))
n_live = 4 * N_DEVICES * max(1, round(N_LIVE / (4 * N_DEVICES)))
print(f"n_live = {n_live} (requested {N_LIVE}, N_DEVICES = {N_DEVICES})")

sampler_config = BlackJAXSwiGConfig(
    blocks=blocks,
    n_live=n_live,
    n_delete_frac=DELETE_FRAC,
    num_inner_steps_per_dim=STEPS_PER_DIM, 
    termination_dlogz=0.5, # used to be 0.1
    num_gibbs_sweeps=1,
    n_devices=N_DEVICES,
    checkpoint_dir=str(OUT / "ckpt"),
    checkpoint_interval=120,
)

print(BlackJAXSwiGConfig.model_fields.keys())

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
    "M_c": r"$\mathcal{M}_c^{\rm det}\,[M_\odot]$",
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
    "intrinsic_only": INTRINSIC_ONLY,
    "injection": {
        "eos": EOS,
        "m1_src": m1_src,
        "m2_src": m2_src,
        "m1_det": m1_det,
        "m2_det": m2_det,
        "z": z,
        "cosmology": COSMOLOGY,
        "frame": "M_c detector frame; lambdas from source-frame masses",
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
        "f_max": f_max_likelihood,
        "cutoff_threshold": CUTOFF_THRESHOLD,
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

# # throughput vs batch size; us_per_eval taken at the sampler's deletion batch (n_delete)
# n_delete = int(sampler_config.n_live * sampler_config.n_delete_frac)
# f = jax.jit(jax.vmap(jim.evaluate_posterior))
# timings = {}
# for B in sorted({256, 1024, 4096, 16384, n_delete, sampler_config.n_live}):
#     x = jim.prior.sample(jax.random.key(42), B)
#     x = jnp.stack([x[k] for k in jim.prior.parameter_names]).T
#     f(x).block_until_ready()
#     t = time.perf_counter(); f(x).block_until_ready()
#     timings[B] = time.perf_counter() - t
#     print(f"{B:6d}  {timings[B]*1e3:8.2f} ms/batch  {timings[B]/B*1e6:7.2f} µs/eval")
# us_per_eval = timings[n_delete] / n_delete * 1e6

n_delete = int(sampler_config.n_live * sampler_config.n_delete_frac)
logp = jax.vmap(jim.evaluate_posterior)
f = jax.jit(logp)

@jax.jit
def chain(x, K):
    """K strictly sequential batched evals in a while_loop with a traced bound, like the sampler's."""
    def body(c):
        i, x, acc = c
        l = logp(x)
        x = x + jnp.where(jnp.isfinite(l), 1e-30 * l, 0.0)[:, None]   # data dependency
        return i + 1, x, acc + l
    return jax.lax.while_loop(lambda c: c[0] < K, body, (0, x, jnp.zeros(x.shape[0])))[2]

def med_time(fn, *a, reps=20):
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def draw(B):
    s = jim.prior.sample(jax.random.key(42), B)
    return jnp.stack([s[k] for k in jim.prior.parameter_names]).T

timings, per_trip = {}, {}
K1, K2 = 100, 600
for B in sorted({256, 512, 1024, 2048, 4096, 16384, n_delete, sampler_config.n_live}):
    x = draw(B)
    timings[B]  = med_time(f, x)
    per_trip[B] = (med_time(chain, x, K2, reps=3) - med_time(chain, x, K1, reps=3)) / (K2 - K1)
    print(f"{B:6d}  single {timings[B]*1e3:6.3f} ms   in-loop {per_trip[B]*1e3:6.3f} ms/trip")
us_per_eval = timings[n_delete] / n_delete * 1e6

if PROFILE:
    with jax.profiler.trace(str(OUT / "trace"), create_perfetto_trace=True):
        jax.block_until_ready(chain(draw(n_delete), 20))

# sample
compile_timer.reset()
start_time = time.time()
jim.sample()
sampling_time = time.time() - start_time
compile_s = compile_timer.total

# save immediately to be safe
base = {
    "us_per_posterior_eval": us_per_eval,
    "ms_per_batch": {str(B): t * 1e3 for B, t in timings.items()},
    "sampling_time_s": sampling_time,
}
chains = save_results(OUT, jim, base)

# jimgw deletes ckpt/jax_cache on completion, but JAX has already latched the
# persistent cache as enabled; reset it so later jits don't try to write there
from jax._src import compilation_cache
compilation_cache.reset_cache()
print(f"Sampling took {sampling_time / 60:.2f} mins")


# post-run check: re-evaluate the max-likelihood sample + random posterior samples with the exact
# likelihood; a large difference means the heterodyne was wrong where the sampler went
N_CHECK, CHECK_TOL = 50, 0.5
exact = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    fixed_parameters=fixed_parameters,
    trigger_time=gps,
    f_min=f_min,
    f_max=f_max_likelihood,
    phase_marginalization=phase_marginalization,
)
exact_log_likelihood = jax.jit(
    lambda p: exact.evaluate(MassRatioToSymmetricMassRatioTransform.forward(p))
)
samples = jim.get_samples()

diag   = jim.get_diagnostics()
n_iter = diag["n_iterations"]                       # adjust key if different
S      = STEPS_PER_DIM * len(jim.prior.parameter_names) * sampler_config.num_gibbs_sweeps
kbar   = diag["n_likelihood_evaluations"] / (n_iter * n_delete * S)
t_iter = (sampling_time - compile_s) / n_iter
n_trip = t_iter / per_trip[n_delete]
gamma_eff = n_trip / (S * kbar)
print(f"compile {compile_s:.1f}s | t_iter {t_iter:.2f}s | kbar {kbar:.2f} | trips/iter {n_trip:.0f} | gamma_eff {gamma_eff:.1f}")

het_logl = np.asarray(samples["log_likelihood"])
n_post = len(het_logl)
idx = np.r_[np.argmax(het_logl), np.random.default_rng(0).choice(n_post, min(N_CHECK, n_post), replace=False)]
ex_logl = np.array([
    float(exact_log_likelihood({k: jnp.asarray(samples[k][i]) for k in jim.prior.parameter_names}))
    for i in idx
])
max_dlogl = float(np.max(np.abs(het_logl[idx] - ex_logl)))

# for a correct likelihood the peak logL is ~ (network matched-filter SNR)^2 / 2
half_rho_sq = 0.5 * sum(abs(complex(ifo.match_filtered_snr)) ** 2 for ifo in ifos)
peak_logl = float(het_logl.max())
print(f"max |logL_het - logL_exact| over {len(idx)} samples: {max_dlogl:.3g}")
print(f"peak logL {peak_logl:.1f} vs SNR_mf^2/2 {half_rho_sq:.1f}")
if max_dlogl > CHECK_TOL:
    print(f"WARNING: heterodyne disagrees with exact likelihood by {max_dlogl:.3g} > {CHECK_TOL}; posterior unreliable")

# chains = save_results(OUT, jim, {
#     "n_live": sampler_config.n_live,
#     "n_delete": n_delete,
#     "us_per_posterior_eval": us_per_eval,
#     "ms_per_batch": {str(B): t * 1e3 for B, t in timings.items()},
#     "sampling_time_s": sampling_time,
#     "check_max_abs_dlogl_het_vs_exact": max_dlogl,
#     "check_n_samples": len(idx),
#     "check_passed": max_dlogl <= CHECK_TOL,
#     "peak_log_likelihood": peak_logl,
#     "half_matched_filter_snr_sq": half_rho_sq,
# })

chains = save_results(OUT, jim, {
    # run configuration
    "n_live": sampler_config.n_live,
    "n_delete": n_delete,
    "steps_per_dim": STEPS_PER_DIM,
    "delete_frac": DELETE_FRAC,
    "sampler_seed": SAMPLER_SEED,

    # benchmarks
    "us_per_posterior_eval": us_per_eval,
    "single_ms": {str(B): t * 1e3 for B, t in timings.items()},      # one jitted vmap call, median
    "per_trip_ms": {str(B): t * 1e3 for B, t in per_trip.items()},   # per while_loop trip, marginal

    # timing decomposition
    "sampling_time_s": sampling_time,
    "compile_s": compile_s,
    "compile_n": compile_timer.n,          # 0 ⇒ log parsing failed; compile_s is not trustworthy
    "n_iterations": n_iter,
    "t_iter": t_iter,                      # s per iteration, net of compilation
    "kbar": kbar,                          # mean likelihood calls per slice step per lane
    "n_trip": n_trip,                      # sequential loop trips per iteration (inferred)
    "gamma_eff": gamma_eff,                # n_trip / (S · kbar): stragglers + bookkeeping overhead

    # correctness checks
    "check_max_abs_dlogl_het_vs_exact": max_dlogl,
    "check_n_samples": len(idx),
    "check_passed": max_dlogl <= CHECK_TOL,
    "peak_log_likelihood": peak_logl,
    "half_matched_filter_snr_sq": half_rho_sq,
})

print(f"Likelihood evaluations: {jim.get_diagnostics()['n_likelihood_evaluations']:,}")

truths = [truth_values[k] for k in jim.prior.parameter_names]


fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
    truths=truths,
    quantiles=[0.05, 0.5, 0.95],
    show_titles=True,
)
fig.savefig(OUT / "corner.png", dpi=150)