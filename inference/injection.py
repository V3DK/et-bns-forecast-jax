"""Rebuild one catalogue event on the host: injected ET strain plus the binned heterodyned likelihood.

Follows swig.py (same detectors, PSD, band, duration rule, merger cutoff and priors). The difference is
that every per-event number comes from a catalogue row instead of module constants. swig.py is left as
is while the CSD3 runs that use it are in progress.
"""
from dataclasses import dataclass
from pathlib import Path

import bilby
import bilby.gw.conversion as conv
import jax
import jax.numpy as jnp
import lal
import lalsimulation as lalsim
import numpy as np
import pandas as pd
from bilby.gw.conversion import lambda_from_mass_and_family

from jimgw.core.single_event.detector import get_ET
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

from heterodyne import BinnedHeterodynedLikelihoodFD, DEFAULT_N_BINS

EOS = "MPA1"
F_REF = 20.0
F_MIN = 5.0
F_MAX = 2000.0
SAMPLING_FREQUENCY = 4096.0
POST_TRIGGER = 2.0          # inject_signal puts 2 s of data after the trigger
SAFETY = 1.1                # duration margin over the longest template's chirp time
CUTOFF_THRESHOLD = 0.2      # see merger_cutoff_frequency
MC_HALF_WIDTH_FRAC = 0.01   # M_c prior: injected M_c +- 1%
PSD_FILE = Path(bilby.__file__).parent / "gw/detector/noise_curves/ET_D_psd.txt"

_EOS_FAMILY = lalsim.CreateSimNeutronStarFamily(lalsim.SimNeutronStarEOSByName(EOS))


def make_waveform():
    return RippleIMRPhenomD_NRTidalv2(f_ref=F_REF)


def read_catalogue(path):
    """Rows as dicts with Python scalars, indexed by event_id."""
    df = pd.read_csv(path)
    assert list(df.event_id) == list(range(len(df))), "event_id must be 0..N-1 in order"
    return [{k: (int(v) if k in ("event_id", "noise_seed", "sampler_seed") else float(v))
             for k, v in row.items()} for row in df.to_dict("records")]


def injection_parameters(row):
    """Likelihood-space parameters (eta, detector-frame M_c) of a catalogue row."""
    z = row["z"]
    m1_det, m2_det = (1 + z) * row["m1_src"], (1 + z) * row["m2_src"]
    p = {
        "M_c": conv.component_masses_to_chirp_mass(m1_det, m2_det),
        "eta": conv.component_masses_to_symmetric_mass_ratio(m1_det, m2_det),
        "s1_z": 0.0,
        "s2_z": 0.0,
        "lambda_1": lambda_from_mass_and_family(row["m1_src"], _EOS_FAMILY),
        "lambda_2": lambda_from_mass_and_family(row["m2_src"], _EOS_FAMILY),
        "iota": row["iota"],
        "d_L": row["d_L"],
        "ra": row["ra"],
        "dec": row["dec"],
        "psi": row["psi"],
        "t_c": 0.0,
        "phase_c": 0.0,
    }
    return {k: float(v) for k, v in p.items()}


def mc_bounds(mc):
    half = MC_HALF_WIDTH_FRAC * mc
    return mc - half, mc + half


def data_duration(mc_lo):
    """Power-of-two duration holding the longest template in the prior from F_MIN to merger.

    Longest = lowest M_c, equal masses, max prior spin (conservative bound), as in swig.py.
    """
    m_eq = mc_lo * 2 ** 0.2 * lal.MSUN_SI
    chirp_time = lalsim.SimInspiralChirpTimeBound(F_MIN, m_eq, m_eq, 0.05, 0.05)
    return float(2 ** np.ceil(np.log2(SAFETY * chirp_time + POST_TRIGGER)))


def merger_cutoff_frequency(waveform, params, f_lo, f_hi, threshold, n=20000):
    """First frequency where |h| drops below `threshold` x its inspiral trend (|h| ~ f^-7/6).

    The heterodyne divides by the reference waveform, so it breaks where the reference
    tapers to zero after merger; the likelihood band must stop before that. (Copy of swig.py.)
    """
    f = jnp.linspace(f_lo, f_hi, n)
    h = waveform(f, {k: jnp.asarray(v) for k, v in params.items()})
    amp = np.sqrt(np.abs(np.asarray(h["p"])) ** 2 + np.abs(np.asarray(h["c"])) ** 2)
    f = np.asarray(f)
    trend = amp * f ** (7 / 6)
    trend /= np.median(trend[f < 3 * f_lo])
    below = np.nonzero(trend < threshold)[0]
    return float(f[below[0]]) if below.size else float(f_hi)


@dataclass
class Injection:
    row: dict
    params: dict              # likelihood space
    ifos: list                # jimgw detectors holding data + PSD
    gps: float
    duration: float
    f_max_likelihood: float
    mc_lo: float
    mc_hi: float


def inject(row, waveform):
    """Simulate the event's strain in the three ET detectors (signal + Gaussian noise)."""
    params = injection_parameters(row)
    mc_lo, mc_hi = mc_bounds(params["M_c"])
    duration = data_duration(mc_lo)
    gps = row["gps"]

    ifos = get_ET()
    noise_key = jax.random.key(row["noise_seed"])
    for ifo, key in zip(ifos, jax.random.split(noise_key, len(ifos))):   # independent noise per ifo
        ifo.load_and_set_psd(psd_file=str(PSD_FILE))
        ifo.inject_signal(
            duration=duration,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=gps,
            waveform_model=waveform,
            parameters=params,
            f_min=F_MIN,
            f_max=F_MAX,
            zero_noise=False,
            rng_key=key,
        )
    f_max_lik = min(F_MAX, merger_cutoff_frequency(waveform, params, F_MIN, F_MAX, CUTOFF_THRESHOLD))
    return Injection(row=row, params=params, ifos=ifos, gps=gps, duration=duration,
                     f_max_likelihood=f_max_lik, mc_lo=mc_lo, mc_hi=mc_hi)


def make_likelihood(inj, waveform, n_bins=DEFAULT_N_BINS, fixed_names=()):
    """Heterodyned likelihood with exactly `n_bins` bins, referenced to the injection.

    `fixed_names` are held at their injected values (the INTRINSIC_ONLY setup). Raises if jimgw
    dropped bins beyond the reference waveform's support: a different n_bins means a different
    compiled program.
    """
    likelihood = BinnedHeterodynedLikelihoodFD(
        inj.ifos,
        waveform=waveform,
        fixed_parameters={k: inj.params[k] for k in fixed_names},
        trigger_time=inj.gps,
        f_min=F_MIN,
        f_max=inj.f_max_likelihood,
        reference_parameters=inj.params,
        phase_marginalization=True,
        n_bins=n_bins,
    )
    if likelihood.n_bins != n_bins:
        raise RuntimeError(f"event {inj.row['event_id']}: heterodyne has {likelihood.n_bins} bins, "
                           f"expected {n_bins} (bins dropped beyond the reference waveform's support)")
    return likelihood
