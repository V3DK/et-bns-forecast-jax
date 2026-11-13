"""Pure-JAX log-likelihood and log-prior for one event, with all event data passed in as `EventData`.

jimgw's likelihood and prior hold per-event arrays and floats as Python attributes, so a jitted sampler
step embeds them as constants and recompiles for every event (driver-plan.md, section 1). Here, every
event-specific number comes in through the `ev` argument. `EventModel` holds only what is the same for
every event: the waveform, the ET detector geometry, the parameter layout and the fixed prior bounds.

Rule: nothing derived from an event may be closed over. Everything per-event goes through `ev`.

The functions copy jimgw 0.5.2's `HeterodynedTransientLikelihoodFD._likelihood`, `_generate_waveform`,
`_evaluate_from_waveform`, `GroundBased2G.fd_response` and the `CombinePrior` in swig.py. They are checked
against jimgw by check_event_model.py (test T1).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from ripplegw.interfaces import DistanceScaledWaveform

from jimgw.core.single_event.detector import get_ET
from jimgw.core.utils import log_i0

# Prior order in swig.py, which is also jimgw's sampling-array order.
ALL_PARAMS = ("M_c", "q", "lambda_1", "lambda_2",
              "s1_z", "s2_z", "iota", "d_L", "t_c", "ra", "dec", "psi")
INTRINSIC_PARAMS = ALL_PARAMS[:4]
# Held at injected values when intrinsic_only (order of EventData.fixed; as FIXED in swig.py).
FIXED_PARAMS = ("s1_z", "s2_z", "iota", "d_L", "ra", "dec", "psi", "t_c")
DETECTOR_NAMES = ("ET1", "ET2", "ET3")

# Event-independent priors (swig.py). M_c is uniform on [ev.mc_lo, ev.mc_hi].
FIXED_PRIORS = {
    "q": ("uniform", 0.5, 1.0),
    "lambda_1": ("uniform", 0.0, 5000.0),
    "lambda_2": ("uniform", 0.0, 5000.0),
    "s1_z": ("uniform", -0.05, 0.05),
    "s2_z": ("uniform", -0.05, 0.05),
    "iota": ("sine",),
    "d_L": ("powerlaw", 10.0, 2000.0, 2.0),
    "t_c": ("uniform", -0.1, 0.1),
    "ra": ("uniform", 0.0, 2 * jnp.pi),
    "dec": ("cosine",),
    "psi": ("uniform", 0.0, jnp.pi),
}


def sampled_params(intrinsic_only):
    return INTRINSIC_PARAMS if intrinsic_only else ALL_PARAMS


def gibbs_blocks(intrinsic_only):
    """SwiG blocks as in swig.py. Part of the compiled program: change them only between warm jobs."""
    if intrinsic_only:
        return [list(INTRINSIC_PARAMS)]
    return [
        ["M_c", "q", "lambda_1", "lambda_2"],
        ["s1_z", "s2_z"],
        ["iota", "d_L"],
        ["ra", "dec", "psi"],
        ["t_c"],
    ]


class EventData(NamedTuple):
    """Everything per-event that the kernel reads. Shapes are fixed by n_bins (3 detectors)."""
    f_lo: jax.Array       # [n_bins] lower bin edges
    f_hi: jax.Array       # [n_bins] upper bin edges
    bin_w: jax.Array      # [n_bins]
    ref_lo: jax.Array     # [n_det, n_bins] complex: reference detector response at f_lo
    ref_hi: jax.Array     # [n_det, n_bins] complex
    coeffs: jax.Array     # [n_det, 4, n_bins] complex: A0, A1, B0, B1
    t_offset: jax.Array   # [n_det] trigger_time - detector start_time (s); small, unlike GPS
    gmst: jax.Array       # scalar (rad)
    mc_lo: jax.Array      # scalar: M_c prior bounds
    mc_hi: jax.Array      # scalar
    fixed: jax.Array      # [len(FIXED_PARAMS)] injected values, read only when intrinsic_only


def event_data_from_likelihood(likelihood, mc_lo, mc_hi, injection_params):
    """Pack a set-up `BinnedHeterodynedLikelihoodFD` into an `EventData` (host side)."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("enable jax_enable_x64 before building EventData")
    names = tuple(likelihood.detector_names)
    if names != DETECTOR_NAMES:
        raise ValueError(f"detectors {names}, expected {DETECTOR_NAMES}")
    if not likelihood.phase_marginalization:
        raise ValueError("EventModel implements the phase-marginalised likelihood only")
    trigger = float(likelihood.trigger_time)
    return EventData(
        f_lo=jnp.asarray(likelihood.freq_grid_low),
        f_hi=jnp.asarray(likelihood.freq_grid_high),
        bin_w=jnp.asarray(likelihood.bin_widths),
        ref_lo=jnp.stack([likelihood.waveform_low_ref[n] for n in names]),
        ref_hi=jnp.stack([likelihood.waveform_high_ref[n] for n in names]),
        coeffs=jnp.stack([likelihood.summary_data[n] for n in names]),
        # trigger and start_time are within a factor 2 of each other, so this subtraction is exact
        t_offset=jnp.array([trigger - float(d.start_time) for d in likelihood.detectors]),
        gmst=jnp.asarray(likelihood.gmst, dtype=jnp.float64),
        mc_lo=jnp.asarray(mc_lo, dtype=jnp.float64),
        mc_hi=jnp.asarray(mc_hi, dtype=jnp.float64),
        fixed=jnp.array([injection_params[k] for k in FIXED_PARAMS], dtype=jnp.float64),
    )


# ---------------------------------------------------------------- priors (match jimgw's log_prob)

def _uniform(x, lo, hi):
    u = (x - lo) / (hi - lo)
    return jnp.where((u >= 0.0) & (u <= 1.0), -jnp.log(hi - lo), -jnp.inf)


def _sine(x):
    # SinePrior: uniform on cos(x) in [-1, 1], Jacobian |sin x|, bounded to [0, pi]
    return jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.abs(jnp.sin(x))) - jnp.log(2.0), -jnp.inf)


def _cosine(x):
    # CosinePrior: SinePrior on x + pi/2, bounded to [-pi/2, pi/2]
    inside = (x >= -jnp.pi / 2) & (x <= jnp.pi / 2)
    return jnp.where(inside, _sine(x + jnp.pi / 2), -jnp.inf)


def _powerlaw(x, xmin, xmax, alpha):
    a1 = 1.0 + alpha
    norm = xmax ** a1 - xmin ** a1
    u = (x ** a1 - xmin ** a1) / norm
    return jnp.where((u >= 0.0) & (u <= 1.0), jnp.log(jnp.abs(a1 * x ** alpha / norm)), -jnp.inf)


def _fixed_prior(name, x):
    kind, *args = FIXED_PRIORS[name]
    return {"uniform": _uniform, "sine": _sine, "cosine": _cosine, "powerlaw": _powerlaw}[kind](x, *args)


# ---------------------------------------------------------------- model

class EventModel:
    """Pure functions of (x, ev). `x` is the sampling array in `self.names` order."""

    def __init__(self, waveform, intrinsic_only):
        if not isinstance(waveform, DistanceScaledWaveform):
            raise TypeError("the waveform cache assumes a DistanceScaledWaveform (unit-distance cache)")
        self.waveform = waveform
        self.intrinsic_only = intrinsic_only
        self.names = sampled_params(intrinsic_only)
        self.n_dims = len(self.names)
        # data-free detectors: only their geometry (antenna_pattern, delay_from_geocenter) is used
        self.detectors = get_ET()
        assert tuple(d.name for d in self.detectors) == DETECTOR_NAMES

    # --- prior

    def log_prior(self, x, ev):
        out = jnp.zeros(())
        for i, name in enumerate(self.names):
            out = out + (_uniform(x[i], ev.mc_lo, ev.mc_hi) if name == "M_c" else _fixed_prior(name, x[i]))
        return out

    # --- likelihood

    def likelihood_params(self, x, ev):
        """Sampling array -> likelihood-space dict, as Jim + SingleEventLikelihood._prepare_parameters."""
        p = {name: x[i] for i, name in enumerate(self.names)}
        if self.intrinsic_only:
            p.update({name: ev.fixed[j] for j, name in enumerate(FIXED_PARAMS)})
        q = p.pop("q")
        p["eta"] = q / (1 + q) ** 2           # MassRatioToSymmetricMassRatioTransform
        p["phase_c"] = 0.0                    # phase marginalisation
        p["gmst"] = ev.gmst
        return p

    def _response(self, d, f, h_sky, p, ev):
        """GroundBased2G.fd_response with `trigger_time - start_time` replaced by ev.t_offset[d]."""
        det = self.detectors[d]
        antenna = det.antenna_pattern(p["ra"], p["dec"], p["psi"], p["gmst"])
        time_shift = det.delay_from_geocenter(p["ra"], p["dec"], p["gmst"])
        time_shift = time_shift + (ev.t_offset[d] + p["t_c"])     # same association as jimgw
        h_det = jax.tree_util.tree_map(lambda h, a: h * a, h_sky, antenna)
        projected = jnp.sum(jnp.stack(jax.tree_util.tree_leaves(h_det)), axis=0)
        return projected * jnp.exp(-2j * jnp.pi * f * time_shift)

    def _likelihood(self, p, h_low, h_high, ev):
        """HeterodynedTransientLikelihoodFD._likelihood with phase marginalisation."""
        log_l = jnp.zeros(())
        d_inner_h = jnp.zeros((), dtype=jnp.complex128)
        for d in range(len(self.detectors)):
            r_low = self._response(d, ev.f_lo, h_low, p, ev) / ev.ref_lo[d]
            r_high = self._response(d, ev.f_hi, h_high, p, ev) / ev.ref_hi[d]
            r0 = (r_low + r_high) / 2
            r1 = (r_high - r_low) / ev.bin_w
            A0, A1, B0, B1 = ev.coeffs[d, 0], ev.coeffs[d, 1], ev.coeffs[d, 2], ev.coeffs[d, 3]
            d_inner_h += jnp.sum(A0 * r0.conj() + A1 * r1.conj())
            optimal = jnp.sum(B0 * jnp.abs(r0) ** 2 + 2 * B1 * (r0 * r1.conj()).real)
            log_l += -optimal.real / 2
        return log_l + log_i0(jnp.absolute(d_inner_h))

    def log_likelihood(self, x, ev):
        p = self.likelihood_params(x, ev)
        return self._likelihood(p, self.waveform(ev.f_lo, p), self.waveform(ev.f_hi, p), ev)

    def build_cache(self, x, ev):
        """Bin-edge polarisations at unit distance (d_L is not a cache dependency)."""
        p = self.likelihood_params(x, ev)
        return {"low": self.waveform.at_unit_distance(ev.f_lo, p),
                "high": self.waveform.at_unit_distance(ev.f_hi, p)}

    def ll_from_cache(self, x, cache, ev):
        p = self.likelihood_params(x, ev)
        scale = 1.0 / p["d_L"]
        low = {k: v * scale for k, v in cache["low"].items()}
        high = {k: v * scale for k, v in cache["high"].items()}
        return self._likelihood(p, low, high, ev)

    def log_posterior(self, x, ev):
        return self.log_prior(x, ev) + self.log_likelihood(x, ev)
