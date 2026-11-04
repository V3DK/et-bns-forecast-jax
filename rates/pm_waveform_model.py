"""
GWBench-compatible post-merger waveform model for BNS systems.

Implements the analytic model of Soultanis, Bauswein & Stergioulas (2022),
arXiv:2111.08353, calibrated to the MPA1 EOS.

Two distinct roles
------------------
hfpc(f, ...)
    Master waveform function consumed by GWBench.  All model parameters are
    treated as *independent* Fisher variables; the M_tot empirical relations
    are NOT used here.  f is in Hz (GWBench convention); internal conversion
    to kHz is handled inside hfpc.

Helper functions (chirp_mass, fpeak_ts, A_peak, ...)
    Implement the M_tot empirical relations from the paper.  Used only by the
    injection notebook to produce the true parameter values for a given M_tot.
    They are not called by hfpc.

Unit conventions
----------------
  GWBench / hfpc : f in Hz, DL in Mpc, tc in s
  Internal model : f in kHz, t in ms, DL in km
  Amplitude      : h_plus / h_cross return r*h [km]; hfpc divides by DL [km]
                   to obtain dimensionless strain.
"""

import numpy as np
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.constants import G, c

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SOLAR_MASS = 1.98892e30 # in kg
MPC_TO_M  = 3.085_677_581_5e22   # 1 Mpc in metres
GM_SUN_OVER_C2 = G * SOLAR_MASS / c**2  # GM_sun / c^2 in metres  (~1.477 km)
MS_TO_S   = 1e-3                  # 1 ms in s
KHZ_TO_HZ = 1e3                   # 1 kHz in Hz

# Default EOS parameter
R18 = 12.55   # radius [km] of a 1.8 M_sun non-rotating NS (MPA1)

# Time grid parameters
T_MAX_MS = 100.0    # post-merger signal duration [ms]
DT_MS    = 0.02     # time step [ms]  ->  f_Nyquist = 25 kHz >> 8 kHz


deriv_mod       = 'numdifftools'
wf_symbs_string = ('f '
                   'fpeak_ts_ zeta_drift_ t_star_ '
                   'f_spiral_ f_2m0_ f_2p0_ '
                   'A_peak_ A_spiral_ A_2m0_ A_2p0_ '
                   'tau_peak_ tau_spiral_ tau_2m0_ tau_2p0_ '
                   'phi_peak_ phi_spiral_ phi_2m0_ phi_2p0_ '
                   'N_ s_ '
                   'DL tc phic iota')

# ===========================================================================
# Section 1 – M_tot empirical relations  (injection use only)
#             All equation numbers refer to arXiv:2111.08353
# ===========================================================================

def chirp_mass(M_tot: float) -> float:
    """Chirp mass [M_sun] for an equal-mass binary with total mass M_tot."""
    return (M_tot / 2.0) * 2.0 ** (-0.2)


def zeta_drift(M_tot: float) -> float:
    """Linear frequency-drift rate zeta_drift [kHz^2]. Eq. (10)."""
    return (-1.420 * M_tot**3
            + 11.085 * M_tot**2
            - 28.834 * M_tot
            + 24.943)


def t_star(M_tot: float) -> float:
    """Transition time t* [ms]. Eq. (12)."""
    return -8.523 * M_tot**2 + 40.179 * M_tot - 40.741


def fpeak_ts(M_tot: float) -> float:
    """Dominant-mode frequency fpeak(t*) [kHz]. Derived from Eqs. (10-12)."""
    fp0 = 0.908 * M_tot**2 - 3.974 * M_tot + 7.058   # Eq. (11)
    return fp0 + zeta_drift(M_tot) * t_star(M_tot)


def fpeak_mean(M_tot: float) -> float:
    """Time-averaged <fpeak(t)> [kHz] over [0, t*]."""
    return fpeak_ts(M_tot) - 0.5 * zeta_drift(M_tot) * t_star(M_tot)


def A_peak(M_tot: float) -> float:
    """Dimensionless amplitude of fpeak component. Eq. (13)."""
    return -0.409 * M_tot**2 + 3.657 * M_tot - 6.130


def tau_peak(M_tot: float) -> float:
    """Damping timescale tau_peak [ms]. Eq. (14)."""
    return 7.782 * M_tot**2 - 53.040 * M_tot + 93.542


def tau_spiral(M_tot: float) -> float:
    """Damping timescale tau_spiral [ms]. Eq. (15)."""
    return -0.874 * M_tot**2 + 3.521 * M_tot - 2.005


def tau_2m0(M_tot: float) -> float:
    """Damping timescale tau_{2-0} [ms]. Eq. (16)."""
    return 2.057 * M_tot**2 - 10.804 * M_tot + 14.606


def tau_2p0(M_tot: float) -> float:
    """Damping timescale tau_{2+0} [ms]. Eq. (17)."""
    return 8.469 * M_tot**2 - 48.785 * M_tot + 71.671


def A_spiral(M_tot: float) -> float:
    """Dimensionless amplitude A_spiral. Eq. (18)."""
    return 2.649 * M_tot**2 - 13.580 * M_tot + 17.752


def A_2m0(M_tot: float) -> float:
    """Dimensionless amplitude A_{2-0}. Eq. (19)."""
    return -1.704 * M_tot**2 + 10.004 * M_tot - 13.909


def A_2p0(M_tot: float) -> float:
    """Dimensionless amplitude A_{2+0}. Eq. (20)."""
    return 0.816 * M_tot**2 - 3.920 * M_tot + 4.734


def N_norm(M_tot: float) -> float:
    """Global normalization factor N. Eq. (21)."""
    return -0.485 * M_tot + 2.025


def f_0() -> float:
    """Quasi-radial oscillation frequency f_0 [kHz]. Fixed at 1 kHz."""
    return 1.0


def f_spiral(M_tot: float, R18: float = R18) -> float:
    """Spiral-mode frequency f_spiral [kHz]."""
    Mc  = chirp_mass(M_tot)
    lhs = (6.264
           + 1.929  * Mc
           - 0.645  * R18
           + 0.881  * Mc**2
           - 0.311  * R18 * Mc
           + 0.030  * R18**2)
    return lhs * Mc


def f_2m0(M_tot: float) -> float:
    """Lower combination tone f_{2-0} [kHz]."""
    return fpeak_mean(M_tot) - f_0()


def f_2p0(M_tot: float) -> float:
    """Upper combination tone f_{2+0} [kHz]."""
    return fpeak_mean(M_tot) + f_0()


def phi_peak(M_tot: float) -> float:
    """Initial phase phi_peak [rad]. Eq. (C1)."""
    return 18.957 * M_tot - 46.321 if M_tot <= 2.7 else 43.425 * M_tot - 113.152


def phi_spiral(M_tot: float) -> float:
    """Initial phase phi_spiral [rad]. Eq. (C2)."""
    return 17.580 * M_tot - 42.199 if M_tot <= 2.7 else 40.448 * M_tot - 104.258


def phi_2m0(M_tot: float) -> float:
    """Initial phase phi_{2-0} [rad]. Eq. (C3)."""
    return 18.541 * M_tot - 43.911 if M_tot <= 2.7 else 43.613 * M_tot - 112.705


def phi_2p0(M_tot: float) -> float:
    """Initial phase phi_{2+0} [rad]. Eq. (C4)."""
    return 16.064 * M_tot - 41.163 if M_tot <= 2.7 else 43.309 * M_tot - 115.341


def tukey_s(M_tot: float) -> float:
    """Tukey roll-off parameter s."""
    return 0.075 if M_tot <= 2.9 else 0.1


def get_params(M_tot: float, R18: float = R18) -> dict:
    """
    Collect all analytic-model parameters for a given M_tot.
    Returns a dict with all frequencies in kHz, times in ms, phases in rad.
    This is the single entry-point the notebook should use to build inj_params.
    """
    return dict(
        M_tot      = M_tot,
        fpeak_ts   = fpeak_ts(M_tot),
        zeta_drift = zeta_drift(M_tot),
        t_star     = t_star(M_tot),
        f_spiral   = f_spiral(M_tot, R18),
        f_2m0      = f_2m0(M_tot),
        f_2p0      = f_2p0(M_tot),
        f_0        = f_0(),
        A_peak     = A_peak(M_tot),
        A_spiral   = A_spiral(M_tot),
        A_2m0      = A_2m0(M_tot),
        A_2p0      = A_2p0(M_tot),
        tau_peak   = tau_peak(M_tot),
        tau_spiral = tau_spiral(M_tot),
        tau_2m0    = tau_2m0(M_tot),
        tau_2p0    = tau_2p0(M_tot),
        phi_peak   = phi_peak(M_tot),
        phi_spiral = phi_spiral(M_tot),
        phi_2m0    = phi_2m0(M_tot),
        phi_2p0    = phi_2p0(M_tot),
        N          = N_norm(M_tot),
        s          = tukey_s(M_tot),
    )


# ===========================================================================
# Section 2 – Time-domain waveform engine  (shared by hfpc)
# ===========================================================================

def _phi_peak_t(t: np.ndarray, fpeak_ts_: float, zeta_drift_: float,
                t_star_: float, phi_peak_: float) -> np.ndarray:
    """
    Instantaneous GW phase Phi_peak(t) [rad] of the dominant component.
    Eq. (3): piecewise quadratic (t <= t*) then linear (t > t*).
    All inputs in kHz / ms / rad.
    """
    fp0   = fpeak_ts_ - zeta_drift_ * t_star_
    phi   = np.empty_like(t, dtype=float)

    mask_early = t <= t_star_
    mask_late  = ~mask_early

    te = t[mask_early]
    phi[mask_early] = 2.0 * np.pi * (fp0 * te + 0.5 * zeta_drift_ * te**2) + phi_peak_

    phi_ts = 2.0 * np.pi * (fp0 * t_star_ + 0.5 * zeta_drift_ * t_star_**2) + phi_peak_
    fp_ts  = fp0 + zeta_drift_ * t_star_

    tl = t[mask_late]
    phi[mask_late] = phi_ts + 2.0 * np.pi * fp_ts * (tl - t_star_)

    return phi


def _h_td(t: np.ndarray,
           fpeak_ts_: float, zeta_drift_: float, t_star_: float,
           f_spiral_: float, f_2m0_: float, f_2p0_: float,
           A_peak_: float, A_spiral_: float, A_2m0_: float, A_2p0_: float,
           tau_peak_: float, tau_spiral_: float, tau_2m0_: float, tau_2p0_: float,
           phi_peak_: float, phi_spiral_: float, phi_2m0_: float, phi_2p0_: float,
           N_: float, s_: float,
           cross: bool = False) -> np.ndarray:
    """
    Compute r*h_{+,x}(t) [km] on the supplied time grid.
    All frequencies in kHz, times in ms.
    cross=False -> h_+  (sin phases)
    cross=True  -> h_x  (sin phases + pi/2, i.e. cos phases)
    """
    ps = np.pi / 2.0 if cross else 0.0

    Phi_peak = _phi_peak_t(t, fpeak_ts_, zeta_drift_, t_star_, phi_peak_)
    W        = tukey(len(t), alpha=s_)

    h_peak = (A_peak_
              * np.exp(-t / tau_peak_)
              * np.sin(Phi_peak + ps)
              * W)

    h_spi  = (A_spiral_
              * np.exp(-t / tau_spiral_)
              * np.sin(2.0 * np.pi * f_spiral_ * t + phi_spiral_ + ps))

    h_2m0  = (A_2m0_
              * np.exp(-t / tau_2m0_)
              * np.sin(2.0 * np.pi * f_2m0_    * t + phi_2m0_   + ps))

    h_2p0  = (A_2p0_
              * np.exp(-t / tau_2p0_)
              * np.sin(2.0 * np.pi * f_2p0_    * t + phi_2p0_   + ps))

    return N_ * (h_peak + h_spi + h_2m0 + h_2p0)


# ===========================================================================
# Section 3 – Master waveform function for GWBench
# ===========================================================================

def hfpc(f,
         fpeak_ts_, zeta_drift_, t_star_,
         f_spiral_, f_2m0_, f_2p0_,
         A_peak_, A_spiral_, A_2m0_, A_2p0_,
         tau_peak_, tau_spiral_, tau_2m0_, tau_2p0_,
         phi_peak_, phi_spiral_, phi_2m0_, phi_2p0_,
         N_, s_,
         DL, tc, phic, iota):
    """
    Frequency-domain post-merger waveform for GWBench.

    All model parameters are treated as independent Fisher variables.
    The M_tot empirical relations are NOT used here.

    Parameters
    ----------
    f          : np.ndarray  — frequency array from GWBench [Hz]
    fpeak_ts_  : float       — fpeak(t*) [kHz]
    zeta_drift_: float       — linear frequency-drift rate [kHz^2]
    t_star_    : float       — transition time t* [ms]
    f_spiral_  : float       — spiral-mode frequency [kHz]
    f_2m0_     : float       — lower combination tone [kHz]
    f_2p0_     : float       — upper combination tone [kHz]
    A_peak_    : float       — amplitude of dominant component [dimensionless]
    A_spiral_  : float       — amplitude of spiral component [dimensionless]
    A_2m0_     : float       — amplitude of f_{2-0} component [dimensionless]
    A_2p0_     : float       — amplitude of f_{2+0} component [dimensionless]
    tau_peak_  : float       — damping time of dominant component [ms]
    tau_spiral_: float       — damping time of spiral component [ms]
    tau_2m0_   : float       — damping time of f_{2-0} component [ms]
    tau_2p0_   : float       — damping time of f_{2+0} component [ms]
    phi_peak_  : float       — initial phase of dominant component [rad]
    phi_spiral_: float       — initial phase of spiral component [rad]
    phi_2m0_   : float       — initial phase of f_{2-0} component [rad]
    phi_2p0_   : float       — initial phase of f_{2+0} component [rad]
    N_         : float       — global normalization factor
    s_         : float       — Tukey roll-off parameter
    DL         : float       — luminosity distance [Mpc]
    tc         : float       — coalescence time offset [s]
    phic       : float       — coalescence phase [rad]
    iota       : float       — inclination angle [rad]

    Returns
    -------
    hfp : np.ndarray  — plus  polarisation, dimensionless strain [Hz^{-1}]
    hfc : np.ndarray  — cross polarisation, dimensionless strain [Hz^{-1}]

    Notes
    -----
    Procedure
      1. Build a uniform time grid [0, T_MAX_MS] with step DT_MS [ms].
      2. Compute r*h_{+,x}(t) [km] via _h_td (time-domain engine).
      3. FFT to the frequency domain; dt factor gives units of km*ms = km*1e-3*s.
      4. Multiply by GM_sun/c^2 [m] (geometrized -> SI length) then
         divide by DL [m] (DL [Mpc] * MPC_TO_M) to get dimensionless strain.
      5. Apply inclination factors F_+ = (1+cos^2 iota)/2, F_x = cos(iota).
      6. Multiply by the coalescence phase factor
            pf(f) = exp(i * 2*pi * (f*tc - phic/(2*pi)))
         where f is in Hz and tc in s, matching the lal_bns convention.
      7. Interpolate the FFT grid onto the GWBench frequency array f.
         Points outside the FFT range are set to zero.
    """
    # ------------------------------------------------------------------
    # 1. Time grid  (ms)
    # ------------------------------------------------------------------
    t_ms = np.arange(0.0, T_MAX_MS + DT_MS, DT_MS)
    N_t  = len(t_ms)
    dt_s = DT_MS * MS_TO_S          # time step in seconds

    # ------------------------------------------------------------------
    # 2. Time-domain strain  r*h [km]
    # ------------------------------------------------------------------
    rh_plus  = _h_td(t_ms,
                     fpeak_ts_, zeta_drift_, t_star_,
                     f_spiral_, f_2m0_, f_2p0_,
                     A_peak_, A_spiral_, A_2m0_, A_2p0_,
                     tau_peak_, tau_spiral_, tau_2m0_, tau_2p0_,
                     phi_peak_, phi_spiral_, phi_2m0_, phi_2p0_,
                     N_, s_, cross=False)

    rh_cross = _h_td(t_ms,
                     fpeak_ts_, zeta_drift_, t_star_,
                     f_spiral_, f_2m0_, f_2p0_,
                     A_peak_, A_spiral_, A_2m0_, A_2p0_,
                     tau_peak_, tau_spiral_, tau_2m0_, tau_2p0_,
                     phi_peak_, phi_spiral_, phi_2m0_, phi_2p0_,
                     N_, s_, cross=True)

    # ------------------------------------------------------------------
    # 3. FFT  ->  units: km * s  (km * dt_s per sample)
    # ------------------------------------------------------------------
    Hfp_fft  = np.fft.rfft(rh_plus)  * dt_s
    Hfc_fft  = np.fft.rfft(rh_cross) * dt_s
    f_fft_hz = np.fft.rfftfreq(N_t, d=dt_s)   # Hz

    # ------------------------------------------------------------------
    # 4. Convert to dimensionless strain  [s = Hz^{-1}]
    #    Model returns r*h in geometrized units (c=G=M_sun=1), so
    #    multiply by GM_sun/c^2 [m] to make it dimensionful, then
    #    divide by DL [m].
    # ------------------------------------------------------------------
    DL_m    = DL * MPC_TO_M
    Hfp_fft = Hfp_fft * (GM_SUN_OVER_C2 / DL_m)
    Hfc_fft = Hfc_fft * (GM_SUN_OVER_C2 / DL_m)

    # ------------------------------------------------------------------
    # 5. Inclination factors
    # ------------------------------------------------------------------
    F_plus  = 0.5 * (1.0 + np.cos(iota)**2)
    F_cross = np.cos(iota)

    Hfp_fft = F_plus  * Hfp_fft
    Hfc_fft = F_cross * Hfc_fft

    # ------------------------------------------------------------------
    # 6. Coalescence phase factor  pf(f) = exp(i*2*pi*(f*tc - phic/(2*pi)))
    # ------------------------------------------------------------------
    pf_fft = np.exp(1j * 2.0 * np.pi * (f_fft_hz * tc - phic / (2.0 * np.pi)))
    Hfp_fft = pf_fft * Hfp_fft
    Hfc_fft = pf_fft * Hfc_fft

    # ------------------------------------------------------------------
    # 7. Interpolate onto GWBench frequency array
    #    fill_value=0 for frequencies outside the FFT grid
    # ------------------------------------------------------------------
    interp_p = interp1d(f_fft_hz, Hfp_fft, kind='linear',
                        bounds_error=False, fill_value=0.0)
    interp_c = interp1d(f_fft_hz, Hfc_fft, kind='linear',
                        bounds_error=False, fill_value=0.0)

    hfp = interp_p(f)
    hfc = interp_c(f)

    return hfp, hfc