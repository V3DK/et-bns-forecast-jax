"""
gwfast waveform wrapper for the Soultanis, Bauswein & Stergioulas (2022)
post-merger model (MPA1), built on the user's pm_model.py.

SNR-only.  Fisher matrices are NOT supported (amplitude is a tabulated,
non-differentiable function of mass; phase is set to zero).

How it works
------------
gwfast's SNRInteg only ever calls  wf_model.Ampl(f, **evParams)  and
multiplies by Fp*(1+cos^2 i)/2 and Fx*cos i itself, summing |A+|^2+|Ax|^2.
That is exact for the PM model because h_x is h_+ shifted by pi/2, so
H_x(f) = i H_+(f) at positive frequencies (up to negligible negative-frequency
leakage).  So Ampl only needs the face-on |H_+(f)| in the detector frame:

    |H_obs(f)| = (1+z) |RH_src((1+z) f ; M_src)| / d_L

with M_src = (Mc_det / (1+z)) / eta^(3/5).

|RH_src(f_s; M)| is tabulated once on a mass grid (nearest-node lookup in M:
linear interpolation between nodes with shifted spectral lines would
underestimate the power by up to ~50%).
"""

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18

from gwfast.waveforms import WaveFormModel
import pm_waveform_model as pm

_KEYS = ('fpeak_ts zeta_drift t_star f_spiral f_2m0 f_2p0 '
         'A_peak A_spiral A_2m0 A_2p0 tau_peak tau_spiral tau_2m0 tau_2p0 '
         'phi_peak phi_spiral phi_2m0 phi_2p0 N s').split()

GPC_TO_M = 3.085677581491367e25


class PM_MPA1(WaveFormModel):
    """
    Post-merger waveform for gwfast (SNR only).

    Parameters
    ----------
    M_lo, M_hi : source-frame total-mass range of the table [Msun].
                 Set to the paper's calibrated range.  Events outside it get
                 zero amplitude (explicit choice; see out_of_range).
    dM         : mass-grid spacing [Msun] (nearest-node lookup).
    f_src_lo, f_src_hi : source-frame band kept in the table [Hz].
    n_fft      : zero-padded FFT length (dt = pm.DT_MS); 2**16 -> df ~ 0.76 Hz.
    amp_norm   : length [m] converting the model's r*h to SI.  Default copies
                 hfpc (GM_sun/c^2).  CHECK THIS: hfpc's docstring says r*h is in
                 km, in which case amp_norm should be 1e3.
    cosmo      : astropy cosmology used to invert d_L -> z.  Must match the one
                 used to build the catalogue.
    """

    def __init__(self, M_lo=2.4, M_hi=3.0, dM=0.002,
                 f_src_lo=500., f_src_hi=6000., n_fft=2**16,
                 amp_norm=pm.GM_SUN_OVER_C2, cosmo=Planck18, **kwargs):
        # is_LAL=True: skips gwfast's JAX warm-up (which would run a BBH
        # FisherMatr through this numpy model and fail).  SNRInteg is unaffected.
        super().__init__('BNS', fcutPar=f_src_hi, is_LAL=True, **kwargs)
        self.f_src_hi = f_src_hi

        # ---- spectral table |RH_src(f_s; M)| [m * s] at 1 m distance
        self.M_grid = np.arange(M_lo, M_hi + 0.5 * dM, dM)
        t  = np.arange(0.0, pm.T_MAX_MS + pm.DT_MS, pm.DT_MS)
        dt = pm.DT_MS * pm.MS_TO_S
        fs = np.fft.rfftfreq(n_fft, dt)
        band = (fs >= f_src_lo) & (fs <= f_src_hi)
        self.fs = fs[band]
        self.table = np.empty((len(self.M_grid), band.sum()))
        for j, M in enumerate(self.M_grid):
            p = pm.get_params(M)
            rh = pm._h_td(t, *[p[k] for k in _KEYS])          # same taper as hfpc
            self.table[j] = np.abs(np.fft.rfft(rh, n=n_fft)[band]) * dt * amp_norm

        # ---- d_L [Gpc] -> z
        zt = np.concatenate([[0.0], np.geomspace(1e-6, 20.0, 20000)])
        self._zt  = zt
        self._dLt = cosmo.luminosity_distance(zt).to_value(u.Gpc)

    # ------------------------------------------------------------ helpers
    def z_of_dL(self, dL):
        return np.interp(np.asarray(dL, float), self._dLt, self._zt)

    def M_src(self, Mc, eta, dL):
        z = self.z_of_dL(dL)
        return np.asarray(Mc, float) / (1 + z) / np.asarray(eta, float)**0.6, z

    @staticmethod
    def out_of_range(M_src, M_lo, M_hi):
        return (M_src < M_lo) | (M_src > M_hi)

    # ------------------------------------------------------------ gwfast API
    def Ampl(self, f, **kw):
        """
        f : (n_f, n_ev) detector-frame grid from SNRInteg [Hz]
        returns face-on |H_+| (dimensionless strain / Hz), same shape.
        """
        f = np.asarray(f, float)
        M, z = self.M_src(kw['Mc'], kw['eta'], kw['dL'])
        dL_m = np.asarray(kw['dL'], float) * GPC_TO_M

        j = np.clip(np.rint((M - self.M_grid[0]) / (self.M_grid[1] - self.M_grid[0])),
                    0, len(self.M_grid) - 1).astype(int)
        bad = self.out_of_range(M, self.M_grid[0], self.M_grid[-1])

        f2 = f.reshape(f.shape[0], -1)
        out = np.empty_like(f2)
        for k in range(f2.shape[1]):                        # loop over events
            fsrc = (1 + z.flat[k]) * f2[:, k]
            out[:, k] = np.interp(fsrc, self.fs, self.table[j.flat[k]], left=0., right=0.)
        out *= np.atleast_1d((1 + z) / dL_m)[None, :]
        out[:, np.atleast_1d(bad)] = 0.0
        return out.reshape(f.shape)

    def Phi(self, f, **kw):          # irrelevant for SNR
        return np.zeros_like(np.asarray(f, float))

    def hphc(self, f, **kw):         # only reached by GWstrain / Fisher
        A = self.Ampl(f, **kw)
        return A, 1j * A

    def tau_star(self, f, **kw):     # PM lasts ~0.1 s; Earth motion irrelevant
        return np.zeros_like(np.asarray(f, float))

    def fcut(self, **kw):
        """Detector-frame upper limit of the SNR grid."""
        z = self.z_of_dL(kw['dL'])
        return self.f_src_hi / (1 + z)