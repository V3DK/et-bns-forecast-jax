"""
gwfast waveform wrapper for the NRPMw post-merger model
(Breschi et al. 2022, arXiv:2205.09112), vendored from bajes 1.3.0 in
_nrpmw_bajes.py.  Spec: NRPMw.md, plan: NRPMw_plan.md.

SNR-only (no Fisher).  Interface mirrors pm_gwfast.PM_MPA1.

How it works
------------
gwfast's SNRInteg calls  Ampl(f, **evParams)  and multiplies by
Fp*(1+cos^2 i)/2 and Fx*cos i.  bajes builds  hp = h22 (1+cos^2 i)/2,
hc = h22 cos i e^{-i pi/2},  so the cross term vanishes exactly and
Ampl = |h22(f)|  (h22 already contains sqrt(5/4pi) M^2/D).

Frames: every NRPMw QUR quantity is in units of M, so passing
(mtot = M_det, distance = d_L) gives the observed h(f_obs) directly.
The exception is t_coll, which is a physical source-frame time:
    NRPMw_t_coll = t_coll / (M_src * MTSUN_SI)   [units of M]
(equivalently t_coll(1+z) / (M_det MTSUN_SI)), so z is needed.

Model variant
-------------
attach=True (default here, by user choice) drops the fusion (merger)
wavelet, i.e. the PM-only variant ('NRPMw_attach' in bajes).  This deviates
from NRPMw.md's 4-wavelet spec; the quoted ~0.13 fully-informed
unfaithfulness refers to the full model.  In attach mode the bounce wavelet
is extended 5 ms *backward* from t_1 (bajes' `_dt`, computed with
mtot = M_det, i.e. 5 ms in the detector frame).

Masks applied inside Ampl (amplitude set to 0)
-------------------------------------------------
* source-frame band  f_src in [f_src_lo, f_src_hi]  (NRPMw.md: 1-8 kHz)
* calibration box    M_src in cal['M'], q in cal['q'], k2t in cal['k2t']
  -> N_det is a lower bound w.r.t. out-of-calibration events.
Prompt collapse is NOT applied here (done in rate_calculation.py).
"""

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18

from gwfast.waveforms import WaveFormModel
import _nrpmw_bajes as nrpmw

MTSUN_SI = nrpmw.MTSUN_SI
CAL_BOX  = dict(M=(2.4, 3.4), q=(1.0, 2.0), k2t=(22.0, 458.0))   # Breschi+22 calibration set


def q_of_eta(eta):
    """q = m1/m2 >= 1 from symmetric mass ratio."""
    eta = np.asarray(eta, float)
    d = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    return (1.0 - 2.0 * eta + d) / (2.0 * eta)


def kappa2T(m1, m2, L1, L2):
    """kappa_2^T, identical to bajes' lambda_2_kappa (NRPMw.md)."""
    return nrpmw.lambda_2_kappa(np.asarray(m1, float), np.asarray(m2, float),
                                np.asarray(L1, float), np.asarray(L2, float))


def calibration_flags(M, q, k2t, cal=CAL_BOX):
    """Boolean arrays: inside each cut of the calibration box (source-frame M)."""
    ok_M = (M >= cal['M'][0]) & (M <= cal['M'][1])
    ok_q = (q >= cal['q'][0]) & (q <= cal['q'][1])
    ok_k = (k2t >= cal['k2t'][0]) & (k2t <= cal['k2t'][1])
    return ok_M, ok_q, ok_k


class PM_NRPMw(WaveFormModel):
    """
    NRPMw post-merger waveform for gwfast (SNR only).

    Parameters
    ----------
    t_coll_ms : collapse time after merger, source frame [ms]   (spec: 8)
    df2_geom  : f_2 drift, dimensionless M^2 alpha_peak         (spec: 2e-6)
    phi_pm    : PM phase jump [rad]                             (spec: pi/2)
    f_src_lo, f_src_hi : source-frame band [Hz]                 (spec: 1-8 kHz)
    attach    : True -> PM-only variant (no fusion wavelet)
    cal       : calibration box; events outside get zero amplitude
    cosmo     : astropy cosmology for d_L -> z (must match the catalogue)
    """

    def __init__(self, t_coll_ms=8.0, df2_geom=2e-6, phi_pm=np.pi / 2,
                 f_src_lo=1000., f_src_hi=8000., attach=True,
                 cal=CAL_BOX, cosmo=Planck18, **kwargs):
        # is_LAL=True: skip gwfast's JAX warm-up (numpy model). SNRInteg unaffected.
        super().__init__('BNS', fcutPar=f_src_hi, is_LAL=True, **kwargs)
        self.t_coll_s = t_coll_ms * 1e-3
        self.df2_geom = df2_geom
        self.phi_pm   = phi_pm
        self.f_src_lo, self.f_src_hi = f_src_lo, f_src_hi
        self.attach   = attach
        self.cal      = cal

        zt = np.concatenate([[0.0], np.geomspace(1e-6, 20.0, 20000)])
        self._zt  = zt
        self._dLt = cosmo.luminosity_distance(zt).to_value(u.Gpc)

        self._cache = None
        self.n_calls = 0            # NRPMw evaluations
        self.n_sanity = 0           # events with exact-zero in-band bins (bajes _sanity_check)

    # ------------------------------------------------------------ helpers
    def z_of_dL(self, dL):
        return np.interp(np.asarray(dL, float), self._dLt, self._zt)

    def event_params(self, **kw):
        """Per-event arrays: z, M_det, M_src, q, k2t, in_cal."""
        z     = self.z_of_dL(kw['dL'])
        eta   = np.asarray(kw['eta'], float)
        M_det = np.asarray(kw['Mc'], float) / eta**0.6
        M_src = M_det / (1 + z)
        q     = q_of_eta(eta)
        m1, m2 = M_src * q / (1 + q), M_src / (1 + q)
        k2t   = kappa2T(m1, m2, kw['Lambda1'], kw['Lambda2'])
        in_cal = np.logical_and.reduce(calibration_flags(M_src, q, k2t, self.cal))
        return z, M_det, M_src, q, k2t, in_cal

    def nrpmw_params(self, M_det, M_src, q, L1, L2, chi1z, chi2z, dL_Gpc):
        """Fresh bajes params dict for one event (bajes mutates its input)."""
        return dict(mtot=float(M_det), q=float(q),
                    lambda1=float(L1), lambda2=float(L2),
                    s1z=float(chi1z), s2z=float(chi2z),
                    distance=float(dL_Gpc) * 1e3,                 # Mpc
                    phi_ref=0.0, seglen=1.0,                      # seglen >> t_coll
                    NRPMw_t_coll=self.t_coll_s / (float(M_src) * MTSUN_SI),
                    NRPMw_df_2=self.df2_geom,
                    NRPMw_phi_pm=self.phi_pm)

    def h22(self, f, M_det, M_src, q, L1, L2, chi1z=0., chi2z=0., dL_Gpc=1.):
        """Complex h22(f) [1/Hz] for one event on detector-frame f [Hz]."""
        p = self.nrpmw_params(M_det, M_src, q, L1, L2, chi1z, chi2z, dL_Gpc)
        return nrpmw.NRPMw(np.asarray(f, float), p, attach=self.attach)

    def fits(self, M_det, M_src, q, L1, L2, chi1z=0., chi2z=0.):
        """QUR outputs + wavelet times (geometric units) for diagnostics."""
        p = self.nrpmw_params(M_det, M_src, q, L1, L2, chi1z, chi2z, 1.0)
        return nrpmw._nrpmw_fits(p, attach=self.attach)

    # ------------------------------------------------------------ gwfast API
    def _key(self, f, kw):
        return [np.asarray(f)] + [np.asarray(kw[k]) for k in
                                  ('Mc', 'eta', 'dL', 'Lambda1', 'Lambda2', 'chi1z', 'chi2z')]

    def Ampl(self, f, **kw):
        """
        f : (n_f, n_ev) detector-frame grid from SNRInteg [Hz]
        returns |h22| (face-on |h_+|, 1/Hz), same shape.
        """
        key = self._key(f, kw)
        if self._cache is not None and len(key) == len(self._cache[0]) and \
                all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(key, self._cache[0])):
            return self._cache[1].copy()

        f = np.asarray(f, float)
        z, M_det, M_src, q, _, in_cal = self.event_params(**kw)
        L1, L2 = np.asarray(kw['Lambda1'], float), np.asarray(kw['Lambda2'], float)
        c1 = np.broadcast_to(np.asarray(kw.get('chi1z', 0.), float), M_det.shape)
        c2 = np.broadcast_to(np.asarray(kw.get('chi2z', 0.), float), M_det.shape)
        dL = np.asarray(kw['dL'], float)

        f2  = f.reshape(f.shape[0], -1)
        out = np.zeros_like(f2)
        for k in np.flatnonzero(np.atleast_1d(in_cal)):
            fk   = f2[:, k]
            fsrc = (1 + z.flat[k]) * fk
            band = (fsrc >= self.f_src_lo) & (fsrc <= self.f_src_hi)
            if not band.any():
                continue
            a = np.abs(self.h22(fk[band], M_det.flat[k], M_src.flat[k], q.flat[k],
                                L1.flat[k], L2.flat[k], c1.flat[k], c2.flat[k], dL.flat[k]))
            self.n_calls += 1
            if np.any(a == 0.0):
                self.n_sanity += 1
            out[band, k] = a
        out = out.reshape(f.shape)
        self._cache = ([a.copy() for a in key], out.copy())
        return out

    def Phi(self, f, **kw):          # irrelevant for SNR
        return np.zeros_like(np.asarray(f, float))

    def hphc(self, f, **kw):         # only reached by GWstrain / Fisher
        A = self.Ampl(f, **kw)
        return A, -1j * A            # bajes: hc = h cos(i) e^{-i pi/2}

    def tau_star(self, f, **kw):     # PM lasts < 0.1 s; Earth motion irrelevant
        return np.zeros_like(np.asarray(f, float))

    def fcut(self, **kw):
        """Detector-frame upper limit of the SNR grid."""
        return self.f_src_hi / (1 + self.z_of_dL(kw['dL']))
