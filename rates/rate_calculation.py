import os, copy
import numpy as np
import jax
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as COSMO

import scipy.integrate as igt
if not hasattr(igt, 'cumtrapz'):
    igt.cumtrapz = igt.cumulative_trapezoid
if not hasattr(igt, 'simps'):
    igt.simps = igt.simpson
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

import lal
import lalsimulation as lalsim
from bilby.gw.conversion import lambda_from_mass_and_family

import gwfast.gwfastGlobals as glob
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD_NRTidalv2
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from pm_gwfast import PM_MPA1
from pm_nrpmw_gwfast import PM_NRPMw, kappa2T, calibration_flags, CAL_BOX

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------- config
RHO_TH_INSP = 12.0 # network SNR >= 12 used in arXiv:2303.15923, 
RHO_TH_PM = 8.0 # a lower SNR of 8 justifiable when counterpart exists (see arXiv:2303.15923, Section 4 intro)

NZ, NS = 60, 5000
Z_MAX  = 10.0
NZ_PM, Z_MAX_PM = 60, 0.5

SEED   = 10
RES = 1000

# inspiral
FMIN_INSP  = 5.0                # Hz
BATCH_INSP = 20_000

# post-merger
PM_MODEL = 'NRPMw'              # 'NRPMw' (Breschi+22, PM-only/attach) or 'SBS22' (Soultanis+22, MPA1)
BATCH_PM = 5_000
if PM_MODEL == 'SBS22':
    FMIN_PM = 300 # Hz, checked that 100-500 makes no difference
    M_LO, M_HI = 2.4, 3.1       # calibrated range of the PM fits (source-frame M_tot)
elif PM_MODEL == 'NRPMw':
    F_SRC_LO_PM, F_SRC_HI_PM = 1000., 8000.   # source-frame band (NRPMw.md); cut inside Ampl
    T_COLL_MS, DF2_GEOM, PHI_PM = 8.0, 2e-6, np.pi / 2
    FMIN_PM = F_SRC_LO_PM / (1 + Z_MAX_PM)    # ≈ 667 Hz detector frame: grid covers the band at all z
    M_LO, M_HI = CAL_BOX['M']   # 2.4–3.4; q and kappa2T cuts below
else:
    raise ValueError(PM_MODEL)
T_OBS, DUTY, Z_REF = 1.0, 1.0, 0.0 # T_OBS 1 year

# prompt collapse threshold
# see Koeppel 2019 (arXiv:1901.09977) for M_tov, R_tov -> M_thr
# see Fan 24 (arXiv:2309.12644) for 68.3% interval on M_tov
Z90 = 1.6449 # widen to 90% (assumes gaussian)
M_TOV_0, M_TOV_ERR = 2.25, (0.07, 0.08)     # Msun, (minus, plus), 68.3%
R_TOV_0, R_TOV_ERR = 11.90, (0.60, 0.63)    # km,   (minus, plus), 68.3%
K_B, K_C  = 1.01, 1.34
K_A       = 2.0 * K_B / (2.0 - K_C)
GM_SUN_KM = (const.G * const.M_sun / const.c**2).to_value(u.km) 
 
def m_thr_koeppel(M_tov, R_tov):
    C = GM_SUN_KM * M_tov / R_tov
    return (K_A - K_B / (1.0 - K_C * C)) * M_tov

_M = (M_TOV_0 - Z90 * M_TOV_ERR[0], M_TOV_0 + Z90 * M_TOV_ERR[1])
_R = (R_TOV_0 - Z90 * R_TOV_ERR[0], R_TOV_0 + Z90 * R_TOV_ERR[1])
M_THR_MIN = m_thr_koeppel(_M[0], _R[0])
M_THR_MAX = m_thr_koeppel(_M[1], _R[1])
_corners  = [m_thr_koeppel(m, r) for m in _M for r in _R]
assert np.isclose(min(_corners), M_THR_MIN) and np.isclose(max(_corners), M_THR_MAX), \
    "M_thr not monotonic over the box; like-sign corners do not bound it"

# 90% credible interval on BNS R0 (https://iopscience.iop.org/article/10.3847/2041-8213/ae771e)
R0_min, R0_max = 8.8, 250.0     # Gpc^-3 yr^-1

# DTD: P(t_d) ∝ t_d^-ALPHA on [T_D_MIN, T_D_MAX]
T_D_MIN = 0.02 # 0.02 Gyr = 20 Myr from arXiv:1201.3563
T_D_MAX = COSMO.age(0).to_value(u.Gyr)
ALPHA = 1.0

Z_FORM_MAX = 15.0 # value doesn't really matter:
# for low z, delay time has to be large to reach Z_FORM_MAX -> unlikely due to DTD
# for high z, cap hits but SNR for ET is already basically 0
NTD = 128 # checked that above 128 resolution stops changing anything

PSD = os.path.join(glob.detPath, 'ET-0000A-18.txt')   # ET-D, ASD
ETS = copy.deepcopy(glob.detectors['ETS'])

EOS = "MPA1"
_fam = lalsim.CreateSimNeutronStarFamily(lalsim.SimNeutronStarEOSByName(EOS))
M_MAX_EOS = lalsim.SimNeutronStarMaximumMass(_fam) / lal.MSUN_SI
M_MIN_EOS = lalsim.SimNeutronStarFamMinimumMass(_fam) / lal.MSUN_SI

M_NS_LO = 0.8
M_NS_HI = M_TOV_0 + Z90 * M_TOV_ERR[1]   # ≈ 2.38 Msun; must be defined after M_TOV_*

assert M_MIN_EOS <= M_NS_LO and M_NS_HI < M_MAX_EOS, \
    f"{EOS} family [{M_MIN_EOS:.3f}, {M_MAX_EOS:.3f}] does not cover [{M_NS_LO}, {M_NS_HI}]"

_m_tab   = np.linspace(M_NS_LO, M_NS_HI, 601)
_lam_tab = np.array([lambda_from_mass_and_family(m, _fam) for m in _m_tab])
_logLam  = CubicSpline(_m_tab, np.log(_lam_tab))

def Lambda_MPA1(m_src):
    m_src = np.asarray(m_src, float)
    if m_src.min() < M_NS_LO or m_src.max() > M_NS_HI:
        raise ValueError("source-frame mass outside tabulated Lambda range")
    return np.exp(_logLam(m_src))

# spline accuracy check at midpoints
_mid = 0.5 * (_m_tab[1:] + _m_tab[:-1])[::50]
_err = np.max(np.abs(Lambda_MPA1(_mid) / np.array([lambda_from_mass_and_family(m, _fam) for m in _mid]) - 1))
print(f"{EOS}: M_max = {M_MAX_EOS:.3f} Msun, Lambda(1.4) = {Lambda_MPA1(1.4):.1f}, "
      f"spline rel. err = {_err:.1e}")


# ---------------------------------------------------------------- cosmology
def dVc_dz(z):
    """All-sky dV_c/dz [Gpc^3]."""
    dv = COSMO.differential_comoving_volume(np.asarray(z, float))
    return (4.0 * np.pi * u.sr * dv).to_value(u.Gpc**3)

_zt  = np.concatenate([[0.0], np.geomspace(1e-6, Z_FORM_MAX, 20_000)])
_tLt = COSMO.lookback_time(_zt).to_value(u.Gyr)
t_L     = lambda z: np.interp(z, _zt, _tLt)
z_of_tL = lambda t: np.interp(t, _tLt, _zt)
T_FORM_MAX = t_L(Z_FORM_MAX)

# ---------------------------------------------------------------- SFR / rate
def psi_MF17(z):
    """Madau & Fragos (2017) SFRD [Msun yr^-1 Mpc^-3]."""
    zp1 = 1.0 + np.asarray(z, float)
    return 0.01 * zp1**2.6 / (1.0 + (zp1 / 3.2)**6.2)

DTD_NORM = (np.log(T_D_MAX / T_D_MIN) if np.isclose(ALPHA, 1.0)
            else (T_D_MAX**(1 - ALPHA) - T_D_MIN**(1 - ALPHA)) / (1 - ALPHA))
P_dtd = lambda td: td**(-ALPHA) / DTD_NORM

def rate_dtd(z):
    z    = np.atleast_1d(np.asarray(z, float))
    tLz  = t_L(z)
    tcap = np.minimum(T_D_MAX, T_FORM_MAX - tLz)
    ok   = tcap > T_D_MIN
    out  = np.zeros_like(z)
    if not ok.any():
        return out
    # u = ln t_d, uniform on [ln T_D_MIN, ln tcap(z)] for each z
    s  = np.linspace(0.0, 1.0, NTD)                      # (NTD,)
    lu = np.log(T_D_MIN) + s[None, :] * np.log(tcap[ok] / T_D_MIN)[:, None]  # (Nz, NTD)
    td = np.exp(lu)
    f  = psi_MF17(z_of_tL(tLz[ok, None] + td)) * P_dtd(td) * td
    out[ok] = simpson(f, x=lu, axis=1)
    return out

def merger_rate_density(z, R0=1.0, z_ref=0.0):
    return R0 * rate_dtd(z) / rate_dtd(z_ref)[0] # scale rate to R0 local value we have


# ---------------------------------------------------------------- mass population

def sample_m1m2(rng, n, eps=0.35, per_binary=False,
                mu1=1.32, sigma1=0.11, mu2=1.80, sigma2=0.21,
                m_min=M_NS_LO, m_max=M_NS_HI):
    # Component masses from Sarin, Lasky & Ashton 2020 (arXiv:2001.06102), Eq. 12
    # per_binary=False: each star picks its Gaussian iid
    # per_binary=True:  one Gaussian is chosen per binary and both stars are drawn from it
    ma, mb = np.empty(n), np.empty(n)
    todo = np.ones(n, bool)
    while todo.any():
        idx = np.flatnonzero(todo)
        k = idx.size
        ha = rng.random(k) < eps
        hb = ha if per_binary else (rng.random(k) < eps)
        a = np.where(ha, rng.normal(mu2, sigma2, k), rng.normal(mu1, sigma1, k))
        b = np.where(hb, rng.normal(mu2, sigma2, k), rng.normal(mu1, sigma1, k))
        ok = (a >= m_min) & (a <= m_max) & (b >= m_min) & (b <= m_max)
        ma[idx[ok]], mb[idx[ok]] = a[ok], b[ok]
        todo[idx[ok]] = False
    return np.maximum(ma, mb), np.minimum(ma, mb)

# ---------------------------------------------------------------- catalogue (shared extrinsics)
rng = np.random.default_rng(SEED)

def make_catalogue(zg, ns):
    """Population masses + isotropic extrinsics on a z grid (ns draws per z node)."""
    n   = len(zg) * ns
    z   = np.repeat(zg, ns)
    dL  = COSMO.luminosity_distance(zg).to_value(u.Gpc)
    m1, m2 = sample_m1m2(rng, n)
    Mt  = m1 + m2
    eta = m1 * m2 / Mt**2
    ev = dict(
        Mc      = Mt * eta**0.6 * (1.0 + z),     # detector frame
        eta     = eta,
        Lambda1 = Lambda_MPA1(m1),               # source-frame masses; Lambda1 <-> heavier
        Lambda2 = Lambda_MPA1(m2),
        dL      = np.repeat(dL, ns),
        theta   = np.arccos(rng.uniform(-1, 1, n)),
        phi     = rng.uniform(0, 2 * np.pi, n),
        iota    = np.arccos(rng.uniform(-1, 1, n)),
        psi     = rng.uniform(0, np.pi, n),
        tcoal   = np.zeros(n), Phicoal = np.zeros(n),
        chi1z   = np.zeros(n), chi2z   = np.zeros(n),
    )
    return z, ev, m1, m2

# inspiral: full grid
z_grid       = np.geomspace(1e-3, Z_MAX, NZ)
z_i, ev_insp, m1_i, m2_i = make_catalogue(z_grid, NS)

# post-merger: dedicated low-z grid. Lambda1/2 are kept (NRPMw needs them; PM_MPA1 ignores them)
z_grid_pm    = np.geomspace(1e-3, Z_MAX_PM, NZ_PM) # checked that P_det -> 0 around Z_MAX_PM
z_p, ev_pm, m1_p, m2_p = make_catalogue(z_grid_pm, NS)
Mt  = m1_p + m2_p
q_p   = m1_p / m2_p
k2t_p = kappa2T(m1_p, m2_p, ev_pm['Lambda1'], ev_pm['Lambda2'])

# prompt collapse (NRPMw has no ringdown -> PM zeroed above M_thr): Koeppel band
has_PM_lo = Mt < M_THR_MIN      
has_PM_hi = Mt < M_THR_MAX      
if PM_MODEL == 'NRPMw':
    ok_M, ok_q, ok_k = calibration_flags(Mt, q_p, k2t_p)
    in_cal = ok_M & ok_q & ok_k   # must match the mask applied inside PM_NRPMw.Ampl
else:
    in_cal = (Mt >= M_LO) & (Mt <= M_HI)
print(f"M_thr in [{M_THR_MIN:.3f}, {M_THR_MAX:.3f}] Msun   "
      f"f_PM in [{has_PM_lo.mean():.3f}, {has_PM_hi.mean():.3f}]   "
      f"outside calibration = {(~in_cal).mean():.3f}   max(Mt) = {Mt.max():.3f}")
if PM_MODEL == 'NRPMw':
    for lab, sel in (('all', np.ones_like(Mt, bool)), ('Mt<M_thr_max', has_PM_hi)):
        print(f"  outside cal [{lab}]: M {(~ok_M[sel]).mean():.3f}  q {(~ok_q[sel]).mean():.3f}  "
              f"k2t {(~ok_k[sel]).mean():.3f}  any {(~in_cal[sel]).mean():.3f}   "
              f"k2t range [{k2t_p[sel].min():.0f}, {k2t_p[sel].max():.0f}]")

# ---------------------------------------------------------------- intrinsic rate integral
z_f     = np.geomspace(z_grid[0], z_grid[-1], 4000)
dN_intr = merger_rate_density(z_f, 1.0, Z_REF) * dVc_dz(z_f) / (1.0 + z_f)
N_intr  = T_OBS * simpson(dN_intr, x=z_f)
# print(f"rho_th = {RHO_TH}, intrinsic/R0 = {N_intr:.4g} /yr")

# ---------------------------------------------------------------- helpers
def make_net(wf, fmin):
    sig = GWSignal(wf, psd_path=PSD, is_ASD=True,
                   detector_shape=ETS['shape'], det_lat=ETS['lat'],
                   det_long=ETS['long'], det_xax=ETS['xax'],
                   useEarthMotion=False, fmin=fmin, verbose=False)
    return DetNet({'ETS': sig}, verbose=False)

def compute_snr(net, ev, batch):
    n   = len(ev['dL'])
    snr = np.empty(n)
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        snr[sl] = np.asarray(net.SNR({k: v[sl] for k, v in ev.items()}, res=RES))
    return snr

def rate_from_det(det, zg):
    P     = det.reshape(len(zg), NS).mean(axis=1)
    err   = np.sqrt(P * (1 - P) / NS)
    P_f   = np.interp(np.log(z_f), np.log(zg), P, right=0.0)
    N_hat = T_OBS * DUTY * simpson(dN_intr * P_f, x=z_f)
    return P, err, N_hat

HDR = f"{'fmin[Hz]':>8} {'N_det/R0':>10} {'eff':>9} {'N_det 90% range [/yr]':>26}"
def row(fmin, N_hat):
    return (f"{fmin:8.4g} {N_hat:10.4g} {N_hat/N_intr:9.3g} "
            f"{R0_min*N_hat:12.3g} – {R0_max*N_hat:.3g}")

# ---------------------------------------------------------------- (1) inspiral
summary = []
def report(line):
    """Print a results line and keep it for the summary file."""
    print(line)
    summary.append(line)

net_insp = make_net(IMRPhenomD_NRTidalv2(), FMIN_INSP)
report("[inspiral]  IMRPhenomD_NRTidalv2")
report(HDR)
snr_i = compute_snr(net_insp, ev_insp, BATCH_INSP)
P, err, N_hat = rate_from_det(snr_i >= RHO_TH_INSP, z_grid)
report(row(FMIN_INSP, N_hat))

# ---------------------------------------------------------------- (2) post-merger
if PM_MODEL == 'NRPMw':
    report(f"\n[post-merger]  NRPMw (attach/PM-only, fully informed)  t_coll = {T_COLL_MS:g} ms (src)  "
           f"M^2 alpha_peak = {DF2_GEOM:g}  phi_PM = {PHI_PM:.4g}  "
           f"f_src in [{F_SRC_LO_PM:g}, {F_SRC_HI_PM:g}] Hz")
    report(f"outside NRPMw calibration box: {(~in_cal).mean():.3f} (all), "
           f"{(~in_cal[has_PM_hi]).mean():.3f} (Mt < M_thr_max)")
    wf_pm = PM_NRPMw(t_coll_ms=T_COLL_MS, df2_geom=DF2_GEOM, phi_pm=PHI_PM,
                     f_src_lo=F_SRC_LO_PM, f_src_hi=F_SRC_HI_PM, attach=True, cosmo=COSMO)
else:
    report("\n[post-merger]  PM_MPA1")
    wf_pm = PM_MPA1(M_lo=M_LO, M_hi=M_HI, cosmo=COSMO)
report(HDR)

snr_p = compute_snr(make_net(wf_pm, FMIN_PM), ev_pm, BATCH_PM)
# out-of-calibration events have SNR=0 inside Ampl -> N_det is a lower bound

det_pm_lo = (snr_p >= RHO_TH_PM) & has_PM_lo
det_pm_hi = (snr_p >= RHO_TH_PM) & has_PM_hi
_, _, N_pm_lo = rate_from_det(det_pm_lo, z_grid_pm)
_, _, N_pm_hi = rate_from_det(det_pm_hi, z_grid_pm)


report(row(FMIN_PM, N_pm_lo) + f"   [M_thr = {M_THR_MIN:.3f}]")
report(row(FMIN_PM, N_pm_hi) + f"   [M_thr = {M_THR_MAX:.3f}]")
report(f"envelope (R0 x M_thr): {R0_min*N_pm_lo:.3g} – {R0_max*N_pm_hi:.3g} /yr   "
       f"N(M_thr_max)/N(M_thr_min) = {N_pm_hi/N_pm_lo:.3g}")
if PM_MODEL == 'NRPMw':
    report(f"NRPMw evaluations: {wf_pm.n_calls}   events with NaN/inf-zeroed bins: {wf_pm.n_sanity}")

# ---------------------------------------------------------------- save outputs
# SBS22 keeps its old name (plot.ipynb reads it)
OUT = (f'ETS_rates_PMfmin{FMIN_PM:g}.npz' if PM_MODEL == 'SBS22'
       else f'ETS_rates_PM{PM_MODEL}_fmin{FMIN_PM:.0f}.npz')
np.savez_compressed(
    OUT,
    # integration grid + intrinsic rate (per R0)
    z_f=z_f, dN_intr=dN_intr, N_intr=N_intr,
    # MC grids and raw per-event outputs
    z_grid=z_grid, z_grid_pm=z_grid_pm, NS=NS,
    snr_i=snr_i, snr_p=snr_p, Mt=Mt,
    # settings needed to reproduce selections/labels
    T_OBS=T_OBS, DUTY=DUTY,
    RHO_TH_INSP=RHO_TH_INSP, RHO_TH_PM=RHO_TH_PM,
    FMIN_INSP=FMIN_INSP, FMIN_PM=FMIN_PM,
    M_THR_MIN=M_THR_MIN, M_THR_MAX=M_THR_MAX,
    R0_min=R0_min, R0_max=R0_max,
    # PM model bookkeeping
    PM_MODEL=PM_MODEL, q_p=q_p, k2t_p=k2t_p, in_cal=in_cal,
    **(dict(n_sanity=wf_pm.n_sanity, T_COLL_MS=T_COLL_MS, DF2_GEOM=DF2_GEOM, PHI_PM=PHI_PM,
            F_SRC_LO_PM=F_SRC_LO_PM, F_SRC_HI_PM=F_SRC_HI_PM) if PM_MODEL == 'NRPMw' else {}),
)
print(f"saved -> {OUT}")

# same stem and folder as the PDF written by plot.ipynb
OUT_TXT = OUT.replace('.npz', '.txt')
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write("\n".join(summary) + "\n")
print(f"saved -> {OUT_TXT}")