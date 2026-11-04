"""
ET-D (ETS) BNS detection rates with gwfast:
  (1) inspiral    : TaylorF2_RestrictedPN, 1.4+1.4 Msun, fmin = 5 Hz
  (2) post-merger : PM_MPA1, Galactic-DNS M_tot population, fmin scanned 100 -> 1000 Hz
Both share cosmology, merger-rate model, z grid (to z=10) and extrinsic draws.
"""
import os, copy
import numpy as np
import jax
import astropy.units as u
from astropy.cosmology import Planck18 as COSMO

import scipy.integrate as igt
if not hasattr(igt, 'cumtrapz'):
    igt.cumtrapz = igt.cumulative_trapezoid
if not hasattr(igt, 'simps'):
    igt.simps = igt.simpson
from scipy.integrate import simpson

import gwfast.gwfastGlobals as glob
from gwfast.waveforms import TaylorF2_RestrictedPN
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from pm_gwfast import PM_MPA1

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------- config
RHO_TH_INSP = 8.0
RHO_TH_PM = 5.0

NZ, NS = 60, 5000
NZ_PM, Z_MAX_PM = 60, 0.5
SEED   = 10
Z_MAX  = 10.0
RES = 1000

# inspiral
FMIN_INSP  = 5.0                # Hz
BATCH_INSP = 20_000

# post-merger
FMIN_PM = 300 # Hz
BATCH_PM = 5_000
M_LO, M_HI = 2.0, 3.2           # calibrated range of the PM fits (source-frame M_tot)
M_THR      = 3.2                # MPA1 prompt-collapse threshold, guess / approximate (see: https://arxiv.org/abs/1307.5191)
T_OBS, DUTY, Z_REF = 1.0, 1.0, 0.0

# 90% credible interval on BNS R0, https://arxiv.org/abs/2508.18083
R0_min, R0_max = 7.6, 250.0     # Gpc^-3 yr^-1
# R0_min, R0_max = 13.0, 170.0     # Gpc^-3 yr^-1

# DTD: P(t_d) ∝ t_d^-ALPHA on [T_D_MIN, T_D_MAX]
T_D_MIN, T_D_MAX, ALPHA = 0.02, COSMO.age(0).to_value(u.Gyr), 1.0
Z_FORM_MAX, NTD = 15.0, 10

PSD = os.path.join(glob.detPath, 'ET-0000A-18.txt')   # ET-D, ASD
ETS = copy.deepcopy(glob.detectors['ETS'])

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
    """R(z) = ∫ psi(z_f(z, t_d)) P(t_d) dt_d, t_d ∈ [T_D_MIN, min(T_D_MAX, t_L(Z_FORM_MAX) - t_L(z))]."""
    out = []
    for zi in np.atleast_1d(z):
        tLz, tcap = t_L(zi), min(T_D_MAX, T_FORM_MAX - t_L(zi))
        if tcap <= T_D_MIN:
            out.append(0.0); continue
        td = np.geomspace(T_D_MIN, tcap, NTD)
        out.append(simpson(psi_MF17(z_of_tL(tLz + td)) * P_dtd(td), x=td))
    return np.array(out)

def merger_rate_density(z, R0=1.0, z_ref=0.0):
    return R0 * rate_dtd(z) / rate_dtd(z_ref)[0]

# ---------------------------------------------------------------- PM mass population
def sample_mtot(rng, n, m_lo=1.0, m_hi=2.5): # see mass model from https://arxiv.org/abs/2508.18083
    m1 = rng.uniform(m_lo, m_hi, n)
    m2 = rng.uniform(m_lo, m_hi, n)
    return m1 + m2

# ---------------------------------------------------------------- catalogue (shared extrinsics)
rng = np.random.default_rng(SEED)
eta = 0.25                                       # both models equal-mass

def make_base(zg, ns):
    n  = len(zg) * ns
    dL = COSMO.luminosity_distance(zg).to_value(u.Gpc)
    return np.repeat(zg, ns), dict(
        eta     = np.full(n, eta),
        dL      = np.repeat(dL, ns),
        theta   = np.arccos(rng.uniform(-1, 1, n)),
        phi     = rng.uniform(0, 2 * np.pi, n),
        iota    = np.arccos(rng.uniform(-1, 1, n)),
        psi     = rng.uniform(0, np.pi, n),
        tcoal   = np.zeros(n), Phicoal = np.zeros(n),
        chi1z   = np.zeros(n), chi2z   = np.zeros(n),
    )

# inspiral: fixed 1.4 + 1.4, full grid
z_grid       = np.geomspace(1e-3, Z_MAX, NZ)
z_i, base_i  = make_base(z_grid, NS)
M_INSP       = 2.8
ev_insp      = dict(base_i, Mc=M_INSP * eta**0.6 * (1.0 + z_i))

# post-merger: dedicated low-z grid
z_grid_pm    = np.geomspace(1e-3, Z_MAX_PM, NZ_PM)
z_p, base_p  = make_base(z_grid_pm, NS)
Mt     = sample_mtot(rng, len(z_p))
ev_pm  = dict(base_p, Mc=Mt * eta**0.6 * (1.0 + z_p))
has_PM = Mt < M_THR
in_cal = (Mt >= M_LO) & (Mt <= M_HI)
print(f"f_PM = {has_PM.mean():.3f}   outside calibration = {(~in_cal).mean():.3f}")

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
print("\n[inspiral]  TaylorF2, 1.4+1.4")
print(HDR)
snr_i = compute_snr(make_net(TaylorF2_RestrictedPN(), FMIN_INSP), ev_insp, BATCH_INSP)
P, err, N_hat = rate_from_det(snr_i >= RHO_TH_INSP, z_grid)
# np.savez(f'pdet_ETS_insp_fmin{FMIN_INSP:g}.npz', z=z_grid, P_det=P, P_err=err,
#          rho_th=RHO_TH, fmin=FMIN_INSP)
print(row(FMIN_INSP, N_hat))

# ---------------------------------------------------------------- (2) post-merger, fmin scan
print("\n[post-merger]  PM_MPA1")
print(HDR)
wf_pm   = PM_MPA1(M_lo=M_LO, M_hi=M_HI, cosmo=COSMO)

snr_p = compute_snr(make_net(wf_pm, FMIN_PM), ev_pm, BATCH_PM)
# out-of-calibration events have SNR=0 inside Ampl -> N_det is a lower bound
P, err, N_hat = rate_from_det((snr_p >= RHO_TH_PM) & has_PM, z_grid_pm)
# np.savez(f'pdet_ETS_PM_fmin{fmin:g}.npz', z=z_grid, P_det=P, P_err=err,
#             rho_th=RHO_TH, fmin=fmin, snr=snr_p, Mt=Mt)
print(row(FMIN_PM, N_hat))






# ---------------------------------------------------------------- plot: cumulative detection rate vs z
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def dNdz_det(det, zg):
    """Detected dN/dz per R0 on z_f."""
    P   = det.reshape(len(zg), NS).mean(axis=1)
    P_f = np.interp(np.log(z_f), np.log(zg), P, right=0.0)
    return T_OBS * DUTY * dN_intr * P_f

curves = [
    (rf'inspiral ($\rho_{{\rm th}}={RHO_TH_INSP:g}$, $f_{{\rm min}}={FMIN_INSP:g}$ Hz)',
     dNdz_det(snr_i >= RHO_TH_INSP, z_grid), 'C0', 0.30),
    (rf'post-merger ($\rho_{{\rm th}}={RHO_TH_PM:g}$, $f_{{\rm min}}={FMIN_PM:g}$ Hz)',
     dNdz_det((snr_p >= RHO_TH_PM) & has_PM, z_grid_pm), 'C3', 0.30),
]

fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
for lab, dn, c, a in curves:
    cum    = cumulative_trapezoid(dn, z_f, initial=0.0)
    lo, hi = R0_min * cum, R0_max * cum
    ax.fill_between(z_f, lo, hi, color=c, alpha=a, lw=0, label=lab)
    ax.plot(z_f, lo, color=c, lw=0.8)
    ax.plot(z_f, hi, color=c, lw=0.8)
    print(f"{lab:>60s}: N(<z_max) = {lo[-1]:.3g} – {hi[-1]:.3g} /yr")

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(z_f[0], z_f[-1]); ax.set_ylim(bottom=1e-3)
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$N(<z)$ [yr$^{-1}$]')
ax.grid(which='both', alpha=0.2)
ax.legend(fontsize=8, loc='lower right')
ax.set_title(rf'ET-D (ETS), $R_0\in[{R0_min:g},\,{R0_max:g}]$ Gpc$^{{-3}}$ yr$^{{-1}}$ (90% CI)', fontsize=10)
fig.savefig(f'Ncum_vs_z_ETS_PMfmin{FMIN_PM:g}.pdf')
plt.show()