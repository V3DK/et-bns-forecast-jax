"""Injection catalogue for the many-event driver: one CSV row per BNS event.

Every row is drawn from its own RNG, seeded by (MASTER_SEED, event_id). Row i is therefore the same
however many rows are generated, and the catalogue can be extended without changing earlier events:

    python make_catalogue.py --n 1000                       # rows 0..999
    python make_catalogue.py --n 9000                       # appends rows 1000..9999

The driver rebuilds each injection (strain, noise, likelihood) from its row, so no data is stored.

Population (driver-plan.md, "Decisions"):
- z uniform in comoving volume (Planck18) on [0, Z_MAX]; no (1+z) time-dilation factor.
- Source-frame masses uniform on [M_MIN, M_MAX], sorted so m1 >= m2 (q >= 0.58, inside the q prior).
- Isotropic orientation and sky; psi uniform on [0, pi]; gps uniform over one year from GPS_START.
- Spins zero, t_c = phase_c = 0 (set in injection.py, not stored here). EOS fixed (MPA1).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bilby.gw.cosmology import get_cosmology

MASTER_SEED = 20261113
COSMOLOGY = "Planck18"
Z_MAX = 0.1
M_MIN, M_MAX = 1.1, 1.9            # source frame, Msun; MPA1 Lambda(1.1) ~ 1920, inside [0, 5000]
GPS_START = 1735257618.0           # 2035-01-01T00:00:00 UTC
GPS_SPAN = 365.25 * 86400.0

COLUMNS = ["event_id", "z", "d_L", "m1_src", "m2_src", "iota", "ra", "dec", "psi", "gps",
           "noise_seed", "sampler_seed"]

DEFAULT_OUT = Path(__file__).resolve().parent / "catalogue.csv"


class _RedshiftSampler:
    """Inverse CDF of comoving volume on [0, Z_MAX], tabulated once."""

    def __init__(self, n_grid=20001):
        self.cosmo = get_cosmology(COSMOLOGY)
        self.z = np.linspace(0.0, Z_MAX, n_grid)
        v = self.cosmo.comoving_volume(self.z).value
        self.cdf = v / v[-1]

    def __call__(self, u):
        return float(np.interp(u, self.cdf, self.z))

    def d_L(self, z):
        return float(self.cosmo.luminosity_distance(z).value)


def draw_row(event_id, z_sampler):
    # Draw order is part of the catalogue definition: append new draws at the end only.
    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED, spawn_key=(event_id,)))
    z = z_sampler(rng.uniform())
    m1, m2 = sorted(rng.uniform(M_MIN, M_MAX, size=2), reverse=True)
    iota = np.arccos(rng.uniform(-1.0, 1.0))
    ra = rng.uniform(0.0, 2 * np.pi)
    dec = np.arcsin(rng.uniform(-1.0, 1.0))
    psi = rng.uniform(0.0, np.pi)
    gps = GPS_START + rng.uniform(0.0, GPS_SPAN)
    noise_seed, sampler_seed = rng.integers(0, 2**31 - 1, size=2)
    return {
        "event_id": event_id, "z": z, "d_L": z_sampler.d_L(z),
        "m1_src": float(m1), "m2_src": float(m2),
        "iota": float(iota), "ra": float(ra), "dec": float(dec), "psi": float(psi), "gps": float(gps),
        "noise_seed": int(noise_seed), "sampler_seed": int(sampler_seed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of rows to generate")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    z_sampler = _RedshiftSampler()
    if args.out.exists():
        old = pd.read_csv(args.out)
        if list(old.columns) != COLUMNS or list(old.event_id) != list(range(len(old))):
            raise SystemExit(f"{args.out}: unexpected columns or event_ids; not appending")
        # the generator must not have changed since the file was written
        for i in {0, len(old) - 1}:
            redo = draw_row(i, z_sampler)
            if not np.allclose([redo[c] for c in COLUMNS], old.loc[i, COLUMNS].to_numpy(float), rtol=0, atol=0):
                raise SystemExit(f"{args.out}: row {i} no longer reproduces; generator changed")
        start = len(old)
    else:
        old, start = None, 0

    new = pd.DataFrame([draw_row(i, z_sampler) for i in range(start, start + args.n)], columns=COLUMNS)
    out = new if old is None else pd.concat([old, new], ignore_index=True)
    tmp = args.out.with_suffix(".tmp")
    out.to_csv(tmp, index=False, float_format="%.17g")
    tmp.replace(args.out)
    print(f"{args.out}: rows {start}..{start + args.n - 1} written ({len(out)} total)")


if __name__ == "__main__":
    main()
