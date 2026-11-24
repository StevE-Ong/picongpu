#!/usr/bin/env python3
"""Check the Bremsstrahlung lookup tables written by PIConGPU (dumpTables = true).

Copyright 2026 Jian Fuh Ong
This file is part of PIConGPU. License: GPLv3+

The tables are compared with an independent float64 evaluation of the same formulas in SI units:
  - scaled spectrum kappa dsigma/dkappa: screened Bethe-Heitler form of PIConGPU 0.6.0 (Salvat et al. 2006)
  - stopping power table: int_{MIN_KAPPA Ekin}^{kappa Ekin} W dsigma/dW dW (scipy.integrate.quad)
  - photon angle table: inverse of the closed-form CDF of the boosted dipole distribution (scipy brentq)

The table values are float_X (float32 by default), so the expected relative difference is ~1e-7 plus the
error of the Simpson integral of the stopping power table.

Usage: check_tables.py <directory with bremsstrahlung_*.dat>
"""

import glob
import os
import re
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# PIConGPU SI constants (include/picongpu/param/physicalConstants.param, unitless/simulation.unitless)
C = 2.99792458e8
MU0 = 1.25663706127e-6
EPS0 = 1.0 / MU0 / C / C
ME = 9.1093837139e-31
QE = 1.602176634e-19
HBAR = 1.054571817e-34
MEV = QE * 1.0e6

BOHR = np.pi * 4.0 * EPS0 * HBAR**2 / (ME * QE**2)
RE = QE**2 / (np.pi * 4.0 * EPS0 * ME * C**2)
ALPHA = QE**2 / (np.pi * 4.0 * EPS0 * HBAR * C)

MIN_KAPPA = 1.0e-10
MAX_DELTA = 0.95


def dcs(ekin, kappa, z):
    """d sigma / d W in m^2 / J (same formula as ScaledSpectrum::dcs, SI units)"""
    w = kappa * ekin
    eps = w / (ekin + ME * C**2)
    r = z ** (-1.0 / 3.0) * BOHR
    gamma = ekin / (ME * C**2) + 1.0
    b = r * ME * C / HBAR / (2.0 * gamma) * eps / (1.0 - eps)
    lr = np.log(r * ME * C / HBAR)
    at = np.arctan(1.0 / b)
    phi1 = 4.0 * lr + 2.0 - 2.0 * np.log(1.0 + b * b) - 4.0 * b * at
    phi2 = (
        4.0 * lr
        + 7.0 / 3.0
        - 2.0 * np.log(1.0 + b * b)
        - 6.0 * b * at
        - b * b * (4.0 - 4.0 * b * at - 3.0 * np.log(1.0 + 1.0 / (b * b)))
    )
    return (
        RE**2 * ALPHA * z * z / w * (eps * eps * phi1 + 4.0 / 3.0 * (1.0 - eps) * phi2)
    )


def stopping_table(ekin, kappa, z):
    lo, hi = MIN_KAPPA * ekin, kappa * ekin
    if hi <= lo:
        return 0.0
    val, _ = quad(
        lambda w: w * dcs(ekin, w / ekin, z),
        lo,
        hi,
        epsabs=0.0,
        epsrel=1e-12,
        limit=200,
    )
    return val


def angle_cdf(theta, gamma):
    t = 1.0 + gamma * gamma * theta * theta
    return 1.0 - 1.5 / t + 1.5 / t**2 - 1.0 / t**3


def angle_theta(delta, gamma):
    if delta <= 0.0:
        return 0.0
    if angle_cdf(np.pi, gamma) <= delta:
        return np.pi
    return brentq(
        lambda th: angle_cdf(th, gamma) - delta, 0.0, np.pi, xtol=1e-16, rtol=1e-15
    )


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0.0, np.abs(a / b - 1.0), np.abs(a))


def main(directory):
    ok = True
    for path in sorted(
        glob.glob(os.path.join(directory, "bremsstrahlung_spectrum_Z*.dat"))
    ):
        z = float(re.search(r"_Z(\d+)\.dat$", path).group(1))
        data = np.loadtxt(path)
        ekin = data[:, 0] * MEV
        kappa = data[:, 1]
        spec_ref = np.array([e * k * dcs(e, k, z) for e, k in zip(ekin, kappa)])
        stop_ref = np.array([stopping_table(e, k, z) for e, k in zip(ekin, kappa)])
        r_spec = rel(data[:, 2], spec_ref)
        mask = stop_ref > 0.0
        r_stop = rel(data[mask, 3], stop_ref[mask])
        print(f"{os.path.basename(path)}: {len(data)} entries, Z = {z:g}")
        print(
            f"  scaled spectrum: max rel. diff {r_spec.max():.3e} (median {np.median(r_spec):.3e})"
        )
        print(
            f"  stopping power:  max rel. diff {r_stop.max():.3e} (median {np.median(r_stop):.3e})"
        )
        print(
            f"  stopping power at kappa = 0 (expected 0): max |value| {np.abs(data[~mask, 3]).max():.3e}"
        )
        ok &= np.all(np.isfinite(data))

    path = os.path.join(directory, "bremsstrahlung_photonAngle.dat")
    if os.path.exists(path):
        data = np.loadtxt(path)
        ref = np.array([angle_theta(d, g) for d, g in zip(data[:, 0], data[:, 1])])
        diff = np.abs(data[:, 2] - ref)
        r = rel(data[:, 2], ref)
        print(f"{os.path.basename(path)}: {len(data)} entries")
        print(
            f"  theta: max abs diff {diff.max():.3e} rad, max rel. diff (theta > 0) {r[ref > 0].max():.3e}"
        )
        print(
            f"  delta range [{data[:, 0].min():g}, {data[:, 0].max():g}], gamma range "
            f"[{data[:, 1].min():g}, {data[:, 1].max():g}]"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
