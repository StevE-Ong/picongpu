#!/usr/bin/env python3
"""Monte Carlo check of the Bremsstrahlung test (stationary case: no pusher, no deflection, no drag, no recoil).

Copyright 2026 Jian Fuh Ong
This file is part of PIConGPU. License: GPLv3+

Observables, each against the model's own distribution (no pass/fail threshold, numbers with Poisson errors):
  1. electron and ion start positions are independent (fraction of electrons on an ion position)
  2. macro-photons per electron per step vs R n v dt sigma(kappa > kappa_c)
  3. photon kappa = E_photon / Ekin spectrum in bins of ln(kappa) vs R n v dt int_bin kappa dsigma/dkappa dln(kappa)
  4. photon polar angle to the electron direction vs the truncated CDF delta(theta) / MAX_DELTA

Usage: validate_mc.py <run directory> --ekin-mev 10 --z 79 --density 5.90e28 [--dt 3.5e-15] [--ratio 10]
"""

import argparse
import importlib.util
import os

import numpy as np
import openpmd_api as io
from scipy.integrate import quad

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "check_tables", os.path.join(_here, "check_tables.py")
)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

ME_MEV = 0.51099895


def table_model(
    ekin, z, n_ekin=32, n_kappa=32, emin_mev=0.5, emax_mev=200.0, min_kappa=1.0e-10
):
    """kappa dsigma/dkappa as the code evaluates it: bilinear interpolation of the node values
    (Ekin nodes logarithmic, kappa nodes linear, node 0 at MIN_KAPPA), see ScaledSpectrum.hpp"""
    ln_e = np.log(np.array([emin_mev, emax_mev]) * ct.MEV)
    e_nodes = np.exp(ln_e[0] + (ln_e[1] - ln_e[0]) * np.arange(n_ekin) / (n_ekin - 1))
    k_nodes = np.arange(n_kappa) / (n_kappa - 1.0)
    k_nodes[0] = min_kappa
    pos_e = (np.log(ekin) - ln_e[0]) / (ln_e[1] - ln_e[0]) * (n_ekin - 1)
    i0 = min(int(pos_e), n_ekin - 2)
    f0 = pos_e - i0
    rows = [
        np.array([e * k * ct.dcs(e, k, z) for k in k_nodes])
        for e in (e_nodes[i0], e_nodes[i0 + 1])
    ]
    nodes = (1.0 - f0) * rows[0] + f0 * rows[1]

    def value(kappa):
        pos = np.clip(kappa * (n_kappa - 1), 0.0, n_kappa - 1)
        j = np.minimum(pos.astype(int), n_kappa - 2)
        f = pos - j
        return (1.0 - f) * nodes[j] + f * nodes[j + 1]

    return value


def read_species(series, step, name, records):
    it = series.iterations[step]
    sp = it.particles[name]
    out = {}
    for rec, comps in records.items():
        for comp in comps:
            rc = sp[rec][comp] if comp else sp[rec][io.Mesh_Record_Component.SCALAR]
            data = rc.load_chunk()
            series.flush()
            out[(rec, comp)] = np.asarray(data, dtype=np.float64) * rc.unit_SI
    return out


def positions(series, step, name):
    d = read_species(
        series,
        step,
        name,
        {"position": ["x", "y", "z"], "positionOffset": ["x", "y", "z"]},
    )
    return np.stack(
        [d[("position", c)] + d[("positionOffset", c)] for c in "xyz"], axis=1
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "runs", nargs="+", help="run directories (independent seeds are pooled)"
    )
    ap.add_argument("--ekin-mev", type=float, default=10.0)
    ap.add_argument("--z", type=float, default=79.0)
    ap.add_argument(
        "--density", type=float, default=5.90e28, help="ion density in m^-3"
    )
    ap.add_argument("--dt", type=float, default=3.5e-15)
    ap.add_argument("--ratio", type=float, default=10.0, help="WEIGHTING_RATIO")
    ap.add_argument("--cutoff-kev", type=float, default=5.0)
    ap.add_argument("--max-delta", type=float, default=0.95)
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument(
        "--table-scale",
        type=int,
        default=1,
        help="PARAM_BREMS_NUM_SAMPLES_SCALE of the build",
    )
    args = ap.parse_args()

    ekin = args.ekin_mev * ct.MEV
    gamma = 1.0 + args.ekin_mev / ME_MEV
    v = ct.C * np.sqrt(1.0 - 1.0 / gamma**2)
    kc = args.cutoff_kev * 1.0e3 * ct.QE / ekin
    norm = (
        args.ratio * args.density * v * args.dt
    )  # per electron per step, times a cross section
    tab = table_model(
        ekin, args.z, n_ekin=32 * args.table_scale, n_kappa=32 * args.table_scale
    )

    def exact(lk):
        return ekin * np.exp(lk) * ct.dcs(ekin, np.exp(lk), args.z)

    def table(lk):
        return tab(np.exp(lk))

    edges = np.linspace(np.log(kc), 0.0, args.nbins + 1)
    sig_bins = np.array(
        [
            quad(exact, edges[i], edges[i + 1], epsrel=1e-12)[0]
            for i in range(args.nbins)
        ]
    )
    sig_bins_tab = np.array(
        [
            quad(table, edges[i], edges[i + 1], epsrel=1e-10, limit=200)[0]
            for i in range(args.nbins)
        ]
    )
    sig_hard, sig_tab = sig_bins.sum(), sig_bins_tab.sum()

    exposure = 0.0  # electron steps
    counts = np.zeros(args.nbins)
    u_all = []
    print(
        "[1,2] per run: electrons, electrons on an ion position, steps, macro-photons, ratio to table model"
    )
    for run in args.runs:
        series = io.Series(
            os.path.join(run, "openPMD", "simData_%T.bp"), io.Access.read_only
        )
        steps = sorted(series.iterations)
        s0, s1 = steps[0], steps[-1]
        pe = positions(series, s0, "e")
        pi = positions(series, s0, "i")
        on_ion = len(
            np.intersect1d(
                np.round(pe, 12).view("f8,f8,f8"), np.round(pi, 12).view("f8,f8,f8")
            )
        )
        ne = len(pe)
        ph = read_species(
            series, s1, "ph", {"momentum": ["x", "y", "z"], "weighting": [None]}
        )
        w = ph[("weighting", None)]
        p = (
            np.stack([ph[("momentum", c)] for c in "xyz"], axis=1) / w[:, None]
        )  # per real photon (macroWeighted)
        kappa = np.linalg.norm(p, axis=1) * ct.C / ekin
        c, _ = np.histogram(np.log(kappa), bins=edges)
        counts += c
        exp_run = norm * sig_tab * ne * (s1 - s0)
        exposure += ne * (s1 - s0)
        theta = np.arccos(np.clip(p[:, 0] / np.linalg.norm(p, axis=1), -1.0, 1.0))
        u_all.append(gamma * theta)
        print(
            f"    {os.path.basename(os.path.normpath(run))}: {ne} {on_ion} {s1 - s0} {len(kappa)} "
            f"{len(kappa) / exp_run:.5f} +- {np.sqrt(len(kappa)) / exp_run:.5f}"
        )
        del series

    nph = counts.sum()
    ex = norm * sig_bins * exposure
    et = norm * sig_bins_tab * exposure
    print(f"[2] pooled: macro-photons {nph:.0f}")
    print(
        f"    exact formula:      ratio {nph / ex.sum():.5f} +- {np.sqrt(nph) / ex.sum():.5f}"
    )
    print(
        f"    table-interpolated: ratio {nph / et.sum():.5f} +- {np.sqrt(nph) / et.sum():.5f}"
        f"  (table/exact = {sig_tab / sig_hard:.5f})"
    )

    print(
        "[3] kappa spectrum: bin [kappa_lo, kappa_hi], counts, ratio to exact, ratio to table model +- Poisson, pull"
    )
    pulls = (counts - et) / np.sqrt(et)
    for i in range(args.nbins):
        print(
            f"    [{np.exp(edges[i]):.3e}, {np.exp(edges[i + 1]):.3e}] {counts[i]:9.0f} {counts[i] / ex[i]:.4f} "
            f"{counts[i] / et[i]:.4f} +- {np.sqrt(et[i]) / et[i]:.4f} {pulls[i]:+.2f}"
        )
    print(
        f"    chi2 to table model = {np.sum(pulls**2):.1f} for {args.nbins} bins "
        f"(the total rate is included, so all {args.nbins} bins are free)"
    )

    u = np.concatenate(u_all)
    u_max = gamma * ct.angle_theta(args.max_delta, gamma)
    print(
        f"[4] angle: gamma = {gamma:.4f}, u_max = gamma theta(MAX_DELTA) = {u_max:.4f}, max u in data = {u.max():.4f}"
    )
    print(
        "    u = gamma theta bin, counts, model = N (delta(hi) - delta(lo)) / MAX_DELTA, ratio +- Poisson, pull"
    )
    u_edges = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, u_max])
    cu, _ = np.histogram(u, bins=u_edges)
    cdf = np.array([ct.angle_cdf(q / gamma, gamma) for q in u_edges]) / args.max_delta
    mu = len(u) * np.diff(cdf)
    pu = (cu - mu) / np.sqrt(mu)
    for i in range(len(cu)):
        print(
            f"    [{u_edges[i]:.2f}, {u_edges[i + 1]:.2f}] {cu[i]:9d} {mu[i]:11.1f} {cu[i] / mu[i]:.4f} "
            f"+- {1 / np.sqrt(mu[i]):.4f} {pu[i]:+.2f}"
        )
    print(
        f"    chi2 = {np.sum(pu**2):.1f} for {len(cu)} bins ({len(cu) - 1} free, the total is fixed)"
    )
    print(
        f"    photons above u_max: {np.sum(u > u_max)} (u_max at the exact gamma; the table interpolates in ln gamma)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
