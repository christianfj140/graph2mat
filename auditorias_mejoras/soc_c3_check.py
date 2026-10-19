#!/usr/bin/env python3
"""Roadmap 8.1: does SIESTA's spin-orbit channel g rotate as Graph2Mat assumes?

Bi2Se3 (F0 cell, all atoms on the c axis) is invariant under the threefold rotation C3 about z.
Every atom maps to itself; only the periodic images permute. On the real F0 Hamiltonian
(``experimentos_soc_f0_bi2se3_20261019/soc_k6``) the TRS channels must then satisfy

    h0_ij(C3 T) = D_i h0_ij(T) D_j^T            (orbital rotation, siesta_spherical)
    g_ij(C3 T)  = sum_b R_ab D_i g_b,ij(T) D_j^T (g is a Cartesian axial vector)

with D the orbital rotation Graph2Mat uses (``e3nn_irreps.D_from_matrix(cob R cob^T)``, as in
test_models.py). Wrong conventions (spin index left unrotated, rotated with R^T) are reported as
controls: they must be clearly worse. SIESTA's real-space mesh breaks the symmetry slightly, so
the reference is a small but nonzero residual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import graph2mat.bindings.e3nn  # noqa: F401  (e3nn setup)
import sisl
import torch
from graph2mat import AtomicTableWithEdges
from graph2mat.core.data.spin_orbit import siesta_to_soc_channels

RUN = Path(__file__).resolve().parent / "experimentos_soc_f0_bi2se3_20261019/soc_k6"


def blocks(H, channels):
    """{(i, j, sc_off tuple): (n_i, n_j, 4)} for every stored atom pair."""
    out, geom = {}, H.geometry
    no = geom.no
    for isc, off in enumerate(geom.lattice.sc_off):
        sub = channels[:, isc * no:(isc + 1) * no]
        for i in range(geom.na):
            oi = slice(geom.firsto[i], geom.firsto[i + 1])
            for j in range(geom.na):
                oj = slice(geom.firsto[j], geom.firsto[j + 1])
                b = sub[oi, oj]
                if np.any(b):
                    out[(i, j, tuple(int(x) for x in off))] = b
    return out


def main() -> int:
    geometry = sisl.get_sile(RUN / "RUN.fdf").read_geometry()
    H = sisl.get_sile(RUN / "bi2se3.TSHS").read_hamiltonian(geometry=geometry)
    raw = np.stack([H.tocsr(c).toarray() for c in range(8)], -1)
    ch = blocks(H, siesta_to_soc_channels(raw))
    assert np.allclose(geometry.xyz[:, :2], 0), "all atoms must sit on the c axis"

    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    cell = geometry.cell
    M = np.linalg.solve(cell.T, R @ cell.T)               # R a_k = sum_l a_l M_lk
    assert np.allclose(M, np.round(M)), "C3 must map the lattice onto itself"
    M = np.round(M).astype(int)

    table = AtomicTableWithEdges(geometry.atoms.atom)
    cob = table.change_of_basis
    D = [table.basis[int(sp)].e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ R @ cob.T)).numpy()
         for sp in geometry.atoms.species]

    scale = max(np.abs(b).max() for b in ch.values())
    variants = {"g Cartesian R (assumed)": R, "g unrotated (control)": np.eye(3), "g with R^T (control)": R.T}
    worst = {name: [0.0, 0.0] for name in variants}  # max |error| in h0, in g
    missing = 0
    for (i, j, off), b in ch.items():
        target = ch.get((i, j, tuple(int(x) for x in M @ np.array(off))))
        if target is None:
            missing += 1
            continue
        h0 = D[i] @ b[..., 0] @ D[j].T
        g = np.einsum("xy,ayz->axz", D[i], np.einsum("xyb,zy->bxz", b[..., 1:], D[j]))  # (3, ni, nj)
        for name, Rs in variants.items():
            g_rot = np.einsum("ab,bxz->xza", Rs, g)
            worst[name][0] = max(worst[name][0], np.abs(h0 - target[..., 0]).max())
            worst[name][1] = max(worst[name][1], np.abs(g_rot - target[..., 1:]).max())
    print(f"blocks {len(ch)}, image not stored after C3: {missing}, max|channel| {scale:.3f} eV")
    for name, (eh, eg) in worst.items():
        print(f"{name:28s} max|err| h0 {eh:.3e} eV  g {eg:.3e} eV  (g rel {eg / scale:.2e})")
    ok = worst["g Cartesian R (assumed)"][1]
    controls = min(worst[n][1] for n in variants if "control" in n)
    print("PASS" if ok < 1e-3 and controls > 100 * ok else "FAIL")
    return 0 if ok < 1e-3 and controls > 100 * ok else 1


if __name__ == "__main__":
    sys.exit(main())
