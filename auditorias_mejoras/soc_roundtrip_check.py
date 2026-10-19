#!/usr/bin/env python3
"""Roadmap 8.1, Task 3: spin-orbit label roundtrip on the real F0 Bi2Se3 data.

SIESTA TSHS (8 components + S) -> labels with ``matrix_component_policy="spin_orbit"``
(4 TRS channels) -> orthogonal ``spin-orbit`` sisl Hamiltonian -> ``attach_overlap``.

The roundtrip is not expected to reproduce the stored matrix exactly, for two reasons
that are not reconstruction errors:

- TRS: the stored matrix has a small SCF time-reversal residual that the channels drop.
  Dropped part per component <= residual * |H|max (checked).
- Support: graph2mat only keeps blocks of atom pairs closer than the sum of their orbital
  radii. SIESTA's H also has entries beyond it (Kleinman-Bylander projector tails), which
  cannot come back. Reported as ``outside support``.

So the hard gate compares the reconstruction with ``reference`` = the stored matrix with
the TRS-odd part removed and the out-of-support blocks zeroed (same S): every component
and the eigenvalues at Gamma and 3 random k must agree to ``TOL`` (1e-8 eV).
With ``symmetric_matrix=True`` the stored H is only Hermitian up to numerical noise and the
reconstruction keeps one direction of each edge, so each tolerance grows by a bound computed
from the stored matrix alone (not from the reconstruction): the max Hermiticity defect for the
components, and ``2 sqrt(||a||_1 ||a||_inf) / lambda_min(S(k))`` for the eigenvalues, with
``a`` the defect summed over images and channels (see ``hermiticity_defect``). The raw
comparison (eigenvalue shift versus the stored matrix) is printed for information.
Exit code 1 if any gate fails. Run with the MD venv python (this directory is not a
package)::

    MD_vs_AtomicDisplacement/.venv/bin/python auditorias_mejoras/soc_roundtrip_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array
import sisl

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.core.data.spin_orbit import (
    SOC_TRS_RTOL,
    SOC_TRANSPOSE_SIGN,
    attach_overlap,
    siesta_to_soc_channels,
    soc_channels_to_siesta,
    soc_trs_residual,
)

DATA = Path(__file__).resolve().parent / "experimentos_soc_f0_bi2se3_20261019" / "soc_k6"
TOL = 1e-8  # eV


def support_mask(H, data):
    """Dense bool mask (no, no * n_s) of the blocks graph2mat keeps: on-site + edges."""
    g = H.geometry
    mask = np.zeros((g.no, g.no * g.n_s), dtype=bool)
    first = g.firsto
    blocks = [(i, i, 0) for i in range(g.na)]
    blocks += [(i, j, isc) for (i, j), isc in zip(data.edge_index.T.tolist(), data.neigh_isc.tolist())]
    for i, j, isc in blocks:
        mask[first[i] : first[i + 1], isc * g.no + first[j] : isc * g.no + first[j + 1]] = True
    return mask


def hermiticity_defect(H):
    """Hermiticity defect of the spin-orbit ``H``, from the data alone.

    ``c_ij(R) - sign * c_ji(-R)^T`` over the 4 TRS channels; zero for an exactly Hermitian
    matrix. ``symmetric_matrix=True`` labels keep one direction of every edge and rebuild
    the other, so every reconstructed element equals the stored one or its transposed
    counterpart: they differ from the stored matrix by at most this defect.

    Returns ``(max defect, a)`` with ``a[r, c] = sum_R sum_channels |defect|`` ``(no, no)``;
    a spinor element is bounded by that sum, so the spinor defect has 2-norm at most
    ``2 sqrt(||a||_1 ||a||_inf)`` at any k.
    """
    g = H.geometry
    dense = np.stack([H.tocsr(i).toarray() for i in range(8)], -1).reshape(g.no, g.n_s, g.no, 8)
    c = siesta_to_soc_channels(dense)
    perm = g.lattice.sc_index(-g.lattice.sc_off)
    swapped = c[:, perm].transpose(2, 1, 0, 3)
    defect = np.abs(c - swapped * SOC_TRANSPOSE_SIGN)
    return defect.max(), defect.sum(axis=(1, 3))


def defect_eig_bound(a, Sk):
    """Eigenvalue shift bound ``2 sqrt(||a||_1 ||a||_inf) / lambda_min(S(k))`` (Weyl)."""
    return 2 * np.sqrt(np.abs(a).sum(0).max() * np.abs(a).sum(1).max()) / np.linalg.eigvalsh(Sk)[0]


def main():
    geometry = sisl.get_sile(DATA / "RUN.fdf").read_geometry()
    H = sisl.get_sile(DATA / "bi2se3.TSHS").read_hamiltonian(geometry=geometry)
    raw8 = H._csr._D[:, :8]
    rel = soc_trs_residual(raw8)
    h_max = np.abs(raw8).max()
    print(f"H: no={H.no} nsc={H.nsc.tolist()} nnz={H.nnz}  TRS residual {rel:.2e} (limit {SOC_TRS_RTOL:.0e}), |H|max {h_max:.2f} eV")
    if not (H.spin.is_spinorbit and not H.orthogonal and rel <= SOC_TRS_RTOL):
        print("FAIL: TSHS is not a TRS non-orthogonal spin-orbit Hamiltonian")
        return 1

    proj = H.copy()
    d = proj._csr._D
    d[:, :8] = soc_channels_to_siesta(siesta_to_soc_channels(d[:, :8]))
    S = sisl.SparseOrbital.fromsp(H.geometry, H.tocsr(H.S_idx))
    ks = [np.zeros(3), *np.random.default_rng(7).uniform(-0.5, 0.5, size=(3, 3))]
    ok = True
    for symmetric in (False, True):
        processor = MatrixDataProcessor(
            basis_table=AtomicTableWithEdges(H.geometry.atoms.atom),
            out_matrix="hamiltonian",
            symmetric_matrix=symmetric,
            sub_point_matrix=False,
            matrix_component_policy="spin_orbit",
        )
        data = BasisMatrixData.new(H, processor)
        rec = attach_overlap(processor.matrix_from_data(data, out_format="sisl_H", threshold=None), S)
        mask = support_mask(H, data)
        ref = sisl.Hamiltonian.fromsp(
            H.geometry,
            [csr_array(proj.tocsr(i).toarray() * mask) for i in range(8)],
            S=H.tocsr(H.S_idx),
            spin=sisl.Spin("spin-orbit"),
        )
        asym, defect = hermiticity_defect(ref) if symmetric else (0.0, None)
        print(f"\nsymmetric_matrix={symmetric}: {data.point_labels.shape[0]} node + {data.edge_labels.shape[0]} edge label rows"
              + (f"; reference asymmetry {asym:.2e}" if symmetric else ""))

        print(f"{'comp':>4} {'rec - reference':>16} {'TRS drop':>11} {'outside support':>16}")
        for i in range(9):
            full = H.tocsr(i).toarray()
            e_ref = np.abs(rec.tocsr(i).toarray() - ref.tocsr(i).toarray()).max()
            trs = np.abs((proj.tocsr(i).toarray() - full) * mask).max()
            outside = np.abs(full * ~mask).max()
            good = e_ref <= TOL + asym and trs <= rel * h_max * 1.0001 + TOL
            ok &= bool(good)
            print(f"{i:>4} {e_ref:16.3e} {trs:11.3e} {outside:16.3e}{'' if good else '  <-- FAIL'}")

        print(f"{'k':>26} {'rec - reference':>16} {'rec - stored':>13} {'defect bound':>13}")
        for k in ks:
            e_rec = rec.eigh(k)
            d_ref = np.abs(e_rec - ref.eigh(k)).max()
            d_raw = np.abs(e_rec - H.eigh(k)).max()
            # Symmetric labels: shift bounded by the Hermiticity defect of the reference.
            weyl = defect_eig_bound(defect, ref.Sk(k, format="array")) if symmetric else 0.0
            good = d_ref <= TOL + weyl
            ok &= bool(good)
            print(f"{np.array2string(k, precision=3):>26} {d_ref:16.3e} {d_raw:13.3e} {weyl:13.2e}{'' if good else '  <-- FAIL'}")
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
