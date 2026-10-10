"""Roadmap 3.11 (CRIT-04): a sisl matrix whose ``lattice.sc_off`` is not sisl's default
order must give the same labels as the default order. The graph indexes images with a
fresh default Lattice; block keys taken raw from the matrix columns would read another
image's block without any error."""

import numpy as np
import pytest
import sisl

from graph2mat import BasisMatrixData, BasisTableWithEdges, MatrixDataProcessor, OrbitalConfiguration, PointBasis


def _chain_hamiltonian(sc_order):
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=0.8)])
    geometry = sisl.Geometry([[0, 0, 0], [0.7, 0, 0]], atoms=[atom, atom.copy()],
                             lattice=sisl.Lattice([1.5, 20, 20], nsc=[3, 1, 1]))
    geometry.lattice.sc_off = geometry.lattice.sc_off[sc_order]
    H = sisl.Hamiltonian(geometry)
    for i in range(2):
        H[i, i] = 10. * (i + 1)
        for shift in (-1, 0, 1):
            for j in range(2):
                if (i, j, shift) != (i, i, 0):
                    # A value unique to each (i, j, image).
                    H[i, j + 2 * geometry.lattice.sc_index([shift, 0, 0])] = 100 * i + 10 * j + shift
    return H


def _labels(H):
    table = BasisTableWithEdges([PointBasis(1, R=np.array([0.8]), basis=[1], basis_convention="siesta_spherical")])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=False, sub_point_matrix=False)
    data = BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(H, labels=True), data_processor=processor)
    shifts = np.rint(np.asarray(data.shifts) @ np.linalg.inv(np.asarray(data.cell).reshape(3, 3))).astype(int)
    return {(int(i), int(j), int(s[0])): float(v)
            for i, j, s, v in zip(data.edge_index[0], data.edge_index[1], shifts, data.edge_labels)}


def test_permuted_sc_off_gives_the_same_labels():
    default = _labels(_chain_hamiltonian([0, 1, 2]))
    permuted = _labels(_chain_hamiltonian([0, 2, 1]))
    assert default == permuted
    # And the labels are the physical ones: H[i, j, image] = 100 i + 10 j + image.
    for (i, j, shift), value in default.items():
        assert value == 100 * i + 10 * j + shift
