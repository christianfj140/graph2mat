"""Roadmap 8.3: point types come from Z, so two species with the same Z and different
basis (C_bulk/C_surf) map to one table entry. The labels must not be built silently
with block sizes that do not match the matrix."""

import numpy as np
import pytest
import sisl

from graph2mat import BasisMatrixData, BasisTableWithEdges, MatrixDataProcessor, OrbitalConfiguration, PointBasis


def _orbitals(n):
    names = ["2s", "2pyZ1", "2pzZ1", "2pxZ1"][:n]
    return [sisl.AtomicOrbital(name, R=1.) for name in names]


def test_same_z_species_with_other_basis_raise():
    small = sisl.Atom(6, orbitals=_orbitals(1), tag="C_s")
    big = sisl.Atom(6, orbitals=_orbitals(4), tag="C_sp")
    geometry = sisl.Geometry([[0, 0, 0], [1.2, 0, 0]], atoms=[small, big], lattice=[10, 10, 10])
    H = sisl.Hamiltonian(geometry)
    H[0, 0], H[1, 1] = 1., 2.
    table = BasisTableWithEdges([PointBasis.from_sisl_atom(big)])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    with pytest.raises(ValueError, match="same Z"):
        BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(H), data_processor=processor)


def test_matching_basis_still_builds():
    atom = sisl.Atom(6, orbitals=_orbitals(4), tag="C")
    geometry = sisl.Geometry([[0, 0, 0], [1.2, 0, 0]], atoms=[atom, atom.copy()], lattice=[10, 10, 10])
    H = sisl.Hamiltonian(geometry)
    H[0, 0], H[1, 1] = 1., 2.
    table = BasisTableWithEdges([PointBasis.from_sisl_atom(atom)])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    data = BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(H), data_processor=processor)
    assert len(data.point_labels) == 2 * 16
