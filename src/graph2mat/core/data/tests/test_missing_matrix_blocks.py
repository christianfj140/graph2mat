"""Sparse absence is zero for H and an unknown label for DM/EDM."""

import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat import (
    BasisConfiguration, BasisMatrixData, BasisTableWithEdges, Formats,
    MatrixDataProcessor, OrbitalConfiguration, PointBasis,
)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_zero_blocks_in_dense_or_sparse_configuration_roundtrip(sparse, diagonal, symmetric):
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1]), PointBasis("B", R=2, basis=[2])])
    matrix = np.eye(3) if diagonal else np.zeros((3, 3))
    config = BasisConfiguration(
        point_types=["A", "B"], positions=np.array([[0., 0., 0.], [.7, 0., 0.]]),
        basis=table, cell=np.eye(3) * 20, pbc=(False, False, False),
        matrix=csr_array(matrix) if sparse else matrix,
    )
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False)
    data = BasisMatrixData.from_config(config, processor)
    np.testing.assert_array_equal(data.point_labels, [1., 1., 0., 0., 1.] if diagonal else [0.] * 5)
    np.testing.assert_array_equal(data.edge_labels, np.zeros(2 if symmetric else 4))
    np.testing.assert_array_equal(processor.matrix_from_data(data, out_format="numpy"), matrix)


@pytest.mark.parametrize("kind,matrix_cls", [
    ("hamiltonian", sisl.Hamiltonian),
    ("density_matrix", sisl.DensityMatrix),
    ("energy_density_matrix", sisl.EnergyDensityMatrix),
])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("has_value", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_absent_physics_blocks_keep_fill_policy_and_component_axis(kind, matrix_cls, components, periodic, has_value, symmetric):
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=.8)])
    lattice = sisl.Lattice([1.5 if periodic else 10., 10., 10.], nsc=[3 if periodic else 1, 1, 1])
    geometry = sisl.Geometry([[0., 0., 0.], [.5, 0., 0.]], atoms=[atom, atom.copy()], lattice=lattice)
    matrix = matrix_cls(geometry, spin="polarized" if components == 2 else "unpolarized")
    references = [np.zeros((2, 2 * geometry.n_s)) for _ in range(components)]
    if has_value:
        references[0][0, 0] = 2.5
        if components == 2:
            references[1][0, 0] = -1.25
    matrix._csr = matrix._csr.fromsp([csr_array(reference) for reference in references])
    table = BasisTableWithEdges([PointBasis(1, R=np.array([.8]), basis=[1])])
    policy = "spin_h_only" if kind == "hamiltonian" and components == 2 else "raw_components"
    processor = MatrixDataProcessor(
        basis_table=table, out_matrix=kind, symmetric_matrix=symmetric, sub_point_matrix=False,
        n_matrix_components=components, matrix_component_policy=policy,
    )
    config = OrbitalConfiguration.from_matrix(matrix, pbc=(periodic, False, False), matrix_component_policy=policy)
    before = {key: value.copy() for key, value in config.matrix.block_dict.items()}
    data = BasisMatrixData.from_config(config, processor)
    fill = 0. if kind == "hamiltonian" else np.nan
    want_nodes = np.full((2, components), fill)
    if has_value:
        want_nodes[0] = [2.5] if components == 1 else [2.5, -1.25]
    np.testing.assert_allclose(data.point_labels, want_nodes[:, 0] if components == 1 else want_nodes, equal_nan=True)
    assert data.edge_labels.size > 0
    assert data.edge_labels.ndim == (1 if components == 1 else 2)
    assert np.all(data.edge_labels == 0.) if kind == "hamiltonian" else np.isnan(data.edge_labels).all()
    # Missing blocks are supplied lazily, without densifying/mutating the input.
    assert set(config.matrix.block_dict) == set(before)
    for key, value in before.items():
        np.testing.assert_array_equal(config.matrix.block_dict[key], value)
    if periodic:
        assert np.any(data.neigh_isc != 0)
    if kind == "hamiltonian":
        restored = data.convert_to(Formats.SISL_H)
        for channel, reference in enumerate(references):
            np.testing.assert_array_equal(restored.tocsr(channel).toarray(), reference)
