from types import SimpleNamespace

import numpy as np
import sisl
import pytest
from scipy.sparse import csr_array

from graph2mat import (
    MatrixDataProcessor,
    PointBasis,
    BasisTableWithEdges,
    OrbitalConfiguration,
    BasisMatrixData,
    conversions,
    Formats,
)


def _single_orbital_geometry():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    return sisl.Geometry([[0.0, 0.0, 0.0]], atoms=[atom], lattice=[10, 10, 10])


def _two_atom_single_orbital_geometry():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    return sisl.Geometry(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atoms=[atom, atom.copy()],
        lattice=[10, 10, 10],
    )


def _single_orbital_basis_table():
    return BasisTableWithEdges([PointBasis(1, R=np.array([2.0]), basis=[1])])


def _nonspin_nonorthogonal_hamiltonian():
    geometry = _two_atom_single_orbital_geometry()
    h_csr = csr_array([[1.0, 0.1], [0.2, 2.0]])
    s_csr = csr_array([[10.0, 0.3], [0.4, 20.0]])
    hamiltonian = sisl.Hamiltonian.fromsp(geometry, h_csr, S=s_csr)
    return hamiltonian, h_csr, s_csr


def _symmetric_nonspin_nonorthogonal_hamiltonian():
    geometry = _two_atom_single_orbital_geometry()
    h_csr = csr_array([[1.0, 0.125], [0.125, 2.0]])
    s_csr = csr_array([[10.0, 0.75], [0.75, 20.0]])
    hamiltonian = sisl.Hamiltonian.fromsp(geometry, h_csr, S=s_csr)
    return hamiltonian, h_csr, s_csr


def _h_only_symmetric_processor():
    return MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
    )


def _single_item_batch(data):
    arrays = data.numpy_arrays()

    class SingleItemBatch:
        num_graphs = 1

        def numpy_arrays(self):
            return SimpleNamespace(
                ptr=np.array([0, len(arrays.point_types)]),
                n_edges=np.array([arrays.edge_index.shape[1]]),
                point_types=arrays.point_types,
                edge_types=arrays.edge_types,
            )

        def get_example(self, index):
            assert index == 0
            return data

    return SingleItemBatch()


def test_hamiltonian_collinear_read_preserves_two_components():
    geometry = _single_orbital_geometry()
    h = sisl.Hamiltonian(geometry, spin=sisl.Spin("polarized"))
    h._csr = h._csr.fromsp([csr_array([[1.0]]), csr_array([[2.0]])])

    config = OrbitalConfiguration.from_matrix(h, labels=True)
    block = config.matrix.block_dict[(0, 0, 0)]

    assert block.shape == (1, 1, 2)
    np.testing.assert_allclose(block[0, 0, 0], 1.0)
    np.testing.assert_allclose(block[0, 0, 1], 2.0)


def test_nonspin_nonorthogonal_hamiltonian_h_only_filters_overlap():
    hamiltonian, h_csr, s_csr = _nonspin_nonorthogonal_hamiltonian()

    assert not hamiltonian.spin.is_polarized
    assert not hamiltonian.orthogonal
    assert hamiltonian.S_idx == 1

    config = OrbitalConfiguration.from_matrix(
        hamiltonian, labels=True, matrix_component_policy="h_only"
    )

    np.testing.assert_allclose(config.matrix.block_dict[0, 0, 0], [[h_csr[0, 0]]])
    np.testing.assert_allclose(config.matrix.block_dict[1, 1, 0], [[h_csr[1, 1]]])
    np.testing.assert_allclose(config.matrix.block_dict[0, 1, 0], [[h_csr[0, 1]]])
    np.testing.assert_allclose(config.matrix.block_dict[1, 0, 0], [[h_csr[1, 0]]])
    assert config.matrix.block_dict[0, 0, 0].ndim == 2
    assert config.matrix.block_dict[0, 0, 0][0, 0] != s_csr[0, 0]


def test_nonspin_nonorthogonal_hamiltonian_h_only_flat_labels_and_roundtrip():
    hamiltonian, h_csr, _ = _nonspin_nonorthogonal_hamiltonian()
    processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
    )

    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)

    assert data.point_labels.ndim == 1
    assert data.edge_labels.ndim == 1
    np.testing.assert_allclose(data.point_labels, np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.sort(data.edge_labels), np.array([0.1, 0.2]))

    roundtrip = data.convert_to(Formats.SISL_H)
    assert not roundtrip.spin.is_polarized
    assert roundtrip._csr.data.shape[1] == 1
    np.testing.assert_allclose(
        roundtrip.tocsr().toarray(),
        h_csr.toarray(),
    )


def test_h_only_symmetric_nonorthogonal_labels_roundtrip_h_not_overlap():
    hamiltonian, h_csr, s_csr = _symmetric_nonspin_nonorthogonal_hamiltonian()
    processor = _h_only_symmetric_processor()

    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)

    assert processor.n_matrix_components == 1
    assert data.point_labels.ndim == 1
    assert data.edge_labels.ndim == 1
    np.testing.assert_allclose(data.point_labels, np.array([1.0, 2.0]))
    assert not np.isin(data.point_labels, s_csr.diagonal()).any()
    assert not np.isin(data.edge_labels, s_csr.data).any()

    roundtrip = data.convert_to(Formats.SISL_H)

    assert not roundtrip.spin.is_polarized
    assert roundtrip._csr.data.shape[1] == 1
    np.testing.assert_allclose(roundtrip.tocsr().toarray(), h_csr.toarray())
    np.testing.assert_allclose(
        roundtrip.tocsr().toarray(),
        roundtrip.tocsr().toarray().T,
    )


def test_h_only_symmetric_yield_from_batch_reconstructs_exact_h_labels():
    hamiltonian, h_csr, _ = _symmetric_nonspin_nonorthogonal_hamiltonian()
    processor = _h_only_symmetric_processor()
    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)
    batch = _single_item_batch(data)
    predictions = {
        "node_labels": data.point_labels.copy(),
        "edge_labels": data.edge_labels.copy(),
    }

    reconstructed = next(
        processor.yield_from_batch(
            batch,
            predictions=predictions,
            as_matrix=True,
            out_format=Formats.SISL_H,
        )
    )

    np.testing.assert_allclose(reconstructed.tocsr().toarray(), h_csr.toarray())


def test_h_only_symmetric_yield_from_batch_rejects_extra_edge_labels():
    hamiltonian, _h_csr, _s_csr = _symmetric_nonspin_nonorthogonal_hamiltonian()
    processor = _h_only_symmetric_processor()
    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)
    batch = _single_item_batch(data)
    predictions = {
        "node_labels": data.point_labels.copy(),
        "edge_labels": np.concatenate([data.edge_labels.copy(), np.array([999.0])]),
    }

    with pytest.raises(
        ValueError, match="Predicted edge labels were not fully consumed"
    ):
        list(
            processor.yield_from_batch(
                batch,
                predictions=predictions,
                as_matrix=False,
            )
        )


def test_nonspin_nonorthogonal_hamiltonian_raw_components_preserved():
    hamiltonian, h_csr, s_csr = _nonspin_nonorthogonal_hamiltonian()
    processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="raw_components",
    )

    config = OrbitalConfiguration.from_matrix(
        hamiltonian, labels=True, matrix_component_policy="raw_components"
    )
    block = config.matrix.block_dict[0, 0, 0]

    assert block.shape == (1, 1, 2)
    np.testing.assert_allclose(block[0, 0], np.array([h_csr[0, 0], s_csr[0, 0]]))

    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)

    assert data.point_labels.shape == (2, 2)
    np.testing.assert_allclose(
        data.point_labels,
        np.array([[h_csr[0, 0], s_csr[0, 0]], [h_csr[1, 1], s_csr[1, 1]]]),
    )


def test_nonspin_nonorthogonal_hamiltonian_h_and_overlap_serializes_explicit_overlap():
    hamiltonian, h_csr, s_csr = _nonspin_nonorthogonal_hamiltonian()
    processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="h_and_overlap",
    )

    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)
    matrix = data.convert_to(Formats.SISL_H)

    assert not matrix.spin.is_polarized
    assert not matrix.orthogonal
    assert matrix.S_idx == 1
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), h_csr.toarray())
    np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), s_csr.toarray())


def test_raw_hamiltonian_components_do_not_serialize_as_spin():
    hamiltonian, _, _ = _nonspin_nonorthogonal_hamiltonian()
    processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="raw_components",
    )

    data = BasisMatrixData.new(hamiltonian, data_processor=processor, labels=True)

    try:
        data.convert_to(Formats.SISL_H)
    except ValueError as exc:
        assert "raw_components" in str(exc)
    else:
        raise AssertionError("raw_components should not serialize as spin implicitly")


def test_spin_nonorthogonal_hamiltonian_h_only_excludes_overlap():
    geometry = _single_orbital_geometry()
    h = sisl.Hamiltonian.fromsp(
        geometry,
        [csr_array([[1.0]]), csr_array([[2.0]])],
        S=csr_array([[3.0]]),
        spin=sisl.Spin("polarized"),
    )

    assert h.spin.is_polarized
    assert not h.orthogonal
    assert h.S_idx == 2

    config = OrbitalConfiguration.from_matrix(
        h, labels=True, matrix_component_policy="h_only"
    )
    block = config.matrix.block_dict[(0, 0, 0)]

    assert block.shape == (1, 1, 2)
    np.testing.assert_allclose(block[0, 0], np.array([1.0, 2.0]))


def test_flatten_nodes_and_edges_multicomponent():
    from graph2mat.core.data.matrices import OrbitalMatrix

    matrix = OrbitalMatrix(
        block_dict={(0, 0, 0): np.array([[[1.0, 2.0]]])},
        nsc=np.array([1, 1, 1]),
        orbital_count=np.array([1]),
    )

    node_labels, edge_labels = matrix.to_flat_nodes_and_edges(
        edge_index=np.empty((2, 0), dtype=np.int64),
        edge_sc_shifts=np.empty((0,), dtype=np.int64),
    )

    assert node_labels.shape == (1, 2)
    assert edge_labels.shape == (0, 2)
    np.testing.assert_allclose(node_labels[0], np.array([1.0, 2.0]))


def test_yield_from_batch_multicomponent_labels():
    basis_table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=2,
    )

    class FakeBatch:
        num_graphs = 2

        def __init__(self):
            self._arrays = type(
                "A",
                (),
                dict(
                    ptr=np.array([0, 1, 2]),
                    n_edges=np.array([3, 3]),
                    point_types=np.array([0, 0]),
                    edge_types=np.array([0, 0, 0, 0, 0, 0]),
                ),
            )()
            self._examples = [type("E", (), {})(), type("E", (), {})()]

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            return self._examples[index]

    batch = FakeBatch()
    predictions = {
        "node_labels": np.array([[10.0, 11.0], [20.0, 21.0]]),
        "edge_labels": np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]]),
    }

    outputs = list(
        processor.yield_from_batch(batch, predictions=predictions, as_matrix=False)
    )
    assert outputs[0].point_labels.shape == (1, 2)
    assert outputs[1].edge_labels.shape == (2, 2)


def test_nodes_edges_to_polarized_hamiltonian():
    geometry = _single_orbital_geometry()
    converter = conversions.get_converter(Formats.NODESEDGES, Formats.SISL_H)

    h = converter(
        node_vals=np.array([[1.0, 2.0]]),
        edge_vals=np.empty((0, 2)),
        edge_index=np.empty((2, 0), dtype=np.int64),
        geometry=geometry,
        matrix_component_policy="spin_h_only",
    )

    assert isinstance(h, sisl.Hamiltonian)
    assert h._csr.data.shape[1] == 2
    np.testing.assert_allclose(h.tocsr(0).toarray()[0, 0], 1.0)
    np.testing.assert_allclose(h.tocsr(1).toarray()[0, 0], 2.0)
