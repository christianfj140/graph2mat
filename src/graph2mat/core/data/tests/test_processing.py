import pytest

import numpy as np
import warnings
from types import SimpleNamespace

from graph2mat import (
    PointBasis,
    BasisTableWithEdges,
    MatrixDataProcessor,
    BasisConfiguration,
    BasisMatrixData,
    OrbitalConfiguration,
)


def test_orbital_matrix_point_annotation_is_resolvable():
    from typing import Dict, get_type_hints
    from graph2mat.core.data.matrices.physics.orbital_matrix import OrbitalMatrix

    hints = get_type_hints(OrbitalMatrix.get_point_matrices)
    assert hints["return"] == Dict[int, np.ndarray]


def test_neighbour_images_require_edge_indices():
    with pytest.raises(AssertionError, match="neigh_isc.*edge_index"):
        BasisMatrixData(neigh_isc=np.array([0]), already_basis=True)


@pytest.mark.parametrize("field", ["positions", "shifts", "cell"])
def test_cartesian_arrays_require_data_processor(field):
    shape = (3, 3) if field == "cell" else (1, 3)
    with pytest.raises(ValueError, match="data_processor"):
        BasisMatrixData(node_attrs=np.ones((1, 1)), **{field: np.ones(shape)})


@pytest.mark.parametrize("reference_getter", [None, lambda point: None])
def test_reconstruction_rejects_missing_point_reference(reference_getter):
    table = BasisTableWithEdges(
        [PointBasis("A", R=2, basis=[1])], get_point_matrix=reference_getter
    )
    processor = MatrixDataProcessor(basis_table=table, sub_point_matrix=True)
    arrays = {
        "point_labels": np.array([2.]), "edge_labels": np.empty(0),
        "point_types": np.array([0]), "edge_types": np.empty(0, dtype=int),
        "edge_index": np.empty((2, 0), dtype=int), "neigh_isc": np.empty(0, dtype=int),
        "nsc": np.ones(3, dtype=int),
    }
    with pytest.raises(ValueError, match="point.*reference|reference.*point"):
        processor.labels_to("numpy", arrays)
    np.testing.assert_array_equal(arrays["point_labels"], [2.])


@pytest.fixture(scope="module")
def positions():
    return np.array([[0.1, 0, 0], [6.0, 0, 0]])


@pytest.fixture(scope="module", params=["cartesian", "spherical", "siesta_spherical"])
def basis_convention(request):
    return request.param


@pytest.fixture(scope="module")
def basis_table(basis_convention):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention=basis_convention)
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention=basis_convention)

    return BasisTableWithEdges([point_1, point_2])


@pytest.mark.parametrize("load_matrix", [False, True])
@pytest.mark.parametrize("n_ats", [1, 2])
@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
@pytest.mark.parametrize("new_method", ["from_config", "new"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_init_data(
    positions, basis_table, basis_convention, n_ats, load_matrix, new_method, config_cls,
    symmetric,
):
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=basis_table, symmetric_matrix=symmetric, sub_point_matrix=False
    )

    positions = positions[:n_ats]

    matrix = None
    if load_matrix:
        size = 1 if n_ats == 1 else 6
        matrix = np.random.default_rng(0).random((size, size))
        if symmetric:
            matrix = (matrix + matrix.T) / 2

    config = config_cls(
        point_types=["A", "B"][:n_ats],
        positions=positions,
        basis=basis_table,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
        matrix=matrix,
    )

    # Test from_config method
    new = getattr(BasisMatrixData, new_method)
    data = new(config, processor)

    if basis_convention == "cartesian":
        assert np.all(data.positions == positions)
    else:
        assert (data.positions != positions).sum() == n_ats * 2
    if load_matrix:
        reconstructed = processor.matrix_from_data(data, out_format="numpy")
        np.testing.assert_allclose(reconstructed, matrix, rtol=0, atol=1e-12)


def test_sub_point_matrix_disabled_for_hamiltonian(basis_table):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        processor = MatrixDataProcessor(
            basis_table=basis_table,
            symmetric_matrix=True,
            sub_point_matrix=True,
            out_matrix="hamiltonian",
        )

    assert processor.sub_point_matrix is False
    assert any("sub_point_matrix is only supported for density_matrix" in str(w.message) for w in caught)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_matrix_from_data_uses_predictions_without_mutating_reference(backend):
    from graph2mat import BasisMatrixData

    if backend == "torch":
        from graph2mat.bindings.torch import TorchBasisMatrixData
        data_cls = TorchBasisMatrixData
    else:
        data_cls = BasisMatrixData
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1], basis_convention="cartesian")])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    config = BasisConfiguration(
        point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
        basis=table, cell=np.eye(3) * 10, pbc=(False, False, False), matrix=np.ones((2, 2)),
    )
    data = data_cls.new(config, processor)
    reference_nodes, reference_edges = data.point_labels, data.edge_labels
    node_values = reference_nodes + 1
    node_values[1] += 2
    edge_values = reference_edges + 2
    matrix = processor.matrix_from_data(data, predictions={
        "node_labels": node_values, "edge_labels": edge_values,
    }, out_format="scipy_csr")
    np.testing.assert_array_equal(matrix.toarray(), [[2., 3.], [3., 4.]])
    assert data.point_labels is reference_nodes
    assert data.edge_labels is reference_edges
    np.testing.assert_array_equal(data.numpy_arrays().point_labels, [1., 1.])
    np.testing.assert_array_equal(data.numpy_arrays().edge_labels, [1.])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_nonorthogonal_analytic_levels_and_gauge_shift(backend, symmetric):
    import sisl
    from scipy.linalg import cholesky, solve_triangular
    from scipy.sparse import csr_array
    from graph2mat import AtomicTableWithEdges

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[20.] * 3)
    h = np.array([[1.375, -.125], [-.125, 1.375]])
    s = np.array([[1., .25], [.25, 1.]])
    original = sisl.Hamiltonian.fromsp(geometry, csr_array(h), S=csr_array(s))
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix="hamiltonian",
        symmetric_matrix=symmetric, sub_point_matrix=False,
        n_matrix_components=2, matrix_component_policy="h_and_overlap",
    )
    data_cls = BasisMatrixData
    if backend == "torch":
        from graph2mat.bindings.torch import TorchBasisMatrixData
        data_cls = TorchBasisMatrixData
    data = data_cls.new(original, processor)
    if backend == "torch":
        # Dyadic input coefficients are exact in FP32; do gauge arithmetic in FP64.
        data.point_labels = data.point_labels.double()
        data.edge_labels = data.edge_labels.double()
    nodes, edges = (a.copy() for a in (data.numpy_arrays().point_labels, data.numpy_arrays().edge_labels))
    matrix = processor.matrix_from_data(data, threshold=None)
    levels = matrix.eigenvalue().eig
    np.testing.assert_allclose(levels, [1., 2.], rtol=1e-10, atol=1e-10)
    l = cholesky(s, lower=True)
    left = solve_triangular(l, h, lower=True)
    cholesky_h = solve_triangular(l, left.T, lower=True).T
    np.testing.assert_allclose(np.linalg.eigvalsh(cholesky_h), levels, rtol=1e-10, atol=1e-10)
    predicted_nodes, predicted_edges = nodes.copy(), edges.copy()
    predicted_nodes[:, 0] += .7 * predicted_nodes[:, -1]
    predicted_edges[:, 0] += .7 * predicted_edges[:, -1]
    if backend == "torch":
        import torch
        predicted_nodes = torch.from_numpy(predicted_nodes)
        predicted_edges = torch.from_numpy(predicted_edges)
    shifted = processor.matrix_from_data(data, predictions={
        "node_labels": predicted_nodes, "edge_labels": predicted_edges,
    }, threshold=None)
    np.testing.assert_allclose(shifted.tocsr(0).toarray(), h + .7 * s, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(shifted.tocsr(shifted.S_idx).toarray(), s)
    shifted_levels = shifted.eigenvalue().eig
    np.testing.assert_allclose(shifted_levels - levels, .7, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.diff(shifted_levels), np.diff(levels), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(data.numpy_arrays().point_labels, nodes)
    np.testing.assert_array_equal(data.numpy_arrays().edge_labels, edges)
    np.testing.assert_array_equal(original.tocsr(0).toarray(), h)
    np.testing.assert_array_equal(original.tocsr(original.S_idx).toarray(), s)


@pytest.mark.parametrize("singular", [False, True])
def test_nonorthogonal_controls_detect_wrong_or_singular_overlap(singular):
    import sisl
    from scipy.sparse import csr_array
    from graph2mat import AtomicTableWithEdges

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[20.] * 3)
    h = np.array([[1.375, -.125], [-.125, 1.375]])
    s = np.ones((2, 2)) if singular else np.eye(2)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix="hamiltonian",
        symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=2, matrix_component_policy="h_and_overlap",
    )
    data = BasisMatrixData.new(sisl.Hamiltonian.fromsp(geometry, csr_array(h), S=csr_array(s)), processor)
    before = data.point_labels.copy(), data.edge_labels.copy()
    reconstructed = processor.matrix_from_data(data, threshold=None)
    np.testing.assert_array_equal(reconstructed.tocsr(reconstructed.S_idx).toarray(), s)
    if singular:
        with pytest.raises(np.linalg.LinAlgError):
            reconstructed.eigenvalue()
    else:
        assert not np.allclose(reconstructed.eigenvalue().eig, [1., 2.], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(data.point_labels, before[0])
    np.testing.assert_array_equal(data.edge_labels, before[1])


def test_yield_from_batch_with_symmetric_matrix_handles_odd_edges_per_graph():
    basis_table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    class FakeBatch:
        num_graphs = 2

        def __init__(self):
            self._arrays = SimpleNamespace(
                ptr=np.array([0, 1, 2]),
                n_edges=np.array([3, 3]),
                point_types=np.array([0, 0]),
                edge_types=np.array([0, 0, 0, 0, 0, 0]),
            )
            self._examples = [SimpleNamespace(), SimpleNamespace()]

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            return self._examples[index]

    batch = FakeBatch()
    predictions = {
        "node_labels": np.array([10.0, 20.0]),
        "edge_labels": np.array([1.0, 2.0, 3.0, 4.0]),
    }

    outputs = list(processor.yield_from_batch(batch, predictions=predictions, as_matrix=False))

    assert len(outputs) == 2
    np.testing.assert_array_equal(outputs[0].point_labels, np.array([10.0]))
    np.testing.assert_array_equal(outputs[1].point_labels, np.array([20.0]))
    np.testing.assert_array_equal(outputs[0].edge_labels, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(outputs[1].edge_labels, np.array([3.0, 4.0]))


def test_symmetric_unique_edge_mask_matches_label_count_not_only_position():
    basis_table = BasisTableWithEdges(
        [
            PointBasis("A", R=2, basis=[1]),
            PointBasis("B", R=2, basis=[2]),
        ]
    )
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    mask = processor._get_symmetric_unique_edge_mask(
        edge_types=np.array([0, 1, 0]),
        expected_nlabels=3,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False]))


def test_symmetric_unique_edge_mask_handles_same_species_by_label_length():
    basis_table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    mask = processor._get_symmetric_unique_edge_mask(
        edge_types=np.array([0, 0]),
        expected_nlabels=1,
    )

    np.testing.assert_array_equal(mask, np.array([True, False]))


def test_irreps_from_data_uses_label_count_for_symmetric_edge_representatives():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    basis_table = BasisTableWithEdges(
        [
            PointBasis("A", R=2, basis=[1]),
            PointBasis("B", R=2, basis=[2]),
        ]
    )
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    point_types = np.array([1, 0, 0])
    edge_index = np.array(
        [
            [0, 1, 1],
            [1, 2, 0],
        ]
    )
    edge_types = basis_table.point_type_to_edge_type(point_types[edge_index])
    edge_labels = torch.arange(3, dtype=torch.float64)

    class FakeData:
        def __init__(self):
            self.point_labels = torch.arange(
                processor.basis_table.point_block_pointer(point_types)[-1],
                dtype=torch.float64,
            )
            self.edge_labels = edge_labels

        def numpy_arrays(self):
            return {
                "point_types": point_types,
                "edge_index": edge_index,
                "edge_types": edge_types,
            }

    result = processor.irreps_from_data(FakeData())

    np.testing.assert_array_equal(edge_types, np.array([-1, 0, 1]))
    assert result["edge_labels"].numel() == len(edge_labels)


def _h2o_like_symmetric_mismatch_case():
    basis_table = BasisTableWithEdges(
        [
            PointBasis("A", R=2, basis=[3]),
            PointBasis("B", R=2, basis=[4]),
            PointBasis("C", R=2, basis=[6]),
            PointBasis("D", R=2, basis=[7]),
            PointBasis("E", R=2, basis=[8]),
        ]
    )
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    point_types = np.array([3, 3, 0, 0, 4, 4, 1, 1, 2, 3])
    edge_index = np.array(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
        ]
    )
    edge_types = basis_table.point_type_to_edge_type(point_types[edge_index])
    edge_labels = np.arange(155, dtype=float)

    class FakeExample:
        def __init__(self):
            self.edge_labels = edge_labels.copy()

        def numpy_arrays(self):
            return {
                "point_types": point_types,
                "edge_index": edge_index,
                "edge_types": edge_types,
            }

    class FakeBatch:
        num_graphs = 1

        def __init__(self):
            self._arrays = SimpleNamespace(
                ptr=np.array([0, len(point_types)]),
                n_edges=np.array([edge_index.shape[1]]),
                point_types=point_types,
                edge_types=edge_types,
            )
            self._example = FakeExample()

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            assert index == 0
            return self._example

    return (
        basis_table,
        processor,
        point_types,
        edge_index,
        edge_types,
        edge_labels,
        FakeBatch(),
    )


def test_irreps_from_data_handles_h2o_like_symmetric_edge_label_length():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    (
        _basis_table,
        processor,
        point_types,
        edge_index,
        edge_types,
        edge_labels,
        _batch,
    ) = _h2o_like_symmetric_mismatch_case()

    class FakeData:
        def __init__(self):
            self.point_labels = torch.zeros(
                processor.basis_table.point_block_pointer(point_types)[-1],
                dtype=torch.float64,
            )
            self.edge_labels = torch.as_tensor(edge_labels, dtype=torch.float64)

        def numpy_arrays(self):
            return {
                "point_types": point_types,
                "edge_index": edge_index,
                "edge_types": edge_types,
            }

    result = processor.irreps_from_data(FakeData())

    assert len(edge_labels) == 155
    assert result["edge_labels"].numel() == len(edge_labels)


def test_debug_edge_label_accounting_reports_symmetric_prediction_mismatch():
    (
        _basis_table,
        processor,
        point_types,
        edge_index,
        edge_types,
        _edge_labels,
        batch,
    ) = _h2o_like_symmetric_mismatch_case()
    predictions = {
        "node_labels": np.zeros(
            processor.basis_table.point_block_pointer(point_types)[-1]
        ),
        "edge_labels": np.zeros(155),
    }

    report = processor._debug_edge_label_accounting(batch, predictions=predictions)

    assert report["symmetric_matrix"] is True
    assert report["n_edges"] == 5
    assert report["edge_index_shape"] == (2, 5)
    np.testing.assert_array_equal(report["edge_types"], edge_types)
    np.testing.assert_array_equal(report["point_types"], point_types)
    np.testing.assert_array_equal(report["basis_sizes"], np.array([3, 4, 6, 7, 8]))
    np.testing.assert_array_equal(
        report["selected_unique_edge_mask"],
        np.array([True, True, True, True, True]),
    )
    np.testing.assert_array_equal(report["selected_edge_indices"], np.arange(5))
    np.testing.assert_array_equal(report["selected_edge_index"], edge_index)
    np.testing.assert_array_equal(report["selected_edge_types"], edge_types)
    np.testing.assert_array_equal(
        report["per_selected_edge_block_sizes"],
        np.array([49, 9, 64, 16, 42]),
    )
    assert report["expected_edge_label_len"] == 180
    assert report["actual_edge_labels_len"] == 155
    assert report["actual_prediction_edge_labels_len"] == 155
    assert report["fallback_pointer_based_expected_len"] == 155
    assert report["expected_matches_example_edge_labels"] is False


def test_yield_from_batch_uses_example_edge_label_length_for_symmetric_predictions():
    (
        _basis_table,
        processor,
        point_types,
        _edge_index,
        _edge_types,
        edge_labels,
        batch,
    ) = _h2o_like_symmetric_mismatch_case()
    predictions = {
        "node_labels": np.zeros(
            processor.basis_table.point_block_pointer(point_types)[-1]
        ),
        "edge_labels": edge_labels.copy(),
    }

    outputs = list(
        processor.yield_from_batch(batch, predictions=predictions, as_matrix=False)
    )

    assert len(outputs) == 1
    np.testing.assert_array_equal(outputs[0].edge_labels, edge_labels)


def test_yield_from_batch_reconstructs_original_edge_labels_from_predictions():
    (
        _basis_table,
        processor,
        point_types,
        _edge_index,
        _edge_types,
        edge_labels,
        batch,
    ) = _h2o_like_symmetric_mismatch_case()
    predictions = {
        "node_labels": np.zeros(
            processor.basis_table.point_block_pointer(point_types)[-1]
        ),
        "edge_labels": batch.get_example(0).edge_labels.copy(),
    }

    reconstructed = next(
        processor.yield_from_batch(batch, predictions=predictions, as_matrix=False)
    )

    np.testing.assert_array_equal(reconstructed.edge_labels, edge_labels)


def test_yield_from_batch_fails_when_prediction_edge_labels_are_too_short():
    (
        _basis_table,
        processor,
        point_types,
        _edge_index,
        _edge_types,
        _edge_labels,
        batch,
    ) = _h2o_like_symmetric_mismatch_case()
    predictions = {
        "node_labels": np.zeros(
            processor.basis_table.point_block_pointer(point_types)[-1]
        ),
        "edge_labels": np.zeros(154),
    }

    with pytest.raises(ValueError, match="expected 155, got 154"):
        list(
            processor.yield_from_batch(
                batch, predictions=predictions, as_matrix=False
            )
        )


def test_yield_from_batch_fails_when_prediction_edge_labels_are_too_long():
    (
        _basis_table,
        processor,
        point_types,
        _edge_index,
        _edge_types,
        _edge_labels,
        batch,
    ) = _h2o_like_symmetric_mismatch_case()
    predictions = {
        "node_labels": np.zeros(
            processor.basis_table.point_block_pointer(point_types)[-1]
        ),
        "edge_labels": np.zeros(156),
    }

    with pytest.raises(ValueError, match="consumed=155, total=156"):
        list(
            processor.yield_from_batch(
                batch, predictions=predictions, as_matrix=False
            )
        )


def test_yield_from_batch_handles_empty_edge_predictions():
    basis_table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    class FakeExample:
        edge_labels = np.array([])

    class FakeBatch:
        num_graphs = 1

        def __init__(self):
            self._arrays = SimpleNamespace(
                ptr=np.array([0, 1]),
                n_edges=np.array([0]),
                point_types=np.array([0]),
                edge_types=np.array([], dtype=np.int64),
            )
            self._example = FakeExample()

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            assert index == 0
            return self._example

    predictions = {
        "node_labels": np.array([1.0]),
        "edge_labels": np.array([]),
    }

    outputs = list(
        processor.yield_from_batch(
            FakeBatch(), predictions=predictions, as_matrix=False
        )
    )

    assert len(outputs) == 1
    np.testing.assert_array_equal(outputs[0].edge_labels, np.array([]))


def test_from_config_rejects_shifts_beyond_nsc():
    """A neighbour in the next cell with nsc=1 would wrap to the unit cell (wrong block)."""
    import numpy as np
    import pytest

    from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
    from graph2mat.core.data.processing import BasisMatrixData

    basis = [PointBasis("A", R=1.5, basis="0e", basis_convention="cartesian")]
    processor = MatrixDataProcessor(basis_table=BasisTableWithEdges(basis), symmetric_matrix=True,
                                    sub_point_matrix=False)
    config = BasisConfiguration(point_types=["A"], positions=np.zeros((1, 3)), basis=basis,
                                cell=np.diag([2.0, 20.0, 20.0]), pbc=(True, False, False))
    assert BasisMatrixData.from_config(config, processor).n_edges > 0
    with pytest.raises(ValueError, match="nsc"):
        BasisMatrixData.from_config(config, processor, nsc=[1, 1, 1])
