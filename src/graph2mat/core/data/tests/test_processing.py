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
def test_init_data(
    positions, basis_table, basis_convention, n_ats, load_matrix, new_method, config_cls
):
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=basis_table, symmetric_matrix=True, sub_point_matrix=False
    )

    positions = positions[:n_ats]

    matrix = None
    if load_matrix:
        matrix = np.random.rand(6, 6)

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
