"""Regression of the shared NumPy/Torch data-container API (roadmap 3.6)."""

import numpy as np
import pytest

from graph2mat import (
    BasisConfiguration, BasisMatrixData, BasisTableWithEdges,
    MatrixDataProcessor, PointBasis,
)
from graph2mat.bindings.torch import TorchBasisMatrixData


@pytest.fixture(params=[BasisMatrixData, TorchBasisMatrixData])
def data_cls(request):
    return request.param


def _data(data_cls, labelled=False):
    table = BasisTableWithEdges([
        PointBasis("A", R=2, basis=[1], basis_convention="siesta_spherical"),
        PointBasis("B", R=2, basis=[1], basis_convention="siesta_spherical"),
    ])
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
    )
    config = BasisConfiguration(
        point_types=["B", "A", "B"],
        positions=np.array([[.1, .2, .3], [.4, .5, .6], [1.1, 1.2, 1.3]]),
        basis=table, cell=np.eye(3)*10, pbc=(False, False, False),
        matrix=np.ones((3, 3)) if labelled else None,
    )
    return data_cls.new(config, processor)


def test_label_filters_survive_copy_and_backend_conversion(data_cls):
    original = _data(data_cls, labelled=True)
    arrays = {
        key: original.numpy_arrays()[key] for key in original._init_keys
        if original._data.get(key) is not None
    }
    arrays.update(
        labels_point_filter=np.array([True, False, True]),
        labels_edge_filter=np.array([False, True, False]),
    )
    data = data_cls(
        **arrays, data_processor=original.metadata["data_processor"], already_basis=True,
    )
    for copied in (data, data.copy(), data.copy(cls=BasisMatrixData), data.copy(cls=TorchBasisMatrixData)):
        np.testing.assert_array_equal(
            copied.numpy_arrays()["labels_point_filter"], [True, False, True],
        )
        np.testing.assert_array_equal(
            copied.numpy_arrays()["labels_edge_filter"], [False, True, False],
        )
        np.testing.assert_array_equal(copied.numpy_arrays()["point_labels"], np.ones(3))


def test_unlabelled_subgraph_reindexes_nodes_without_rotating_coordinates(data_cls):
    data = _data(data_cls)
    selected = data.node_types_subgraph(np.array([1]))
    assert selected.num_nodes == 2 and selected.n_edges == 2
    np.testing.assert_array_equal(selected.numpy_arrays()["edge_index"], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(selected.numpy_arrays()["point_types"], [1, 1])
    np.testing.assert_array_equal(
        selected.numpy_arrays()["positions"], data.numpy_arrays()["positions"][[0, 2]],
    )
    np.testing.assert_array_equal(selected.numpy_arrays()["cell"], data.numpy_arrays()["cell"])
    assert selected.metadata["data_processor"] is data.metadata["data_processor"]
    assert getattr(selected, "point_labels", None) is None
    assert getattr(selected, "edge_labels", None) is None


def test_unlabelled_subgraph_allows_empty_selection(data_cls):
    selected = _data(data_cls).node_types_subgraph(np.array([99]))
    assert selected.num_nodes == 0 and selected.n_edges == 0
    assert selected.positions.shape == (0, 3)
    assert selected.edge_index.shape == (2, 0)


def test_labelled_subgraph_keeps_explicit_rejection(data_cls):
    with pytest.raises(ValueError, match="labels"):
        _data(data_cls, labelled=True).node_types_subgraph(np.array([1]))
