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
