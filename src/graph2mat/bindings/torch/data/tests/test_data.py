import pytest

import numpy as np
import torch
import sisl

from graph2mat import (
    PointBasis,
    BasisTableWithEdges,
    MatrixDataProcessor,
    BasisConfiguration,
    OrbitalConfiguration,
)

from graph2mat.bindings.torch import TorchBasisMatrixData, TorchBasisMatrixDataset


def test_torch_predict_supplies_batch_for_every_node_and_ptr():
    """A real pooling operation must see all three nodes in the single graph."""
    processor = MatrixDataProcessor(
        basis_table=BasisTableWithEdges([PointBasis(1, R=1.5, basis=[1])]),
        symmetric_matrix=True, sub_point_matrix=False,
    )
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
                             atoms=[1, 1, 1], lattice=[10]*3)

    def pooled_model(data):
        torch.testing.assert_close(data.ptr, torch.tensor([0, 3]))
        totals = torch.zeros(1).scatter_add(0, data.batch, torch.tensor([1., 2., 3.]))
        return {"node_labels": totals[data.batch],
                "edge_labels": torch.zeros(data.edge_index.shape[1]//2)}

    matrix = processor.torch_predict(pooled_model, geometry)
    np.testing.assert_array_equal(matrix.toarray(), np.eye(3)*6)


@pytest.fixture(scope="module")
def positions():
    return np.array([[0, 0, 0], [6.0, 0, 0]])


@pytest.fixture(scope="module", params=["cartesian", "spherical", "siesta_spherical"])
def basis_convention(request):
    return request.param


@pytest.fixture(scope="module")
def basis_table(basis_convention):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention=basis_convention)
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention=basis_convention)

    return BasisTableWithEdges([point_1, point_2])


@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
@pytest.mark.parametrize("new_method", ["from_config", "new"])
def test_init_data(positions, basis_table, basis_convention, new_method, config_cls):
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=basis_table, symmetric_matrix=True, sub_point_matrix=False
    )

    config = config_cls(
        point_types=["A", "B"],
        positions=positions,
        basis=basis_table,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    # Test from_config method
    new = getattr(TorchBasisMatrixData, new_method)
    data = new(config, processor)

    assert isinstance(data.positions, torch.Tensor)

    if basis_convention == "cartesian":
        assert np.all(data.positions.numpy() == positions)
    else:
        assert (data.positions.numpy() != positions).sum() == 2


@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
@pytest.mark.parametrize("new_method", ["from_config", "new"])
def test_dataset(positions, basis_table, basis_convention, new_method, config_cls):
    # The data processor.
    processor = MatrixDataProcessor(
        basis_table=basis_table, symmetric_matrix=True, sub_point_matrix=False
    )

    config_1 = config_cls(
        point_types=["A", "B"],
        positions=positions,
        basis=basis_table,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    config_2 = config_cls(
        point_types=["B", "A"],
        positions=positions,
        basis=basis_table,
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )

    dataset = TorchBasisMatrixDataset([config_1, config_2], data_processor=processor)

    assert len(dataset) == 2

    assert isinstance(dataset[1].positions, torch.Tensor)

    if basis_convention == "cartesian":
        assert np.all(dataset[1].positions.numpy() == positions)
    else:
        assert (dataset[1].positions.numpy() != positions).sum() == 2
    pass
