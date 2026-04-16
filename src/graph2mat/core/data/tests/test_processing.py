import pytest

import numpy as np
import warnings

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
