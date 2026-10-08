import pytest
import numpy as np

from graph2mat import PointBasis, BasisTableWithEdges


@pytest.mark.parametrize(
    "basis_convention", ["cartesian", "spherical", "siesta_spherical"]
)
def test_init_table(basis_convention):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention=basis_convention)
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention=basis_convention)

    table = BasisTableWithEdges([point_1, point_2])

    assert table.basis_convention == basis_convention


def test_different_convention():
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention="cartesian")
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention="spherical")

    with pytest.raises(AssertionError):
        table = BasisTableWithEdges([point_1, point_2])


def test_no_basis():
    point_1 = PointBasis("A", R=2, basis_convention="cartesian")
    point_2 = PointBasis("B", R=5)

    table = BasisTableWithEdges([point_1, point_2])


@pytest.mark.parametrize("other_missing", [False, True])
def test_table_equality_supports_missing_point_reference(other_missing):
    basis = [PointBasis("A", R=2., basis=[1], basis_convention="cartesian")]
    left = BasisTableWithEdges(basis, get_point_matrix=lambda point: None)
    right = BasisTableWithEdges(basis, get_point_matrix=lambda point: None if other_missing else np.zeros((1, 1)))
    assert (left == right) is other_missing
