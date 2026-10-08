import pytest
import numpy as np
import pickle
import sisl

from graph2mat import PointBasis, BasisTableWithEdges, AtomicTableWithEdges


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


@pytest.mark.parametrize("radial", [False, True])
@pytest.mark.parametrize("mixed_shells", [False, True])
def test_atomic_table_pickle_preserves_every_orbital_cutoff(radial, mixed_shells):
    orbitals = []
    radii = []
    for l, radius in ((0, 1.4), (2, 2.1), (3, 2.8)) if mixed_shells else ((0, 1.4),):
        if radial:
            r = np.linspace(0., radius, 100)
            orbitals.extend(sisl.SphericalOrbital(l, (r, np.exp(-r)), R=radius).toAtomicOrbital(n=l+1))
        else:
            orbitals.extend(sisl.AtomicOrbital(n=l+1, l=l, m=m, R=radius) for m in range(-l, l+1))
        radii.extend([radius] * (2*l+1))
    atom = sisl.Atom("H", orbitals=orbitals)
    table = AtomicTableWithEdges([atom])
    restored = pickle.loads(pickle.dumps(table))
    np.testing.assert_array_equal(restored.basis[0].R, radii)
    np.testing.assert_array_equal([orb.R for orb in restored.atoms[0]], radii)
    for orbital, radius in zip(restored.atoms[0], radii):
        assert orbital.orb.R == radius
    assert restored.maxR() == table.maxR()
    assert restored.basis[0].basis == table.basis[0].basis


def test_atomic_table_loads_legacy_atoms_state_without_new_metadata():
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    restored = AtomicTableWithEdges.__new__(AtomicTableWithEdges)
    restored.__setstate__({"atoms": [atom]})
    np.testing.assert_array_equal(restored.basis[0].R, [1.4])
    assert restored.atoms[0] is atom
