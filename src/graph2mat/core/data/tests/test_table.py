import pytest
import numpy as np
import pickle
import sisl
from copy import deepcopy

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


@pytest.mark.parametrize("radial", [False, True])
def test_atomic_table_deepcopy_preserves_runtime_state_and_orbital_cutoffs(radial):
    radius = 2.1
    if radial:
        r = np.linspace(0., radius, 100)
        orbitals = sisl.SphericalOrbital(2, (r, np.exp(-r)), R=radius).toAtomicOrbital(n=3)
    else:
        orbitals = [sisl.AtomicOrbital(n=3, l=2, m=m, R=radius) for m in range(-2, 3)]
    table = AtomicTableWithEdges([sisl.Atom("H", orbitals=orbitals)])
    table.point_matrix = [np.eye(5) * 7.]
    expected_basis = deepcopy(table.basis)
    duplicated = deepcopy(table)
    np.testing.assert_array_equal(duplicated.point_matrix[0], table.point_matrix[0])
    assert duplicated.basis == expected_basis
    np.testing.assert_array_equal([orb.R for orb in duplicated.atoms[0]], [radius] * 5)
    np.testing.assert_array_equal([orb.orb.R for orb in duplicated.atoms[0]], [radius] * 5)
    duplicated.point_matrix[0][0, 0] = -9
    assert table.point_matrix[0][0, 0] == 7
    duplicated.R[0] = -1
    assert table.R[0] == radius


def test_atomic_table_deepcopy_does_not_deserialize_portable_file_metadata():
    table = AtomicTableWithEdges([sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])])
    table.file_names = ["original.ion.xml"]
    table.file_contents = ["metadata is not an instruction to deserialize during deepcopy"]
    copied = deepcopy(table)
    assert copied.file_names == table.file_names and copied.file_names is not table.file_names
    assert copied.file_contents == table.file_contents and copied.file_contents is not table.file_contents
    assert copied.basis == table.basis


def test_atomic_table_deepcopy_preserves_distinct_atomic_and_radial_support():
    table = AtomicTableWithEdges([sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])])
    table.atoms[0][0].orb._R = 1.3
    copied = deepcopy(table)
    assert copied.atoms[0][0].R == table.atoms[0][0].R
    assert copied.atoms[0][0].orb.R == table.atoms[0][0].orb.R
