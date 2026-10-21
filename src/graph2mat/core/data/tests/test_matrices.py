import sisl

from graph2mat.core.data.matrices import get_matrix_cls
from graph2mat.core.data.matrices.physics.orbital_matrix import OrbitalMatrix


def test_hamiltonian_matrix_class_resolution():
    assert get_matrix_cls("hamiltonian") is OrbitalMatrix
    assert get_matrix_cls(sisl.Hamiltonian) is OrbitalMatrix


def test_grouping_rejects_shells_in_a_different_order():
    """Audit M13: '1x0e+1x1o' and '1x1o+1x0e' were grouped together (basis_shape) or masked
    as if l-sorted (max), so one species' rows got the other's orbital order."""
    import pytest

    from graph2mat import BasisTableWithEdges, PointBasis

    A = PointBasis("A", R=2.0, basis="1x0e+1x1o")
    B = PointBasis("B", R=2.0, basis="1x1o+1x0e")
    with pytest.raises(ValueError, match="different order"):
        BasisTableWithEdges([A, B]).group("basis_shape")
    with pytest.raises(ValueError, match="sorted by l"):
        BasisTableWithEdges([A, B]).group("max")
    BasisTableWithEdges([A, PointBasis("C", R=2.0, basis="1x0e+1x1o")]).group("basis_shape")


def test_max_grouping_of_an_atomic_table():
    """Audit (upstream): group('max') rebuilt the table with the subclass, which failed for
    AtomicTableWithEdges (AtomUnknown 'all')."""
    import sisl

    from graph2mat.core.data.table import AtomicTableWithEdges

    H = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=1.0)])
    C = sisl.Atom(6, orbitals=[sisl.AtomicOrbital("2s", R=1.5)] + [sisl.AtomicOrbital(f"2p{c}", R=1.5) for c in "yzx"])
    table, *_ = AtomicTableWithEdges([H, C]).group("max")
    assert len(table.basis) == 1
