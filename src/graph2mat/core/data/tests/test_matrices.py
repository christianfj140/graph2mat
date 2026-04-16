import sisl

from graph2mat.core.data.matrices import get_matrix_cls
from graph2mat.core.data.matrices.physics.orbital_matrix import OrbitalMatrix


def test_hamiltonian_matrix_class_resolution():
    assert get_matrix_cls("hamiltonian") is OrbitalMatrix
    assert get_matrix_cls(sisl.Hamiltonian) is OrbitalMatrix
