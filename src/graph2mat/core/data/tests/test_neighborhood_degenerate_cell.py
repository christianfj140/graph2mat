"""Roadmap 3.11 (PHYS_H-17): a periodic direction needs a real lattice vector. A zero
cell used to be replaced by a 1 Angstrom identity cell with pbc, giving absurd images."""

import numpy as np
import pytest

from graph2mat.core.data.neighborhood import get_neighborhood

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


def test_periodic_direction_with_zero_cell_raises():
    with pytest.raises(ValueError, match="periodic"):
        get_neighborhood(POSITIONS, cutoff=2.0, pbc=(True, True, True), cell=np.zeros((3, 3)))


def test_molecule_without_cell_still_works():
    edge_index, unit_shifts, _ = get_neighborhood(POSITIONS, cutoff=2.0, pbc=(False, False, False),
                                                  cell=np.zeros((3, 3)))
    assert edge_index.shape == (2, 2) and not unit_shifts.any()


def test_slab_with_zero_vacuum_vector_but_no_pbc_there():
    cell = np.diag([2.46, 2.46, 0.0])
    get_neighborhood(POSITIONS, cutoff=2.0, pbc=(True, True, False), cell=cell)
