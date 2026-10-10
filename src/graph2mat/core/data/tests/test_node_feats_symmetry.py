"""Roadmap 3.8 (MATH_EQUIV-22): node features must transform as their declared irreps."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from graph2mat import BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.core.data.node_feats import WaterDipole, WaterDipoleInv


@pytest.fixture
def processor():
    table = BasisTableWithEdges([PointBasis(1, R=2, basis="1x0e"), PointBasis(8, R=2, basis="1x0e")])
    return MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)


def _water(rotation=np.eye(3)):
    positions = np.array([[0.0, 0.0, 0.1], [0.76, 0.3, -0.5], [-0.76, -0.2, -0.45]]) @ rotation.T
    return SimpleNamespace(positions=positions, point_types=np.array([8, 1, 1]))


def test_water_dipole_is_a_vector_and_its_invariant_a_scalar(processor):
    rotation = Rotation.from_euler("xyz", [0.4, -1.1, 2.0]).as_matrix()
    cob = processor.basis_table.change_of_basis
    rotated_basis = cob @ rotation @ np.linalg.inv(cob)

    vector, vector_rotated = (WaterDipole.get_feature(_water(r), processor) for r in (np.eye(3), rotation))
    np.testing.assert_allclose(vector_rotated, vector @ rotated_basis.T, atol=1e-12)
    assert np.abs(vector).max() > 1e-3

    scalar, scalar_rotated = (WaterDipoleInv.get_feature(_water(r), processor) for r in (np.eye(3), rotation))
    np.testing.assert_allclose(scalar_rotated, scalar, atol=1e-12)
    np.testing.assert_allclose(scalar[:, 0], np.linalg.norm(vector, axis=1), atol=1e-12)
