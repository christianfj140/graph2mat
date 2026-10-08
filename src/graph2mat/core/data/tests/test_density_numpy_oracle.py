"""Density postprocessing also works in the base wheel without Torch."""

import numpy as np
import pytest
import sisl

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.core.data.metrics import normalized_density_error


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("components", [1, 2])
def test_numpy_density_matches_orbital_integral_and_valid_observation_policy(symmetric, components):
    r = np.linspace(0, 1.3, 101)
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", (r, np.exp(-r)), q0=1)])
    geometry = sisl.Geometry([[2., 2., 2.]], atoms=atom, lattice=[4., 4., 4.])
    geometry.set_nsc([1, 1, 1])
    matrix = sisl.DensityMatrix(geometry, spin="polarized" if components == 2 else "unpolarized")
    matrix[0, 0] = np.ones(components)
    table = AtomicTableWithEdges([atom])
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", symmetric_matrix=symmetric,
                                    sub_point_matrix=True, n_matrix_components=components)
    data = BasisMatrixData.new(matrix, data_processor=processor, labels=True)
    ref, edge_ref = data.point_labels.copy(), data.edge_labels.copy()
    grid = sisl.Grid(.3, geometry=geometry)
    coordinates = np.indices(grid.shape).reshape(3, -1).T @ grid.dcell + grid.origin
    integral = (atom.orbitals[0].psi(coordinates - geometry.xyz[0]) ** 2).sum() * grid.dvolume
    loss, stats = normalized_density_error(nodes_pred=ref + 1., nodes_ref=ref, edges_pred=edge_ref, edges_ref=edge_ref,
                                          batch=data, basis_table=table, grid_spacing=.3, log_verbose=True)
    assert loss == pytest.approx(components * integral, rel=1e-10)
    assert stats["total_electrons"] == 1 and stats["n_structures"] == 1
    missing = np.full_like(ref, np.nan)
    value, stats = normalized_density_error(nodes_pred=np.full_like(ref, np.inf), nodes_ref=missing,
                                           edges_pred=edge_ref, edges_ref=edge_ref, batch=data,
                                           basis_table=table, grid_spacing=.3, log_verbose=True)
    assert value == 0 and stats["avg_error_percent"] == 0
    np.testing.assert_array_equal(data.point_labels, ref)
    with pytest.raises(ValueError, match="finite"):
        normalized_density_error(nodes_pred=np.full_like(ref, np.inf), nodes_ref=ref,
                                 edges_pred=edge_ref, edges_ref=edge_ref, batch=data, basis_table=table, grid_spacing=.3)
