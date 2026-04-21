import numpy as np
import pytest
import sisl
import torch
from scipy.sparse import csr_array

from graph2mat import BasisTableWithEdges, OrbitalConfiguration, PointBasis
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.model import LitBasisMatrixModel


def _polarized_single_orbital_config():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry([[0.0, 0.0, 0.0]], atoms=[atom], lattice=[10, 10, 10])

    h = sisl.Hamiltonian(geometry, spin=sisl.Spin("polarized"))
    h._csr = h._csr.fromsp([csr_array([[1.0]]), csr_array([[2.0]])])

    return OrbitalConfiguration.from_matrix(h, labels=True)


def test_datamodule_raises_on_matrix_component_mismatch():
    config = _polarized_single_orbital_config()
    basis_table = BasisTableWithEdges([PointBasis(1, R=np.array([2.0]), basis=[1])])

    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=1,
        train_runs=[config],
    )

    with pytest.raises(ValueError, match="n_matrix_components mismatch"):
        datamodule.setup("fit")


def test_shape_validation_raises_clear_error_message():
    out = {
        "node_labels": torch.zeros(4),
        "edge_labels": torch.zeros(2),
    }
    batch = {
        "point_labels": torch.zeros(4, 2),
        "edge_labels": torch.zeros(2, 2),
    }

    with pytest.raises(ValueError, match="n_matrix_components"):
        LitBasisMatrixModel._validate_pred_ref_shapes(out, batch)
