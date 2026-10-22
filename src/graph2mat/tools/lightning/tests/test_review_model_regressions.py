"""Model/data representation checks missing from the audit fix."""

from types import SimpleNamespace

import numpy as np
import pytest
import sisl
import torch

from graph2mat import AtomicTableWithEdges, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.models.long_range import label_element_atoms
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


def _check_agreement(model_table, data_table):
    model = SimpleNamespace(
        basis_table=model_table, hparams={},
        trainer=SimpleNamespace(datamodule=SimpleNamespace(
            basis_table=data_table, matrix_component_policy="h_only", magnetic_moments=None,
            edge_cutoff_padding=0.0,
        )),
    )
    LitMACEMatrixModel._check_data_agreement(model)


@pytest.mark.parametrize("model_basis,data_basis", [("3x0e", "1x1o"),
                                                    ("1x0e+1x1o", "1x1o+1x0e")])
def test_model_data_agreement_rejects_equal_size_different_shells(model_basis, data_basis):
    model = BasisTableWithEdges([PointBasis(6, R=1.0, basis=model_basis)])
    data = BasisTableWithEdges([PointBasis(6, R=1.0, basis=data_basis)])
    with pytest.raises(ValueError, match="basis"):
        _check_agreement(model, data)


def test_model_data_agreement_rejects_different_basis_conventions():
    model = BasisTableWithEdges([PointBasis(6, R=1.0, basis="1x0e+1x1o", basis_convention="cartesian")])
    data = BasisTableWithEdges([PointBasis(6, R=1.0, basis="1x0e+1x1o", basis_convention="siesta_spherical")])
    with pytest.raises(ValueError, match="basis"):
        _check_agreement(model, data)


def test_model_data_agreement_accepts_equivalent_explicit_bases():
    model = BasisTableWithEdges([PointBasis(6, R=1.0, basis="2x0e+1x1o", basis_convention="spherical")])
    data = BasisTableWithEdges([PointBasis(6, R=1.0, basis="1x0e+1x0e+1x1o", basis_convention="yzx")])
    _check_agreement(model, data)


def test_long_range_common_spin_shift_with_real_h_and_overlap_input():
    atoms = [sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s", R=1.0)]),
             sisl.Atom(6, orbitals=[sisl.AtomicOrbital("2s", R=1.0)])]
    table = AtomicTableWithEdges(atoms)
    geometry = sisl.Geometry([[0, 0, 0], [1, 0.1, 0], [1.6, 1.3, 0]],
                             atoms=[atoms[0], atoms[1], atoms[0]], lattice=[10] * 3)
    matrix = sisl.Hamiltonian(geometry, spin="polarized", orthogonal=False)
    for orbital in range(3):
        matrix[orbital, orbital, :] = [0, 0, 1]
    for i, j in ((0, 1), (1, 0), (1, 2), (2, 1)):
        matrix[i, j, :] = [0, 0, 0.1]
    processor = MatrixDataProcessor(table, out_matrix="hamiltonian", sub_point_matrix=False,
                                    symmetric_matrix=True, matrix_component_policy="h_and_overlap",
                                    n_matrix_components=3)
    data = TorchBasisMatrixData.new(matrix, processor)
    torch.manual_seed(3)
    lit = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+1x1o",
                            edge_hidden_irreps="2x0e+1x1o", num_bessel=3, max_ell=1,
                            num_interactions=1, correlation=1, lean_mace=True,
                            n_matrix_components=3, n_output_components=2, long_range=True,
                            long_range_smearing=1.0).eval()
    torch.nn.init.normal_(lit.model.charge_readout.weight)
    with torch.no_grad():
        output = lit(data)
        head = lit.model.charge_readout
        lit.model.charge_readout = None
        base = lit(data)
        lit.model.charge_readout = head
    owners, senders, receivers = label_element_atoms(
        np.asarray(data.point_types), np.asarray(data.edge_types), np.asarray(data.edge_index),
        len(data.edge_labels), table,
    )
    potential = output["node_potential"]
    expected_nodes = potential[owners] * data.point_labels[:, -1]
    expected_edges = (potential[senders] + potential[receivers]) / 2 * data.edge_labels[:, -1]
    assert expected_nodes.abs().max() > 1e-4
    for column in (0, 1):
        torch.testing.assert_close(output["node_labels"][:, column] - base["node_labels"][:, column], expected_nodes)
        torch.testing.assert_close(output["edge_labels"][:, column] - base["edge_labels"][:, column], expected_edges)
