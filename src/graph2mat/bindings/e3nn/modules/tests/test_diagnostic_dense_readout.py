import numpy as np
import pytest
import torch
from e3nn import o3

from graph2mat import Graph2Mat, PointBasis
from graph2mat.bindings.e3nn import (
    E3nnDiagnosticDenseEdgeBlock,
    E3nnDiagnosticDenseNodeBlock,
)
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock
from graph2mat.bindings.e3nn.modules.graph2mat import E3nnGraph2Mat
from graph2mat.core.data.metrics import (
    coefficient_space_mse,
    coefficients_to_labels,
    hamiltonian_composite_loss,
    target_coefficients_from_labels,
)


class _Batch(dict):
    def __getattr__(self, key):
        return self[key]


def _assert_has_nonzero_parameter_gradient(module):
    grads = [
        param.grad
        for param in module.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert grads
    assert any(torch.any(grad != 0) for grad in grads)


def _parameter_jacobian(module, output):
    params = [param for param in module.parameters() if param.requires_grad]
    rows = []
    flat_output = output.reshape(-1)

    for value in flat_output:
        grads = torch.autograd.grad(value, params, retain_graph=True)
        rows.append(torch.cat([grad.reshape(-1) for grad in grads]))

    return torch.stack(rows)


class _NoCoefficientNodeBlock:
    def __init__(self, i_basis, j_basis, **kwargs):
        self.i_basis = i_basis
        self.j_basis = j_basis

    def __call__(self, **kwargs):
        n_samples = len(next(iter(kwargs.values())))
        return torch.zeros(
            n_samples,
            self.i_basis.basis_size,
            self.j_basis.basis_size,
        )


class _NoCoefficientEdgeBlock(_NoCoefficientNodeBlock):
    pass


def test_dense_node_readout_shape_and_backward():
    torch.manual_seed(0)
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    readout = E3nnDiagnosticDenseNodeBlock(irreps_in, irreps_out)

    node_feats = torch.randn(5, irreps_in.dim)
    output = readout(node_feats=node_feats)

    assert output.shape == (5, irreps_out.dim)

    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(readout)


def test_dense_edge_readout_shape_and_backward():
    torch.manual_seed(0)
    irreps_in = o3.Irreps("2x0e + 1x1o")
    irreps_out = o3.Irreps("5x0e + 1x1o")
    readout = E3nnDiagnosticDenseEdgeBlock(irreps_in, irreps_out)

    left = torch.randn(4, irreps_in.dim)
    right = torch.randn(4, irreps_in.dim)
    output = readout(edge_messages=(left, right))

    assert output.shape == (4, irreps_out.dim)

    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(readout)


def test_dense_node_readout_inside_irreps_matrix_block():
    torch.manual_seed(0)
    basis = PointBasis("H", R=2.0, basis="2x0e + 1x1o")
    irreps_in = o3.Irreps("6x0e")
    block = E3nnIrrepsMatrixBlock(
        i_basis=basis,
        j_basis=basis,
        symmetry="ij=ji",
        operation_cls=E3nnDiagnosticDenseNodeBlock,
        irreps={"node_feats_irreps": irreps_in},
    )

    node_feats = torch.randn(3, irreps_in.dim)
    output = block(node_feats=node_feats)

    assert output.shape == (3, basis.basis_size, basis.basis_size)

    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(block.operation)


def test_irreps_matrix_block_coefficient_roundtrip():
    torch.manual_seed(0)
    basis = PointBasis("H", R=2.0, basis="2x0e + 1x1o")
    irreps_in = o3.Irreps("6x0e")
    block = E3nnIrrepsMatrixBlock(
        i_basis=basis,
        j_basis=basis,
        symmetry="ij=ji",
        operation_cls=E3nnDiagnosticDenseNodeBlock,
        irreps={"node_feats_irreps": irreps_in},
    )

    node_feats = torch.randn(3, irreps_in.dim)
    coefficients = block.forward_coefficients(node_feats=node_feats)
    labels = block(node_feats=node_feats)

    assert torch.allclose(block.coefficients_to_block(coefficients), labels)
    assert torch.allclose(block.block_to_coefficients(labels), coefficients, atol=1e-6)


def test_default_e3nn_readout_coefficients_are_opt_in():
    torch.manual_seed(0)
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    irreps_in = o3.Irreps("3x0e")
    readout = E3nnGraph2Mat(
        unique_basis=basis,
        irreps={
            "node_feats_irreps": irreps_in,
        },
        symmetric=True,
        preprocessing_edges=None,
    )
    batch = _Batch(
        point_types=torch.tensor([0, 0], dtype=torch.long),
        edge_types=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    node_feats = torch.randn(2, irreps_in.dim)

    default_output = readout(data=batch, node_feats=node_feats)

    assert len(default_output) == 2

    node_labels, edge_labels, coefficients = readout(
        data=batch,
        node_feats=node_feats,
        return_coefficients=True,
    )

    assert node_labels.shape == default_output[0].shape
    assert edge_labels.shape == default_output[1].shape
    assert set(coefficients["node"]) == {"node:0"}
    assert set(coefficients["edge"]) == {"edge:(0, 0, 0)"}
    assert coefficients["metadata"]["node"]["node:0"]["kind"] == "node"
    assert coefficients["metadata"]["node"]["node:0"]["block_shape"] == (1, 1)
    assert coefficients["metadata"]["edge"]["edge:(0, 0, 0)"]["kind"] == "edge"
    assert coefficients["metadata"]["edge"]["edge:(0, 0, 0)"]["block_shape"] == (
        1,
        1,
    )


def test_dense_edge_readout_inside_irreps_matrix_block():
    torch.manual_seed(0)
    h_basis = PointBasis("H", R=2.0, basis="2x0e + 1x1o")
    o_basis = PointBasis("O", R=2.0, basis="4x0e + 3x1o")
    irreps_in = o3.Irreps("5x0e + 2x1o")
    block = E3nnIrrepsMatrixBlock(
        i_basis=h_basis,
        j_basis=o_basis,
        symmetry="ij",
        operation_cls=E3nnDiagnosticDenseEdgeBlock,
        irreps={"node_feats_irreps": None, "edge_messages_irreps": irreps_in},
    )

    left = torch.randn(2, irreps_in.dim)
    right = torch.randn(2, irreps_in.dim)
    output = block(edge_messages=(left, right))

    assert output.shape == (2, h_basis.basis_size, o_basis.basis_size)

    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(block.operation)


def test_dense_node_readout_parameter_jacobian_full_rank_for_distinct_inputs():
    torch.manual_seed(1)
    dtype = torch.float64
    irreps_in = o3.Irreps("5x0e")
    irreps_out = o3.Irreps("4x0e")
    readout = E3nnDiagnosticDenseNodeBlock(
        irreps_in,
        irreps_out,
        hidden_dim=8,
        activation=torch.nn.Tanh,
    ).to(dtype=dtype)

    node_feats = torch.randn(3, irreps_in.dim, dtype=dtype)
    output = readout(node_feats=node_feats)
    jacobian = _parameter_jacobian(readout, output)

    assert jacobian.shape[0] == output.numel()
    assert torch.linalg.matrix_rank(jacobian, tol=1e-10).item() == output.numel()


def test_graph2mat_return_coefficients_and_coefficient_loss_backward():
    torch.manual_seed(0)
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    irreps_in = o3.Irreps("3x0e")
    readout = E3nnGraph2Mat(
        unique_basis=basis,
        irreps={
            "node_feats_irreps": irreps_in,
        },
        symmetric=True,
        preprocessing_edges=None,
        node_operation=E3nnDiagnosticDenseNodeBlock,
        edge_operation=E3nnDiagnosticDenseEdgeBlock,
    )
    batch = _Batch(
        point_types=torch.tensor([0, 0], dtype=torch.long),
        edge_types=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    node_feats = torch.randn(2, irreps_in.dim)

    node_labels, edge_labels = readout(
        data=batch,
        node_feats=node_feats,
    )
    assert len((node_labels, edge_labels)) == 2
    node_labels_with_coeffs, edge_labels_with_coeffs, coefficients = readout(
        data=batch,
        node_feats=node_feats,
        return_coefficients=True,
    )

    assert torch.allclose(node_labels, node_labels_with_coeffs)
    assert torch.allclose(edge_labels, edge_labels_with_coeffs)
    assert set(coefficients["node"]) == {"node:0"}
    assert set(coefficients["edge"]) == {"edge:(0, 0, 0)"}
    assert coefficients["metadata"]["node"]["node:0"]["operation_key"] == "node:0"

    out = {
        "node_labels": node_labels_with_coeffs,
        "edge_labels": edge_labels_with_coeffs,
        "node_coefficients": coefficients["node"],
        "edge_coefficients": coefficients["edge"],
        "coefficient_metadata": coefficients["metadata"],
    }
    model = type("Model", (), {"matrix_readouts": readout})()
    target_coefficients = target_coefficients_from_labels(
        readout=readout,
        batch=batch,
        basis_table=readout.basis_table,
        nodes_ref=node_labels_with_coeffs.detach(),
        edges_ref=edge_labels_with_coeffs.detach(),
    )
    reconstructed = coefficients_to_labels(
        readout=readout,
        batch=batch,
        basis_table=readout.basis_table,
        coefficients=target_coefficients,
    )

    assert torch.allclose(
        reconstructed["node_labels"], node_labels_with_coeffs.detach(), atol=1e-6
    )
    assert torch.allclose(
        reconstructed["edge_labels"], edge_labels_with_coeffs.detach(), atol=1e-6
    )

    loss, stats = coefficient_space_mse(
        nodes_pred=node_labels_with_coeffs,
        nodes_ref=node_labels_with_coeffs.detach(),
        edges_pred=edge_labels_with_coeffs,
        edges_ref=edge_labels_with_coeffs.detach(),
        batch=batch,
        basis_table=readout.basis_table,
        out=out,
        model=model,
    )

    assert loss.item() < 1e-12
    assert stats["coefficient_blocks"].item() == 2

    composite_loss, composite_stats = hamiltonian_composite_loss(
        beta=0.01,
        coefficient_mse_weight=0.1,
    )(
        nodes_pred=node_labels_with_coeffs,
        nodes_ref=node_labels_with_coeffs.detach(),
        edges_pred=edge_labels_with_coeffs,
        edges_ref=edge_labels_with_coeffs.detach(),
        batch=batch,
        basis_table=readout.basis_table,
        out=out,
        model=model,
    )

    assert composite_loss.item() < 1e-12
    assert composite_stats["coefficient_mse"].item() < 1e-12

    perturbed_out = {
        **out,
        "node_coefficients": {
            key: value + 0.1 for key, value in out["node_coefficients"].items()
        },
    }
    loss, _stats = coefficient_space_mse(
        nodes_pred=node_labels_with_coeffs,
        nodes_ref=node_labels_with_coeffs.detach(),
        edges_pred=edge_labels_with_coeffs,
        edges_ref=edge_labels_with_coeffs.detach(),
        batch=batch,
        basis_table=readout.basis_table,
        out=perturbed_out,
        model=model,
    )
    assert loss.item() > 0
    loss.backward()
    _assert_has_nonzero_parameter_gradient(readout.self_interactions[0].operation)


def test_coefficient_space_loss_fails_when_coefficients_are_missing():
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    irreps_in = o3.Irreps("3x0e")
    readout = E3nnGraph2Mat(
        unique_basis=basis,
        irreps={
            "node_feats_irreps": irreps_in,
        },
        symmetric=True,
        preprocessing_edges=None,
    )
    batch = _Batch(
        point_types=torch.tensor([0, 0], dtype=torch.long),
        edge_types=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    labels = torch.zeros(2)
    model = type("Model", (), {"matrix_readouts": readout})()

    with pytest.raises(ValueError, match="return_coefficients=True"):
        coefficient_space_mse(
            nodes_pred=labels,
            nodes_ref=labels,
            edges_pred=labels[:1],
            edges_ref=labels[:1],
            batch=batch,
            basis_table=readout.basis_table,
            out={"node_labels": labels, "edge_labels": labels[:1]},
            model=model,
        )


def test_graph2mat_return_coefficients_fails_for_unsupported_matrix_block():
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    readout = Graph2Mat(
        unique_basis=basis,
        symmetric=True,
        node_operation=_NoCoefficientNodeBlock,
        edge_operation=_NoCoefficientEdgeBlock,
    )
    batch = _Batch(
        point_types=np.array([0]),
        edge_types=np.array([], dtype=np.int64),
        edge_index=np.empty((2, 0), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="forward_coefficients"):
        readout(
            data=batch,
            node_feats=np.zeros((1, 1)),
            return_coefficients=True,
        )
