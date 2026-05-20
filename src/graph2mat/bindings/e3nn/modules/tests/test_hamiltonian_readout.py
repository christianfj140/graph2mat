import pytest
import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn import (
    E3nnGraph2Mat,
    E3nnHamiltonianEdgeBlock,
    E3nnHamiltonianNodeBlock,
    E3nnSimpleEdgeBlock,
    E3nnSimpleNodeBlock,
)
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock


class _Batch(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


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
        grads = torch.autograd.grad(
            value,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = []
        for param, grad in zip(params, grads):
            if grad is None:
                pieces.append(torch.zeros_like(param).reshape(-1))
            else:
                pieces.append(grad.reshape(-1))
        rows.append(torch.cat(pieces))
    return torch.stack(rows)


def test_hamiltonian_node_readout_shape_and_backward():
    torch.manual_seed(0)
    irreps_in = o3.Irreps("5x0e + 2x1o")
    irreps_out = o3.Irreps("4x0e + 1x1o")
    readout = E3nnHamiltonianNodeBlock(irreps_in, irreps_out)

    node_feats = torch.randn(4, irreps_in.dim)
    output = readout(node_feats=node_feats)

    assert output.shape == (4, irreps_out.dim)
    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(readout)


def test_hamiltonian_edge_readout_shape_and_backward():
    torch.manual_seed(0)
    irreps_in = o3.Irreps("4x0e + 2x1o")
    irreps_out = o3.Irreps("6x0e + 1x1o")
    readout = E3nnHamiltonianEdgeBlock(irreps_in, irreps_out)

    left = torch.randn(5, irreps_in.dim)
    right = torch.randn(5, irreps_in.dim)
    output = readout(edge_messages=(left, right))

    assert output.shape == (5, irreps_out.dim)
    output.square().sum().backward()
    _assert_has_nonzero_parameter_gradient(readout)


def test_hamiltonian_readout_rejects_unmarked_bottleneck():
    with pytest.raises(ValueError, match="allow_bottleneck=True"):
        E3nnHamiltonianNodeBlock(
            irreps_in=o3.Irreps("4x0e"),
            irreps_out=o3.Irreps("8x0e"),
            hidden_dim=4,
        )


def test_hamiltonian_readout_can_require_scalar_context():
    with pytest.raises(ValueError, match="requires scalar 0e context"):
        E3nnHamiltonianEdgeBlock(
            irreps_in=o3.Irreps("1x1o"),
            irreps_out=o3.Irreps("1x0e"),
            require_scalar_context=True,
        )


def test_hamiltonian_node_readout_inside_irreps_matrix_block():
    torch.manual_seed(0)
    basis = PointBasis("H", R=2.0, basis="2x0e + 1x1o")
    irreps_in = o3.Irreps("6x0e + 1x1o")
    block = E3nnIrrepsMatrixBlock(
        i_basis=basis,
        j_basis=basis,
        symmetry="ij=ji",
        operation_cls=E3nnHamiltonianNodeBlock,
        irreps={"node_feats_irreps": irreps_in},
    )

    node_feats = torch.randn(3, irreps_in.dim)
    coefficients = block.forward_coefficients(node_feats=node_feats)
    output = block(node_feats=node_feats)

    assert coefficients.shape == (3, block._irreps_out.dim)
    assert output.shape == (3, basis.basis_size, basis.basis_size)
    assert torch.allclose(block.coefficients_to_block(coefficients), output)


def test_hamiltonian_edge_readout_inside_irreps_matrix_block():
    torch.manual_seed(0)
    h_basis = PointBasis("H", R=2.0, basis="2x0e + 1x1o")
    o_basis = PointBasis("O", R=2.0, basis="3x0e + 2x1o")
    irreps_in = o3.Irreps("5x0e + 2x1o")
    block = E3nnIrrepsMatrixBlock(
        i_basis=h_basis,
        j_basis=o_basis,
        symmetry="ij",
        operation_cls=E3nnHamiltonianEdgeBlock,
        irreps={"node_feats_irreps": None, "edge_messages_irreps": irreps_in},
    )

    left = torch.randn(2, irreps_in.dim)
    right = torch.randn(2, irreps_in.dim)
    coefficients = block.forward_coefficients(edge_messages=(left, right))
    output = block(edge_messages=(left, right))

    assert coefficients.shape == (2, block._irreps_out.dim)
    assert output.shape == (2, h_basis.basis_size, o_basis.basis_size)
    assert torch.allclose(block.coefficients_to_block(coefficients), output)


def test_e3nn_graph2mat_selects_hamiltonian_readout_classes():
    torch.manual_seed(0)
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    irreps_in = o3.Irreps("4x0e")
    readout = E3nnGraph2Mat(
        unique_basis=basis,
        irreps={"node_feats_irreps": irreps_in},
        symmetric=True,
        preprocessing_edges=None,
        node_operation=E3nnHamiltonianNodeBlock,
        edge_operation=E3nnHamiltonianEdgeBlock,
    )
    batch = _Batch(
        point_types=torch.tensor([0, 0], dtype=torch.long),
        edge_types=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )

    node_labels, edge_labels, coefficients = readout(
        data=batch,
        node_feats=torch.randn(2, irreps_in.dim),
        return_coefficients=True,
    )

    assert isinstance(readout.self_interactions[0].operation, E3nnHamiltonianNodeBlock)
    assert isinstance(
        readout.interactions["(0, 0, 0)"].operation,
        E3nnHamiltonianEdgeBlock,
    )
    assert node_labels.numel() == 2
    assert edge_labels.numel() == 1
    assert set(coefficients["node"]) == {"node:0"}
    assert set(coefficients["edge"]) == {"edge:(0, 0, 0)"}


def test_default_e3nn_graph2mat_readout_classes_unchanged():
    basis = [PointBasis("H", R=2.0, basis="1x0e")]
    readout = E3nnGraph2Mat(
        unique_basis=basis,
        irreps={"node_feats_irreps": o3.Irreps("4x0e")},
        symmetric=True,
        preprocessing_edges=None,
    )

    assert isinstance(readout.self_interactions[0].operation, E3nnSimpleNodeBlock)
    assert isinstance(readout.interactions["(0, 0, 0)"].operation, E3nnSimpleEdgeBlock)


def test_hamiltonian_edge_readout_improves_synthetic_parameter_rank():
    torch.manual_seed(4)
    dtype = torch.float64
    irreps_in = o3.Irreps("4x0e")
    irreps_out = o3.Irreps("5x0e")
    default_readout = E3nnSimpleEdgeBlock(irreps_in, irreps_out).to(dtype=dtype)
    hamiltonian_readout = E3nnHamiltonianEdgeBlock(
        irreps_in,
        irreps_out,
        hidden_dim=16,
    ).to(dtype=dtype)
    left = torch.randn(3, irreps_in.dim, dtype=dtype)
    right = torch.randn(3, irreps_in.dim, dtype=dtype)

    default_output = default_readout(edge_messages=(left, right))
    hamiltonian_output = hamiltonian_readout(edge_messages=(left, right))
    default_rank = torch.linalg.matrix_rank(
        _parameter_jacobian(default_readout, default_output),
        tol=1e-10,
    )
    hamiltonian_rank = torch.linalg.matrix_rank(
        _parameter_jacobian(hamiltonian_readout, hamiltonian_output),
        tol=1e-10,
    )

    assert hamiltonian_output.shape == default_output.shape
    assert hamiltonian_rank > default_rank
