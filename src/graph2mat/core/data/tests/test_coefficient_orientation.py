"""Physical blocks retain their direction, component order and observation mask."""

import pytest
import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn import E3nnDiagnosticDenseEdgeBlock, E3nnDiagnosticDenseNodeBlock
from graph2mat.bindings.e3nn.modules.graph2mat import E3nnGraph2Mat
from graph2mat.core.data import metrics


class Batch(dict):
    def __getattr__(self, key):
        return self[key]


FEATURES = o3.Irreps("2x0e+1x1o+1x2e+1x3o")
BASES = [("1x0e", "1x1o"), ("1x0e+1x1o", "1x0e"), ("1x2e", "1x3o")]


def grouped_readout(bases, grouping, symmetric, n_components):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return E3nnGraph2Mat(
            unique_basis=[PointBasis(str(i), R=2, basis=basis or []) for i, basis in enumerate(bases)],
            irreps={"node_feats_irreps": FEATURES}, basis_grouping=grouping,
            symmetric=symmetric, n_matrix_components=n_components, preprocessing_edges=None,
            node_operation=E3nnDiagnosticDenseNodeBlock, edge_operation=E3nnDiagnosticDenseEdgeBlock,
        )
    finally:
        torch.set_default_dtype(previous)


def heteronuclear_batch():
    # Two local pairs, the second has a different atom of species A.
    return Batch(
        point_types=torch.tensor([0, 1, 0]), edge_types=torch.tensor([1, -1, 1, -1]),
        edge_index=torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]]),
    )


@pytest.mark.parametrize("bases", BASES)
@pytest.mark.parametrize("grouping", ["point_type", "basis_shape"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("n_components", [1, 2])
def test_signed_grouped_targets_match_literal_physical_blocks(bases, grouping, symmetric, n_components):
    readout = grouped_readout(bases, grouping, symmetric, n_components)
    batch = heteronuclear_batch()
    dims = [o3.Irreps(basis).dim for basis in bases]
    channels = () if n_components == 1 else (n_components,)
    nodes = torch.cat([
        (torch.eye(dims[k], dtype=torch.float64) * (i + 1)).unsqueeze(-1).expand(-1, -1, n_components)
        .reshape(-1, n_components)
        for i, k in enumerate([0, 1, 0])
    ]).reshape((-1,) + channels)
    signs = [1, 1] if symmetric else [1, -1, 1, -1]
    blocks = []
    for i, sign in enumerate(signs):
        shape = tuple(dims if sign > 0 else dims[::-1]) + channels
        blocks.append((torch.arange(1, dims[0] * dims[1] * n_components + 1, dtype=torch.float64)
                       + i * 100).reshape(shape) / 17)
    edges = torch.cat([block.reshape((-1,) + channels) for block in blocks]).requires_grad_()
    target = metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)

    # The fixture's basis_shape lexicographic order reverses all three pairs.
    expected = {}
    for sign, block in zip(signs, blocks):
        grouped_sign = sign if grouping == "point_type" else -sign
        operation_sign = abs(grouped_sign) if symmetric else grouped_sign
        module_key = "(0, 1, 1)" if operation_sign > 0 else "(1, 0, -1)"
        q = readout.interactions[module_key].change_of_basis
        physical = block.transpose(0, 1) if symmetric and grouped_sign < 0 else block
        coeff = (torch.einsum("zij,ijc->cz", q, physical).reshape(-1) if n_components > 1
                 else torch.einsum("zij,ij->z", q, physical))
        expected.setdefault(f"edge:{module_key}", []).append(coeff)
    assert set(target["edge"]) == set(expected)
    for key, values in expected.items():
        torch.testing.assert_close(target["edge"][key], torch.stack(values), atol=1e-12, rtol=1e-12)

    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, target)
    torch.testing.assert_close(labels["node_labels"], nodes, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(labels["edge_labels"], edges, atol=1e-12, rtol=1e-12)
    energy = sum(value.square().sum() for value in target["edge"].values())
    gradient, = torch.autograd.grad(energy, edges)
    torch.testing.assert_close(gradient, 2 * edges, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("n_components", [1, 2])
def test_coefficients_reconstruct_real_grouped_forward_and_gradients(grouping, symmetric, n_components):
    torch.manual_seed(41)
    readout = grouped_readout(BASES[0], grouping, symmetric, n_components)
    batch = heteronuclear_batch()
    features = torch.randn(3, FEATURES.dim, dtype=torch.float64, requires_grad=True)
    nodes, edges, coefficients = readout(batch, node_feats=features, return_coefficients=True)
    if grouping == "max" and n_components == 2:
        batch["edge_labels"] = edges.detach()
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, coefficients)
    torch.testing.assert_close(labels["node_labels"], nodes, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(labels["edge_labels"], edges, atol=1e-12, rtol=1e-12)
    original_grad, = torch.autograd.grad(nodes.square().sum() + edges.square().sum(), features, retain_graph=True)
    reconstructed_grad, = torch.autograd.grad(sum(value.square().sum() for value in labels.values()), features)
    torch.testing.assert_close(reconstructed_grad, original_grad, atol=1e-12, rtol=1e-12)
    if grouping != "max":
        target = metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)
        for kind in ("node", "edge"):
            for key, value in coefficients[kind].items():
                torch.testing.assert_close(target[kind][key], value, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("grouping", ["point_type", "basis_shape"])
@pytest.mark.parametrize("symmetric", [True, False])
def test_ghost_points_and_edges_require_no_coefficients(grouping, symmetric):
    readout = grouped_readout(["1x0e", "", "1x1o"], grouping, symmetric, 2)
    batch = Batch(
        point_types=torch.tensor([0, 1, 2]), edge_types=torch.tensor([1, -1, 2, -2, 4, -4]),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
    )
    nodes = torch.cat([torch.ones(1, 2), torch.eye(3).reshape(-1, 1).expand(-1, 2)]).double()
    edges = torch.tensor([[1., 2.], [3., 4.], [5., 6.]] * (1 if symmetric else 2), dtype=torch.float64)
    target = metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, target)
    torch.testing.assert_close(labels["node_labels"], nodes, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(labels["edge_labels"], edges, atol=1e-12, rtol=1e-12)
    assert len(target["node"]) == 2
    assert len(target["edge"]) == (1 if symmetric else 2)


def test_padded_targets_do_not_silently_impose_zero_labels():
    readout = grouped_readout(BASES[0], "max", True, 2)
    batch = heteronuclear_batch()
    with pytest.raises(ValueError, match="padding.*masked|masked.*padding"):
        metrics.target_coefficients_from_labels(
            readout, batch, readout.basis_table, torch.zeros(11, 2), torch.zeros(6, 2),
        )


def test_max_grouping_without_padding_keeps_exact_targets():
    readout = grouped_readout(["1x0e", "1x0e"], "max", True, 2)
    batch = heteronuclear_batch()
    nodes = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    edges = torch.tensor([[7., 8.], [9., 10.]], dtype=torch.float64)
    target = metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, target)
    torch.testing.assert_close(labels["node_labels"], nodes)
    torch.testing.assert_close(labels["edge_labels"], edges)
