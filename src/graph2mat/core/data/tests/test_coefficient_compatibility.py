"""Coefficient targets keep component axes and shared readout coordinates."""

from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn import (
    E3nnDiagnosticDenseEdgeBlock,
    E3nnDiagnosticDenseNodeBlock,
)
from graph2mat.bindings.e3nn.modules.graph2mat import E3nnGraph2Mat
from graph2mat.core.data import metrics


def component_readout(n_components=2, basis="1x0e", two_types=False, features="2x0e+1x1o+1x2e+1x3o"):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return E3nnGraph2Mat(
            unique_basis=[PointBasis(kind, R=2, basis=basis) for kind in (("A", "B") if two_types else ("A",))],
            irreps={"node_feats_irreps": o3.Irreps(features)},
            symmetric=True, n_matrix_components=n_components, preprocessing_edges=None,
            node_operation=E3nnDiagnosticDenseNodeBlock,
            edge_operation=E3nnDiagnosticDenseEdgeBlock,
        )
    finally:
        torch.set_default_dtype(previous)


def test_two_components_keep_literal_species_and_edge_order():
    readout = component_readout(two_types=True)
    batch = SimpleNamespace(
        point_types=torch.tensor([0, 1, 0]), edge_types=torch.tensor([1, -1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    nodes = torch.tensor([[1., 2.], [7., 8.], [3., 4.]], dtype=torch.float64)
    edges = torch.tensor([[5., 6.]], dtype=torch.float64)
    target = metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)
    torch.testing.assert_close(target["node"]["node:0"], torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64))
    torch.testing.assert_close(target["node"]["node:1"], nodes[1:2])
    torch.testing.assert_close(target["edge"]["edge:(0, 1, 1)"], edges)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, target)
    torch.testing.assert_close(labels["node_labels"], nodes)
    torch.testing.assert_close(labels["edge_labels"], edges)


@pytest.mark.parametrize("basis", ["1x0e", "1x1o", "1x2e", "1x3o"])
def test_multiple_components_roundtrip_shells_and_projection_gradients(basis):
    readout = component_readout(basis=basis)
    batch = SimpleNamespace(
        point_types=torch.tensor([0, 0]), edge_types=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    expected = {}
    references = {}
    for kind, key, operation, n_blocks in (
        ("node", "node:0", readout.self_interactions[0], 2),
        ("edge", "edge:(0, 0, 0)", readout.interactions["(0, 0, 0)"], 1),
    ):
        dim = operation.change_of_basis.shape[0]
        value = torch.arange(1, n_blocks * 2 * dim + 1, dtype=torch.float64).reshape(n_blocks, 2 * dim) / 17
        value.requires_grad_()
        block = torch.einsum("ncz,zij->nijc", value.reshape(n_blocks, 2, dim), operation.change_of_basis)
        references[kind] = block.reshape(-1, 2)
        expected[kind] = {key: value}
    target = metrics.target_coefficients_from_labels(
        readout, batch, readout.basis_table, references["node"], references["edge"],
    )
    for kind in expected:
        for key, value in expected[kind].items():
            torch.testing.assert_close(target[kind][key], value, rtol=1e-12, atol=1e-12)
            gradient, = torch.autograd.grad(target[kind][key].square().sum(), value, retain_graph=True)
            torch.testing.assert_close(gradient, 2 * value, rtol=1e-12, atol=1e-12)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, target)
    for kind in expected:
        torch.testing.assert_close(labels[f"{kind}_labels"], references[kind], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("has_tensors", [False, True])
def test_empty_multicomponent_coefficients_keep_component_axis(has_tensors):
    readout = component_readout()
    batch = SimpleNamespace(
        point_types=torch.empty(0, dtype=torch.long), edge_types=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    coefficients = {"node": {}, "edge": {}}
    if has_tensors:
        coefficients["node"]["node:0"] = torch.empty((0, 2), dtype=torch.float64)
        coefficients["edge"]["edge:(0, 0, 0)"] = torch.empty((0, 2), dtype=torch.float64)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, coefficients)
    for label in labels.values():
        assert label.shape == (0, 2)
        if has_tensors:
            assert label.dtype == torch.float64
    target = metrics.target_coefficients_from_labels(
        readout, batch, readout.basis_table, torch.empty(0, 2), torch.empty(0, 2),
    )
    assert target["node"] == target["edge"] == {}


@pytest.mark.parametrize("n_columns", [None, 1, 3])
def test_component_mismatch_is_rejected_without_broadcasting(n_columns):
    readout = component_readout()
    batch = SimpleNamespace(
        point_types=torch.tensor([0]), edge_types=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    nodes = torch.zeros(1) if n_columns is None else torch.zeros(1, n_columns)
    edges = torch.zeros(0) if n_columns is None else torch.zeros(0, n_columns)
    with pytest.raises(ValueError, match="components"):
        metrics.target_coefficients_from_labels(readout, batch, readout.basis_table, nodes, edges)


@pytest.mark.parametrize("n_components", [1, 2])
def test_compatible_interaction_readouts_share_targets_and_loss(n_components):
    readout = component_readout(n_components)
    other = component_readout(n_components, features="3x0e+2x1o+1x2e+1x3o")
    readouts = torch.nn.ModuleList([readout, other])
    batch = SimpleNamespace(
        point_types=torch.tensor([0, 0]), edge_types=torch.tensor([0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    shape = () if n_components == 1 else (n_components,)
    nodes = torch.zeros((2,) + shape, dtype=torch.float64)
    edges = torch.zeros((1,) + shape, dtype=torch.float64)
    target = metrics.target_coefficients_from_labels(readouts, batch, readout.basis_table, nodes, edges)
    offset = torch.tensor(1., requires_grad=True)
    out = {f"{kind}_coefficients": {key: value + offset for key, value in target[kind].items()}
           for kind in ("node", "edge")}
    out.update(node_labels=nodes + offset, edge_labels=edges + offset)
    loss, stats = metrics.coefficient_space_mse(
        nodes_pred=out["node_labels"], edges_pred=out["edge_labels"],
        nodes_ref=nodes, edges_ref=edges, batch=batch, basis_table=readout.basis_table,
        out=out, model=SimpleNamespace(matrix_readouts=readouts),
    )
    assert float(loss) == pytest.approx(1.)
    assert float(stats["coefficient_blocks"]) == 2
    loss.backward()
    assert float(offset.grad) == pytest.approx(2.)
    reconstructed = metrics.coefficients_to_labels(readouts, batch, readout.basis_table, target)
    torch.testing.assert_close(reconstructed["node_labels"], nodes)
    torch.testing.assert_close(reconstructed["edge_labels"], edges)


@pytest.mark.parametrize("empty", [False, True])
def test_incompatible_or_empty_interaction_readouts_fail_explicitly(empty):
    readout = component_readout(1)
    readouts = torch.nn.ModuleList([] if empty else [readout, component_readout(2)])
    batch = SimpleNamespace(
        point_types=torch.tensor([0]), edge_types=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    with pytest.raises(ValueError, match="readout"):
        metrics.target_coefficients_from_labels(readouts, batch, readout.basis_table, torch.zeros(1), torch.zeros(0))
