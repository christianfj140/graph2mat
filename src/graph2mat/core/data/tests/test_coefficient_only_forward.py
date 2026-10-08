"""The coefficient route must skip dense blocks/resorting, preserving matrix MSE."""

import pytest
import torch

from graph2mat.bindings.torch.data.coefficient_targets import (
    CoefficientMatrixMSEProjector, coefficient_matrix_mse,
)
from graph2mat.bindings.torch.modules.graph2mat import torch_type_layout
from graph2mat.core.data import metrics
from graph2mat.core.data.tests.test_coefficient_orientation import Batch, FEATURES, grouped_readout
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR


def forbidden(*args, **kwargs):
    raise AssertionError("dense reconstruction/resorting entered coefficient-only forward")


@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "float64", "cast32to64"])
@pytest.mark.parametrize("with_layout", [True, False])
def test_graph_coefficient_only_matches_dense_mse_and_gradients(monkeypatch, grouping, symmetric, components, dtype, with_layout):
    torch.manual_seed(17)
    readout = grouped_readout(["2x0e+1x1o+1x2e", "1x0e+1x1o+1x3o"], grouping, symmetric, components)
    if dtype != "float64":
        readout.float()
    if dtype == "cast32to64":
        readout.double()
    batch = Batch(point_types=torch.tensor([0, 1, 0]), edge_types=torch.tensor([0, 0, 1, -1, 1, -1]),
                  edge_index=torch.tensor([[0, 2, 0, 1, 2, 1], [2, 0, 1, 0, 1, 2]]))
    if with_layout:
        batch[LAYOUT_ATTR] = torch_type_layout(batch.point_types.numpy(), batch.edge_types.numpy(), readout.basis_table)
    features = torch.randn(3, FEATURES.dim, dtype=readout.self_interactions[0].change_of_basis.dtype, requires_grad=True)
    nodes, edges = readout(batch, node_feats=features)
    nodes_ref, edges_ref = torch.randn_like(nodes), torch.randn_like(edges)
    batch["edge_labels"] = edges_ref
    targets = CoefficientMatrixMSEProjector(readout)(batch, nodes_ref, edges_ref)
    expected = ((nodes - nodes_ref).square().sum() + 2 * (edges - edges_ref).square().sum()) / (nodes.numel() + 2 * edges.numel())
    parameters = [features, *readout.parameters()]
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    state_keys = set(readout.state_dict())
    for operation in [*readout.self_interactions, *readout.interactions.values()]:
        monkeypatch.setattr(operation, "coefficients_to_block", forbidden)
    monkeypatch.setattr(readout, "_get_nodelabels_resort_index", forbidden)
    monkeypatch.setattr(readout, "_get_edgelabels_resort_index", forbidden)
    coefficients = readout(batch, node_feats=features, coefficients_only=True)
    assert set(coefficients) == {"node", "edge", "metadata"}
    assert set(readout.state_dict()) == state_keys
    actual, _ = coefficient_matrix_mse(coefficients, targets, edge_weight=2.)
    tolerance = 5e-6 if dtype == "float32" else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    gradients = torch.autograd.grad(actual, parameters, allow_unused=True)
    for gradient, reference in zip(gradients, expected_gradients):
        assert (gradient is None) == (reference is None)
        if gradient is not None:
            torch.testing.assert_close(gradient, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("points", [[], [1]])
def test_coefficient_only_ghost_and_empty_graphs(monkeypatch, grouping, points):
    readout = grouped_readout(["1x0e", ""], grouping, True, 2)
    batch = Batch(point_types=torch.tensor(points, dtype=torch.long), edge_types=torch.empty(0, dtype=torch.long),
                  edge_index=torch.empty(2, 0, dtype=torch.long))
    for operation in [*readout.self_interactions, *readout.interactions.values()]:
        if operation is not None:
            monkeypatch.setattr(operation, "coefficients_to_block", forbidden)
    coefficients = readout(batch, node_feats=torch.randn(len(points), FEATURES.dim, dtype=torch.float64), coefficients_only=True)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, coefficients) if not coefficients["node"] else None
    assert coefficients["edge"] == {}
    if grouping != "max" or not points:
        assert coefficients["node"] == {} and labels["node_labels"].shape == (0, 2)
    else:
        assert coefficients["node"]["node:0"].shape == (1, 2)


@pytest.mark.parametrize("only", [False, True])
def test_control_flag_does_not_steal_an_operation_data_keyword(only):
    from graph2mat.core.modules.graph2mat import Graph2Mat

    class Operation(torch.nn.Module):
        def forward(self, coefficients_only):
            return 2 * coefficients_only

        def forward_coefficients(self, coefficients_only):
            return 2 * coefficients_only.reshape(1, 1)

    block, coefficients = Graph2Mat._evaluate_block(
        Operation(), True, only, **{"coefficients_only": torch.tensor([[[3.]]])},
    )
    torch.testing.assert_close(coefficients, torch.tensor([[6.]]))
    if only:
        assert block is None
    else:
        torch.testing.assert_close(block, torch.tensor([[[6.]]]))


def test_coefficient_only_rejects_an_operation_without_coefficient_api(monkeypatch):
    readout = grouped_readout(["1x0e"], "point_type", True, 1)
    batch = Batch(point_types=torch.tensor([0]), edge_types=torch.empty(0, dtype=torch.long),
                  edge_index=torch.empty(2, 0, dtype=torch.long))
    monkeypatch.setattr(readout.self_interactions[0], "forward_coefficients", None)
    with pytest.raises(ValueError, match="forward_coefficients"):
        readout(batch, node_feats=torch.zeros(1, FEATURES.dim, dtype=torch.float64), coefficients_only=True)
