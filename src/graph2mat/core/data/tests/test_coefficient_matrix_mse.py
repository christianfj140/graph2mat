"""Coefficient training must preserve the physical matrix MSE and gradients."""

import importlib
import pickle

import pytest
import torch

from graph2mat.core.data import metrics
from graph2mat.core.data.tests.test_coefficient_orientation import (
    Batch, FEATURES, grouped_readout, heteronuclear_batch,
)


def target_api():
    name = "graph2mat.bindings.torch.data.coefficient_targets"
    assert importlib.util.find_spec(name) is not None, "CPU coefficient projector not implemented"
    return importlib.import_module(name)


def predictions_for(readout, batch):
    operation = next(op for op in readout.self_interactions if op is not None)
    _, _, outputs = readout(batch, node_feats=torch.randn(
        len(batch.point_types), FEATURES.dim, dtype=operation.change_of_basis.dtype,
    ), return_coefficients=True)
    return {kind: {key: torch.randn_like(value).requires_grad_() for key, value in outputs[kind].items()}
            for kind in ("node", "edge")}


def dense_reference(readout, batch, predictions, nodes_ref, edges_ref, edge_weight):
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, predictions)
    node_error, edge_error = metrics.get_predictions_error(
        labels["node_labels"], nodes_ref, labels["edge_labels"], edges_ref,
    )
    count = node_error.numel() + edge_weight * edge_error.numel()
    return ((node_error.square().sum() + edge_weight * edge_error.square().sum()) / (count or 1),
            node_error.numel(), edge_error.numel())


def forbid_reconstruction(*args, **kwargs):
    raise AssertionError("matrix reconstruction entered the coefficient objective")


@pytest.mark.parametrize("bases", [
    ["1x0e", "1x1o"], ["2x0e+1x1o+1x2e", "1x0e+2x1o+1x3o"],
])
@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("dtype", ["float64", "float32", "cast32to64"])
def test_cpu_targets_preserve_matrix_mse_and_gradients(monkeypatch, bases, grouping, symmetric, components, dtype):
    torch.manual_seed(64)
    readout = grouped_readout(bases, grouping, symmetric, components)
    if dtype != "float64":
        readout.float()
    if dtype == "cast32to64":
        readout.double()
    batch = heteronuclear_batch()
    predictions = predictions_for(readout, batch)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, predictions)
    nodes_ref = torch.randn_like(labels["node_labels"])
    edges_ref = torch.randn_like(labels["edge_labels"])
    batch["edge_labels"] = edges_ref
    values = [value for kind in ("node", "edge") for value in predictions[kind].values()]
    references = []
    for weight in (0., 1., 2.):
        expected, node_count, edge_count = dense_reference(readout, batch, predictions, nodes_ref, edges_ref, weight)
        gradients = torch.autograd.grad(expected, values, retain_graph=True)
        references.append((weight, expected, gradients, node_count, edge_count))
    for operation in [*readout.self_interactions, *readout.interactions.values()]:
        if operation is not None:
            monkeypatch.setattr(operation, "coefficients_to_block", forbid_reconstruction)
    api = target_api()
    targets = api.CoefficientMatrixMSEProjector(readout)(batch, nodes_ref, edges_ref)
    tolerance = 5e-6 if dtype == "float32" else 1e-12
    for weight, expected, expected_gradients, node_count, edge_count in references:
        actual, stats = api.coefficient_matrix_mse(predictions, targets, edge_weight=weight)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        assert stats["node_count"] == node_count
        assert stats["edge_count"] == edge_count
        gradients = torch.autograd.grad(actual, values, retain_graph=True)
        for gradient, expected_gradient in zip(gradients, expected_gradients):
            torch.testing.assert_close(gradient, expected_gradient, atol=tolerance, rtol=tolerance)


def test_target_residual_cannot_be_dropped():
    readout = grouped_readout(["1x1o"], "point_type", True, 1)
    batch = Batch(point_types=torch.tensor([0]), edge_types=torch.empty(0, dtype=torch.long),
                  edge_index=torch.empty(2, 0, dtype=torch.long))
    nodes = torch.tensor([[0., 2., 0.], [-2., 0., 0.], [0., 0., 0.]], dtype=torch.float64).ravel()
    predictions = {"node": {"node:0": torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)}, "edge": {}}
    api = target_api()
    targets = api.CoefficientMatrixMSEProjector(readout)(batch, nodes, torch.empty(0, dtype=torch.float64))
    loss, stats = api.coefficient_matrix_mse(predictions, targets)
    torch.testing.assert_close(loss, torch.tensor(8 / 9, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert stats["node_count"] == 9
    gradient, = torch.autograd.grad(loss, predictions["node"]["node:0"])
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("components", [1, 2])
def test_outer_autocast_does_not_lower_matrix_loss_precision(components):
    torch.manual_seed(7)
    readout = grouped_readout(["1x0e+1x1o+1x2e", "1x1o"], "max", True, components).float()
    batch = heteronuclear_batch()
    predictions = predictions_for(readout, batch)
    labels = metrics.coefficients_to_labels(readout, batch, readout.basis_table, predictions)
    nodes, edges = torch.randn_like(labels["node_labels"]), torch.randn_like(labels["edge_labels"])
    batch["edge_labels"] = edges
    expected, _, _ = dense_reference(readout, batch, predictions, nodes, edges, 2.)
    values = [value for kind in ("node", "edge") for value in predictions[kind].values()]
    expected_gradients = torch.autograd.grad(expected, values)
    api = target_api()
    targets = api.CoefficientMatrixMSEProjector(readout)(batch, nodes, edges)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = api.coefficient_matrix_mse(predictions, targets, edge_weight=2.)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
    for actual_gradient, expected_gradient in zip(torch.autograd.grad(actual, values), expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=5e-6, rtol=5e-6)


def test_whole_missing_blocks_exclude_nan_predictions():
    readout = grouped_readout(["1x0e", "1x1o"], "point_type", True, 2)
    batch = heteronuclear_batch()
    predictions = predictions_for(readout, batch)
    with torch.no_grad():
        predictions["node"]["node:0"][0] = float("nan")
        predictions["edge"]["edge:(0, 1, 1)"][0] = float("nan")
    nodes = torch.zeros(11, 2, dtype=torch.float64)
    edges = torch.zeros(6, 2, dtype=torch.float64)
    nodes[0, 1] = float("nan")
    edges[:3, 0] = float("nan")
    expected, _, _ = dense_reference(readout, batch, predictions, nodes, edges, 2.)
    api = target_api()
    targets = api.CoefficientMatrixMSEProjector(readout)(batch, nodes, edges)
    actual, stats = api.coefficient_matrix_mse(predictions, targets, edge_weight=2.)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert stats["node_count"] == 20
    assert stats["edge_count"] == 6
    actual.backward()
    for kind in ("node", "edge"):
        for value in predictions[kind].values():
            assert torch.isfinite(value.grad).all()
    assert (predictions["node"]["node:0"].grad[0] == 0).all()
    assert (predictions["edge"]["edge:(0, 1, 1)"].grad[0] == 0).all()


def test_partial_nan_requests_explicit_matrix_fallback():
    readout = grouped_readout(["1x1o"], "point_type", True, 1)
    batch = Batch(point_types=torch.tensor([0]), edge_types=torch.empty(0, dtype=torch.long))
    nodes = torch.zeros(9, dtype=torch.float64)
    nodes[1] = float("nan")
    api = target_api()
    with pytest.raises(api.CoefficientMSEUnsupported, match="partial.*NaN"):
        api.CoefficientMatrixMSEProjector(readout)(batch, nodes, torch.empty(0, dtype=torch.float64))


@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
def test_ghost_nodes_and_no_edges_have_connected_zero_loss(grouping):
    readout = grouped_readout(["1x0e", ""], grouping, True, 2)
    batch = Batch(point_types=torch.tensor([1]), edge_types=torch.empty(0, dtype=torch.long),
                  edge_index=torch.empty(2, 0, dtype=torch.long))
    predictions = predictions_for(readout, batch)
    # Keep the empty loss usable even when no operation has outputs at all.
    if not predictions["node"]:
        predictions["node"]["node:0"] = torch.empty(0, 2, dtype=torch.float64, requires_grad=True)
    api = target_api()
    targets = api.CoefficientMatrixMSEProjector(readout)(
        batch, torch.empty(0, 2, dtype=torch.float64), torch.empty(0, 2, dtype=torch.float64),
    )
    loss, stats = api.coefficient_matrix_mse(predictions, targets)
    assert loss == 0 and stats["node_count"] == stats["edge_count"] == 0
    loss.backward()
    for value in predictions["node"].values():
        assert value.grad is not None and (value.grad == 0).all()


def test_projector_pickle_owns_only_cpu_snapshots():
    readout = grouped_readout(["1x0e", "1x1o"], "max", True, 2)
    api = target_api()
    projector = api.CoefficientMatrixMSEProjector(torch.nn.ModuleList([readout, readout]))

    def check(value):
        assert not isinstance(value, torch.nn.Module)
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and not value.requires_grad
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
    check(projector.__dict__)
    restored = pickle.loads(pickle.dumps(projector))
    batch = heteronuclear_batch()
    predictions = predictions_for(readout, batch)
    nodes, edges = torch.randn(11, 2, dtype=torch.float64), torch.randn(6, 2, dtype=torch.float64)
    original = projector(batch, nodes, edges)
    # A snapshot must not be changed by later mutations to model Q.
    with torch.no_grad():
        readout.self_interactions[0].change_of_basis.zero_()
    copied = restored(batch, nodes, edges)
    torch.testing.assert_close(api.coefficient_matrix_mse(predictions, original)[0],
                               api.coefficient_matrix_mse(predictions, copied)[0])


def test_interaction_readouts_require_identical_effective_q():
    one = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    other = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    with torch.no_grad():
        other.self_interactions[1].change_of_basis.mul_(2)
    api = target_api()
    with pytest.raises(ValueError, match="coordinates|change.of.basis"):
        api.CoefficientMatrixMSEProjector(torch.nn.ModuleList([one, other]))


def test_modulelist_coordinates_do_not_depend_on_registration_order():
    one = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    other = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    other.interactions = torch.nn.ModuleDict(dict(reversed(list(other.interactions.items()))))
    api = target_api()
    projector = api.CoefficientMatrixMSEProjector(torch.nn.ModuleList([one, other]))
    batch = heteronuclear_batch()
    predictions = predictions_for(one, batch)
    nodes, edges = torch.zeros(11, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
    expected, _, _ = dense_reference(one, batch, predictions, nodes, edges, 1.)
    loss, _ = api.coefficient_matrix_mse(predictions, projector(batch, nodes, edges))
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)


def test_projector_does_not_pickle_irrelevant_table_initializers():
    readout = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    readout.basis_table._init_args["get_point_matrix"] = lambda basis: None
    projector = target_api().CoefficientMatrixMSEProjector(readout)
    pickle.loads(pickle.dumps(projector))


@pytest.mark.parametrize("failure", ["uncertified", "partial_mask"])
def test_uncertified_coordinates_request_matrix_fallback(failure):
    readout = grouped_readout(["1x1o", "1x0e"], "max", True, 1)
    if failure == "uncertified":
        readout.self_interactions[0]._coefficient_groups = None
    else:
        readout.basis_filters[0, 1] = False
    api = target_api()
    with pytest.raises(api.CoefficientMSEUnsupported):
        api.CoefficientMatrixMSEProjector(readout)


@pytest.mark.parametrize("failure", ["components", "length", "infinite", "predicted_shape", "edge_weight"])
def test_matrix_mse_boundary_validation(failure):
    readout = grouped_readout(["1x0e", "1x1o"], "max", True, 2)
    batch = heteronuclear_batch()
    predictions = predictions_for(readout, batch)
    nodes, edges = torch.zeros(11, 2, dtype=torch.float64), torch.zeros(6, 2, dtype=torch.float64)
    api = target_api()
    projector = api.CoefficientMatrixMSEProjector(readout)
    if failure == "components":
        with pytest.raises(ValueError, match="components|shape"):
            projector(batch, nodes[:, :1], edges)
    elif failure == "length":
        with pytest.raises(ValueError, match="length|rows"):
            projector(batch, nodes[:-1], edges)
    elif failure == "infinite":
        nodes[0, 0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            projector(batch, nodes, edges)
    else:
        targets = projector(batch, nodes, edges)
        if failure == "predicted_shape":
            predictions["node"]["node:0"] = predictions["node"]["node:0"][:-1]
            with pytest.raises(ValueError, match="shape"):
                api.coefficient_matrix_mse(predictions, targets)
        else:
            with pytest.raises(ValueError, match="edge_weight"):
                api.coefficient_matrix_mse(predictions, targets, edge_weight=-1.)
