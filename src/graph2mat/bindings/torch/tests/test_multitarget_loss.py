"""Independent objectives, coefficient support and physical-reference contracts."""

import importlib

import numpy as np
import pytest
import torch
from scipy.sparse import csr_array
from torch_geometric.data import Batch

from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data import multitarget
from graph2mat.core.data.metrics import elementwise_mse, gauge_invariant_mse
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices
from graph2mat.core.data.tests.test_multitarget import TARGETS, processor_for
from graph2mat.tools.lightning.data import TypeLayoutCollate


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def sample(spin=1, *, masks=None, physical_labels=None, matrices=None):
    matrices = analytic_matrices(spin=spin) if matrices is None else matrices
    processor = processor_for(matrices, spin)
    data = multitarget.prepare_multitarget_data(processor, **matrices,
        observation_masks=masks, physical_labels=physical_labels, data_cls=TorchBasisMatrixData)
    batch = TypeLayoutCollate(Batch.from_data_list, processor.basis_table)([data])
    return batch, processor


def objective(processor, h_loss="elementwise_mse", weights=None, kwargs=None):
    cls = importlib.import_module("graph2mat.bindings.torch.multitarget").MultitargetMatrixLoss
    losses = {name: {"name": h_loss if name == "hamiltonian" else "elementwise_mse",
                     "kwargs": dict((kwargs or {}).get(name, {}))} for name in TARGETS}
    return cls(processor.matrix_target_layout,
               weights=weights or dict(zip(TARGETS, [2., 3., 5., 7.])), losses=losses)


def evaluate(loss, data, processor, nodes, edges):
    return loss(nodes_pred=nodes, nodes_ref=data.point_labels,
                edges_pred=edges, edges_ref=data.edge_labels, batch=data,
                basis_table=processor.basis_table)


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("edge_weight", [1., 2.])
@pytest.mark.parametrize("h_loss", ["elementwise_mse", "gauge_invariant_mse"])
def test_weighted_loss_equals_independent_targets(spin, edge_weight, h_loss):
    data, p = sample(spin)
    nodes = (data.point_labels + .17).requires_grad_()
    edges = (data.edge_labels - .11).requires_grad_()
    loss = objective(p, h_loss, kwargs={t: {"edge_weight": edge_weight} for t in TARGETS})
    actual, stats = evaluate(loss, data, p, nodes, edges)
    expected = nodes.new_zeros(())
    for target, weight in zip(TARGETS, [2., 3., 5., 7.]):
        columns = p.matrix_target_layout.slices[target]
        rn, re = data.point_labels[:, columns], data.edge_labels[:, columns]
        metric = elementwise_mse(edge_weight=edge_weight)
        if target == "hamiltonian" and h_loss == "gauge_invariant_mse":
            scol = p.matrix_target_layout.slices["overlap"]
            rn = torch.cat([rn, data.point_labels[:, scol]], dim=1)
            re = torch.cat([re, data.edge_labels[:, scol]], dim=1)
            metric = gauge_invariant_mse(edge_weight=edge_weight, overlap_weight=0.)
        value, _ = metric(nodes_pred=nodes[:, columns], nodes_ref=rn,
            edges_pred=edges[:, columns], edges_ref=re, batch=data, basis_table=p.basis_table)
        expected = expected + weight * value
        torch.testing.assert_close(stats[f"target/{target}/loss"], value.detach())
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    gradients = torch.autograd.grad(actual, (nodes, edges))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_component_mask_preserves_other_spin():
    masks = {"density_matrix": [csr_array(np.eye(2, dtype=bool)), csr_array(np.ones((2, 2), dtype=bool))]}
    data, p = sample(2, masks=masks)
    nodes, edges = data.point_labels.nan_to_num().clone(), data.edge_labels.nan_to_num().clone()
    nodes[:, 3] += 1.
    nodes[:, 4] += 2.
    edges[:, 3] = 1000.  # Unknown: must not discard the known other spin, nor enter arithmetic.
    edges[:, 4] += 3.
    nodes.requires_grad_()
    edges.requires_grad_()
    weights = dict(zip(TARGETS, [0., 0., 1., 0.]))
    value, _ = evaluate(objective(p, weights=weights), data, p, nodes, edges)
    assert float(value) == pytest.approx((2*1. + 2*4. + 9.) / 5, abs=1e-14)
    g = torch.autograd.grad(value, edges)[0]
    assert g[0, 3] == 0. and g[0, 4] != 0.


def test_zero_weight_nan_is_not_evaluated():
    data, p = sample()
    nodes, edges = data.point_labels.clone(), data.edge_labels.clone()
    nodes[:, 3] = edges[:, 3] = float("nan")
    nodes[:, 0] += .1
    nodes.requires_grad_()
    edges.requires_grad_()
    loss = objective(p, weights=dict(zip(TARGETS, [1., 1., 1., 0.])))
    value, _ = evaluate(loss, data, p, nodes, edges)
    assert torch.isfinite(value)
    for g in torch.autograd.grad(value, (nodes, edges)):
        assert torch.isfinite(g).all()
        assert (g[:, 3] == 0).all()


def test_h_gauge_uses_reference_s():
    data, p = sample()
    nodes, edges = data.point_labels.clone(), data.edge_labels.clone()
    nodes[:, 0] += .7 * nodes[:, 1]
    edges[:, 0] += .7 * edges[:, 1]
    nodes[:, 1] *= 3
    edges[:, 1] *= 4
    value, stats = evaluate(objective(p, "gauge_invariant_mse"), data, p, nodes, edges)
    assert abs(float(stats["target/hamiltonian/loss"])) < 1e-28
    assert float(stats["target/overlap/loss"]) > 0.
    assert float(value) > 0.


def test_spin_gauge_keeps_opposite_spin_shifts_in_multitarget_loss():
    data, p = sample(spin=2)
    nodes, edges = data.point_labels.clone(), data.edge_labels.clone()
    h = p.matrix_target_layout.slices["hamiltonian"]
    s = p.matrix_target_layout.slices["overlap"]
    signs = nodes.new_tensor([1., -1.])
    nodes[:, h] += nodes[:, s] * signs
    edges[:, h] += edges[:, s] * signs
    nodes.requires_grad_()
    edges.requires_grad_()
    weights = {target: float(target == "hamiltonian") for target in TARGETS}
    value, stats = evaluate(objective(p, "gauge_invariant_mse", weights=weights), data, p, nodes, edges)
    expected = (data.point_labels[:, s].square().sum() + data.edge_labels[:, s].square().sum()
                ) / (len(nodes) + len(edges))
    torch.testing.assert_close(value, expected)
    assert float(stats["target/hamiltonian/gauge_shift_mean_abs"]) < 1e-14
    gradients = torch.autograd.grad(value, (nodes, edges))
    assert any(gradient[:, h].abs().max() > 0 for gradient in gradients)


@pytest.mark.parametrize("weights", [[0., 0., 0., 0.], [-1., 1., 1., 1.],
                                    [float("nan"), 1., 1., 1.], [1., float("inf"), 1., 1.]])
def test_invalid_weights_rejected(weights):
    _, p = sample()
    with pytest.raises(ValueError, match="weight"):
        objective(p, weights=dict(zip(TARGETS, weights)))


def test_missing_support_and_invalid_predictions_fail_with_target_context():
    data, p = sample()
    data.point_labels[:, 2] = float("nan")
    data.edge_labels[:, 2] = float("nan")
    data.matrix_known_nodes[:, 2] = False
    data.matrix_known_edges[:, 2] = False
    with pytest.raises(ValueError, match="density_matrix.*(observ|structure)|structure.*density_matrix"):
        evaluate(objective(p), data, p, data.point_labels.nan_to_num(), data.edge_labels.nan_to_num())
    data, p = sample()
    prediction = data.point_labels.clone()
    prediction[0, 0] = float("inf")
    with pytest.raises(ValueError, match="hamiltonian.*finite|finite.*hamiltonian"):
        evaluate(objective(p), data, p, prediction, data.edge_labels)


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("target,expected", [("hamiltonian", 1/288),
    ("density_matrix", 1/800), ("energy_density_matrix", 1/200)])
def test_physical_targets_match_single_target(spin, target, expected):
    requested = {target: ["energy"]}
    data, p = sample(spin, physical_labels=requested)
    weights = {name: float(name == target) for name in TARGETS}
    loss = objective(p, weights=weights, kwargs={target: {"energy_weight": 1.}})
    nodes, edges = data.point_labels.clone(), data.edge_labels.clone()
    col = p.matrix_target_layout.slices[target].start
    nodes[0, col] += .1
    nodes.requires_grad_()
    edges.requires_grad_()
    value, stats = evaluate(loss, data, p, nodes, edges)
    # Spin2 fixture has half occupation per spin; the chosen H-channel partner halves.
    expected_physical = expected / 4 if spin == 2 and target == "hamiltonian" else expected
    assert float(stats[f"target/{target}/physical_energy_mse"]) == pytest.approx(expected_physical, abs=1e-14)
    view = multitarget.target_data_view(data, p.matrix_target_layout, target)
    columns = p.matrix_target_layout.slices[target]
    independent, _ = elementwise_mse(energy_weight=1.)(
        nodes_pred=nodes[:, columns], nodes_ref=data.point_labels[:, columns],
        edges_pred=edges[:, columns], edges_ref=data.edge_labels[:, columns],
        batch=view, basis_table=p.basis_table)
    torch.testing.assert_close(value, independent, rtol=1e-12, atol=1e-14)
    assert torch.autograd.grad(value, nodes)[0][0, col] != 0.


def test_fractional_dm_and_periodic_physical_references():
    for matrices, population in [(analytic_matrices(fractional=True), [.6, .6]),
                                 (periodic_matrices(), [.84])]:
        data, p = sample(matrices=matrices, physical_labels={"density_matrix": ["charge", "energy"]})
        view = multitarget.target_data_view(data, p.matrix_target_layout, "density_matrix")
        torch.testing.assert_close(view.physical_population_reference, torch.tensor(population))
        value, stats = evaluate(objective(p, weights=dict(zip(TARGETS, [0., 0., 1., 0.])),
            kwargs={"density_matrix": {"electron_weight": 2., "mulliken_weight": 3., "energy_weight": 4.}}),
            data, p, data.point_labels, data.edge_labels)
        assert abs(float(value)) < 1e-28
        assert abs(float(stats["target/density_matrix/physical_delta_n_mean"])) < 1e-14


def test_gauge_physics_uses_same_alignment_and_reference_s():
    data, p = sample(physical_labels={"hamiltonian": ["energy"]})
    loss = objective(p, "gauge_invariant_mse", weights=dict(zip(TARGETS, [1., 0., 0., 0.])),
                     kwargs={"hamiltonian": {"energy_weight": 3.}})
    nodes, edges = data.point_labels.clone(), data.edge_labels.clone()
    nodes[:, 0] += .7 * nodes[:, 1]
    edges[:, 0] += .7 * edges[:, 1]
    nodes[:, 1] *= 4
    edges[:, 1] *= 2
    value, stats = evaluate(loss, data, p, nodes, edges)
    assert abs(float(value)) < 1e-28
    assert abs(float(stats["target/hamiltonian/physical_energy_mse"])) < 1e-28


def test_physical_partners_absent_fail_early():
    data, p = sample()
    loss = objective(p, kwargs={"density_matrix": {"electron_weight": 1.}})
    with pytest.raises(ValueError, match="prepared|physical|reference"):
        evaluate(loss, data, p, data.point_labels, data.edge_labels)


def test_multiple_physical_targets_stay_flat_and_reuse_one_graph(monkeypatch):
    original = TorchBasisMatrixData.from_config.__func__
    calls = []
    def observe(cls, *args, **kwargs):
        calls.append(1)
        return original(cls, *args, **kwargs)
    monkeypatch.setattr(TorchBasisMatrixData, "from_config", classmethod(observe))
    data, p = sample(physical_labels={"hamiltonian": ["energy"],
        "density_matrix": ["charge", "energy"], "energy_density_matrix": ["energy"]})
    assert len(calls) == 1
    assert all(key.count("__") <= 1 for key in data.keys())
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    assert "physical_node_owner" in getattr(data, LAYOUT_ATTR)
    for target in ("hamiltonian", "density_matrix", "energy_density_matrix"):
        view = multitarget.target_data_view(data, p.matrix_target_layout, target)
        assert view.physical_spin.tolist() == [1]


def test_physical_mask_cannot_hide_nonzero_reference_partner():
    masks = {"density_matrix": [csr_array(np.eye(2, dtype=bool))]}
    with pytest.raises(ValueError, match="unknown|observed"):
        sample(masks=masks, physical_labels={"density_matrix": ["charge"]})


@pytest.mark.parametrize("bad", ["mask_dtype", "mask_shape", "signature"])
def test_bad_masks_and_semantic_signature_rejected(bad):
    data, p = sample()
    if bad == "mask_dtype":
        data.matrix_known_nodes = data.matrix_known_nodes.long()
    elif bad == "mask_shape":
        data.matrix_known_nodes = data.matrix_known_nodes[:, :1]
    else:
        data.matrix_target_signature[0, 1] = 2
    with pytest.raises(ValueError, match="mask|layout|signature"):
        evaluate(objective(p), data, p, data.point_labels, data.edge_labels)
