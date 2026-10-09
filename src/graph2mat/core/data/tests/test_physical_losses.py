"""Literal periodic/atomic oracles, independent derivatives and physical invariances."""

from copy import deepcopy

import numpy as np
import pytest
import sisl
import torch
from scipy.sparse import csr_array
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data import physical
from graph2mat.core.data.metrics import elementwise_mse, gauge_invariant_mse
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices, prepared_sample
from graph2mat.tools.lightning.data import TypeLayoutCollate


def evaluate(metric, data, table, nodes=None, edges=None):
    return metric(nodes_pred=data.point_labels if nodes is None else nodes, nodes_ref=data.point_labels,
        edges_pred=data.edge_labels if edges is None else edges, edges_ref=data.edge_labels,
        batch=data, basis_table=table)


@pytest.mark.parametrize("baseline", [False, True])
def test_dm_terms_have_literal_loss_and_gradient(baseline):
    data, table = prepared_sample(baseline=baseline, data_cls=TorchBasisMatrixData)
    n = data.point_labels.clone()
    n[0] += .1
    n.requires_grad_()
    e = data.edge_labels.clone().requires_grad_()
    value, stats = evaluate(elementwise_mse(electron_weight=2., mulliken_weight=3.), data, table, n, e)
    assert float(value) == pytest.approx(7/300, abs=1e-14)
    assert float(torch.autograd.grad(value, n)[0][0]) == pytest.approx(7/15, abs=1e-13)
    assert float(stats["physical_electron_mse"]) == pytest.approx(1/400, abs=1e-14)
    assert float(stats["physical_mulliken_mse"]) == pytest.approx(1/200, abs=1e-14)
    assert float(stats["physical_delta_n_mean"]) == pytest.approx(.1, abs=1e-14)
    assert float(stats["physical_delta_n_rmse"]) == pytest.approx(.1, abs=1e-14)
    assert all(not value.requires_grad for name, value in stats.items()
        if name.startswith("physical_") and torch.is_tensor(value))


@pytest.mark.parametrize("target,expected,derivative", [("hamiltonian", 1/288, 5/72),
    ("energy_density_matrix", 1/200, .1), ("density_matrix", 1/800, .025)])
def test_energy_perturbation_is_not_tautological(target, expected, derivative):
    data, table = prepared_sample(target=target, data_cls=TorchBasisMatrixData)
    n = data.point_labels.clone()
    n[0] += .1
    n.requires_grad_()
    e = data.edge_labels.clone().requires_grad_()
    fn = lambda a, b: physical.physical_loss_terms(a, b, batch=data, basis_table=table, energy_weight=1.)[0]
    value = fn(n, e)
    assert float(value) == pytest.approx(expected, abs=1e-14)
    assert float(torch.autograd.grad(value, n, retain_graph=True)[0][0]) == pytest.approx(derivative, abs=1e-14)
    assert torch.autograd.gradcheck(fn, (n, e), eps=1e-6, atol=1e-7)


@pytest.mark.parametrize("symmetric", [False, True])
def test_half_edge_and_directed_self_image_agree(symmetric):
    m = periodic_matrices()
    table = AtomicTableWithEdges(m["density"].geometry.atoms.atom)
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix",
        symmetric_matrix=symmetric, sub_point_matrix=False)
    data = physical.prepare_physical_data(processor, **m, requested=("charge",))
    layout = physical.physical_layout(data, table)
    pop = physical.physical_contractions(data.physical_reference_nodes, data.physical_reference_edges,
        data.physical_overlap_nodes, data.physical_overlap_edges, layout=layout,
        reverse_partner_edges=data.physical_overlap_reverse_edges)
    np.testing.assert_allclose(pop, [.84], atol=1e-14)
    edge_pred = data.edge_labels + .1
    loss, _ = physical.physical_loss_terms(data.point_labels, edge_pred, batch=data,
        basis_table=table, electron_weight=1.)
    assert float(loss) == pytest.approx(.04**2, abs=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_charge_occupancy_channels_without_extra_factor_two(backend, spin, dtype, symmetric):
    cls = TorchBasisMatrixData if backend == "torch" else BasisMatrixData
    data, table = prepared_sample(spin=spin, symmetric=symmetric, data_cls=cls, fractional=True)
    n, e = data.point_labels, data.edge_labels
    cast = lambda value: value.to(getattr(torch, dtype)) if backend == "torch" else value.astype(dtype)
    n, e = cast(n).clone() if backend == "torch" else cast(n).copy(), cast(e)
    n[0] += .1/spin
    value, stats = physical.physical_loss_terms(n, e, batch=data, basis_table=table,
        electron_weight=2., mulliken_weight=3.)
    assert float(value) == pytest.approx(.02, abs=2e-7 if dtype == "float32" else 1e-14)
    assert float(stats["physical_delta_n_mean"]) == pytest.approx(.1, abs=2e-7 if dtype == "float32" else 1e-14)


@pytest.mark.parametrize("edge_weight", [1., 2., 7.])
def test_physical_edge_count_is_independent_of_matrix_mse_weight(edge_weight):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    _, stats = evaluate(elementwise_mse(electron_weight=1., edge_weight=edge_weight),
        data, table, edges=data.edge_labels+.1)
    assert float(stats["physical_electron_mse"]) == pytest.approx(.02**2, abs=1e-14)


def test_full_population_and_reference_continuation_outside_graph():
    m = analytic_matrices()
    # Preserve full labels but choose a certified shorter model graph (no edges).
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=.4, q0=1.)])
    table = AtomicTableWithEdges([atom])
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False)
    data = physical.prepare_physical_data(processor, **m, requested=("charge", "energy"))
    assert data.edge_labels.size == 0
    charge, _ = physical.physical_loss_terms(data.point_labels, data.edge_labels,
        batch=data, basis_table=table, electron_weight=1.)
    energy, _ = physical.physical_loss_terms(data.point_labels, data.edge_labels,
        batch=data, basis_table=table, energy_weight=1.)
    assert float(charge) == pytest.approx(1/36, abs=1e-14)
    assert float(energy) == pytest.approx(0., abs=1e-28)
    # The actual writer-truncated HD-ES would be .5*5/6 - 5/36 = 5/18 per atom.
    assert (5/18)**2 > float(energy)


@pytest.mark.parametrize("target", ["hamiltonian", "energy_density_matrix"])
def test_charge_objectives_reject_non_density_prediction(target):
    data, table = prepared_sample(target=target)
    with pytest.raises(ValueError, match="DM|density|charge"):
        physical.physical_loss_terms(data.point_labels, data.edge_labels,
            batch=data, basis_table=table, electron_weight=1.)


@pytest.mark.parametrize("field,value", [("physical_version", 2), ("physical_spin", 3),
    ("physical_kind", 9), ("physical_capabilities", 1)])
def test_active_term_rejects_incompatible_prepared_contract(field, value):
    data, table = prepared_sample()
    setattr(data, field, np.array([value]))
    data._data[field] = getattr(data, field)
    with pytest.raises(ValueError, match="physical|reference|spin|energy|version|kind"):
        physical.physical_loss_terms(data.point_labels, data.edge_labels, batch=data, basis_table=table, energy_weight=1.)


@pytest.mark.parametrize("invalid", [-1., np.nan, np.inf])
@pytest.mark.parametrize("metric", [elementwise_mse, gauge_invariant_mse])
def test_metric_call_validates_physical_weights_even_without_auxiliary(metric, invalid):
    data, table = prepared_sample(target="hamiltonian" if metric is gauge_invariant_mse else "density_matrix",
        gauge=metric is gauge_invariant_mse, data_cls=TorchBasisMatrixData)
    with pytest.raises(ValueError, match="finite|nonnegative"):
        evaluate(metric(energy_weight=invalid), data, table)


def test_zero_weights_do_not_require_or_evaluate_auxiliary_contract(monkeypatch):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    for key in list(data.keys()):
        if key.startswith("physical_"):
            del data[key]
    def forbidden(*args, **kwargs):
        raise AssertionError("physical helper used while disabled")
    monkeypatch.setattr(physical, "physical_loss_terms", forbidden, raising=False)
    expected, expected_stats = evaluate(elementwise_mse(), data, table, data.point_labels+.1)
    actual, stats = evaluate(elementwise_mse(electron_weight=0., mulliken_weight=0., energy_weight=0.),
        data, table, data.point_labels+.1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats.keys() == expected_stats.keys()


@pytest.mark.parametrize("edge_weight", [1., 2.])
def test_gauge_energy_uses_existing_aligned_delta_not_raw_error(edge_weight):
    data, table = prepared_sample(target="hamiltonian", gauge=True, data_cls=TorchBasisMatrixData)
    pred = data.point_labels[:, 0].clone()
    pred[0] += .1
    pred.requires_grad_()
    edges = data.edge_labels[:, 0].clone().requires_grad_()
    base, _ = evaluate(gauge_invariant_mse(edge_weight=edge_weight), data, table, pred, edges)
    total, stats = evaluate(gauge_invariant_mse(edge_weight=edge_weight, energy_weight=1.), data, table, pred, edges)
    c = .1/(2 + edge_weight*.04)
    expected = ((1/12-c)**2 + c**2)/2
    assert float(stats["physical_energy_mse"]) == pytest.approx(expected, abs=1e-14)
    assert float(total-base) == pytest.approx(expected, abs=1e-14)
    assert torch.autograd.gradcheck(lambda a, b: evaluate(gauge_invariant_mse(
        edge_weight=edge_weight, energy_weight=1.), data, table, a, b)[0], (pred, edges), atol=1e-7)


def test_pred_only_gauge_shift_is_removed_only_by_gauge_objective():
    data, table = prepared_sample(target="hamiltonian", gauge=True, data_cls=TorchBasisMatrixData)
    n = data.point_labels[:, 0] + .37*data.point_labels[:, 1]
    e = data.edge_labels[:, 0] + .37*data.edge_labels[:, 1]
    _, aligned = evaluate(gauge_invariant_mse(energy_weight=1.), data, table, n, e)
    raw, _ = physical.physical_loss_terms(n, e, batch=data, basis_table=table, energy_weight=1.)
    assert float(aligned["physical_energy_mse"]) == pytest.approx(0., abs=1e-28)
    assert float(raw) == pytest.approx(.37**2, abs=1e-14)


def test_active_physics_does_not_assemble_or_diagonalize_predictions(monkeypatch):
    data, table = prepared_sample()
    def forbidden(*args, **kwargs):
        raise AssertionError("dense prediction assembly/eigensolver entered physical objective")
    monkeypatch.setattr(MatrixDataProcessor, "matrix_from_data", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    value, _ = physical.physical_loss_terms(data.point_labels+.1, data.edge_labels,
        batch=data, basis_table=table, energy_weight=1., electron_weight=1., mulliken_weight=1.)
    assert float(value) > 0


@pytest.mark.parametrize("target,partner", [("density_matrix", "overlap"), ("energy_density_matrix", "partner")])
def test_unknown_reference_is_excluded_before_nan_times_exact_zero(target, partner):
    data, table = prepared_sample(target=target, data_cls=TorchBasisMatrixData)
    data.physical_reference_edges[:] = float("nan")
    data.physical_reference_known_edges[:] = False
    for key in (f"physical_{partner}_edges", f"physical_{partner}_reverse_edges"):
        data[key][:] = 0.
    n = data.point_labels.clone().requires_grad_()
    e = torch.full_like(data.edge_labels, float("nan"), requires_grad=True)
    weights = {"electron_weight": 1.} if target == "density_matrix" else {"energy_weight": 1.}
    value, _ = physical.physical_loss_terms(n, e, batch=data, basis_table=table, **weights)
    assert torch.isfinite(value)
    gradients = torch.autograd.grad(value, (n, e))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert torch.equal(gradients[1], torch.zeros_like(e))
    data[f"physical_{partner}_edges"][:] = .2
    with pytest.raises(ValueError, match="unknown|observed"):
        physical.physical_loss_terms(n, e, batch=data, basis_table=table, **weights)


def test_nonfinite_prediction_on_observed_reference_is_rejected():
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    n = data.point_labels.clone()
    n[0] = float("nan")
    with pytest.raises(ValueError, match="finite|prediction"):
        physical.physical_loss_terms(n, data.edge_labels, batch=data, basis_table=table, electron_weight=1.)


def test_per_structure_charge_normalization_is_not_matrix_element_weighting():
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    batch = TypeLayoutCollate(Batch.from_data_list, table)([data, deepcopy(data)])
    n = batch.point_labels.clone()
    n[0] += .1
    n[2] += .3
    value, stats = physical.physical_loss_terms(n, batch.edge_labels, batch=batch,
        basis_table=table, electron_weight=1.)
    assert float(value) == pytest.approx(.0125, abs=1e-14)
    assert stats["physical_structures"] == 2
    assert float(stats["physical_delta_n_mean"]) == pytest.approx(.2, abs=1e-14)
    assert float(stats["physical_delta_n_rmse"]) == pytest.approx(np.sqrt(.05), abs=1e-14)


def two_orbital_fields():
    rng = np.random.default_rng(44)
    b = rng.normal(size=(4, 4))
    s = b.T @ b + np.eye(4)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    c = np.linalg.solve(np.linalg.cholesky(s).T, q)
    occupation, epsilon = np.array([1.6, .9, .3, .1]), np.array([-.7, -.2, .3, .9])
    h = s @ c @ np.diag(epsilon) @ c.T @ s
    d = c @ np.diag(occupation) @ c.T
    e = c @ np.diag(occupation*epsilon) @ c.T
    delta = rng.normal(size=(4, 4))*.03
    return dict(s=s, h=h, d=d, e=e), (delta+delta.T)/2


def dense_sample(fields, target, permuted=False):
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.8, q0=1.),
        sisl.AtomicOrbital("2s", R=2.8, q0=0.)])
    xyz = [[0., 0., 0.], [1., 0., 0.]]
    g = sisl.Geometry(xyz[::-1] if permuted else xyz, atoms=[atom, atom], lattice=[20.]*3)
    g.set_nsc([1, 1, 1])
    s = csr_array(fields["s"])
    matrices = dict(overlap=sisl.Overlap.fromsp(g, s),
        hamiltonian=sisl.Hamiltonian.fromsp(g, csr_array(fields["h"]), S=s),
        density=sisl.DensityMatrix.fromsp(g, csr_array(fields["d"]), S=s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, csr_array(fields["e"]), S=s),
        kpoints=np.zeros((1, 3)), kweights=np.ones(1))
    table = AtomicTableWithEdges([atom])
    processor = MatrixDataProcessor(basis_table=table, out_matrix=target,
        symmetric_matrix=True, sub_point_matrix=False)
    return physical.prepare_physical_data(processor, **matrices, requested=("charge", "energy")), table


def flat_dense_delta(delta, data):
    nodes = np.concatenate([delta[2*i:2*i+2, 2*i:2*i+2].ravel() for i in range(2)])
    edges = np.concatenate([delta[2*i:2*i+2, 2*j:2*j+2].ravel() for i, j in data.edge_index[:, ::2].T])
    return nodes, edges


@pytest.mark.parametrize("target", ["density_matrix", "hamiltonian", "energy_density_matrix"])
def test_atom_grouping_and_basis_rotation_match_independent_dense_trace(target):
    fields, delta = two_orbital_fields()
    u = np.zeros((4, 4))
    for atom, angle in enumerate((.31, -.52)):
        u[2*atom:2*atom+2, 2*atom:2*atom+2] = [[np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]]
    partner = fields[{"density_matrix": "h", "hamiltonian": "d", "energy_density_matrix": "s"}[target]]
    dense_r = np.diag(delta @ partner).reshape(2, 2).sum(1)
    expected = np.mean(dense_r**2)
    expected_p = np.diag(fields["d"] @ fields["s"]).reshape(2, 2).sum(1)
    for transformed, perturbation in [(fields, delta),
            ({key: u.T @ value @ u for key, value in fields.items()}, u.T @ delta @ u)]:
        data, table = dense_sample(transformed, target)
        np.testing.assert_allclose(data.physical_population_reference, expected_p, atol=1e-13)
        dn, de = flat_dense_delta(perturbation, data)
        value, _ = physical.physical_loss_terms(data.point_labels+dn, data.edge_labels+de,
            batch=data, basis_table=table, energy_weight=1.)
        assert float(value) == pytest.approx(expected, abs=1e-12)
        if target == "density_matrix":
            dp = np.diag(delta @ fields["s"]).reshape(2, 2).sum(1)
            charge, _ = physical.physical_loss_terms(data.point_labels+dn, data.edge_labels+de,
                batch=data, basis_table=table, electron_weight=2., mulliken_weight=3.)
            assert float(charge) == pytest.approx(2*(dp.sum()/2)**2 + 3*np.mean(dp**2), abs=1e-12)


def test_atomic_permutation_and_endpoint_exchange_preserve_objectives():
    fields, delta = two_orbital_fields()
    order = [2, 3, 0, 1]
    permuted = {key: value[np.ix_(order, order)] for key, value in fields.items()}
    answers = []
    for values, perturbation, swap in [(fields, delta, False),
            (permuted, delta[np.ix_(order, order)], True)]:
        data, table = dense_sample(values, "density_matrix", permuted=swap)
        dn, de = flat_dense_delta(perturbation, data)
        value, _ = physical.physical_loss_terms(data.point_labels+dn, data.edge_labels+de,
            batch=data, basis_table=table, energy_weight=1., electron_weight=1., mulliken_weight=1.)
        answers.append(float(value))
    assert answers[0] == pytest.approx(answers[1], abs=1e-12)


def test_common_reference_energy_origin_shift_is_consistent_for_h_prediction():
    fields, delta = two_orbital_fields()
    shifted = {key: value.copy() for key, value in fields.items()}
    shifted["h"] += .37*fields["s"]
    shifted["e"] += .37*fields["d"]
    answers = []
    for values in (fields, shifted):
        data, table = dense_sample(values, "hamiltonian")
        dn, de = flat_dense_delta(delta, data)
        value, _ = physical.physical_loss_terms(data.point_labels+dn, data.edge_labels+de,
            batch=data, basis_table=table, energy_weight=1.)
        answers.append(float(value))
    assert answers[0] == pytest.approx(answers[1], abs=1e-12)


@pytest.mark.parametrize("nonzero_partner", [False, True])
def test_gauge_excluded_label_row_requires_exact_zero_energy_partner(nonzero_partner):
    data, table = prepared_sample(target="hamiltonian", gauge=True, data_cls=TorchBasisMatrixData)
    if not nonzero_partner:
        data.physical_partner_edges.zero_()
        data.physical_partner_reverse_edges.zero_()
    data.edge_labels[0, -1] = float("nan")
    n = (data.point_labels[:, 0]+.1).clone().requires_grad_()
    e = torch.full((1,), float("nan"), dtype=torch.float64, requires_grad=True)
    if nonzero_partner:
        with pytest.raises(ValueError, match="unknown|observed|excluded"):
            evaluate(gauge_invariant_mse(energy_weight=1.), data, table, n, e)
    else:
        value, stats = evaluate(gauge_invariant_mse(energy_weight=1.), data, table, n, e)
        assert "physical_energy_mse" in stats
        assert torch.isfinite(value)
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(value, (n, e)))


@pytest.mark.parametrize("target,expected", [("hamiltonian", 1/288),
    ("density_matrix", 1/800), ("energy_density_matrix", 1/200)])
def test_energy_collinear_channels_preserve_occupation_convention(target, expected):
    data, table = prepared_sample(target=target, spin=2, data_cls=TorchBasisMatrixData)
    n = data.point_labels.clone()
    n[0] += .1 if target == "hamiltonian" else .05
    value, _ = physical.physical_loss_terms(n, data.edge_labels, batch=data, basis_table=table, energy_weight=1.)
    assert float(value) == pytest.approx(expected, abs=1e-14)


def test_archived_asymmetry_uses_separate_half_edge_partners():
    data, table = prepared_sample()
    data.physical_overlap_edges[:] = .2-1e-9
    data.physical_overlap_reverse_edges[:] = .2+1e-9
    value, _ = physical.physical_loss_terms(data.point_labels, data.edge_labels+.1,
        batch=data, basis_table=table, mulliken_weight=1.)
    # Direct opposite-image contractions, no averaging/symmetrizing of archived S.
    expected = ((5/6*.2 - (5/6+.1)*(.2-1e-9))**2
        + (5/6*.2 - (5/6+.1)*(.2+1e-9))**2)/2
    assert float(value) == pytest.approx(expected, abs=1e-16)


def test_energy_model_parameter_gradient_matches_central_finite_difference():
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    torch.manual_seed(13)
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2, max_ell=1, num_interactions=2, correlation=1,
        n_matrix_components=1).double().eval()
    batch = TypeLayoutCollate(Batch.from_data_list, table)([data])
    def objective():
        out = model(batch)
        return physical.physical_loss_terms(out["node_labels"], out["edge_labels"],
            batch=batch, basis_table=table, energy_weight=1.)[0]
    parameters = tuple(model.parameters())
    value = objective()
    grads = torch.autograd.grad(value, parameters, allow_unused=True)
    candidates = [(p, g) for p, g in zip(parameters, grads) if g is not None and float(g.abs().max()) > 1e-10]
    assert candidates
    parameter, grad = candidates[-1]
    index = int(grad.abs().reshape(-1).argmax())
    original = parameter.detach().reshape(-1)[index].item()
    eps = 1e-6
    with torch.no_grad():
        parameter.reshape(-1)[index] = original+eps
        plus = float(objective())
        parameter.reshape(-1)[index] = original-eps
        minus = float(objective())
        parameter.reshape(-1)[index] = original
    assert (plus-minus)/(2*eps) == pytest.approx(float(grad.reshape(-1)[index]), rel=1e-6, abs=1e-8)


def test_energy_keeps_nonzero_certified_reference_residual_and_edm_sign():
    m = analytic_matrices()
    g = m["energy_density"].geometry
    m["energy_density"] = sisl.EnergyDensityMatrix.fromsp(g,
        m["energy_density"].tocsr(0)-csr_array(np.eye(2)*1e-7), S=m["overlap"].tocsr(0))
    table = AtomicTableWithEdges(g.atoms.atom)
    processor = MatrixDataProcessor(basis_table=table, out_matrix="energy_density_matrix",
        symmetric_matrix=True, sub_point_matrix=False)
    data = physical.prepare_physical_data(processor, **m, requested=("energy",))
    np.testing.assert_allclose(data.physical_energy_reference_residual, [1e-7, 1e-7], atol=1e-15)
    at_reference, _ = physical.physical_loss_terms(data.point_labels, data.edge_labels,
        batch=data, basis_table=table, energy_weight=1.)
    assert float(at_reference) == pytest.approx(1e-14, rel=1e-8, abs=1e-24)
    n = data.point_labels.copy()
    n[0] += .1
    value, _ = physical.physical_loss_terms(n, data.edge_labels,
        batch=data, basis_table=table, energy_weight=1.)
    assert float(value) == pytest.approx(((1e-7-.1)**2+1e-14)/2, abs=1e-14)


@pytest.mark.parametrize("field", ["physical_reference_nodes", "physical_overlap_nodes",
    "physical_partner_edges", "physical_reference_known_edges", "physical_population_reference"])
def test_prepared_auxiliary_shapes_are_not_silently_broadcast(field):
    data, table = prepared_sample()
    setattr(data, field, getattr(data, field)[:0])
    data._data[field] = getattr(data, field)
    with pytest.raises(ValueError, match="shape|physical|reference|mask"):
        physical.physical_loss_terms(data.point_labels, data.edge_labels, batch=data,
            basis_table=table, energy_weight=1., electron_weight=1.)


def test_partial_empty_basis_still_counts_in_physical_loss_denominators():
    from graph2mat.core.data.basis import NoBasisAtom
    m = analytic_matrices()
    g = m["density"].geometry.add(sisl.Geometry([[8., 0., 0.]], atoms=NoBasisAtom("He", R=0.)))
    s = m["overlap"].tocsr(0)
    m.update(overlap=sisl.Overlap.fromsp(g, s),
        density=sisl.DensityMatrix.fromsp(g, m["density"].tocsr(0), S=s),
        hamiltonian=sisl.Hamiltonian.fromsp(g, m["hamiltonian"].tocsr(0), S=s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, m["energy_density"].tocsr(0), S=s))
    table = AtomicTableWithEdges(g.atoms.atom)
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False)
    data = physical.prepare_physical_data(processor, **m, requested=("charge", "energy"))
    n = data.point_labels.copy()
    n[0] += .1
    value, _ = physical.physical_loss_terms(n, data.edge_labels, batch=data,
        basis_table=table, electron_weight=2., mulliken_weight=3., energy_weight=1.)
    assert float(value) == pytest.approx(47/3600, abs=1e-14)


def test_active_physical_metric_requires_prepared_references():
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    del data.physical_version
    with pytest.raises(ValueError, match="prepared|physical|version"):
        evaluate(elementwise_mse(energy_weight=1.), data, table)


def test_unknown_fixed_partner_cannot_be_forged_as_an_observed_zero():
    data, table = prepared_sample(target="hamiltonian")
    data.physical_partner_known_edges[:] = False
    data.physical_partner_edges[:] = 0.
    with pytest.raises(ValueError, match="unknown|observed|partner"):
        physical.physical_loss_terms(data.point_labels, data.edge_labels,
            batch=data, basis_table=table, energy_weight=1.)


def test_gauge_exclusion_does_not_mutate_prepared_observation_masks():
    data, table = prepared_sample(target="hamiltonian", gauge=True, data_cls=TorchBasisMatrixData)
    data.edge_labels[0, -1] = float("nan")
    data.physical_partner_edges.zero_()
    data.physical_partner_reverse_edges.zero_()
    before = data.physical_reference_known_edges.clone()
    physical.physical_loss_terms(data.point_labels[:, 0], data.edge_labels[:, 0],
        batch=data, basis_table=table, energy_weight=1.,
        aligned_nodes=torch.zeros((2, 1), dtype=torch.float64),
        aligned_edges=torch.zeros((1, 1), dtype=torch.float64))
    torch.testing.assert_close(data.physical_reference_known_edges, before, rtol=0, atol=0)
