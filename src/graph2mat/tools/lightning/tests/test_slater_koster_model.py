"""CPU integration and independent periodic orbital checks for optional 7.4."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch

from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, OrbitalConfiguration, PointBasis
from e3nn import o3
from scipy.linalg import block_diag
from pytorch_lightning.cli import LightningArgumentParser
from graph2mat.bindings.e3nn import E3nnGraph2Mat, E3nnSlaterKosterEdgeBlock, HamiltonianLocalContext
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.data.coefficient_targets import (
    CoefficientMatrixMSEProjector, CoefficientMSEUnsupported,
)
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.models.mace import MatrixMACE
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs, data_module_for


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def sk_inputs(**kwargs):
    return training_inputs(edge_block_readout=E3nnSlaterKosterEdgeBlock,
                           edge_block_readout_kwargs={"rank": 2}, **kwargs)


def readouts(model):
    value = model.model.matrix_readouts
    return list(value) if isinstance(value, torch.nn.ModuleList) else [value]


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("mode", ["single", "sum", "mean"])
@pytest.mark.parametrize("reuse", [False, True])
def test_full_mace_loss_gradients_layout_and_absent_type(monkeypatch, components, grouping, mode, reuse):
    model, batch, _ = sk_inputs(components=components, grouping=grouping, mode=mode,
                               preprocessing_edges_reuse_nodes=reuse)
    out = model(batch.clone())
    expected = ((out["node_labels"]-batch.point_labels).square().sum()
                + 2*(out["edge_labels"]-batch.edge_labels).square().sum())
    expected /= batch.point_labels.numel() + 2*batch.edge_labels.numel()
    params = list(model.parameters())
    gradients = torch.autograd.grad(expected, params, allow_unused=True)
    dynamic = batch.clone()
    if hasattr(dynamic, LAYOUT_ATTR):
        object.__delattr__(dynamic, LAYOUT_ATTR)
    actual = model(dynamic)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], out[key], atol=0, rtol=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("certified coefficient route reconstructed matrices")
    for readout in readouts(model):
        for block in [*readout.self_interactions, *readout.interactions.values()]:
            monkeypatch.setattr(block, "coefficients_to_block", forbidden)
    loss = model.training_step(batch.clone(), 0)["loss"]
    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-12)
    for actual, reference in zip(torch.autograd.grad(loss, params, allow_unused=True), gradients):
        assert (actual is None) == (reference is None)
        if actual is not None:
            torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-12)
    unused = {"point_type": ".(1, 1, 2).", "basis_shape": ".(0, 0, 0).", "max": None}[grouping]
    for branch in ("sk_net", "radial_gates"):
        selected = [g for (name, _), g in zip(model.named_parameters(), gradients)
                    if branch in name and (unused is None or unused not in name)]
        assert selected and all(g is not None and torch.isfinite(g).all() for g in selected)
        assert any(g.abs().sum() > 0 for g in selected)


def test_custom_readout_cannot_bypass_context_guard():
    class SKReadout(E3nnGraph2Mat):
        def __init__(self, **kwargs):
            super().__init__(edge_operation=E3nnSlaterKosterEdgeBlock, **kwargs)
    model, _, _ = training_inputs(coefficient_training=False)
    with pytest.raises(ValueError, match="hamiltonian_local_context"):
        MatrixMACE(mace=deepcopy(model.model.mace), graph2mat_cls=SKReadout,
                   unique_basis=model.basis_table,
                   edge_hidden_irreps=o3.Irreps("2x0e+2x1o+2x2e"),
                   hamiltonian_local_context=HamiltonianLocalContext(num_types=2, r_max=4))


def test_mp_cutoff_keeps_matrix_edges_and_invalid_certificate_falls_back(monkeypatch):
    model, batch, samples = sk_inputs(mp_cutoff=1.5, optim_lr=0.)
    out = model(batch.clone())
    assert out["edge_labels"].shape == batch.edge_labels.shape
    lengths = (batch.positions[batch.edge_index[1]] - batch.positions[batch.edge_index[0]] + batch.shifts).norm(dim=-1)
    assert (lengths > 1.5).any()
    block = next(iter(readouts(model)[0].interactions.values()))
    block._coefficient_groups = None
    with pytest.raises(CoefficientMSEUnsupported, match="certified"):
        CoefficientMatrixMSEProjector(model.model.matrix_readouts)
    # Removing a certificate must decline the optimization, not alter the objective.
    calls = []
    forward = model.model.forward
    def spy(*args, **kwargs):
        calls.append(kwargs.get("coefficients_only", False))
        return forward(*args, **kwargs)
    monkeypatch.setattr(model.model, "forward", spy)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, max_epochs=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0)
    module = data_module_for(model, samples)
    with pytest.warns(UserWarning, match="matrix fallback"):
        trainer.fit(model, datamodule=module)
    assert calls == [False]
    assert module._coefficient_mse_projector is None
    expected = ((out["node_labels"]-batch.point_labels).square().sum()
                + 2*(out["edge_labels"]-batch.edge_labels).square().sum())
    expected /= batch.point_labels.numel() + 2*batch.edge_labels.numel()
    torch.testing.assert_close(trainer.callback_metrics["train_loss_epoch"], expected)
    module.teardown(None)


@pytest.mark.parametrize("components", [1, 2])
def test_two_steps_strict_checkpoint_and_trainable_branches(tmp_path, components):
    model, batch, samples = sk_inputs(components=components, coefficient_training=False, optim_lr=1e-3)
    initial = deepcopy(model.state_dict())
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, max_epochs=2,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0)
    module = data_module_for(model, samples)
    trainer.fit(model, datamodule=module)
    assert trainer.global_step == 2
    for branch in ("sk_net", "residual.radial_gates"):
        assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items() if branch in k)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path / "sk.ckpt"
    trainer.save_checkpoint(path)
    restored = LitMACEMatrixModel.load_from_checkpoint(path, map_location="cpu", weights_only=False, strict=True)
    model.eval()
    restored.eval()
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(restored(batch.clone())[key], model(batch.clone())[key], atol=0, rtol=0)
    for block in readouts(restored)[0].interactions.values():
        op = block.operation
        assert isinstance(op, E3nnSlaterKosterEdgeBlock) and op.n_matrix_components == components
        assert all(mul == 2 for mul, _ in op.residual.tensor_products[0].irreps_in1)
        assert all(not v.requires_grad for k, v in op.named_buffers() if "sk_transform" in k)
    module.teardown(None)


def test_existing_cli_class_path_selects_sk():
    parser = LightningArgumentParser()
    parser.add_class_arguments(LitMACEMatrixModel, "model", skip={"basis_table"}, instantiate=False)
    cfg = parser.parse_object({"model": {
        "edge_block_readout": "graph2mat.bindings.e3nn.E3nnSlaterKosterEdgeBlock",
        "edge_block_readout_kwargs": {"rank": 2},
    }})
    assert cfg.model.edge_block_readout is E3nnSlaterKosterEdgeBlock
    model, batch, _ = training_inputs(edge_block_readout=cfg.model.edge_block_readout,
                                     edge_block_readout_kwargs=cfg.model.edge_block_readout_kwargs)
    assert torch.isfinite(model(batch.clone())["edge_labels"]).all()


@pytest.mark.parametrize("convention", ["cartesian", "spherical", "siesta_spherical"])
def test_periodic_spdf_orbitals_rotation_inversion_permutation_supercell(convention):
    def shell(l):
        return [sisl.AtomicOrbital(n=l+1, l=l, m=m, R=1.7) for m in range(-l, l+1)]
    a = sisl.Atom("H", orbitals=shell(0)+shell(1))
    b = sisl.Atom("O", orbitals=sum((shell(l) for l in range(4)), []))
    table = BasisTableWithEdges([PointBasis.from_sisl_atom(at, basis_convention=convention) for at in (a, b)])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    torch.manual_seed(2)
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o+2x2e+2x3o", edge_hidden_irreps="2x0e+2x1o+2x2e+2x3o",
        num_bessel=3, max_ell=3, num_interactions=2, correlation=1,
        edge_block_readout=E3nnSlaterKosterEdgeBlock, edge_block_readout_kwargs={"rank": 2}).eval()
    geom = sisl.Geometry([[0., 0., 0.], [1., .2, .1], [2., .1, .4]], atoms=[a, b, a], lattice=[3.2, 20, 20])
    geom.set_nsc([3, 1, 1])
    def predict(geometry):
        data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geometry), data_processor=processor)
        with torch.no_grad():
            out = model(data)
        return processor.matrix_from_data(data, out, out_format="sisl_DM")
    original = predict(geom)
    matrix = original.tocsr().toarray()
    data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geom), data_processor=processor)
    assert ((data.edge_index[0] == data.edge_index[1]) & data.shifts.ne(0).any(-1)).any()
    assert np.abs(matrix[13:20]).max() > 1e-6
    for index, offset in enumerate(geom.sc_off):
        opposite = geom.sc_index(-offset)
        np.testing.assert_allclose(matrix[:, index*geom.no:(index+1)*geom.no],
            matrix[:, opposite*geom.no:(opposite+1)*geom.no].T, rtol=1e-10, atol=1e-12)
    for angles, sign in [((.31, .72, -.48), 1), ((.91, -.27, .63), -1)]:
        R = sign * o3.angles_to_matrix(*[torch.tensor(x) for x in angles]).numpy()
        rotated = geom.copy()
        rotated.xyz[:] = geom.xyz @ R.T
        rotated.cell[:] = geom.cell @ R.T
        cob = table.change_of_basis
        D = block_diag(*[table.basis[s].e3nn_irreps.D_from_matrix(torch.as_tensor(cob @ R @ cob.T)).numpy()
                         for s in geom.atoms.species])
        expected = np.concatenate([D @ cell @ D.T for cell in np.split(matrix, geom.n_s, axis=1)], axis=1)
        np.testing.assert_allclose(predict(rotated).tocsr().toarray(), expected, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(predict(geom.sub([2, 1, 0])).tocsr().toarray(),
                               original.sub([2, 1, 0]).tocsr().toarray(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(predict(geom.tile(2, 0)).tocsr().toarray(),
                               original.tile(2, 0).tocsr().toarray(), rtol=2e-8, atol=2e-10)


def test_position_derivative_with_frozen_periodic_topology():
    model, batch, _ = sk_inputs(coefficient_training=False)
    assert batch.shifts.ne(0).any()
    batch.positions.requires_grad_(True)
    def objective(data):
        return model(data)["edge_labels"].square().sum()
    derivative = torch.autograd.grad(objective(batch), batch.positions)[0]
    assert torch.isfinite(derivative).all() and derivative.abs().max() > 1e-8
    for i, axis in ((0, 0), (1, 2), (3, 1)):
        values = []
        for sign in (-1, 1):
            changed = batch.clone()
            changed.positions = batch.positions.detach().clone()
            changed.positions[i, axis] += sign * 1e-5
            values.append(objective(changed).detach())
        torch.testing.assert_close((values[1]-values[0])/2e-5, derivative[i, axis], rtol=2e-6, atol=2e-9)


def test_legacy_default_state_keys_are_unchanged():
    # Digest captured at product base 8c8dc3e; prediction equivalence is a local gate.
    model, _, _ = training_inputs(coefficient_training=False)
    digest = hashlib.sha256(json.dumps(list(model.state_dict())).encode()).hexdigest()
    assert digest == "d834a8c7b6f8bbaccaeeca001c5f7fa57dd903b9e89a6ae9884bbb2ea6171bdd"
    assert not any(any(s in k for s in ("sk_net", "sk_transform", "residual")) for k in model.state_dict())


def test_isolated_atom_full_model_has_finite_nodes_and_empty_edges():
    model, _, _ = sk_inputs(coefficient_training=False)
    processor = MatrixDataProcessor(basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False)
    config = BasisConfiguration(point_types=["A"], positions=np.zeros((1, 3)), basis=model.basis_table,
                                cell=np.eye(3)*20, pbc=(False, False, False))
    sample = TorchBasisMatrixData.from_config(config, data_processor=processor)
    out = model(sample)
    assert out["edge_labels"].numel() == 0
    assert out["node_labels"].numel() and torch.isfinite(out["node_labels"]).all()
