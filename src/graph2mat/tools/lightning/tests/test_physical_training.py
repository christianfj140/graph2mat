"""Physical objectives must survive training, stage changes and epoch reduction."""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from scipy.sparse import csr_array
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.metrics import block_type_mse, elementwise_mse, gauge_invariant_mse
from graph2mat.core.data.physical import prepare_physical_data
from graph2mat.core.data.tests.physical_fixtures import prepared_sample
from graph2mat.tools.lightning.data import TypeLayoutCollate
from graph2mat.tools.lightning.model import LitBasisMatrixModel, _EpochWeightedLoss
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def physical_model(table, loss_kwargs, *, stages=None):
    return LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2, max_ell=1, num_interactions=2, correlation=1,
        n_matrix_components=1, coefficient_training=True,
        loss=elementwise_mse, loss_kwargs=loss_kwargs, training_stages=stages).double()


def physical_loader(samples, table, batch_size=2):
    return DataLoader(samples, batch_size=batch_size,
        collate_fn=TypeLayoutCollate(Batch.from_data_list, table, symmetric=True))


def fit_physical(model, samples, table, tmp_path, max_epochs=1, val_samples=None):
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs,
        limit_train_batches=1, limit_val_batches=0 if val_samples is None else 1,
        num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=physical_loader(samples, table),
        val_dataloaders=None if val_samples is None else physical_loader(val_samples, table))
    return trainer


@pytest.mark.parametrize("term", ["electron_weight", "mulliken_weight", "energy_weight"])
def test_each_physical_term_changes_real_model_gradient(term):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {term: .3})
    batch = next(iter(physical_loader([data], table)))
    out = model(batch)
    args = dict(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
        edges_pred=out["edge_labels"], edges_ref=batch.edge_labels, batch=batch, basis_table=table)
    total, _ = model.loss_fn(**args)
    base, _ = elementwise_mse()(**args)
    gradients = torch.autograd.grad(total-base, list(model.parameters()), allow_unused=True)
    actual = [g for g in gradients if g is not None]
    assert actual and all(torch.isfinite(g).all() for g in actual)
    assert sum(float(g.abs().sum()) for g in actual) > 0
    assert model._coefficient_training_reason(batch) == "physical objectives require matrix predictions"


@pytest.mark.parametrize("metric", [elementwise_mse, gauge_invariant_mse])
@pytest.mark.parametrize("term", ["electron_weight", "mulliken_weight", "energy_weight"])
@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_weights_rejected_at_construction(metric, term, bad):
    with pytest.raises(ValueError, match="weight|finite|nonnegative"):
        metric(**{term: bad})


def test_future_physical_weight_rejected_at_model_construction():
    _, table = prepared_sample(data_cls=TorchBasisMatrixData)
    with pytest.raises(ValueError, match="weight|finite|nonnegative"):
        physical_model(table, {}, stages=[{"max_epochs": 1, "loss": "elementwise_mse",
            "loss_kwargs": {"energy_weight": -1.}}])


def test_unsupported_objective_does_not_silently_ignore_physics():
    with pytest.raises(ValueError, match="physical.*support|support.*physical"):
        block_type_mse(energy_weight=1.)


def test_stage_without_loss_does_not_silently_ignore_physical_kwargs():
    _, table = prepared_sample(data_cls=TorchBasisMatrixData)
    with pytest.raises(ValueError, match="loss_kwargs|loss"):
        physical_model(table, {}, stages=[{"max_epochs": 1, "loss_kwargs": {"energy_weight": 1.}}])


def test_zero_weights_keep_coefficient_certificate():
    from graph2mat.bindings.torch.data.coefficient_targets import CoefficientMatrixMSEProjector
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {"electron_weight": 0., "mulliken_weight": 0., "energy_weight": 0.})
    batch = TypeLayoutCollate(Batch.from_data_list, table,
        CoefficientMatrixMSEProjector(model.model.matrix_readouts))([data])
    assert model._coefficient_training_reason(batch) is None
    assert set(model.training_step(batch, 0)) == {"loss"}


@pytest.mark.parametrize("physics", [False, True])
def test_datamodule_two_steps_preserve_default_and_physical_contract(physics, tmp_path):
    from graph2mat.tools.lightning import MatrixDataModule
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {"electron_weight": .3} if physics else {})
    module = MatrixDataModule(basis_table=table, symmetric_matrix=True,
        out_matrix="density_matrix", sub_point_matrix=False, train_runs=[data], val_runs=[],
        batch_size=1, loader_threads=0, physical_labels=("charge",) if physics else None)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, limit_train_batches=1,
        limit_val_batches=0, num_sanity_val_steps=0, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=tmp_path)
    trainer.fit(model, datamodule=module)
    assert trainer.global_step == 2
    assert ("train_physical_electron_mse_epoch" in trainer.callback_metrics) is physics
    module.teardown(None)


@pytest.mark.parametrize("invalid", ["future_energy", "missing", "version", "spin", "target", "validation", "later_sample"])
def test_preflight_fails_before_any_optimizer_step(invalid, tmp_path, monkeypatch):
    data, table = prepared_sample(requested=("charge",) if invalid in ("future_energy", "validation")
        else ("charge", "energy"), data_cls=TorchBasisMatrixData)
    stages = None
    kwargs = {"electron_weight": .3}
    val_samples = None
    if invalid == "future_energy":
        kwargs = {}
        stages = [{"max_epochs": 1, "loss": "elementwise_mse"},
            {"name": "future_energy", "max_epochs": 1, "loss": "elementwise_mse",
             "loss_kwargs": {"energy_weight": 1.}}]
    elif invalid == "missing":
        del data.physical_version
    elif invalid == "version":
        data.physical_version[:] = 17
    elif invalid == "spin":
        data.physical_spin[:] = 2
    elif invalid == "target":
        data, table = prepared_sample(target="hamiltonian", data_cls=TorchBasisMatrixData)
    elif invalid == "validation":
        val_samples = [data]
        data, _ = prepared_sample(data_cls=TorchBasisMatrixData)
        kwargs = {"energy_weight": 1.}
    samples = [data]
    if invalid == "later_sample":
        bad = deepcopy(data)
        del bad.physical_version
        samples = [data, deepcopy(data), bad]
    model = physical_model(table, kwargs, stages=stages)
    assert isinstance(model.configure_optimizers(), torch.optim.Adam)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def forbidden_step(*args, **kwargs):
        raise AssertionError("optimizer.step ran before physical preflight")
    def forbidden_forward(*args, **kwargs):
        raise AssertionError("model forward ran before physical preflight")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_step)
    monkeypatch.setattr(model.model, "forward", forbidden_forward)
    with pytest.raises(ValueError, match="physical|reference|energy|spin|density"):
        fit_physical(model, samples, table, tmp_path, max_epochs=2, val_samples=val_samples)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_physical_iterable_rejected_without_consuming_it(tmp_path):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    class Stream(IterableDataset):
        def __iter__(self):
            raise AssertionError("physical preflight consumed an irreversible stream")
            yield data
    loader = physical_loader(Stream(), table)
    model = physical_model(table, {"energy_weight": 1.})
    trainer = pl.Trainer(accelerator="cpu", max_epochs=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path)
    with pytest.raises(ValueError, match="physical.*inspect|inspect.*physical"):
        trainer.fit(model, train_dataloaders=loader)


def test_preflight_preserves_missing_resource_error_context(tmp_path):
    _, table = prepared_sample(data_cls=TorchBasisMatrixData)
    class MissingReference:
        def __len__(self):
            return 1
        def __getitem__(self, index):
            raise ValueError("missing TSHS reference")
    model = physical_model(table, {"energy_weight": 1.})
    with pytest.raises(ValueError, match="physical preflight train sample 0.*missing TSHS"):
        fit_physical(model, MissingReference(), table, tmp_path)


def epoch_samples(gauge=False):
    """D=ones, S=I, H=.5I, E=.5D: exact N=n and HD=ES."""
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.8, q0=1.)])
    table = AtomicTableWithEdges([atom])
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True,
        sub_point_matrix=False, out_matrix="hamiltonian" if gauge else "density_matrix",
        n_matrix_components=2 if gauge else 1,
        matrix_component_policy="h_and_overlap" if gauge else "h_only")
    samples = []
    for n, error in ((1, 1.), (2, 2.), (3, 4.)):
        g = sisl.Geometry([[i, 0., 0.] for i in range(n)], atoms=atom, lattice=[20.]*3)
        g.set_nsc([1, 1, 1])
        s, d = csr_array(np.eye(n)), csr_array(np.ones((n, n)))
        data = prepare_physical_data(processor, overlap=sisl.Overlap.fromsp(g, s),
            density=sisl.DensityMatrix.fromsp(g, d),
            hamiltonian=sisl.Hamiltonian.fromsp(g, .5*s, S=s),
            energy_density=sisl.EnergyDensityMatrix.fromsp(g, .5*d),
            kpoints=np.zeros((1, 3)), kweights=np.ones(1),
            requested=("energy",) if gauge else ("charge", "energy"), data_cls=TorchBasisMatrixData)
        if gauge:
            data.node_prediction = data.point_labels[:, 0].clone()
            data.edge_prediction = data.edge_labels[:, 0] + error
        else:
            data.node_prediction = data.point_labels + error
            data.edge_prediction = data.edge_labels.clone()
        samples.append(data)
    return samples, table


@pytest.mark.parametrize("gauge", [False, True])
@pytest.mark.parametrize("split", ["val", "test"])
def test_physical_epoch_logs_use_separate_denominators(gauge, split):
    samples, table = epoch_samples(gauge)
    kwargs = {"edge_weight": 2., "energy_weight": 4.} if gauge else {
        "electron_weight": 2., "mulliken_weight": 3., "energy_weight": 4.}
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
        loss=gauge_invariant_mse if gauge else elementwise_mse, loss_kwargs=kwargs)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for size in (1, 2, 3, 1):
        result = evaluate(model, dataloaders=physical_loader(samples, table, size), verbose=False)[0]
        if gauge:
            assert result[f"{split}_loss"] == pytest.approx(52/7 + 4*68/3)
            assert result[f"{split}_physical_energy_mse"] == pytest.approx(68/3)
        else:
            assert result[f"{split}_loss"] == pytest.approx(57/10 + 6*7)
            assert result[f"{split}_physical_electron_mse"] == pytest.approx(7.)
            assert result[f"{split}_physical_mulliken_mse"] == pytest.approx(7.)
            assert result[f"{split}_physical_energy_mse"] == pytest.approx(7/4)
            assert result[f"{split}_physical_delta_n_mean"] == pytest.approx(17/3)
            assert result[f"{split}_physical_delta_n_rmse"] == pytest.approx((161/3)**.5)
        assert result[f"{split}_physical_structures"] == 3


def test_physical_reducer_resets():
    metric = _EpochWeightedLoss(2)
    for values in ((1.,), (3., 2.), (4.,)):
        values = torch.tensor(values, dtype=torch.float64)
        metric.update(torch.stack([values.square().mean(), values.mean()]),
            [len(values), len(values)], (1., .5))
    assert float(metric.compute()) == pytest.approx(8.75)
    metric.reset()
    assert metric.counts.tolist() == [0., 0.]


def test_delta_n_epoch_mean_retains_its_sign():
    samples, table = epoch_samples()
    samples[1].node_prediction = samples[1].point_labels - 2.
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
        loss=elementwise_mse, loss_kwargs={"electron_weight": 1.})
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    result = trainer.validate(model, dataloaders=physical_loader(samples, table, 2), verbose=False)[0]
    assert result["val_physical_delta_n_mean"] == pytest.approx(3.)
    assert result["val_physical_delta_n_rmse"] == pytest.approx((161/3)**.5)


def test_fractional_edge_weight_is_not_clamped_to_one():
    samples, table = epoch_samples()
    sample = samples[1]
    sample.node_prediction = sample.point_labels.clone()
    sample.point_labels[:] = float("nan")
    sample.edge_prediction = sample.edge_labels + 4.
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
        loss=elementwise_mse, loss_kwargs={"edge_weight": .1, "energy_weight": 1.})
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    result = trainer.validate(model, dataloaders=physical_loader([sample], table), verbose=False)[0]
    assert result["val_loss"] == pytest.approx(16.)


def test_config_resolved_metrics_keep_physical_references():
    samples, table = epoch_samples()
    batch = next(iter(physical_loader(samples, table, 3)))
    values, _ = elementwise_mse(electron_weight=2., mulliken_weight=3., energy_weight=4.)(
        nodes_pred=batch.node_prediction, nodes_ref=batch.point_labels,
        edges_pred=batch.edge_prediction, edges_ref=batch.edge_labels,
        batch=batch, basis_table=table, symmetric_matrix=True, config_resolved=True)
    np.testing.assert_allclose(values, [7., 80/3, 104.], atol=1e-12)


def test_stage_switches_reset_physical_epoch_weights(tmp_path):
    samples, table = epoch_samples()
    class TrainPrediction(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.)
    class Recorder(pl.Callback):
        def __init__(self):
            self.losses = []
        def on_train_epoch_end(self, trainer, pl_module):
            self.losses.append(float(trainer.callback_metrics["train_loss_epoch"]))
    stages = [{"max_epochs": 1, "loss": "elementwise_mse", "loss_kwargs": {"energy_weight": weight}}
        for weight in (1., 0., 2.)]
    model = TrainPrediction(model_cls=_PredictionModel, basis_table=table,
        loss=elementwise_mse, training_stages=stages)
    model.init_model()
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=3, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, callbacks=[recorder], default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=physical_loader(samples, table, 2))
    assert recorder.losses == pytest.approx([57/10 + 7/4, 57/10, 57/10 + 7/2])


def test_auxiliary_targets_do_not_change_features():
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {"electron_weight": 1., "energy_weight": 1.}).eval()
    changed = deepcopy(data)
    changed.physical_population_reference += 17.
    changed.physical_reference_nodes += 23.
    changed.physical_partner_nodes += 2.
    with torch.no_grad():
        before, after = model(data), model(changed)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    losses = [model.loss_fn(nodes_pred=before["node_labels"], nodes_ref=d.point_labels,
        edges_pred=before["edge_labels"], edges_ref=d.edge_labels, batch=d, basis_table=table)[0]
        for d in (data, changed)]
    assert float(losses[0]) != float(losses[1])


def test_two_steps_checkpoint_and_future_stage(tmp_path):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    stages = [{"max_epochs": 1, "loss": "elementwise_mse", "loss_kwargs": {}},
        {"name": "physics", "max_epochs": 1, "loss": "elementwise_mse",
         "loss_kwargs": {"energy_weight": .3}}]
    model = physical_model(table, {}, stages=stages)
    keys = set(model.state_dict())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = fit_physical(model, [data], table, tmp_path, max_epochs=2)
    assert trainer.global_step == 2
    assert model._active_training_stage_name == "physics"
    assert model.loss_fn.metric_kwargs == {"energy_weight": .3}
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    path = tmp_path / "physical.ckpt"
    trainer.save_checkpoint(path)
    loaded = LitMACEMatrixModel.load_from_checkpoint(path, weights_only=False).double().eval()
    model.eval()
    assert set(loaded.state_dict()) == keys
    assert loaded.training_stages_config == stages
    loaded._apply_training_stage_for_epoch(1)
    assert loaded._active_training_stage_name == "physics"
    assert loaded.loss_fn.metric_kwargs == {"energy_weight": .3}
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(loaded(data)[key], model(data)[key], rtol=0, atol=0)
    # Old checkpoints have no physical kwargs or auxiliaries and need no new parameters.
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["hyper_parameters"].pop("loss_kwargs", None)
    checkpoint["hyper_parameters"].pop("training_stages", None)
    legacy_path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, legacy_path)
    legacy = LitMACEMatrixModel.load_from_checkpoint(legacy_path, weights_only=False).double().eval()
    assert not legacy.loss_fn.metric_kwargs
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(legacy(data)[key], model(data)[key], rtol=0, atol=0)
    resumed = pl.Trainer(accelerator="cpu", devices=1, max_epochs=3, limit_train_batches=1,
        limit_val_batches=0, num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path)
    resumed.fit(loaded, train_dataloaders=physical_loader([data], table), ckpt_path=path)
    assert resumed.global_step == 3
    assert loaded._active_training_stage_name == "physics"
    assert loaded.loss_fn.metric_kwargs == {"energy_weight": .3}


def _ddp_physical(output, devices):
    from pytorch_lightning.strategies import DDPStrategy

    torch.set_default_dtype(torch.float64)
    samples, table = epoch_samples()
    # DistributedSampler sends indices 0,2 to one rank, both without edges.
    samples = [samples[0], samples[1], deepcopy(samples[0]), samples[2]]
    class Probe(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.)

        def training_step(self, batch, batch_idx):
            if self.trainer.world_size == 2 and self.trainer.global_rank == 0:
                assert batch.edge_labels.numel() == 0, "rank zero must exercise empty edges"
            out = self(batch)
            args = dict(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
                edges_pred=out["edge_labels"], edges_ref=batch.edge_labels,
                batch=batch, basis_table=table)
            total, _ = self.loss_fn(**args)
            base, _ = elementwise_mse()(**args)
            grad = torch.autograd.grad(total-base, self.model.offset, retain_graph=True)[0].detach()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(grad)
                grad /= torch.distributed.get_world_size()
            self.physical_gradient = float(grad)
            return super().training_step(batch, batch_idx)
    model = Probe(model_cls=_PredictionModel, basis_table=table, loss=elementwise_mse,
        loss_kwargs={"electron_weight": 2., "mulliken_weight": 3., "energy_weight": 4.})
    model.init_model()
    strategy = DDPStrategy(process_group_backend="gloo", start_method="popen") if devices == 2 else "auto"
    trainer = pl.Trainer(accelerator="cpu", devices=devices, strategy=strategy,
        max_epochs=1, num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=output)
    loader = physical_loader(samples, table, 4)
    if devices == 2:
        sampler = torch.utils.data.DistributedSampler(samples, num_replicas=2,
            rank=int(os.environ.get("LOCAL_RANK", 0)), shuffle=False)
        loader = DataLoader(samples, batch_size=2, sampler=sampler,
            collate_fn=TypeLayoutCollate(Batch.from_data_list, table, symmetric=True))
    trainer.fit(model, train_dataloaders=loader)
    if trainer.is_global_zero:
        values = {k: float(v) for k, v in trainer.callback_metrics.items()
            if "physical_" in k and k.endswith("_epoch")}
        values["gradient"] = model.physical_gradient
        values["world_size"] = trainer.world_size
        (Path(output) / "physical-ddp.json").write_text(json.dumps(values))


def test_physical_ddp_reduces_structures_and_gradients_with_edgeless_rank(tmp_path):
    results = []
    for devices in (1, 2):
        output = tmp_path / str(devices)
        output.mkdir()
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(output), str(devices)],
            cwd=output, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail(stdout + stderr)
        (output / "training.log").write_text(stdout + stderr)
        assert process.returncode == 0, stdout + stderr
        results.append(json.loads((output / "physical-ddp.json").read_text()))
    assert [r.pop("world_size") for r in results] == [1, 2]
    assert results[0] == pytest.approx(results[1])
    assert results[0]["gradient"] == pytest.approx(24.)
    assert results[0]["train_physical_structures_epoch"] == 4


if __name__ == "__main__":
    _ddp_physical(sys.argv[1], int(sys.argv[2]))
