"""Opt-in training preserves physical matrix MSE, gradients and epoch reductions."""

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.data.coefficient_targets import (
    COEFFICIENT_TARGETS_ATTR, CoefficientMatrixMSEProjector, CoefficientMSEUnsupported,
)
from graph2mat.core.data import metrics
from graph2mat.tools.lightning.data import TypeLayoutCollate
from graph2mat.tools.lightning.model import _EpochMatrixLoss
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.callbacks import MatrixWriter, SamplewiseMetricsLogger


def training_inputs(components=1, grouping="point_type", mode="single", dtype=torch.float64,
                    symmetric=True, **kwargs):
    torch.manual_seed(13)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="1x0e+1x1o+1x2e"),
                                PointBasis("B", R=2, basis="1x0e+1x1o+1x3o")])
    model = LitMACEMatrixModel(
        basis_table=table, symmetric_matrix=symmetric, hidden_irreps="2x0e+2x1o+2x2e",
        edge_hidden_irreps="2x0e+2x1o+2x2e", num_bessel=2, max_ell=2,
        num_interactions=2, correlation=1, n_matrix_components=components,
        basis_grouping=grouping, readout_per_interaction=mode != "single",
        readout_interaction_aggregation="mean" if mode == "mean" else "sum",
        coefficient_training=kwargs.pop("coefficient_training", True),
        loss_kwargs=kwargs.pop("loss_kwargs", {"edge_weight": 2.}), **kwargs,
    ).to(dtype=dtype)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric, sub_point_matrix=False)
    samples = []
    for n in (3, 2):
        config = BasisConfiguration(
            point_types=["A", "B", "A"][:n], positions=np.array([[0., 0., 0.], [1., .2, .1], [2., .1, 0.]])[:n],
            basis=table, cell=np.diag([4., 20., 20.]), pbc=(True, False, False),
        )
        data = TorchBasisMatrixData.from_config(config, data_processor=processor)
        nodes = int(table.point_block_size[data.point_types.numpy()].sum())
        edges = int(table.edge_block_size[data.edge_types.numpy()[::2] if symmetric else abs(data.edge_types.numpy())].sum())
        shape = lambda rows: (rows,) if components == 1 else (rows, components)
        data.point_labels = torch.randn(shape(nodes), dtype=dtype)
        data.edge_labels = torch.randn(shape(edges), dtype=dtype)
        for key, value in data.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                data[key] = value.to(dtype=dtype)
        samples.append(data)
    projector = CoefficientMatrixMSEProjector(model.model.matrix_readouts)
    batch = TypeLayoutCollate(Batch.from_data_list, table, projector, symmetric)(samples)
    return model, batch, samples


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("grouping,mode", [("point_type", "single"), ("max", "sum"), ("basis_shape", "mean")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("symmetric", [False, True])
def test_coefficient_matrix_training_matches_dense_loss_and_parameter_gradients(monkeypatch, components, grouping, mode, dtype, symmetric):
    model, batch, _ = training_inputs(components, grouping, mode, dtype, symmetric)
    out = model(batch)
    expected = ((out["node_labels"] - batch.point_labels).square().sum()
                + 2 * (out["edge_labels"] - batch.edge_labels).square().sum()
                ) / (batch.point_labels.numel() + 2 * batch.edge_labels.numel())
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    state_keys = set(model.state_dict())

    def forbidden(*args, **kwargs):
        raise AssertionError("matrix reconstruction entered certified training_step")

    readouts = model.model.matrix_readouts
    for readout in readouts if isinstance(readouts, torch.nn.ModuleList) else [readouts]:
        for operation in [*readout.self_interactions, *readout.interactions.values()]:
            monkeypatch.setattr(operation, "coefficients_to_block", forbidden)
    result = model.training_step(batch, 0)
    assert set(result) == {"loss"}
    assert set(model.state_dict()) == state_keys
    tolerance = 5e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(result["loss"], expected, rtol=tolerance, atol=tolerance)
    for actual, reference in zip(torch.autograd.grad(result["loss"], parameters, allow_unused=True), gradients):
        assert (actual is None) == (reference is None)
        if actual is not None:
            torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)


def test_coefficient_matrix_training_precomputed_epoch_terms_invalidate_cache_and_reset():
    metric = _EpochMatrixLoss()
    empty = torch.empty(0)
    first = {"node_sse": torch.tensor(3.), "edge_sse": torch.tensor(8.), "node_count": 3, "edge_count": 2}
    second = {"node_sse": torch.tensor(9.), "edge_sse": torch.tensor(48.), "node_count": 1, "edge_count": 3}
    metric(empty, empty, 2, True, 1., 2., matrix_stats=first)
    assert float(metric.compute()) == pytest.approx(19 / 7)
    metric(empty, empty, 2, True, 1., 2., matrix_stats=second)
    assert float(metric.compute()) == pytest.approx(124 / 14)
    assert metric.totals.tolist() == [12., 56.]
    assert metric.counts.tolist() == [4., 5.]
    metric.reset()
    metric(empty, empty, 2, True, 1., 2., matrix_stats=first)
    assert float(metric.compute()) == pytest.approx(19 / 7)


@pytest.mark.parametrize("broken", ["missing", "effective_q_changed"])
def test_coefficient_matrix_training_requires_a_transpose_map_for_effective_q(broken):
    from graph2mat.core.data.tests.test_coefficient_orientation import grouped_readout

    readout = grouped_readout(["1x0e+1x1o"], "point_type", True, 1)
    operation = next(iter(readout.interactions.values()))
    assert operation.symm_transpose
    if broken == "missing":
        operation._coefficient_transpose_index = None
    else:
        index = operation._coefficient_transpose_index
        row = int((index != torch.arange(len(index))).nonzero()[0])
        operation.change_of_basis[row] *= 2.
    with pytest.raises(CoefficientMSEUnsupported, match="transpose"):
        CoefficientMatrixMSEProjector(readout)


def test_coefficient_matrix_training_requires_fixed_q_not_a_trainable_change_of_basis():
    from graph2mat.core.data.tests.test_coefficient_orientation import grouped_readout

    readout = grouped_readout(["1x0e"], "point_type", True, 1)
    readout.self_interactions[0].change_of_basis.requires_grad_(True)
    with pytest.raises(CoefficientMSEUnsupported, match="fixed Q"):
        CoefficientMatrixMSEProjector(readout)


@pytest.mark.parametrize("broken", ["trainable", "missing_map", "mismatched_map"])
def test_coefficient_matrix_training_requires_fixed_q_in_every_interaction_readout(broken):
    from graph2mat.core.data.tests.test_coefficient_orientation import grouped_readout

    readouts = torch.nn.ModuleList([grouped_readout(["1x0e"], "point_type", True, 1) for _ in range(2)])
    if broken == "trainable":
        readouts[1].self_interactions[0].change_of_basis.requires_grad_(True)
    else:
        operation = next(iter(readouts[1].interactions.values()))
        if broken == "missing_map":
            operation._coefficient_transpose_index = None
        else:
            operation._coefficient_transpose_sign *= -1
    with pytest.raises(CoefficientMSEUnsupported, match="fixed Q|transpose"):
        CoefficientMatrixMSEProjector(readouts)


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_coefficient_matrix_training_declines_uncertified_edge_weights_before_backbone(weight):
    model, batch, _ = training_inputs(loss_kwargs={"edge_weight": weight})
    with pytest.warns(UserWarning, match="finite nonnegative edge_weight"):
        assert not model._can_train_coefficients(batch)


@pytest.mark.parametrize("case", ["mae", "huber", "gauge", "partial_nan", "no_targets", "checkpoint",
                                  "long_range", "verbose", "writer", "custom_callback", "custom_transfer"])
def test_coefficient_matrix_training_fallback_runs_one_real_matrix_forward(monkeypatch, tmp_path, case):
    kwargs = {}
    if case in ("mae", "huber", "gauge"):
        kwargs["loss"] = {"mae": metrics.elementwise_mae, "huber": metrics.block_type_huber,
                          "gauge": metrics.gauge_invariant_mse}[case]
    if case == "huber":
        kwargs["loss_kwargs"] = {"beta": 1.}
    if case == "verbose":
        kwargs["loss_kwargs"] = {"edge_weight": 2., "log_verbose": True}
    if case == "checkpoint":
        kwargs["gradient_checkpointing"] = True
    if case == "long_range":
        kwargs["long_range"] = True
    model, batch, samples = training_inputs(components=2, **kwargs)
    if case == "partial_nan":
        samples[0].point_labels[1, 0] = float("nan")
        projector = CoefficientMatrixMSEProjector(model.model.matrix_readouts)
        batch = TypeLayoutCollate(Batch.from_data_list, model.basis_table, projector)(samples)
    if case == "no_targets":
        object.__delattr__(batch, COEFFICIENT_TARGETS_ATTR)
    if case in ("writer", "custom_callback"):
        class NeedsMatrices(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                assert "node_labels" in outputs
        callback = (MatrixWriter(str(tmp_path / "prediction.DM"), splits=["train"])
                    if case == "writer" else NeedsMatrices())
        trainer = pl.Trainer(callbacks=[callback], accelerator="cpu", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.strategy.connect(model)
        model._trainer = trainer
        # Only logging needs a running Trainer loop; real fit tests cover that boundary.
        monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    if case == "custom_transfer":
        class Transformed(type(model)):
            def on_after_batch_transfer(self, data, dataloader_idx):
                data.point_labels += 1
                return data
        model.__class__ = Transformed
        batch = model.on_after_batch_transfer(batch, 0)
    out = model(batch)
    expected, _ = model.loss_fn(nodes_pred=out["node_labels"], edges_pred=out["edge_labels"],
        nodes_ref=batch.point_labels, edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table,
        out=out, model=model.model)
    calls = []
    forward = model.model.forward
    def spy(*args, **kwargs):
        calls.append(kwargs.get("coefficients_only", False))
        return forward(*args, **kwargs)
    monkeypatch.setattr(model.model, "forward", spy)
    with pytest.warns(UserWarning, match="coefficient_training uses matrix fallback"):
        result = model.training_step(batch, 0)
    assert calls == [False]
    assert set(result) == {"loss", "node_labels", "edge_labels"}
    torch.testing.assert_close(result["loss"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("readout_fp32", [False, True])
def test_coefficient_matrix_training_amp_selects_route_before_backbone(readout_fp32):
    model, batch, _ = training_inputs(dtype=torch.float32, readout_fp32=readout_fp32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if readout_fp32:
            assert model._can_train_coefficients(batch)
        else:
            with pytest.warns(UserWarning, match="autocast requires readout_fp32"):
                assert not model._can_train_coefficients(batch)


def test_coefficient_matrix_training_default_flag_off_and_strict_state_loading():
    model, batch, _ = training_inputs(coefficient_training=False)
    assert model.hparams.coefficient_training is False
    result = model.training_step(batch, 0)
    assert set(result) == {"loss", "node_labels", "edge_labels"}
    candidate, _, _ = training_inputs()
    candidate.load_state_dict(model.state_dict(), strict=True)
    actual, expected = candidate(batch), model(batch)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_coefficient_matrix_training_default_preserves_legacy_logging_overrides(enabled):
    model, batch, _ = training_inputs(coefficient_training=enabled)
    class LegacyLogging(type(model)):
        def _log_loss(self, split, loss, out, batch, stats=None):
            self.last_loss = loss.detach()
        def _log_stats(self, split, stats, batch, out):
            self.last_rmse = stats["node_rmse"]
    model.__class__ = LegacyLogging
    if enabled:
        with pytest.warns(UserWarning, match="custom _log_loss"):
            result = model.training_step(batch, 0)
    else:
        result = model.training_step(batch, 0)
    torch.testing.assert_close(model.last_loss, result["loss"])
    assert torch.isfinite(model.last_rmse)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("components", [1, 2])
def test_coefficient_matrix_training_fit_prepares_targets_and_reduces_physical_epoch_stats(monkeypatch, tmp_path, workers, components):
    from graph2mat.tools.lightning import MatrixDataModule

    model, _, samples = training_inputs(components, "max", "sum")
    total_node, total_edge = 0., 0.
    for sample in samples:
        out = model(sample)
        total_node += float((out["node_labels"] - sample.point_labels).square().sum().detach())
        total_edge += float((out["edge_labels"] - sample.edge_labels).square().sum().detach())
    node_count = sum(data.point_labels.numel() for data in samples)
    edge_count = sum(data.edge_labels.numel() for data in samples)
    expected = (total_node + 2 * total_edge) / (node_count + 2 * edge_count)

    class Probe(type(model)):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)
        def training_step(self, batch, batch_idx):
            assert isinstance(getattr(batch, COEFFICIENT_TARGETS_ATTR, None), dict)
            with monkeypatch.context() as local:
                readouts = self.model.matrix_readouts
                for readout in readouts if isinstance(readouts, torch.nn.ModuleList) else [readouts]:
                    for operation in [*readout.self_interactions, *readout.interactions.values()]:
                        def forbidden(*args, **kwargs):
                            raise AssertionError("fit reconstructed a matrix during training")
                        local.setattr(operation, "coefficients_to_block", forbidden)
                result = super().training_step(batch, batch_idx)
            assert set(result) == {"loss"}
            return result

    class Recorder(pl.Callback):
        def __init__(self):
            self.epochs = []
        def on_train_epoch_end(self, trainer, pl_module):
            self.epochs.append({name: float(trainer.callback_metrics[f"train_{name}_epoch"])
                                for name in ("loss", "node_rmse", "edge_rmse")})

    model.__class__ = Probe
    module = MatrixDataModule(basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=components, loader_threads=workers, train_runs=[], val_runs=[], batch_size=1)
    module.setup("fit")
    module.train_dataset, module.val_dataset = samples, samples
    recorder = Recorder()
    metric_file = tmp_path / "validation.csv"
    callbacks = [recorder, SamplewiseMetricsLogger(metrics=[metrics.elementwise_mse], splits=["val"],
        output_file=str(metric_file)), MatrixWriter(str(tmp_path / "unused.DM"), splits=["test"])]
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, logger=False, callbacks=callbacks,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=module)
    assert trainer.global_step == 4
    assert len(recorder.epochs) == 2
    for epoch in recorder.epochs:
        assert epoch == pytest.approx({"loss": expected, "node_rmse": (total_node / node_count) ** .5,
                                     "edge_rmse": (total_edge / edge_count) ** .5}, rel=1e-12, abs=1e-12)
    assert metric_file.read_text().strip()
    assert not (tmp_path / "unused.DM").exists()
    assert module._coefficient_mse_projector is None
    module.teardown(None)


def data_module_for(model, samples):
    from graph2mat.tools.lightning import MatrixDataModule

    module = MatrixDataModule(basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=model.hparams.n_matrix_components, loader_threads=0,
        train_runs=[], val_runs=[], batch_size=len(samples))
    module.setup("fit")
    module.train_dataset, module.val_dataset = samples, []
    return module


def cpu_trainer(**kwargs):
    return pl.Trainer(accelerator="cpu", devices=1, max_epochs=kwargs.pop("max_epochs", 1), logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, **kwargs)


def test_coefficient_matrix_training_restore_after_setup_uses_explicit_matrix_fallback():
    from pytorch_lightning.strategies import SingleDeviceStrategy

    class LateRestore(SingleDeviceStrategy):
        @property
        def restore_checkpoint_after_setup(self):
            return True

    model, _, samples = training_inputs()
    module = data_module_for(model, samples)
    trainer = cpu_trainer(strategy=LateRestore(device=torch.device("cpu")))
    with pytest.warns(UserWarning, match="restore_checkpoint_after_setup"):
        trainer.fit(model, datamodule=module)
    assert module._coefficient_mse_projector is None
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train_loss_epoch"])
    module.teardown(None)


def test_coefficient_matrix_training_resume_snapshots_restored_effective_q_and_legacy_flag(tmp_path):
    old, _, samples = training_inputs(coefficient_training=False, optim_lr=0.)
    module = data_module_for(old, samples)
    trainer = cpu_trainer()
    trainer.fit(old, datamodule=module)
    checkpoint_path = tmp_path / "legacy.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["hyper_parameters"].pop("coefficient_training")
    torch.save(checkpoint, checkpoint_path)
    legacy = LitMACEMatrixModel.load_from_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
    assert legacy.coefficient_training is False
    old_q = old.model.matrix_readouts.self_interactions[0].change_of_basis.clone()
    plain = Batch.from_data_list(samples)
    out = old(plain)
    expected, _ = old.loss_fn(nodes_pred=out["node_labels"], nodes_ref=plain.point_labels,
                             edges_pred=out["edge_labels"], edges_ref=plain.edge_labels)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        fresh, _, _ = training_inputs(optim_lr=0.)
    finally:
        torch.set_default_dtype(previous)
    assert not torch.equal(fresh.model.matrix_readouts.self_interactions[0].change_of_basis, old_q)

    class ResumeProbe(type(fresh)):
        def training_step(self, batch, batch_idx):
            torch.testing.assert_close(self.model.matrix_readouts.self_interactions[0].change_of_basis,
                                       old_q, rtol=0, atol=0)
            result = super().training_step(batch, batch_idx)
            assert set(result) == {"loss"}
            torch.testing.assert_close(result["loss"], expected, rtol=1e-12, atol=1e-12)
            return result
    fresh.__class__ = ResumeProbe
    resumed_module = data_module_for(fresh, samples)
    resumed = cpu_trainer(max_epochs=2)
    resumed.fit(fresh, datamodule=resumed_module, ckpt_path=checkpoint_path)
    assert resumed.global_step == 2
    assert resumed_module._coefficient_mse_projector is None
    module.teardown(None)
    resumed_module.teardown(None)


@pytest.mark.parametrize("mse_first", [False, True])
def test_coefficient_matrix_training_stage_changes_recheck_loss_and_use_one_forward(mse_first):
    stages = [{"name": "mse", "max_epochs": 1},
              {"name": "huber", "max_epochs": 1, "loss": "block_type_huber", "loss_kwargs": {"beta": 1.}}]
    if not mse_first:
        stages.reverse()
    model, _, samples = training_inputs(training_stages=stages, optim_lr=0.)
    routes = []
    class StageProbe(type(model)):
        def training_step(self, batch, batch_idx):
            result = super().training_step(batch, batch_idx)
            routes.append(set(result))
            return result
    model.__class__ = StageProbe
    module = data_module_for(model, samples)
    with pytest.warns(UserWarning, match="objective requires matrix"):
        cpu_trainer(max_epochs=2).fit(model, datamodule=module)
    expected = [{"loss"}, {"loss", "node_labels", "edge_labels"}]
    assert routes == (expected if mse_first else expected[::-1])
    module.teardown(None)


@pytest.mark.parametrize("grouping", ["point_type", "max"])
def test_coefficient_matrix_training_all_ghost_has_connected_device_zero(grouping):
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e"), PointBasis("ghost", R=2, basis=[])])
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True, coefficient_training=True,
        basis_grouping=grouping, hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        max_ell=1, num_bessel=2, n_matrix_components=2).double()
    config = BasisConfiguration(point_types=["ghost"], positions=np.zeros((1, 3)), basis=table,
                                cell=np.eye(3) * 20, pbc=(False, False, False))
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False)
    data = TorchBasisMatrixData.from_config(config, data_processor=processor)
    for key, value in data.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.double()
    data.point_labels = torch.empty(0, 2, dtype=torch.float64)
    data.edge_labels = torch.empty(0, 2, dtype=torch.float64)
    batch = TypeLayoutCollate(Batch.from_data_list, table,
                             CoefficientMatrixMSEProjector(model.model.matrix_readouts))([data])
    result = model.training_step(batch, 0)
    assert set(result) == {"loss"} and result["loss"].device == model.device
    assert result["loss"].item() == 0 and result["loss"].requires_grad
    result["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


def test_coefficient_matrix_training_wholly_unobserved_blocks_have_zero_not_nan_gradients():
    model, _, samples = training_inputs(components=2)
    for sample in samples:
        sample.point_labels.fill_(float("nan"))
        sample.edge_labels.fill_(float("nan"))
    batch = TypeLayoutCollate(Batch.from_data_list, model.basis_table,
                             CoefficientMatrixMSEProjector(model.model.matrix_readouts))(samples)
    result = model.training_step(batch, 0)
    assert set(result) == {"loss"} and result["loss"].item() == 0
    result["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
