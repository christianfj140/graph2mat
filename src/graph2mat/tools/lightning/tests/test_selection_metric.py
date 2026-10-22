"""Roadmap 1.15: a loss-independent validation error and opt-in model selection."""

import math

import pytest
import pytorch_lightning as pl
import torch
from jsonargparse import Namespace
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data import metrics as metrics_module
from graph2mat.tools.lightning.cli import _apply_selection_metric
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel

TABLE = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])


def _samples(components):
    # Node errors [1, 1, 1, 3], valid edge errors [2, 4, 4, 4]; the NaN edge is unobserved.
    samples = [
        Data(num_nodes=3, point_types=torch.zeros(3, dtype=torch.long),
             point_labels=torch.zeros(3), node_prediction=torch.ones(3),
             edge_labels=torch.tensor([0.0, float("nan")]), edge_prediction=torch.tensor([2.0, 99.0]),
             edge_types=torch.zeros(2, dtype=torch.long), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(num_nodes=1, point_types=torch.zeros(1, dtype=torch.long),
             point_labels=torch.zeros(1), node_prediction=torch.tensor([3.0]),
             edge_labels=torch.zeros(3), edge_prediction=torch.full((3,), 4.0),
             edge_types=torch.zeros(3, dtype=torch.long), edge_index=torch.zeros((2, 3), dtype=torch.long)),
    ]
    if components == 2:
        for sample in samples:
            for key in ("point_labels", "node_prediction", "edge_labels", "edge_prediction"):
                sample[key] = torch.stack([sample[key], 2 * sample[key]], dim=1)
    return samples


def _evaluate(model, samples, split="val", batch_size=1):
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    return evaluate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]


def _model(loss=metrics_module.block_type_mse, **kwargs):
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=TABLE, loss=loss, **kwargs)
    model.init_model()
    return model


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("loss,loss_kwargs", [
    (metrics_module.block_type_mse, None),
    (metrics_module.elementwise_mae, None),
    (metrics_module.elementwise_mse, {"edge_weight": 3.}),
    (metrics_module.block_normalized_huber, {"beta": 1.}),
])
def test_matrix_rmse_does_not_depend_on_loss_or_batching(split, components, loss, loss_kwargs):
    # Squared errors: nodes 1+1+1+9, edges 4+16+16+16 = 64 over 8 elements; the second
    # component doubles every error, adding 4*64 over 8 more elements.
    if components == 2 and loss is metrics_module.block_normalized_huber:
        pytest.skip("block_normalized_huber accepts single-component H labels only")
    want = math.sqrt(8) if components == 1 else math.sqrt(320 / 16)
    model = _model(loss, loss_kwargs=loss_kwargs)
    for batch_size in (1, 2, 1):
        result = _evaluate(model, _samples(components), split, batch_size)
        assert result[f"{split}_matrix_rmse"] == pytest.approx(want)


def test_matrix_rmse_compares_only_predicted_components_without_gauge_alignment():
    # H-only prediction with (H, S) labels: S is a label, never a prediction. The node shifts
    # 10 and -20 enter the absolute RMSE: (3*100 + 400 + 3*1 + 9) / 8 = 89.
    samples = []
    for n, error, shift in ((3, 1.0, 10.0), (1, 3.0, -20.0)):
        samples.append(Data(
            num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
            point_labels=torch.tensor([[0.0, 1.0]]).repeat(n, 1), node_prediction=torch.full((n,), shift),
            edge_labels=torch.zeros(n, 2), edge_prediction=torch.full((n,), error),
            edge_types=torch.zeros(n, dtype=torch.long), edge_index=torch.arange(n).repeat(2, 1),
        ))
    result = _evaluate(_model(metrics_module.gauge_invariant_mse), samples, batch_size=2)
    assert result["val_matrix_rmse"] == pytest.approx(math.sqrt(89))
    assert result["val_gauge_h_rmse"] == pytest.approx(math.sqrt(1.5))


def test_selection_metric_drives_hp_metric_and_default_is_unchanged():
    default = _evaluate(_model(), _samples(1))
    assert default["hp_metric"] == pytest.approx(default["val_loss"])
    for name in ("val_matrix_rmse", "val_edge_rmse"):
        result = _evaluate(_model(selection_metric=name), _samples(1), batch_size=2)
        assert result["hp_metric"] == pytest.approx(result[name])
        assert result["hp_metric"] != pytest.approx(result["val_loss"])
    assert _model(selection_metric="val_loss").selection_metric is None


def test_unknown_selection_metric_fails_before_training_continues():
    with pytest.raises(ValueError, match="selection_metric 'val_nope'.*val_matrix_rmse"):
        _evaluate(_model(selection_metric="val_nope"), _samples(1))


@pytest.mark.parametrize("name", ["val_node_rmse", "val_edge_rmse"])
def test_selection_rejects_a_metric_that_disappears_after_a_stage_change(name):
    model = _model(selection_metric=name, training_stages=[
        {"name": "mse", "max_epochs": 1},
        {"name": "mae", "max_epochs": 1, "loss": "block_type_mae"},
    ])
    model._apply_training_stage_for_epoch(0)
    result = _evaluate(model, _samples(1))
    assert result["hp_metric"] == pytest.approx(result[name])
    model._apply_training_stage_for_epoch(1)
    with pytest.raises(ValueError, match=f"selection_metric '{name}'.*current validation"):
        _evaluate(model, _samples(1))


def test_cli_moves_only_the_default_best_checkpoint_monitor():
    def checkpoint(monitor):
        return Namespace(class_path="ModelCheckpoint", init_args=Namespace(monitor=monitor))

    callbacks = [checkpoint("step"), checkpoint("val_loss"),
                 Namespace(class_path="pytorch_lightning.callbacks.ModelCheckpoint",
                           init_args=Namespace(monitor="val_node_rmse"))]
    config = Namespace(model=Namespace(selection_metric="val_matrix_rmse"),
                       trainer=Namespace(callbacks=callbacks))
    _apply_selection_metric(config)
    assert [c.init_args.monitor for c in callbacks] == ["step", "val_matrix_rmse", "val_node_rmse"]

    callbacks = [checkpoint("val_loss")]
    _apply_selection_metric(Namespace(model=Namespace(selection_metric=None),
                                      trainer=Namespace(callbacks=callbacks)))
    assert callbacks[0].init_args.monitor == "val_loss"
