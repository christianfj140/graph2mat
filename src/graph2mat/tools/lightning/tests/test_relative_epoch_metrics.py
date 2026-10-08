"""Relative selections and mixed absolute terms need separate epoch moments."""

import math

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data import metrics
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


RELATIVE_LOSSES = ["block_type_mape", "block_type_mapemaemix", "block_type_mapemsemix", "block_type_mapestdmix"]


def relative_samples(components=1, missing_nodes=False):
    samples = [
        Data(num_nodes=4, point_labels=torch.tensor([0., 2., 2., float("nan")], dtype=torch.float64),
             node_prediction=torch.tensor([1., 3., 3., 99.], dtype=torch.float64),
             edge_labels=torch.tensor([0., 2., float("nan")], dtype=torch.float64),
             edge_prediction=torch.tensor([2., 4., 99.], dtype=torch.float64)),
        Data(num_nodes=1, point_labels=torch.ones(1, dtype=torch.float64), node_prediction=torch.full((1,), 4., dtype=torch.float64),
             edge_labels=torch.ones(3, dtype=torch.float64), edge_prediction=torch.full((3,), 5., dtype=torch.float64)),
    ]
    for sample in samples:
        if missing_nodes:
            sample.point_labels.fill_(float("nan"))
        if components == 2:
            for key in ("point_labels", "node_prediction", "edge_labels", "edge_prediction"):
                sample[key] = torch.stack([sample[key], 2 * sample[key]], dim=1)
    return samples


def relative_model(loss_name):
    model = LitBasisMatrixModel(model_cls=_PredictionModel,
                               basis_table=BasisTableWithEdges([PointBasis("A", R=2, basis=[1])]),
                               loss=getattr(metrics, loss_name))
    model.init_model()
    return model


def relative_expected(loss_name, components=1, missing_nodes=False):
    # Relative nodes [.5,.5,3] and edges [1,4,4,4]; spin repeats those errors.
    mean = (0. if missing_nodes else 4 / 3) + 13 / 4
    if loss_name == "block_type_mapemaemix":
        return mean + 100 * ((0. if missing_nodes else 1.5) + 16 / 5) * (1 if components == 1 else 1.5)
    if loss_name == "block_type_mapemsemix":
        return mean + 10000 * ((0. if missing_nodes else 3.) + 56 / 5) * (1 if components == 1 else 2.5)
    if loss_name == "block_type_mapestdmix":
        node_std = 0. if missing_nodes else math.sqrt(25 / 12 if components == 1 else 5 / 3)
        edge_std = 1.5 if components == 1 else math.sqrt(27 / 14)
        return mean + node_std + edge_std
    return mean


@pytest.mark.parametrize("loss_name", RELATIVE_LOSSES)
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("missing_nodes", [False, True])
@pytest.mark.parametrize("split", ["val", "test"])
def test_relative_epoch_objectives_use_global_selection_and_mixed_term_counts(loss_name, components, missing_nodes, split):
    model = relative_model(loss_name)
    samples = relative_samples(components, missing_nodes)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for size in (1, 2, 1):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(relative_expected(loss_name, components, missing_nodes))
        if split == "val":
            assert result["hp_metric"] == pytest.approx(relative_expected(loss_name, components, missing_nodes))


@pytest.mark.parametrize("loss_name", RELATIVE_LOSSES)
def test_relative_step_literal_value_and_gradients_are_preserved(loss_name):
    nodes = torch.tensor([3., 6.], dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([2., 4.], dtype=torch.float64, requires_grad=True)
    loss, _ = getattr(metrics, loss_name)()(nodes_pred=nodes, nodes_ref=torch.tensor([2., 4.], dtype=torch.float64),
                                          edges_pred=edges, edges_ref=torch.ones(2, dtype=torch.float64))
    want = {"block_type_mape": 2.5, "block_type_mapemaemix": 352.5,
            "block_type_mapemsemix": 75002.5, "block_type_mapestdmix": 2.5 + math.sqrt(2)}[loss_name]
    assert float(loss) == pytest.approx(want)
    loss.backward()
    if loss_name == "block_type_mapemaemix":
        ng, eg = [50.25, 50.125], [50.5, 50.5]
    elif loss_name == "block_type_mapemsemix":
        ng, eg = [10000.25, 20000.125], [10000.5, 30000.5]
    elif loss_name == "block_type_mapestdmix":
        ng, eg = [.25, .125], [.5 - 1 / math.sqrt(2), .5 + 1 / math.sqrt(2)]
    else:
        ng, eg = [.25, .125], [.5, .5]
    torch.testing.assert_close(nodes.grad, torch.tensor(ng, dtype=torch.float64))
    torch.testing.assert_close(edges.grad, torch.tensor(eg, dtype=torch.float64))


@pytest.mark.parametrize("loss_name", RELATIVE_LOSSES)
def test_relative_training_resets_global_moments_between_epochs_and_splits(loss_name):
    class TrainModel(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    class Recorder(pl.Callback):
        def __init__(self):
            self.losses = []

        def on_after_backward(self, trainer, pl_module):
            assert pl_module.model.offset.grad is not None
            assert torch.isfinite(pl_module.model.offset.grad)

        def on_train_epoch_end(self, trainer, pl_module):
            self.losses.append(float(trainer.callback_metrics["train_loss_epoch"]))

    model = TrainModel(model_cls=_PredictionModel, basis_table=relative_model(loss_name).basis_table,
                       loss=getattr(metrics, loss_name))
    model.init_model()
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    samples = relative_samples()
    trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=1),
                val_dataloaders=DataLoader(samples, batch_size=2))
    assert recorder.losses == pytest.approx([relative_expected(loss_name)] * 2)
    assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(relative_expected(loss_name))


@pytest.mark.parametrize("loss_name", RELATIVE_LOSSES)
@pytest.mark.parametrize("missing", [False, True])
def test_empty_relative_selection_is_zero_without_discarding_absolute_terms(loss_name, missing):
    samples = relative_samples()
    for sample in samples:
        sample.point_labels.fill_(float("nan") if missing else 0.)
        sample.edge_labels.fill_(float("nan") if missing else 0.)
        sample.node_prediction.fill_(2.)
        sample.edge_prediction.fill_(3.)
    want = 0. if missing else 500. if loss_name == "block_type_mapemaemix" else 130000. if loss_name == "block_type_mapemsemix" else 0.
    model = relative_model(loss_name)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for size in (1, 2, 1):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(want)


def test_relative_objective_changes_recreate_only_required_variance_state():
    model = relative_model("block_type_mape")
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for name in ("block_type_mape", "block_type_mapestdmix", "block_type_mapemsemix", "block_type_mapestdmix", "block_type_mapemaemix"):
        model.loss_fn = getattr(metrics, name)()
        result = trainer.validate(model, dataloaders=DataLoader(relative_samples(), batch_size=1), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(relative_expected(name))
