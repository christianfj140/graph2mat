"""Epoch metrics must not depend on how matrix elements are batched."""

import math

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data import metrics as metrics_module
from graph2mat.core.data.metrics import (
    block_type_mse, block_type_mse_per_structure, elementwise_mse, gauge_invariant_mse,
)
from graph2mat.tools.lightning.model import LitBasisMatrixModel


class _PredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        return {"node_labels": batch.node_prediction + self.offset,
                "edge_labels": batch.edge_prediction + self.offset}


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("components", [1, 2])
def test_epoch_rmse_is_independent_of_batch_partition(split, components):
    # Four node errors [1, 1, 1, 3] and four valid edge errors [2, 4, 4, 4].
    # Averaging batch RMSEs instead gives 2 and 3, rather than sqrt(3) and sqrt(13).
    samples = [
        Data(num_nodes=3, point_labels=torch.zeros(3), node_prediction=torch.ones(3),
             edge_labels=torch.tensor([0.0, float("nan")]),
             edge_prediction=torch.tensor([2.0, 99.0])),
        Data(num_nodes=1, point_labels=torch.zeros(1), node_prediction=torch.tensor([3.0]),
             edge_labels=torch.zeros(3), edge_prediction=torch.full((3,), 4.0)),
    ]
    if components == 2:
        for sample in samples:
            for key in ("point_labels", "node_prediction", "edge_labels", "edge_prediction"):
                values = sample[key]
                sample[key] = torch.stack([values, 2 * values], dim=1)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table, loss=block_type_mse)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for batch_size in (1, 2, 1):  # Repeated runs also exercise epoch-state reset.
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]
        assert result[f"{split}_node_rmse"] == pytest.approx(math.sqrt(3 if components == 1 else 7.5))
        assert result[f"{split}_edge_rmse"] == pytest.approx(math.sqrt(13 if components == 1 else 32.5))
        assert result[f"{split}_node_max"] == 3 * components
        assert result[f"{split}_edge_max"] == 4 * components
        assert result[f"{split}_node_mean"] == pytest.approx(1.5 if components == 1 else 2.25)
        assert result[f"{split}_edge_mean"] == pytest.approx(3.5 if components == 1 else 5.25)
        assert result[f"{split}_node_std"] == pytest.approx(1 if components == 1 else math.sqrt(39 / 14))
        assert result[f"{split}_edge_std"] == pytest.approx(1 if components == 1 else math.sqrt(79 / 14))
        assert result[f"{split}_loss"] == pytest.approx(16 if components == 1 else 40)
        if components == 2:
            assert result[f"{split}_node_rmse_spin0"] == pytest.approx(math.sqrt(3))
            assert result[f"{split}_node_rmse_spin1"] == pytest.approx(math.sqrt(12))
            assert result[f"{split}_edge_rmse_spin1"] == pytest.approx(math.sqrt(52))


def test_epoch_gauge_rmse_counts_elements_after_alignment():
    # S_node=1, S_edge=0: any node shift is removed. Weighted edge errors are
    # [1, 1, 1, 3], so sum(w*dH^2)=24 and sum(w*elements)=12, giving sqrt(2).
    samples = []
    for n, error, shift in ((3, 1.0, 10.0), (1, 3.0, -20.0)):
        samples.append(Data(
            num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
            point_labels=torch.tensor([[0.0, 1.0]]).repeat(n, 1),
            node_prediction=torch.full((n,), shift),
            edge_labels=torch.zeros(n, 2), edge_prediction=torch.full((n,), error),
            edge_types=torch.zeros(n, dtype=torch.long),
            edge_index=torch.arange(n).repeat(2, 1),
        ))
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=gauge_invariant_mse, loss_kwargs={"edge_weight": 2})
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for batch_size in (1, 2):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=batch_size),
                                  verbose=False)[0]
        assert result["val_gauge_h_rmse"] == pytest.approx(math.sqrt(2))
        assert result["val_overlap_rmse"] == 0


def test_training_rmse_resets_each_epoch():
    class TrainModel(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    class RecordEpoch(pl.Callback):
        def __init__(self):
            self.values = []
            self.losses = []

        def on_train_epoch_end(self, trainer, pl_module):
            self.values.append(float(trainer.callback_metrics["train_node_rmse_epoch"]))
            self.losses.append(float(trainer.callback_metrics["train_loss_epoch"]))

    samples = [Data(num_nodes=n, point_labels=torch.zeros(n), node_prediction=torch.full((n,), error),
                    edge_labels=torch.zeros(2), edge_prediction=torch.zeros(2))
               for n, error in ((3, 1.0), (1, 3.0))]
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    model = TrainModel(model_cls=_PredictionModel, basis_table=table, loss=elementwise_mse)
    model.init_model()
    recorder = RecordEpoch()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=1))
    assert recorder.values == pytest.approx([math.sqrt(3), math.sqrt(3)])
    assert recorder.losses == pytest.approx([1.5, 1.5])


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("loss_name,loss_kwargs,expected", [
    ("block_type_mse", {}, 86 / 5),
    ("elementwise_mse", {}, 73 / 9),
    ("elementwise_mse", {"edge_weight": 2}, 125 / 13),
    ("node_mse", {}, 21 / 5),
    ("edge_mse", {}, 13),
    ("block_type_mae", {}, 53 / 10),
    ("elementwise_mae", {}, 23 / 9),
])
def test_epoch_loss_uses_its_objective_denominators(split, loss_name, loss_kwargs, expected):
    # Node errors [1,1,1,3,3], valid edge errors [2,4,4,4]: different counts
    # in each batch. Global sums are 21/52 squared and 9/14 absolute.
    samples = [
        Data(num_nodes=3, point_labels=torch.zeros(3), node_prediction=torch.ones(3),
             edge_labels=torch.tensor([0., float("nan")]), edge_prediction=torch.tensor([2., 99.])),
        Data(num_nodes=2, point_labels=torch.zeros(2), node_prediction=torch.full((2,), 3.),
             edge_labels=torch.zeros(3), edge_prediction=torch.full((3,), 4.)),
    ]
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=getattr(metrics_module, loss_name), loss_kwargs=loss_kwargs)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for batch_size in (1, 2, 1):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(expected)
        if split == "val":
            assert result["hp_metric"] == pytest.approx(expected)


@pytest.mark.parametrize("missing_edges", ["none", "first", "all"])
def test_epoch_structure_weighted_rmse_preserves_equal_structure_weight(missing_edges):
    samples = [Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
                    point_labels=torch.zeros(n), node_prediction=torch.full((n,), error),
                    edge_labels=torch.zeros(n), edge_prediction=torch.zeros(n),
                    edge_types=torch.zeros(n, dtype=torch.long), edge_index=torch.arange(n).repeat(2, 1))
               for n, error in ((3, 1.0), (1, 3.0))]
    if missing_edges != "none":
        for sample in samples[:2 if missing_edges == "all" else 1]:
            sample.edge_labels = torch.empty(0)
            sample.edge_prediction = torch.empty(0)
            sample.edge_types = torch.empty(0, dtype=torch.long)
            sample.edge_index = torch.empty((2, 0), dtype=torch.long)
    if missing_edges == "first":
        samples[1].edge_prediction.fill_(3)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis="0e", basis_convention="spherical")])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=block_type_mse_per_structure)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for batch_size in (1, 2):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=batch_size),
                                  verbose=False)[0]
        assert result["val_node_rmse"] == pytest.approx(math.sqrt(5))
        assert result["val_edge_rmse"] == pytest.approx(3 if missing_edges == "first" else 0)
