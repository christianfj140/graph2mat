"""Pair means count observed partners; the base objective counts elements."""

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data import metrics
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


def pair_samples(components=1):
    samples = []
    for n, error in ((1, 1.), (2, 3.), (3, 2.)):
        for sign in (1., -1.):
            ref = torch.zeros(n, dtype=torch.float64)
            pred = torch.full((n,), sign * error, dtype=torch.float64)
            edge_ref = torch.empty(0, dtype=torch.float64)
            if components == 2:
                ref, pred, edge_ref = [torch.stack([x, 2 * x], dim=1) for x in (ref, pred, edge_ref)]
            positions = torch.zeros(n, 3, dtype=torch.float64)
            positions[:, 0] = torch.arange(n) * .7 + .01 * sign
            samples.append(Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
                                positions=positions, point_labels=ref, node_prediction=pred,
                                edge_labels=edge_ref, edge_prediction=edge_ref.clone(),
                                edge_types=torch.empty(0, dtype=torch.long),
                                edge_index=torch.empty((2, 0), dtype=torch.long)))
    return samples


def pair_model(**kwargs):
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=metrics.pair_difference_mse, loss_kwargs=kwargs)
    model.init_model()
    return model


@pytest.mark.parametrize("components", [1, 2])
def test_pair_epoch_means_reset_for_pair_preserving_partitions_and_validation(components):
    class TrainModel(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    class Recorder(pl.Callback):
        def __init__(self):
            self.values = []

        def on_after_backward(self, trainer, pl_module):
            assert pl_module.model.offset.grad is not None
            assert torch.isfinite(pl_module.model.offset.grad)

        def on_train_epoch_end(self, trainer, pl_module):
            self.values.append({name: float(trainer.callback_metrics["train_" + name + "_epoch"])
                                for name in ("loss", "pair_rmse", "n_pairs")})

    samples = pair_samples(components)
    factor = 1. if components == 1 else 2.5
    for size in (2, 4, 6, 2):
        model = TrainModel(model_cls=_PredictionModel, basis_table=pair_model().basis_table,
                           loss=metrics.pair_difference_mse, loss_kwargs={"pair_weight": .25})
        model.init_model()
        recorder = Recorder()
        trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0)
        trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=size),
                    val_dataloaders=DataLoader(samples, batch_size=1))
        assert len(recorder.values) == 2
        for values in recorder.values:
            assert values == pytest.approx({"loss": 59 / 6 * factor, "pair_rmse": (56 / 3 * factor) ** .5, "n_pairs": 3})
        assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(31 / 6 * factor)


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("components", [1, 2])
def test_pair_evaluation_is_plain_global_mse_even_if_expected_pairs_would_fail(split, components):
    model = pair_model(expected_pairs=999)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for size in (1, 2, 4, 6, 1):
        result = evaluate(model, dataloaders=DataLoader(pair_samples(components), batch_size=size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(31 / 6 * (1 if components == 1 else 2.5))
        assert f"{split}_pair_rmse" not in result and f"{split}_n_pairs" not in result


@pytest.mark.parametrize("missing", ["node", "edge"])
def test_pair_training_rejects_incomplete_paired_matrix(missing):
    samples = pair_samples()[:2]
    if missing == "edge":
        for sample in samples:
            sample.edge_types = torch.zeros(1, dtype=torch.long)
            sample.edge_index = torch.zeros(2, 1, dtype=torch.long)
            sample.edge_labels = torch.zeros(1, dtype=torch.float64)
            sample.edge_prediction = torch.zeros(1, dtype=torch.float64)
        samples[0].edge_labels[0] = float("nan")
    else:
        samples[0].point_labels[0] = float("nan")
    batch = Batch.from_data_list(samples)
    model = pair_model()
    with pytest.raises(ValueError, match="complete"):
        model.loss_fn(nodes_pred=batch.node_prediction, nodes_ref=batch.point_labels,
                      edges_pred=batch.edge_prediction, edges_ref=batch.edge_labels,
                      batch=batch, basis_table=model.basis_table, model=model.model)


def test_nan_unpaired_structure_does_not_poison_observed_pair_or_gradients():
    samples = pair_samples()[:3]
    samples[2].point_labels[0] = float("nan")
    batch = Batch.from_data_list(samples)
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    model = pair_model(pair_weight=.25)
    loss, stats = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                                edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table, model=model.model)
    assert float(loss) == pytest.approx(11 / 3 + 1)
    assert stats["n_pairs"] == 1 and float(stats["pair_rmse"]) == 2.
    loss.backward()
    torch.testing.assert_close(nodes.grad, torch.tensor([5 / 3, -5 / 3, 0., 2.], dtype=torch.float64))


@pytest.mark.parametrize("components", [1, 2])
def test_pair_symmetric_gamma_literal_loss_and_node_edge_gradients(components):
    samples = pair_samples(components)[2:4]
    for index, sample in enumerate(samples):
        sample.node_prediction.fill_(1.)
        sample.edge_types = torch.zeros(2, dtype=torch.long)
        sample.edge_index = torch.tensor([[0, 1], [1, 0]])
        sample.edge_labels = torch.zeros((1,) if components == 1 else (1, 2), dtype=torch.float64)
        sample.edge_prediction = torch.tensor([2. if index == 0 else -2.], dtype=torch.float64)
        if components == 2:
            sample.node_prediction[:, 1] = 2.
            sample.edge_prediction = torch.stack([sample.edge_prediction, 2 * sample.edge_prediction], dim=1)
    batch = Batch.from_data_list(samples)
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    model = pair_model(pair_weight=.25, edge_weight=2.)
    loss, stats = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                                edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table, model=model.model)
    assert float(loss) == pytest.approx(4.5 if components == 1 else 11.25)
    assert float(stats["pair_rmse"]) == pytest.approx((8. if components == 1 else 20.) ** .5)
    loss.backward()
    torch.testing.assert_close(nodes.grad, torch.full((4,), .25, dtype=torch.float64) if components == 1
                               else torch.tensor([[.125, .25]], dtype=torch.float64).repeat(4, 1))
    torch.testing.assert_close(edges.grad, torch.tensor([2., -2.], dtype=torch.float64) if components == 1
                               else torch.tensor([[1., 2.], [-1., -2.]], dtype=torch.float64))


def test_splitting_training_partners_is_detected_when_pair_guard_is_requested():
    batch = Batch.from_data_list(pair_samples()[:1])
    model = pair_model(expected_pairs=1)
    with pytest.raises(ValueError, match="found 0.*expected 1"):
        model.loss_fn(nodes_pred=batch.node_prediction, nodes_ref=batch.point_labels,
                      edges_pred=batch.edge_prediction, edges_ref=batch.edge_labels,
                      batch=batch, basis_table=model.basis_table, model=model.model)


@pytest.mark.parametrize("missing", ["empty", "nan"])
def test_pair_without_partners_has_connected_zero_for_empty_observations(missing):
    sample = pair_samples()[0]
    if missing == "empty":
        sample.num_nodes = 0
        for field in ("point_types", "point_labels", "node_prediction", "positions"):
            sample[field] = sample[field][:0]
    else:
        sample.point_labels.fill_(float("nan"))
    batch = Batch.from_data_list([sample])
    model = pair_model()
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    loss, stats = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                                edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table, model=model.model)
    assert loss == 0 and stats["n_pairs"] == 0 and stats["pair_rmse"] == 0
    loss.backward()
    assert nodes.grad is not None and not nodes.grad.count_nonzero()
    assert edges.grad is not None and not edges.grad.count_nonzero()
