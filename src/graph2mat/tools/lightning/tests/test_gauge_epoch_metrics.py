"""Gauge H, S and spectral windows have different epoch denominators."""

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data.metrics import gauge_invariant_mse
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


def gauge_samples(components=1, predicts_overlap=False):
    samples = []
    for n, errors, shift, s_error in ((2, [-1., 1.], 10., .5), (4, [-3., 3., -3., 3.], -20., 1.), (1, [0.], 30., 2.)):
        h = torch.tensor(errors, dtype=torch.float64)
        ref = torch.zeros(n, components + 1, dtype=torch.float64)
        ref[:, -1] = 1.
        pred = torch.stack([(i + 1) * h + shift for i in range(components)], dim=1)
        edge_ref = torch.zeros(int(n == 2), components + 1, dtype=torch.float64)
        edge_pred = torch.zeros(len(edge_ref), components, dtype=torch.float64)
        if predicts_overlap:
            pred = torch.cat([pred, ref[:, -1:] + s_error], dim=1)
            edge_pred = torch.cat([edge_pred, edge_ref[:, -1:] + s_error], dim=1)
        elif components == 1:
            pred, edge_pred = pred[:, 0], edge_pred[:, 0]
        c = torch.zeros(n, 4, dtype=torch.float64)
        c[:, :n] = torch.eye(n, dtype=torch.float64)
        samples.append(Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long), point_labels=ref,
                            node_prediction=pred, edge_labels=edge_ref, edge_prediction=edge_pred, spectral_C=c,
                            edge_types=torch.zeros(len(edge_ref), dtype=torch.long),
                            edge_index=torch.tensor([[0], [1]]) if len(edge_ref) else torch.empty((2, 0), dtype=torch.long)))
    return samples


def gauge_model(**kwargs):
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=gauge_invariant_mse, loss_kwargs=kwargs)
    model.init_model()
    return model


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("edge_weight", [.5, 2.])
@pytest.mark.parametrize("components,predicts_overlap,spectral_weight", [
    (1, False, 0.), (1, True, 0.), (2, False, 0.), (2, True, 0.), (1, False, .25), (1, True, .25),
])
def test_gauge_epoch_objective_uses_each_physical_denominator(split, edge_weight, components, predicts_overlap, spectral_weight):
    samples = gauge_samples(components, predicts_overlap)
    model = gauge_model(edge_weight=edge_weight, overlap_weight=.3, spectral_weight=spectral_weight)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    h_mean = 38 * (1 if components == 1 else 2.5) / (7 + edge_weight)
    s_mean = (8.5 + .25 * edge_weight) / (7 + edge_weight) if predicts_overlap else 0.
    want = h_mean + .3 * s_mean + spectral_weight * 38 / 21
    for size in (1, 2, 3, 1):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(want)
        assert result[f"{split}_gauge_h_rmse"] == pytest.approx(h_mean ** .5)
        assert result[f"{split}_overlap_rmse"] == pytest.approx(s_mean ** .5)
        assert result[f"{split}_gauge_shift_mean_abs"] == pytest.approx(20.)
        if spectral_weight:
            assert result[f"{split}_spectral_rmse"] == pytest.approx((38 / 21) ** .5)
            assert result[f"{split}_spectral_elements"] == 21
        if split == "val":
            assert result["hp_metric"] == pytest.approx(want)


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("predicts_overlap", [False, True])
@pytest.mark.parametrize("missing", ["node", "edge", "all"])
def test_gauge_missing_rows_are_masked_before_alignment_and_backward(components, predicts_overlap, missing):
    samples = gauge_samples(components, predicts_overlap)
    if missing == "all":
        for sample in samples:
            sample.point_labels.fill_(float("nan"))
            sample.edge_labels.fill_(float("nan"))
    else:
        samples[0]["point_labels" if missing == "node" else "edge_labels"][0, -1] = float("nan")
    batch = Batch.from_data_list(samples)
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    model = gauge_model(overlap_weight=.3)
    loss, stats = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                                edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table)
    h_mean = 0. if missing == "all" else (36. if missing == "node" else 38.) / 7 * (1 if components == 1 else 2.5)
    if missing == "node" and components == 2:
        # The remaining spin errors [1, 2] align to [-.5, .5] with one common shift.
        h_mean += .5 / 14
    s_mean = 8.5 / 7 if predicts_overlap and missing != "all" else 0.
    assert float(loss) == pytest.approx(h_mean + .3 * s_mean)
    assert float(stats["gauge_h_rmse"]) == pytest.approx(h_mean ** .5)
    loss.backward()
    assert torch.isfinite(nodes.grad).all() and torch.isfinite(edges.grad).all()
    if missing == "all":
        assert not nodes.grad.count_nonzero() and not edges.grad.count_nonzero()
    else:
        assert not (nodes if missing == "node" else edges).grad[0].count_nonzero()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for size in (1, 3, 1):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(h_mean + .3 * s_mean)
        assert result["val_gauge_h_rmse"] == pytest.approx(h_mean ** .5)


@pytest.mark.parametrize("missing", ["node", "edge", "C"])
def test_spectral_window_rejects_incomplete_reference_instead_of_inventing_zero(missing):
    sample = gauge_samples()[0]
    sample[{"node": "point_labels", "edge": "edge_labels", "C": "spectral_C"}[missing]][0, 0] = float("nan")
    batch = Batch.from_data_list([sample])
    model = gauge_model(spectral_weight=1.)
    with pytest.raises(ValueError, match="complete|finite"):
        model.loss_fn(nodes_pred=batch.node_prediction, nodes_ref=batch.point_labels,
                      edges_pred=batch.edge_prediction, edges_ref=batch.edge_labels,
                      batch=batch, basis_table=model.basis_table)


@pytest.mark.parametrize("predicts_overlap", [False, True])
def test_gauge_empty_labels_return_connected_zero(predicts_overlap):
    sample = gauge_samples(1, predicts_overlap)[2]
    sample.num_nodes = 0
    sample.point_types = sample.point_types[:0]
    sample.point_labels = sample.point_labels[:0]
    sample.node_prediction = sample.node_prediction[:0]
    sample.spectral_C = sample.spectral_C[:0]
    batch = Batch.from_data_list([sample])
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    model = gauge_model(spectral_weight=1.)
    loss, stats = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                                edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table)
    assert loss == 0 and all(torch.isfinite(torch.as_tensor(v)) for v in stats.values())
    loss.backward()
    assert nodes.grad is not None and edges.grad is not None


def test_gauge_spectral_literal_step_value_and_gradient():
    sample = gauge_samples(1, True)[0]
    batch = Batch.from_data_list([sample])
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    model = gauge_model(overlap_weight=.3, spectral_weight=.25)
    loss, _ = model.loss_fn(nodes_pred=nodes, nodes_ref=batch.point_labels, edges_pred=edges,
                            edges_ref=batch.edge_labels, batch=batch, basis_table=model.basis_table)
    # H: 2/3, S: .25, window: 2/4. S edge contributes to its own gradient.
    assert float(loss) == pytest.approx(2 / 3 + .075 + .125)
    loss.backward()
    torch.testing.assert_close(nodes.grad, torch.tensor([[-19 / 24, .1], [19 / 24, .1]], dtype=torch.float64))
    torch.testing.assert_close(edges.grad, torch.tensor([[0., .1]], dtype=torch.float64))


def test_empty_spectral_window_has_no_denominator_weight():
    samples = gauge_samples()
    samples[1].spectral_C.zero_()
    model = gauge_model(spectral_weight=.25)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for size in (1, 3, 1):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result["val_spectral_rmse"] == pytest.approx((2 / 5) ** .5)
        assert result["val_spectral_elements"] == 5
        assert result["val_loss"] == pytest.approx(38 / 8 + .25 * 2 / 5)


@pytest.mark.parametrize("predicts_overlap", [False, True])
def test_gauge_spectral_training_resets_epoch_and_split_state(predicts_overlap):
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
                                for name in ("loss", "spectral_rmse", "spectral_elements")})

    samples = gauge_samples(1, predicts_overlap)
    model = TrainModel(model_cls=_PredictionModel, basis_table=gauge_model().basis_table,
                       loss=gauge_invariant_mse, loss_kwargs={"overlap_weight": .3, "spectral_weight": .25})
    model.init_model()
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=1),
                val_dataloaders=DataLoader(samples, batch_size=3))
    want = 38 / 8 + (.3 * 8.75 / 8 if predicts_overlap else 0.) + .25 * 38 / 21
    for values in recorder.values:
        assert values == pytest.approx({"loss": want, "spectral_rmse": (38 / 21) ** .5, "spectral_elements": 21})
    assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(want)


def test_gauge_fractional_edge_count_is_not_clamped_to_one():
    sample = gauge_samples(1, True)[0]
    sample.point_labels.fill_(float("nan"))
    sample.edge_labels = torch.zeros(2, 2, dtype=torch.float64)
    sample.edge_prediction = torch.tensor([[1., .5], [3., .5]], dtype=torch.float64)
    sample.edge_types = torch.zeros(2, dtype=torch.long)
    sample.edge_index = torch.tensor([[0, 1], [1, 0]])
    model = gauge_model(edge_weight=.25, overlap_weight=.3)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    result = trainer.validate(model, dataloaders=DataLoader([sample], batch_size=1), verbose=False)[0]
    assert result["val_loss"] == pytest.approx(5.075)
    assert result["val_gauge_h_rmse"] == pytest.approx(5 ** .5)
    assert result["val_overlap_rmse"] == .5


def test_gauge_and_basic_objectives_can_replace_epoch_accumulators_between_runs():
    from graph2mat.core.data.metrics import elementwise_mse

    samples = gauge_samples(1, True)
    model = gauge_model(overlap_weight=.3, spectral_weight=.25)
    gauge = model.loss_fn
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    want = 38 / 8 + .3 * 8.75 / 8 + .25 * 38 / 21
    for objective, expected in ((gauge, want), (elementwise_mse(), 2746.75 / 16), (gauge, want)):
        model.loss_fn = objective
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=2), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(expected)
