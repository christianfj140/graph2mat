"""Threshold sums/selections and size-domain means retain their denominators."""

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.core.data import metrics
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel
from graph2mat.tools.lightning.data import TypeLayoutCollate


def _empty_graph_samples(counts, gauge=False):
    samples = []
    for count in counts:
        labels = torch.tensor([[0., 1.]]).repeat(count, 1) if gauge else torch.zeros(count)
        prediction = labels + torch.tensor([2., 0.]) if gauge else torch.full((count,), 2.)
        edges = torch.empty((0, 2)) if gauge else torch.empty(0)
        samples.append(Data(num_nodes=count, point_types=torch.zeros(count, dtype=torch.long),
                            point_labels=labels, node_prediction=prediction,
                            edge_labels=edges, edge_prediction=edges.clone(),
                            edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long)))
    return samples


@pytest.mark.parametrize("counts", [(1, 0), (0, 1), (0, 0), (1, 0, 0), ()])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("loss_name", ["block_type_mse_per_structure", "block_type_mse_per_domain", "gauge_invariant_mse"])
def test_structure_count_preserves_trailing_and_empty_graphs(counts, cached, loss_name):
    gauge = loss_name == "gauge_invariant_mse"
    samples = _empty_graph_samples(counts, gauge)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    if samples:
        batch = TypeLayoutCollate(Batch.from_data_list, table)(samples) if cached else Batch.from_data_list(samples)
    else:
        batch = Batch()
        batch._num_graphs = 0
        batch.ptr = torch.tensor([0])
        for key, value in _empty_graph_samples([0], gauge)[0].to_dict().items():
            batch[key] = value
    nodes = batch.node_prediction.clone().requires_grad_()
    edges = batch.edge_prediction.clone().requires_grad_()
    _, _, size = metrics._per_element_structure_ids(batch.point_labels, batch.edge_labels, batch, table)
    assert size == len(counts)
    kwargs = {"domain_threshold_atoms": 3} if loss_name.endswith("domain") else {}
    loss, stats = getattr(metrics, loss_name)()(nodes_pred=nodes, nodes_ref=batch.point_labels,
                                              edges_pred=edges, edges_ref=batch.edge_labels,
                                              batch=batch, basis_table=table, **kwargs)
    assert float(loss) == pytest.approx(0 if gauge else 4 * sum(counts) / (len(counts) or 1))
    if gauge:
        assert float(stats["gauge_shift_mean_abs"]) == pytest.approx(2 * sum(counts) / (len(counts) or 1))
    else:
        assert stats["n_structures"] == len(counts)
    loss.backward()
    assert torch.isfinite(nodes.grad).all() and torch.isfinite(edges.grad).all()


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("loss_name", ["block_type_mse_per_structure", "block_type_mse_per_domain", "gauge_invariant_mse"])
def test_empty_graph_epoch_is_independent_of_batching_and_order(split, loss_name):
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    kwargs = {"domain_threshold_atoms": 3} if loss_name.endswith("domain") else {}
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=getattr(metrics, loss_name), loss_kwargs=kwargs)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for counts in ((1, 0, 0), (0, 0, 1), (0, 0, 0)):
        samples = _empty_graph_samples(counts, loss_name == "gauge_invariant_mse")
        for size in (1, 2, 3, 1):
            loader = DataLoader(samples, batch_size=size)
            loader.collate_fn = TypeLayoutCollate(loader.collate_fn, table)
            result = evaluate(model, dataloaders=loader, verbose=False)[0]
            if loss_name == "gauge_invariant_mse":
                assert result[f"{split}_loss"] == 0
                assert result[f"{split}_gauge_shift_mean_abs"] == pytest.approx(2 * sum(counts) / 3)
            else:
                assert result[f"{split}_loss"] == pytest.approx(4 * sum(counts) / 3)
                assert result[f"{split}_n_structures"] == 3


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("missing_nodes", [False, True])
@pytest.mark.parametrize("loss_name,want_scalar,want_spin", [
    ("block_type_mse_threshold", 57., 301.),
    ("block_type_mse_sigmoid_thresh", 58., 304.),
    ("block_type_mae_sigmoid_thresh", 16., 52.),
])
def test_threshold_epoch_sums_and_valid_selection_means(split, components, missing_nodes, loss_name, want_scalar, want_spin):
    samples = [
        Data(num_nodes=4, point_labels=torch.tensor([0., 0., 0., float("nan")]),
             node_prediction=torch.tensor([0., 1., 1., 99.]),
             edge_labels=torch.tensor([0., 0., float("nan")]), edge_prediction=torch.tensor([0., 2., 99.])),
        Data(num_nodes=1, point_labels=torch.zeros(1), node_prediction=torch.tensor([3.]),
             edge_labels=torch.zeros(3), edge_prediction=torch.full((3,), 4.)),
    ]
    if missing_nodes:
        for sample in samples:
            sample.point_labels.fill_(float("nan"))
        want_scalar = {"block_type_mse_threshold": 48., "block_type_mse_sigmoid_thresh": 49., "block_type_mae_sigmoid_thresh": 13.}[loss_name]
        want_spin = {"block_type_mse_threshold": 256., "block_type_mse_sigmoid_thresh": 257., "block_type_mae_sigmoid_thresh": 41.}[loss_name]
    if components == 2:
        for sample in samples:
            for key in ("point_labels", "node_prediction", "edge_labels", "edge_prediction"):
                sample[key] = torch.stack([sample[key], 2 * sample[key]], dim=1)
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=getattr(metrics, loss_name), loss_kwargs={"threshold": 2.})
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for batch_size in (1, 2, 1):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(want_scalar if components == 1 else want_spin)
        assert result[f"{split}_node_above_threshold_frac"] == (0. if missing_nodes else .25)
        assert result[f"{split}_node_above_threshold_mean"] == (0. if missing_nodes else 3. if components == 1 else 4.5)
        assert result[f"{split}_edge_above_threshold_frac"] == pytest.approx(.6 if components == 1 else .7)
        assert result[f"{split}_edge_above_threshold_mean"] == pytest.approx(4. if components == 1 else 40 / 7)
        if split == "val":
            assert result["hp_metric"] == pytest.approx(want_scalar if components == 1 else want_spin)


@pytest.mark.parametrize("loss_name", ["block_type_mse_sigmoid_thresh", "block_type_mae_sigmoid_thresh"])
def test_sigmoid_threshold_extreme_exponents_have_finite_exact_gradients(loss_name):
    prediction = torch.tensor([0., 1., 2., 4.], requires_grad=True)
    loss, _ = getattr(metrics, loss_name)()(nodes_pred=prediction, nodes_ref=torch.zeros(4),
                                          edges_pred=prediction[:0], edges_ref=torch.empty(0), threshold=2.)
    assert float(loss) == (17. if loss_name == "block_type_mse_sigmoid_thresh" else 5.)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([0., 0., 100001., 8.] if loss_name == "block_type_mse_sigmoid_thresh" else [0., 0., 50000.5, 1.]), rtol=0, atol=0)


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("missing_large", [False, True])
@pytest.mark.parametrize("loss_name", ["block_type_mse_per_structure", "block_type_mse_per_domain"])
def test_size_domain_epoch_means_and_structure_counts(split, missing_large, loss_name):
    samples = [Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
                    point_labels=torch.full((n,), float("nan") if missing_large and n == 3 else 0.),
                    node_prediction=torch.full((n,), error), edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                    edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))
               for n, error in ((1, 1.), (2, 3.), (3, 2.))]
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    kwargs = {"domain_threshold_atoms": 3, "small_domain_weight": .9, "large_domain_weight": .1} if loss_name.endswith("domain") else {}
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=getattr(metrics, loss_name), loss_kwargs=kwargs)
    model.init_model()
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    want = (4.5 if missing_large else 4.9) if loss_name.endswith("domain") else (10 / 3 if missing_large else 14 / 3)
    for batch_size in (1, 2, 3, 1):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(want)
        assert result[f"{split}_n_structures"] == 3
        if loss_name.endswith("domain"):
            assert result[f"{split}_n_large_structures"] == 1
        if split == "val":
            assert result["hp_metric"] == pytest.approx(want)


@pytest.mark.parametrize("loss_name,want", [("block_type_mse_threshold", 18.),
                                          ("block_type_mse_sigmoid_thresh", 18.),
                                          ("block_type_mae_sigmoid_thresh", 6.),
                                          ("block_type_mse_per_domain", 1.8),
                                          ("block_normalized_huber", 1.),
                                          ("hamiltonian_composite_loss", 1.)])
def test_special_training_metrics_reset_without_changing_step_gradients(loss_name, want):
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

    samples = [Data(num_nodes=n, point_types=torch.zeros(n, dtype=torch.long),
                    point_labels=torch.zeros(n), node_prediction=torch.full((n,), error),
                    edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                    edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))
               for n, error in ((3, 1.), (2, 3.))]
    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    kwargs = {"threshold": 2.}
    if loss_name.endswith("domain"):
        kwargs = {"domain_threshold_atoms": 3, "small_domain_weight": .1, "large_domain_weight": .9}
    if loss_name in ("block_normalized_huber", "hamiltonian_composite_loss"):
        table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1]), PointBasis("B", R=2, basis=[2])])
        samples = [Data(num_nodes=3, point_types=torch.zeros(3, dtype=torch.long),
                        point_labels=torch.zeros(3), node_prediction=torch.ones(3),
                        edge_labels=torch.zeros(2), edge_prediction=torch.full((2,), 2.),
                        edge_types=torch.zeros(4, dtype=torch.long), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])),
                   Data(num_nodes=1, point_types=torch.ones(1, dtype=torch.long),
                        point_labels=torch.zeros(4), node_prediction=torch.full((4,), 3.),
                        edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                        edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))]
        kwargs = {"beta": 1., "node_weight": .25, "edge_weight": .5}
        if loss_name == "hamiltonian_composite_loss":
            kwargs["per_block_normalization"] = True
    model = TrainModel(model_cls=_PredictionModel, basis_table=table,
                       loss=getattr(metrics, loss_name), loss_kwargs=kwargs)
    model.init_model()
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=DataLoader(samples, batch_size=1),
                val_dataloaders=DataLoader(samples, batch_size=2))
    assert recorder.losses == pytest.approx([want, want])
    assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(want)
