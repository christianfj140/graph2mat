"""Coefficient diagnostics use observed coefficients and equal operation weights."""

import math
from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch
from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.bindings.e3nn import E3nnDiagnosticDenseEdgeBlock, E3nnDiagnosticDenseNodeBlock
from graph2mat.bindings.e3nn.modules.graph2mat import E3nnGraph2Mat
from graph2mat.core.data import metrics
from graph2mat.tools.lightning.model import LitBasisMatrixModel


class CoefficientPredictions(torch.nn.Module):
    """Literal s-orbital predictions, projected by a real e3nn readout."""

    def __init__(self, n_matrix_components=1):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.matrix_readouts = E3nnGraph2Mat(
            unique_basis=[PointBasis(kind, R=2, basis=[1]) for kind in ("A", "B")],
            irreps={"node_feats_irreps": o3.Irreps("1x0e")}, symmetric=True,
            n_matrix_components=n_matrix_components,
            preprocessing_edges=None, node_operation=E3nnDiagnosticDenseNodeBlock,
            edge_operation=E3nnDiagnosticDenseEdgeBlock,
        )
        self.basis_table = self.matrix_readouts.basis_table

    def forward(self, batch):
        node = batch.node_prediction + self.offset
        edge = batch.edge_prediction + self.offset
        coefficients = metrics.target_coefficients_from_labels(
            self.matrix_readouts, batch, self.basis_table, node, edge,
        )
        return {"node_labels": node, "edge_labels": edge,
                "node_coefficients": coefficients["node"], "edge_coefficients": coefficients["edge"]}


def coefficient_samples(missing_b=False):
    # A errors [1,1,1,3], B [5], A-A edge errors [2,2,4].
    # MSE=(3+25+8)/3=12; MAE=(1.5+5+8/3)/3=55/18.
    samples = []
    for n, kind, error, edge_errors in ((3, 0, 1., [2., 2.]), (1, 0, 3., [4.]), (1, 1, 5., [])):
        edge_index = torch.zeros((2, 2 * len(edge_errors)), dtype=torch.long)
        if n == 3:
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        sample = Data(
            num_nodes=n, point_types=torch.full((n,), kind, dtype=torch.long),
            point_labels=torch.full((n,), float("nan") if missing_b and kind else 0.),
            node_prediction=torch.full((n,), error), edge_labels=torch.zeros(len(edge_errors)),
            edge_prediction=torch.tensor(edge_errors), edge_types=torch.zeros(edge_index.shape[1], dtype=torch.long),
            edge_index=edge_index,
        )
        samples.append(sample)
    return samples


def coefficient_model(loss_name, train=False, n_components=1):
    class TrainModel(LitBasisMatrixModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    table = BasisTableWithEdges([PointBasis(kind, R=2, basis=[1]) for kind in ("A", "B")])
    kwargs = ({"beta": 1., "node_weight": .25, "edge_weight": .5,
               "coefficient_mse_weight": .3} if loss_name == "hamiltonian_composite_loss" else {})
    model = (TrainModel if train else LitBasisMatrixModel)(
        model_cls=CoefficientPredictions, basis_table=table,
        loss=getattr(metrics, loss_name), loss_kwargs=kwargs,
    )
    model.init_model(n_matrix_components=n_components)
    return model


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
def test_coefficient_epochs_resolve_compatible_interaction_readouts(loss_name):
    model = coefficient_model(loss_name)
    readout = model.model.matrix_readouts
    model.model.matrix_readouts = torch.nn.ModuleList([readout, deepcopy(readout)])
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    expected = {"coefficient_space_mse": 12., "coefficient_space_mae": 55 / 18,
                "hamiltonian_composite_loss": 613 / 120}[loss_name]
    for batch_size in (1, 2, 3, 1):
        result = trainer.validate(model, dataloaders=DataLoader(coefficient_samples(), batch_size=batch_size), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(expected)
        assert result["hp_metric"] == pytest.approx(expected)


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae"])
@pytest.mark.parametrize("missing_b", [False, True])
def test_multicomponent_coefficient_epochs_keep_nan_denominators_and_gradients(loss_name, missing_b):
    model = coefficient_model(loss_name, n_components=2)
    samples = coefficient_samples()
    for sample in samples:
        for name in ("point_labels", "edge_labels", "node_prediction", "edge_prediction"):
            value = sample[name]
            sample[name] = torch.stack([value, 2 * value], dim=1)
    if missing_b:
        samples[-1].point_labels[:, 1] = float("nan")
    expected = ((25 / 8 if missing_b else 55 / 12) if loss_name == "coefficient_space_mae"
                else (13.75 if missing_b else 30.))
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for batch_size in (1, 2, 3, 1):
        result = trainer.validate(model, dataloaders=DataLoader(samples, batch_size=batch_size), verbose=False)[0]
        assert result["val_loss"] == pytest.approx(expected)
        assert result["val_coefficient_blocks"] == (2 if missing_b else 3)
    batch = next(iter(DataLoader(samples, batch_size=3)))
    out = model(batch)
    loss, _ = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
                           edges_pred=out["edge_labels"], edges_ref=batch.edge_labels,
                           batch=batch, basis_table=model.basis_table, out=out, model=model.model)
    assert float(loss) == pytest.approx(expected)
    loss.backward()
    gradient = 1. if loss_name == "coefficient_space_mae" else (25 / 4 if missing_b else 55 / 6)
    assert float(model.model.offset.grad) == pytest.approx(gradient)


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("missing_b", [False, True])
@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
def test_coefficient_epoch_denominators_and_composite_total(split, missing_b, loss_name):
    pl.seed_everything(0, workers=True)
    model = coefficient_model(loss_name)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    coefficient = (25 / 12 if missing_b else 55 / 18) if loss_name == "coefficient_space_mae" else (5.5 if missing_b else 12.)
    label = 4 / 3 if missing_b else 181 / 120
    want = label + .3 * coefficient if loss_name == "hamiltonian_composite_loss" else coefficient
    prefix = "coefficient_" if loss_name == "hamiltonian_composite_loss" else ""
    suffix = "mae" if loss_name == "coefficient_space_mae" else "rmse"
    for batch_size in (1, 2, 3, 1):
        result = evaluate(model, dataloaders=DataLoader(coefficient_samples(missing_b), batch_size=batch_size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(want)
        if split == "val":
            assert result["hp_metric"] == pytest.approx(want)
        assert result[f"{split}_{prefix}node_node:0_coeff_{suffix}"] == pytest.approx(1.5 if suffix == "mae" else math.sqrt(3))
        assert result[f"{split}_{prefix}node_node:1_coeff_{suffix}"] == (0. if missing_b else 5.)
        assert result[f"{split}_{prefix}edge_edge:(0, 0, 0)_coeff_{suffix}"] == pytest.approx(8 / 3 if suffix == "mae" else math.sqrt(8))
        assert result[f"{split}_{prefix}coefficient_blocks"] == (2 if missing_b else 3)
        if loss_name == "hamiltonian_composite_loss":
            assert result[f"{split}_coefficient_mse"] == pytest.approx(coefficient)
            assert result[f"{split}_label_huber"] == pytest.approx(label)


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
def test_coefficient_losses_exclude_unobserved_blocks_and_have_finite_gradients(loss_name):
    model = coefficient_model(loss_name)
    batch = next(iter(DataLoader(coefficient_samples(missing_b=True), batch_size=3)))
    out = model(batch)
    loss, stats = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
                                edges_pred=out["edge_labels"], edges_ref=batch.edge_labels,
                                batch=batch, basis_table=model.basis_table, out=out, model=model.model)
    want = 25 / 12 if loss_name == "coefficient_space_mae" else 5.5
    if loss_name == "hamiltonian_composite_loss":
        want = 4 / 3 + .3 * want
    assert float(loss) == pytest.approx(want)
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in stats.values())
    loss.backward()
    # Independent derivative at zero: MSE 25/6, MAE 1, composite .25+.5+.3*25/6.
    want_gradient = 1. if loss_name == "coefficient_space_mae" else 25 / 6
    if loss_name == "hamiltonian_composite_loss":
        want_gradient = 2.
    assert float(model.model.offset.grad) == pytest.approx(want_gradient)


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
@pytest.mark.parametrize("empty", [False, True])
def test_coefficient_epochs_without_observations_are_zero_and_differentiable(loss_name, empty):
    model = coefficient_model(loss_name)
    if empty:
        samples = [Data(num_nodes=0, point_types=torch.empty(0, dtype=torch.long),
                        point_labels=torch.empty(0), node_prediction=torch.empty(0),
                        edge_labels=torch.empty(0), edge_prediction=torch.empty(0),
                        edge_types=torch.empty(0, dtype=torch.long), edge_index=torch.empty((2, 0), dtype=torch.long))]
    else:
        samples = coefficient_samples()
        for sample in samples:
            sample.point_labels.fill_(float("nan"))
            sample.edge_labels.fill_(float("nan"))
    batch = next(iter(DataLoader(samples, batch_size=3)))
    out = model(batch)
    loss, stats = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
                                edges_pred=out["edge_labels"], edges_ref=batch.edge_labels,
                                batch=batch, basis_table=model.basis_table, out=out, model=model.model)
    assert float(loss) == 0.
    loss.backward()
    assert model.model.offset.grad is not None and float(model.model.offset.grad) == 0.
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    for evaluate in (trainer.validate, trainer.test, trainer.validate):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=1), verbose=False)[0]
        for key, value in result.items():
            if not key.endswith(("coefficient_mse_weight", "smooth_l1_beta")):
                assert value == 0., key


@pytest.mark.parametrize("loss_name", ["coefficient_space_mse", "coefficient_space_mae", "hamiltonian_composite_loss"])
def test_coefficient_training_epochs_and_validation_use_separate_reset_states(loss_name):
    class Recorder(pl.Callback):
        def __init__(self):
            self.losses = []

        def on_train_epoch_end(self, trainer, pl_module):
            self.losses.append(float(trainer.callback_metrics["train_loss_epoch"]))

    model = coefficient_model(loss_name, train=True)
    recorder = Recorder()
    trainer = pl.Trainer(accelerator="cpu", max_epochs=2, logger=False, callbacks=[recorder],
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=DataLoader(coefficient_samples(), batch_size=1),
                val_dataloaders=DataLoader(coefficient_samples(missing_b=True), batch_size=2))
    train = 55 / 18 if loss_name == "coefficient_space_mae" else 12.
    val = 25 / 12 if loss_name == "coefficient_space_mae" else 5.5
    if loss_name == "hamiltonian_composite_loss":
        train, val = 613 / 120, 179 / 60
    assert recorder.losses == pytest.approx([train, train])
    assert float(trainer.callback_metrics["val_loss"]) == pytest.approx(val)
