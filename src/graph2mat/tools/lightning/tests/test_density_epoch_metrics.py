"""Real density grids and independent orbital-square integrals."""

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.metrics import normalized_density_error
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


def density_samples(symmetric=True, sub_point=False, components=1, zero_electrons=False):
    r = np.linspace(0, 1.3, 101)
    atoms = [sisl.Atom(symbol, orbitals=[sisl.AtomicOrbital("1s", (r, np.exp(-r)), q0=0 if zero_electrons else q)])
             for symbol, q in (("H", 1), ("He", 2))]
    table = AtomicTableWithEdges(atoms)
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", symmetric_matrix=symmetric,
                                    sub_point_matrix=sub_point, n_matrix_components=components)
    samples = []
    for atom, error in zip(atoms, (1., 3.)):
        geometry = sisl.Geometry([[2., 2., 2.]], atoms=atom, lattice=[4., 4., 4.])
        geometry.set_nsc([1, 1, 1])
        matrix = sisl.DensityMatrix(geometry, spin="polarized" if components == 2 else "unpolarized")
        matrix[0, 0] = np.ones(components)
        sample = TorchBasisMatrixData.new(matrix, data_processor=processor, labels=True)
        sample.point_labels, sample.edge_labels = sample.point_labels.double(), sample.edge_labels.double()
        sample.node_prediction, sample.edge_prediction = sample.point_labels + error, sample.edge_labels.clone()
        samples.append(sample)
    grid = sisl.Grid(.3, geometry=geometry)
    coordinates = np.indices(grid.shape).reshape(3, -1).T @ grid.dcell + grid.origin
    integral = (atoms[0].orbitals[0].psi(coordinates - geometry.xyz[0]) ** 2).sum() * grid.dvolume
    return samples, table, integral


def density_model(table):
    model = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=table,
                               loss=normalized_density_error, loss_kwargs={"grid_spacing": .3})
    model.init_model()
    return model


@pytest.mark.parametrize("split", ["val", "test"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("sub_point", [False, True])
@pytest.mark.parametrize("components", [1, 2])
def test_density_epoch_uses_electron_denominator_and_independent_grid_integral(split, symmetric, sub_point, components):
    samples, table, integral = density_samples(symmetric, sub_point, components)
    model = density_model(table)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    evaluate = trainer.validate if split == "val" else trainer.test
    for size in (2, 1, 2):
        result = evaluate(model, dataloaders=DataLoader(samples, batch_size=size), verbose=False)[0]
        assert result[f"{split}_loss"] == pytest.approx(1.25 * components * integral, rel=1e-10)
        assert result[f"{split}_avg_per_config_error_percent"] == pytest.approx(125 * components * integral, rel=1e-10)
        assert result[f"{split}_avg_error_percent"] == pytest.approx(400 / 3 * components * integral, rel=1e-10)
        assert result[f"{split}_total_error"] == pytest.approx(4 * components * integral, rel=1e-10)
        assert result[f"{split}_per_config_total_error"] == pytest.approx(2.5 * components * integral, rel=1e-10)
        assert result[f"{split}_total_electrons"] == 3
        assert result[f"{split}_n_structures"] == 2


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("missing", [False, True])
def test_density_zero_or_unobserved_error_is_zero_without_mutating_inputs(components, missing):
    samples, table, _ = density_samples(components=components)
    data = samples[0]
    ref = data.point_labels.clone()
    pred = ref.clone()
    if missing:
        ref[0] = torch.nan
        pred[0] = torch.inf  # Unobserved rows must not contaminate the density.
    before = data.point_labels.clone()
    loss, stats = normalized_density_error(nodes_pred=pred, nodes_ref=ref,
                                          edges_pred=data.edge_labels, edges_ref=data.edge_labels,
                                          batch=data, basis_table=table, grid_spacing=.3, log_verbose=True)
    assert loss == 0 and stats["avg_error_percent"] == 0
    assert all(np.isfinite(float(value)) for value in stats.values())
    torch.testing.assert_close(data.point_labels, before)
    if missing:
        assert torch.isnan(ref).all() and torch.isinf(pred).all()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_density_observed_nonfinite_prediction_is_not_silenced(nonfinite):
    samples, table, _ = density_samples()
    data = samples[0]
    pred = torch.full_like(data.point_labels, nonfinite)
    with pytest.raises(ValueError, match="finite"):
        normalized_density_error(nodes_pred=pred, nodes_ref=data.point_labels,
                                 edges_pred=data.edge_labels, edges_ref=data.edge_labels,
                                 batch=data, basis_table=table, grid_spacing=.3)


def test_density_normalization_requires_positive_electron_count():
    samples, table, _ = density_samples(zero_electrons=True)
    data = samples[0]
    with pytest.raises(ValueError, match="positive.*electron"):
        normalized_density_error(nodes_pred=data.node_prediction, nodes_ref=data.point_labels,
                                 edges_pred=data.edge_labels, edges_ref=data.edge_labels,
                                 batch=data, basis_table=table, grid_spacing=.3)


def test_density_empty_batch_returns_finite_zero_and_zero_counters():
    _, table, _ = density_samples()
    batch = Batch(_base_cls=TorchBasisMatrixData)
    batch._num_graphs = 0
    batch.ptr = torch.tensor([0])
    batch.n_edges = torch.empty(0, dtype=torch.long)
    batch.point_types = torch.empty(0, dtype=torch.long)
    batch.edge_types = torch.empty(0, dtype=torch.long)
    empty = torch.empty(0)
    loss, stats = normalized_density_error(nodes_pred=empty, nodes_ref=empty, edges_pred=empty, edges_ref=empty,
                                          batch=batch, basis_table=table, grid_spacing=.3, log_verbose=True)
    assert loss == 0 and all(float(value) == 0 for value in stats.values())


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_density_edge_observation_mask_and_nonfinite_errors_are_explicit(symmetric, missing):
    _, table, _ = density_samples(symmetric)
    geometry = sisl.Geometry([[1.8, 2., 2.], [2.2, 2., 2.]], atoms=table.atoms, lattice=[4., 4., 4.])
    geometry.set_nsc([1, 1, 1])
    matrix = sisl.DensityMatrix(geometry)
    matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0] = 1., 1., .1, .1
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", symmetric_matrix=symmetric,
                                    sub_point_matrix=False)
    data = TorchBasisMatrixData.new(matrix, data_processor=processor, labels=True)
    ref, pred = data.edge_labels.clone(), data.edge_labels.clone()
    assert len(ref) > 0
    pred[0] = torch.inf
    if missing:
        ref[0] = torch.nan
        value, _ = normalized_density_error(nodes_pred=data.point_labels, nodes_ref=data.point_labels,
                                            edges_pred=pred, edges_ref=ref, batch=data, basis_table=table, grid_spacing=.3)
        assert value == 0 and torch.isinf(pred[0]) and torch.isnan(ref[0])
    else:
        with pytest.raises(ValueError, match="finite"):
            normalized_density_error(nodes_pred=data.point_labels, nodes_ref=data.point_labels,
                                     edges_pred=pred, edges_ref=ref, batch=data, basis_table=table, grid_spacing=.3)
