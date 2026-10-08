"""CPU targets cross workers/device boundaries without changing physical MSE."""

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from graph2mat import BasisConfiguration, MatrixDataProcessor
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.data.coefficient_targets import (
    CoefficientMatrixMSEProjector, CoefficientMSEUnsupported, coefficient_matrix_mse,
)
from graph2mat.core.data import metrics
from graph2mat.core.data.tests.test_coefficient_orientation import FEATURES, grouped_readout
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.data import TypeLayoutCollate
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel


TARGETS_ATTR = "_graph2mat_coefficient_mse_targets"


def samples_for(readout):
    table = readout.basis_table
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=readout.symmetric, sub_point_matrix=False)
    samples = []
    for index in range(3):
        config = BasisConfiguration(
            point_types=["0", "1", "0"], positions=np.array([[0., 0., 0.], [1., .2, 0.], [2., 0., .1]]),
            basis=table, cell=np.diag([4., 20., 20.]), pbc=(True, False, False),
        )
        data = TorchBasisMatrixData.from_config(config, data_processor=processor)
        features = torch.randn(3, FEATURES.dim, dtype=torch.float64)
        nodes, edges = readout(data, node_feats=features)
        data.point_labels = torch.randn_like(nodes) + index
        data.edge_labels = torch.randn_like(edges) + index
        samples.append(data)
    return samples


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("grouping", ["point_type", "basis_shape", "max"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("components", [1, 2])
def test_collate_targets_preserve_periodic_physical_loss_across_workers(workers, grouping, symmetric, components):
    torch.manual_seed(11)
    readout = grouped_readout(["1x0e+1x1o", "1x1o+1x2e"], grouping, symmetric, components)
    samples = samples_for(readout)
    kwargs = {"persistent_workers": True, "multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(samples, batch_size=2, num_workers=workers, **kwargs)
    loader.collate_fn = TypeLayoutCollate(loader.collate_fn, readout.basis_table,
        coefficient_projector=CoefficientMatrixMSEProjector(readout), symmetric=symmetric)
    for _ in range(2):
        for batch in loader:
            targets = getattr(batch, TARGETS_ATTR)
            assert TARGETS_ATTR not in batch.keys()
            assert getattr(batch.get_example(0), TARGETS_ATTR, None) is None
            assert (getattr(batch, LAYOUT_ATTR, None) is not None) == symmetric
            features = torch.randn(batch.num_nodes, FEATURES.dim, dtype=torch.float64, requires_grad=True)
            nodes, edges = readout(batch, node_feats=features)
            expected, _ = metrics.elementwise_mse()(nodes_pred=nodes, nodes_ref=batch.point_labels,
                                                  edges_pred=edges, edges_ref=batch.edge_labels, edge_weight=2.)
            actual, stats = coefficient_matrix_mse(readout(batch, node_feats=features, coefficients_only=True), targets, 2.)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(torch.autograd.grad(actual, features)[0],
                                       torch.autograd.grad(expected, features)[0], atol=1e-12, rtol=1e-12)
            assert stats["node_count"] == batch.point_labels.numel()
            assert stats["edge_count"] == batch.edge_labels.numel()


def test_collate_fallback_preserves_partial_nan_and_does_not_swallow_invalid_shape():
    readout = grouped_readout(["1x0e+1x1o", "1x1o"], "max", True, 2)
    samples = samples_for(readout)
    samples[0].point_labels[1, 0] = float("nan")
    loader = DataLoader(samples, batch_size=1)
    loader.collate_fn = TypeLayoutCollate(loader.collate_fn, readout.basis_table,
        coefficient_projector=CoefficientMatrixMSEProjector(readout))
    batch = next(iter(loader))
    assert "partial NaN" in getattr(batch, TARGETS_ATTR)
    assert torch.isnan(batch.point_labels[1, 0])
    samples[0].point_labels = samples[0].point_labels[:, :1]
    with pytest.raises(ValueError, match="components"):
        next(iter(loader))


def test_datamodule_attaches_targets_only_to_training_and_can_clear_snapshot():
    readout = grouped_readout(["1x0e", "1x1o"], "point_type", False, 1)
    module = MatrixDataModule(basis_table=readout.basis_table, symmetric_matrix=False, loader_threads=0)
    module.setup("fit")
    module.train_dataset = samples_for(readout)
    module.val_dataset = module.train_dataset
    assert getattr(next(iter(module.train_dataloader())), TARGETS_ATTR, None) is None
    module.set_coefficient_mse_projector(CoefficientMatrixMSEProjector(readout))
    assert isinstance(getattr(next(iter(module.train_dataloader())), TARGETS_ATTR), dict)
    assert getattr(next(iter(module.val_dataloader())), TARGETS_ATTR, None) is None
    module.set_coefficient_mse_projector(None)
    assert getattr(next(iter(module.train_dataloader())), TARGETS_ATTR, None) is None


def test_transfer_moves_terms_inside_lists_without_pyg_store_or_cuda():
    readout = grouped_readout(["1x0e", "1x1o"], "max", True, 2)
    loader = DataLoader(samples_for(readout), batch_size=2)
    loader.collate_fn = TypeLayoutCollate(loader.collate_fn, readout.basis_table,
        coefficient_projector=CoefficientMatrixMSEProjector(readout))
    batch = next(iter(loader))
    wrapper = LitBasisMatrixModel(model_cls=_PredictionModel, basis_table=readout.basis_table)
    moved = wrapper.transfer_batch_to_device(batch, torch.device("meta"), 0)
    targets = getattr(moved, TARGETS_ATTR)
    assert TARGETS_ATTR not in moved.keys()
    for kind in ("node", "edge"):
        for entry in targets[kind].values():
            assert entry["residual"].device.type == "meta"
            assert isinstance(entry["terms"], list)
            for term in entry["terms"]:
                assert all(value.device.type == "meta" for value in term.values())


@pytest.mark.parametrize("complex_q", [False, True])
def test_real_mse_certificate_rejects_complex_inputs_without_discarding_imaginary_parts(complex_q):
    readout = grouped_readout(["1x0e", "1x1o"], "point_type", True, 1)
    samples = samples_for(readout)
    batch = next(iter(DataLoader(samples, batch_size=1)))
    if complex_q:
        readout.self_interactions[0].change_of_basis = readout.self_interactions[0].change_of_basis.to(torch.complex128)
        with pytest.raises(CoefficientMSEUnsupported, match="complex"):
            CoefficientMatrixMSEProjector(readout)
    else:
        with pytest.raises(CoefficientMSEUnsupported, match="complex"):
            CoefficientMatrixMSEProjector(readout)(batch, batch.point_labels.to(torch.complex128) + 1j,
                                                  batch.edge_labels.to(torch.complex128))
