"""Literal periodic error maps, without replacing matrix conversion."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from graph2mat import BasisConfiguration, BasisMatrix, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.tools.lightning import callbacks


def _processor(symmetric):
    table = BasisTableWithEdges([
        PointBasis("A", R=.6, basis="0e", basis_convention="cartesian"),
        PointBasis("B", R=.6, basis="0e", basis_convention="cartesian"),
    ])
    return MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric,
                               sub_point_matrix=False, out_matrix="density_matrix")


def _sample(processor, components=1):
    config = BasisConfiguration(
        point_types=["A"], positions=np.zeros((1, 3)), basis=processor.basis_table,
        cell=np.diag([1., 10., 10.]), pbc=(True, False, False),
        matrix=BasisMatrix({(0, 0, image): np.ones((1, 1)) for image in range(3)},
                           nsc=np.array([3, 1, 1]), basis_count=np.array([1])),
    )
    data = TorchBasisMatrixData.new(config, processor)
    assert data.nsc.tolist() == [[3, 1, 1]]
    assert data.edge_index.shape == (2, 2)
    if components == 2:
        data.point_labels = data.point_labels[:, None].repeat(1, 2)
        data.edge_labels = data.edge_labels[:, None].repeat(1, 2)
    return data


def _trainer(processor):
    return SimpleNamespace(datamodule=SimpleNamespace(
        data_processor=processor, basis_table=processor.basis_table,
        symmetric_matrix=processor.symmetric_matrix, out_matrix=processor.out_matrix,
    ))


def _record_plots(monkeypatch):
    recorded = []
    def capture(matrix, **kwargs):
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        return SimpleNamespace(update_layout=lambda **layout: recorded.append(
            (layout["title"], values.copy(), kwargs["configuration"])
        ))
    monkeypatch.setattr(callbacks, "plot_basis_matrix", capture)
    return recorded


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("missing", ["none", "nodes", "edges", "all"])
def test_periodic_plot_uses_image_columns_and_valid_entry_counts(monkeypatch, symmetric, components, missing):
    processor = _processor(symmetric)
    trainer = _trainer(processor)
    recorded = _record_plots(monkeypatch)
    callback = callbacks.PlotMatrixError(show=False, store_in_logger=False)
    samples = [_sample(processor, components), _sample(processor, components)]
    for sample in samples[:1] if missing != "all" else samples:
        if missing in ("nodes", "all"):
            sample.point_labels[0] = float("nan")
        if missing in ("edges", "all"):
            # A partially absent component excludes the entire label row.
            if components == 2:
                sample.edge_labels[:, 0] = float("nan")
            else:
                sample.edge_labels[:] = float("nan")
    for batch_size in (1, 2, 1):
        recorded.clear()
        callback.on_test_epoch_start(trainer, None)
        for start in range(0, 2, batch_size):
            selected = samples[start:start + batch_size]
            batch = Batch.from_data_list(selected)
            outputs = {}
            for target, key in (("point_labels", "node_labels"), ("edge_labels", "edge_labels")):
                predictions = []
                for index, sample in enumerate(selected, start):
                    error = (1 if target == "point_labels" else 2) + 2 * index
                    value = torch.full_like(sample[target], error)
                    if components == 2:
                        value[:, 1] *= 2
                    predictions.append(torch.nan_to_num(sample[target]) + value)
                outputs[key] = torch.cat(predictions)
            callback.on_test_batch_end(trainer, None, outputs, batch, start)
        callback.on_test_epoch_end(trainer, None)
        want_node = 0 if missing == "all" else 3 if missing == "nodes" else 2
        want_edge = 0 if missing == "all" else 4 if missing == "edges" else 3
        want_node_rmse = 0 if missing == "all" else 3 if missing == "nodes" else np.sqrt(5)
        want_edge_rmse = 0 if missing == "all" else 4 if missing == "edges" else np.sqrt(10)
        wanted = [("MAE", want_node, want_edge), ("RMSE", want_node_rmse, want_edge_rmse)]
        assert len(recorded) == 2 * components
        for component in range(components):
            scale = component + 1
            for name, node, edge in wanted:
                title = name if components == 1 else f"{name}_component{component}"
                plot = next(item for item in recorded if item[0] == title)
                np.testing.assert_allclose(plot[1], [[node * scale, edge * scale, edge * scale]],
                                           rtol=0, atol=1e-12)
                assert plot[2].nsc.tolist() == [3, 1, 1]
        assert callback.matrix_count == 2


@pytest.mark.parametrize("change", ["types", "images", "edge_order"])
@pytest.mark.parametrize("same_batch", [False, True])
def test_plot_rejects_incompatible_matrix_layout_before_accumulating(monkeypatch, change, same_batch):
    processor = _processor(False)
    trainer = _trainer(processor)
    _record_plots(monkeypatch)
    first, second = _sample(processor), _sample(processor)
    if change == "types":
        second.point_types[:] = 1
    elif change == "images":
        second.neigh_isc[:] = 0
    else:
        second.neigh_isc = second.neigh_isc.flip(0)
        second.edge_index = second.edge_index.flip(1)
    callback = callbacks.PlotMatrixError(show=False, store_in_logger=False)
    callback.on_test_epoch_start(trainer, None)
    batches = [Batch.from_data_list([first, second])] if same_batch else [
        Batch.from_data_list([first]), Batch.from_data_list([second]),
    ]
    with pytest.raises(ValueError, match="matrix layout"):
        for index, batch in enumerate(batches):
            callback.on_test_batch_end(trainer, None, {
                "node_labels": batch.point_labels + 1,
                "edge_labels": batch.edge_labels + 1,
            }, batch, index)


def test_plot_allows_md_positions_to_move_with_identical_index_layout(monkeypatch):
    processor = _processor(True)
    trainer = _trainer(processor)
    recorded = _record_plots(monkeypatch)
    first, second = _sample(processor), _sample(processor)
    second.positions += .1
    batch = Batch.from_data_list([first, second])
    callback = callbacks.PlotMatrixError(show=False, store_in_logger=False)
    callback.on_test_epoch_start(trainer, None)
    callback.on_test_batch_end(trainer, None, {
        "node_labels": batch.point_labels + 1, "edge_labels": batch.edge_labels + 2,
    }, batch, 0)
    callback.on_test_epoch_end(trainer, None)
    np.testing.assert_array_equal(recorded[0][1], [[1., 2., 2.]])
