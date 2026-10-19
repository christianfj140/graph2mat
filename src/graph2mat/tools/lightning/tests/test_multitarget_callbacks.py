"""Selected physical exports and per-target metrics never interpret packing as spin."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import sisl
import torch
from torch_geometric.data import Batch

from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.multitarget import prepare_multitarget_data
from graph2mat.core.data.metrics import elementwise_mse
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices
from graph2mat.core.data.tests.test_multitarget import TARGETS, processor_for
from graph2mat.core.data.tests.test_multitarget_reconstruction import predicted_labels
from graph2mat.tools.lightning.callbacks import MatrixWriter, SamplewiseMetricsLogger, PlotMatrixError


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def inputs(tmp_path, spin=1, twice=False):
    matrices = analytic_matrices(spin=spin)
    p = processor_for(matrices, spin)
    data = prepare_multitarget_data(p, **matrices, data_cls=TorchBasisMatrixData)
    data.metadata["path"] = str(tmp_path / "a" / "RUN.fdf")
    samples = [data]
    if twice:
        other = prepare_multitarget_data(p, **matrices, data_cls=TorchBasisMatrixData)
        other.metadata["path"] = str(tmp_path / "b" / "RUN.fdf")
        samples.append(other)
    batch = Batch.from_data_list(samples)
    prediction = {name: torch.tensor(np.tile(value, (len(samples), 1)), dtype=torch.float64)
                  for name, value in predicted_labels(spin).items()}
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=p, basis_table=p.basis_table,
        out_matrix="hamiltonian", symmetric_matrix=True), current_epoch=0,
        sanity_checking=False, is_global_zero=True)
    return p, batch, prediction, trainer


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("target", TARGETS)
def test_writer_selected_target_netcdf_roundtrip(tmp_path, spin, target):
    p, batch, prediction, trainer = inputs(tmp_path, spin)
    path = tmp_path / f"{target}.nc"
    writer = MatrixWriter(str(path), matrix_target=target)
    writer._on_batch_end("predict", trainer, None, prediction, batch, 0, 0)
    result = getattr(sisl.get_sile(path), "read_" + target)()
    expected = p.matrix_from_data(batch, prediction, matrix_target=target)[0]
    for component in range(1 if target == "overlap" else spin):
        np.testing.assert_allclose(result.tocsr(component).toarray(), expected.tocsr(component).toarray(),
                                   rtol=1e-10, atol=1e-10)
    if target != "overlap":
        np.testing.assert_allclose(result.tocsr(result.S_idx).toarray(), [[1.2, .35], [.35, 1.2]],
                                   rtol=1e-10, atol=1e-10)
    metadata = json.loads(path.with_suffix(".nc.metadata.json").read_text())
    assert metadata["matrix_target"] == target
    assert metadata["overlap_source"] == "predicted"
    assert metadata["matrix_target_descriptor"] == p.matrix_target_layout.to_dict()


@pytest.mark.parametrize("bad", ["second_nan", "missing_target", "extension", "collision"])
def test_export_failure_leaves_no_partial_files(tmp_path, bad):
    _, batch, prediction, trainer = inputs(tmp_path, twice=True)
    target = None if bad == "missing_target" else "hamiltonian"
    suffix = ".HSX" if bad == "extension" else ".nc"
    path = tmp_path / ("result" + suffix if bad == "collision" else "$name$" + suffix)
    sentinel = tmp_path / ("result" + suffix if bad == "collision" else "a" + suffix)
    sentinel.write_text("preserve user output")
    if bad == "second_nan":
        prediction["node_labels"][2, 0] = float("nan")
    writer = MatrixWriter(str(path), matrix_target=target)
    with pytest.raises(ValueError, match="target|format|finite|path|collision"):
        writer._on_batch_end("predict", trainer, None, prediction, batch, 0, 0)
    assert sentinel.read_text() == "preserve user output"
    assert not (tmp_path / ("b" + suffix)).exists()


def test_callbacks_do_not_mix_targets(tmp_path):
    p, batch, prediction, trainer = inputs(tmp_path)
    output = tmp_path / "metrics.csv"
    logger = SamplewiseMetricsLogger(metrics=[elementwise_mse], matrix_target="overlap", output_file=output)
    logger._on_batch_end("val", trainer, None, prediction, batch, 0, 0)
    logger.close_file_handle("val")
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    expected = (2*(1.2-1.)**2 + (.35-.2)**2) / 3
    assert float(row["elementwise_mse"]) == pytest.approx(expected, abs=1e-7)
    assert row["matrix_target"] == "overlap" and row["unit"] == "dimensionless"
    missing = tmp_path / "missing.csv"
    logger = SamplewiseMetricsLogger(metrics=[elementwise_mse], output_file=missing)
    with pytest.raises(ValueError, match="target"):
        logger._on_batch_end("val", trainer, None, prediction, batch, 0, 0)
    assert not missing.exists()
    plot = PlotMatrixError(matrix_target="overlap", store_in_logger=False)
    plot._setup()
    plot._on_batch_end(trainer, None, prediction, batch, 0, 0)
    assert plot.node_running_ae.shape == (2, 1)
    np.testing.assert_allclose(plot.node_running_ae, [[.2], [.2]], rtol=0, atol=1e-7)
    plot = PlotMatrixError(store_in_logger=False)
    plot._setup()
    with pytest.raises(ValueError, match="target"):
        plot._on_batch_end(trainer, None, prediction, batch, 0, 0)


def test_legacy_samplewise_selector_rejected_before_truncating(tmp_path):
    from graph2mat.core.data.tests.physical_fixtures import prepared_sample

    data, table = prepared_sample(target="hamiltonian", data_cls=TorchBasisMatrixData)
    p = data.metadata["data_processor"]
    batch = Batch.from_data_list([data])
    prediction = {"node_labels": batch.point_labels + .1, "edge_labels": batch.edge_labels + .1}
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=p, basis_table=table,
        out_matrix="hamiltonian", symmetric_matrix=True), current_epoch=0,
        sanity_checking=False, is_global_zero=True)
    path = tmp_path / "legacy.csv"
    path.write_text("preserve user metrics\n")
    logger = SamplewiseMetricsLogger(metrics=[elementwise_mse], matrix_target="hamiltonian", output_file=path)
    with pytest.raises(ValueError, match="matrix_target.*multitarget"):
        logger._on_batch_end("val", trainer, None, prediction, batch, 0, 0)
    assert path.read_text() == "preserve user metrics\n"
    assert not logger._handles and not logger._file_initialized


def test_samplewise_partial_spin_support_keeps_other_channel(tmp_path):
    _, batch, prediction, trainer = inputs(tmp_path, spin=2)
    batch.edge_labels[0, 3] = float("nan")
    batch.matrix_known_edges[0, 3] = False
    path = tmp_path / "dm.csv"
    logger = SamplewiseMetricsLogger(metrics=[elementwise_mse], matrix_target="density_matrix", output_file=path)
    logger._on_batch_end("val", trainer, None, prediction, batch, 0, 0)
    logger.close_file_handle("val")
    with path.open() as stream:
        row = next(csv.DictReader(stream))
    expected = (2*(.9-5/12)**2 + 2*(.7-5/12)**2 + (.2-5/12)**2) / 5
    assert float(row["elementwise_mse"]) == pytest.approx(expected, abs=1e-14)
