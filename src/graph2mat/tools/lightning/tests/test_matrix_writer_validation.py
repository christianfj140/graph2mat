from types import SimpleNamespace

import numpy as np
import pytest

from graph2mat import BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.tools.lightning import MatrixWriter


class _FakeSparseMatrix:
    def write(self, out_file):
        out_file.write_text("valid prediction")


class _FakeExample:
    def __init__(self, point_types, edge_index, edge_types, edge_labels):
        self.metadata = {}
        self._point_types = point_types
        self._edge_index = edge_index
        self._edge_types = edge_types
        self.edge_labels = edge_labels.copy()

    def numpy_arrays(self):
        return {
            "point_types": self._point_types,
            "edge_index": self._edge_index,
            "edge_types": self._edge_types,
        }

    def convert_to(self, *args, **kwargs):
        return _FakeSparseMatrix()


def _h2o_like_writer_case():
    basis_table = BasisTableWithEdges(
        [
            PointBasis("A", R=2, basis=[3]),
            PointBasis("B", R=2, basis=[4]),
            PointBasis("C", R=2, basis=[6]),
            PointBasis("D", R=2, basis=[7]),
            PointBasis("E", R=2, basis=[8]),
        ]
    )
    processor = MatrixDataProcessor(
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    point_types = np.array([3, 3, 0, 0, 4, 4, 1, 1, 2, 3])
    edge_index = np.array(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
        ]
    )
    edge_types = basis_table.point_type_to_edge_type(point_types[edge_index])
    edge_labels = np.arange(155, dtype=float)

    class FakeBatch:
        num_graphs = 1

        def __init__(self):
            self._arrays = SimpleNamespace(
                ptr=np.array([0, len(point_types)]),
                n_edges=np.array([edge_index.shape[1]]),
                point_types=point_types,
                edge_types=edge_types,
            )
            self._example = _FakeExample(
                point_types=point_types,
                edge_index=edge_index,
                edge_types=edge_types,
                edge_labels=edge_labels,
            )

        def numpy_arrays(self):
            return self._arrays

        def get_example(self, index):
            assert index == 0
            return self._example

    node_label_len = basis_table.point_block_pointer(point_types)[-1]
    return processor, FakeBatch(), node_label_len, edge_labels


def _trainer(processor):
    return SimpleNamespace(
        datamodule=SimpleNamespace(data_processor=processor, out_matrix=None)
    )


def test_matrix_writer_fails_closed_on_too_long_edge_predictions(tmp_path):
    processor, batch, node_label_len, edge_labels = _h2o_like_writer_case()
    out_file = tmp_path / "prediction.mtx"
    predictions = {
        "node_labels": np.zeros(node_label_len),
        "edge_labels": np.concatenate([edge_labels, np.array([999.0])]),
    }

    with pytest.raises(
        ValueError,
        match=(
            "MatrixWriter refused to write predictions.*"
            "batch_idx=3.*consumed=155, total=156"
        ),
    ):
        MatrixWriter(str(out_file), splits=["predict"])._on_batch_end(
            "predict",
            trainer=_trainer(processor),
            pl_module=None,
            prediction=predictions,
            batch=batch,
            batch_idx=3,
            dataloader_idx=0,
        )

    assert not out_file.exists()


def test_matrix_writer_fails_closed_on_too_short_edge_predictions(tmp_path):
    processor, batch, node_label_len, edge_labels = _h2o_like_writer_case()
    out_file = tmp_path / "prediction.mtx"
    predictions = {
        "node_labels": np.zeros(node_label_len),
        "edge_labels": edge_labels[:-1],
    }

    with pytest.raises(
        ValueError,
        match=(
            "MatrixWriter refused to write predictions.*"
            "batch_idx=4.*expected 155, got 154"
        ),
    ):
        MatrixWriter(str(out_file), splits=["predict"])._on_batch_end(
            "predict",
            trainer=_trainer(processor),
            pl_module=None,
            prediction=predictions,
            batch=batch,
            batch_idx=4,
            dataloader_idx=0,
        )

    assert not out_file.exists()


def test_matrix_writer_succeeds_when_prediction_lengths_match(tmp_path):
    processor, batch, node_label_len, edge_labels = _h2o_like_writer_case()
    out_file = tmp_path / "prediction.mtx"
    predictions = {
        "node_labels": np.zeros(node_label_len),
        "edge_labels": edge_labels.copy(),
    }

    MatrixWriter(str(out_file), splits=["predict"])._on_batch_end(
        "predict",
        trainer=_trainer(processor),
        pl_module=None,
        prediction=predictions,
        batch=batch,
        batch_idx=5,
        dataloader_idx=0,
    )

    assert out_file.read_text() == "valid prediction"


def test_matrix_writer_h_only_one_sample_uses_pipeline_edge_label_length(tmp_path):
    processor, batch, node_label_len, edge_labels = _h2o_like_writer_case()
    out_file = tmp_path / "h_only_prediction.mtx"
    predictions = {
        "node_labels": np.zeros(node_label_len),
        "edge_labels": edge_labels.copy(),
    }

    MatrixWriter(str(out_file), splits=["predict"])._on_batch_end(
        "predict",
        trainer=_trainer(processor),
        pl_module=None,
        prediction=predictions,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    assert out_file.exists()


def test_samplewise_logger_survives_validation_inside_a_training_epoch(tmp_path):
    """Validation in the middle of a training epoch used to close the handle that the
    training batches were still writing to (audit LOSS_TRAIN-08)."""
    from graph2mat.core.data.metrics import elementwise_mse
    from graph2mat.tools.lightning.callbacks import SamplewiseMetricsLogger

    logger = SamplewiseMetricsLogger(metrics=[elementwise_mse], output_file=tmp_path / "m.csv")
    logger._writer("train").writerow(["a"])
    logger._writer("val").writerow(["b"])
    logger.close_file_handle("val")
    logger._writer("train").writerow(["c"])
    logger.close_file_handle("train")
    assert sorted((tmp_path / "m.csv").read_text().split()[1:]) == ["a", "b", "c"]


@pytest.mark.parametrize("fail_ok", [True, False])
def test_samplewise_metric_failure_warns_once_or_raises(tmp_path, fail_ok):
    import csv
    import math
    import warnings

    import torch
    from torch_geometric.data import Batch, Data

    from graph2mat.tools.lightning.callbacks import SamplewiseMetricsLogger

    class BrokenMetric:
        def __call__(self, **kwargs):
            raise ValueError("intentional metric failure")

    table = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    batch = Batch.from_data_list([
        Data(num_nodes=1, point_labels=torch.zeros(1), edge_labels=torch.empty(0),
             metadata={"path": str(tmp_path / name / "matrix.DM")})
        for name in ("first", "second")
    ])
    trainer = SimpleNamespace(sanity_checking=False, is_global_zero=True, current_epoch=0,
                              datamodule=SimpleNamespace(basis_table=table,
                                                         out_matrix="density_matrix", symmetric_matrix=True))
    logger = SamplewiseMetricsLogger(metrics=[BrokenMetric], output_file=tmp_path / "metrics.csv")
    logger.fail_ok = fail_ok
    outputs = {"node_labels": batch.point_labels, "edge_labels": batch.edge_labels}
    try:
        if not fail_ok:
            with pytest.raises(ValueError, match="intentional metric failure"):
                logger.on_validation_batch_end(trainer, None, outputs, batch, 0)
        else:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always")
                for index in range(3):
                    logger.on_validation_batch_end(trainer, None, outputs, batch, index)
            failures = [warning for warning in emitted if "BrokenMetric" in str(warning.message)]
            assert len(failures) == 1
            assert "ValueError" in str(failures[0].message)
            assert "intentional metric failure" in str(failures[0].message)
    finally:
        logger.close_file_handle("val")
    with (tmp_path / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if fail_ok:
        assert len(rows) == 6
        assert [row["sample_name"] for row in rows] == ["first", "second"] * 3
        assert all(math.isnan(float(row["BrokenMetric"])) for row in rows)
    else:
        assert not rows


def _gauge_h_only_run(tmp_path, n_runs=3):
    """h_and_overlap labels, a model that predicts H only (gauge recipe), trainer.test."""
    import warnings

    import pytorch_lightning as pl
    import sisl

    from graph2mat import AtomicTableWithEdges
    from graph2mat.core.data import metrics as M
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    basis = AtomicTableWithEdges([atom])
    for i in range(n_runs):
        g = sisl.Geometry([[0.0, 0.0, 0.0], [1.0 + 0.05 * i, 0.0, 0.0], [0.0, 1.1, 0.0]], atoms=atom, lattice=[10.0] * 3)
        H = sisl.Hamiltonian(g, orthogonal=False)
        for a in range(3):
            H[a, a] = (-1.0 - 0.1 * a, 1.0)
        H[0, 1] = H[1, 0] = (-0.5, 0.2)
        H[0, 2] = H[2, 0] = (-0.4, 0.15)
        (tmp_path / f"run{i}").mkdir()
        H.write(tmp_path / f"run{i}" / "sample.nc")
    pl.seed_everything(0)
    dm = MatrixDataModule(out_matrix="hamiltonian", root_dir=str(tmp_path), basis_table=basis,
                          matrix_component_policy="h_and_overlap", n_matrix_components=2,
                          sub_point_matrix=False, loader_threads=0, batch_size=3,
                          train_runs="*/sample.nc", test_runs="*/sample.nc")
    model = LitMACEMatrixModel(basis_table=basis, hidden_irreps="4x0e+2x1o", edge_hidden_irreps="2x0e+1x1o",
                               max_ell=1, num_interactions=1, symmetric_matrix=True, n_matrix_components=2,
                               n_output_components=1, loss=M.gauge_invariant_mse)
    return dm, model, pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False, callbacks=[])


def test_matrix_writer_attaches_the_reference_overlap_to_h_only_predictions(tmp_path):
    """Audit G4: a gauge H-only model (h_and_overlap labels) used to be written with S = I."""
    import json

    import sisl

    from graph2mat.tools.lightning.callbacks import MatrixWriter

    dm, model, trainer = _gauge_h_only_run(tmp_path)
    trainer.callbacks.append(MatrixWriter("pred", splits=["test"]))
    trainer.test(model, datamodule=dm, verbose=False)
    written = sorted(p for p in tmp_path.glob("run*/pred/*") if not p.name.endswith(".json"))
    assert written
    for path in written:
        run = path.parent.parent
        m = sisl.get_sile(path).read_hamiltonian()
        ref = sisl.get_sile(run / "sample.nc").read_hamiltonian()
        assert not m.orthogonal
        np.testing.assert_allclose(m.tocsr(m.S_idx).toarray(), ref.tocsr(ref.S_idx).toarray(), atol=1e-6)
        metadata = json.loads(path.with_name(path.name + ".metadata.json").read_text())
        assert metadata["overlap_source"] == "reference labels"


def test_matrix_writer_refuses_to_overwrite_another_structure(tmp_path):
    """Audit M7: several structures resolving to one output file used to overwrite each other."""
    from graph2mat.tools.lightning.callbacks import MatrixWriter

    dm, model, trainer = _gauge_h_only_run(tmp_path)
    trainer.callbacks.append(MatrixWriter("../pred.HSX", splits=["test"]))  # one file for every run
    with pytest.raises(ValueError, match="two structures"):
        trainer.test(model, datamodule=dm, verbose=False)


def test_samplewise_metrics_compare_h_only_predictions_with_the_h_labels(tmp_path):
    """Audit M8: (H, S) labels against H-only predictions gave NaN for every metric."""
    import csv

    from graph2mat.tools.lightning.callbacks import SamplewiseMetricsLogger

    dm, model, trainer = _gauge_h_only_run(tmp_path)
    out = tmp_path / "metrics.csv"
    trainer.callbacks.append(SamplewiseMetricsLogger(splits=["test"], output_file=str(out)))
    trainer.test(model, datamodule=dm, verbose=False)
    rows = list(csv.DictReader(out.open()))
    assert rows
    for name in ("block_type_mse", "elementwise_mse", "node_mse", "edge_mse"):
        assert all(np.isfinite(float(row[name])) for row in rows), name


def test_samplewise_metrics_write_one_file_per_ddp_rank(tmp_path):
    """Audit M9: under DDP only rank 0 wrote, covering 1/world_size of the split."""
    from types import SimpleNamespace

    from graph2mat.tools.lightning.callbacks import PlotMatrixError, SamplewiseMetricsLogger

    logger = SamplewiseMetricsLogger(output_file=str(tmp_path / "metrics.csv"))
    logger.setup(SimpleNamespace(world_size=1, global_rank=0), None, "test")
    assert logger.output_file == str(tmp_path / "metrics.csv")
    logger.setup(SimpleNamespace(world_size=2, global_rank=1), None, "test")
    assert logger.output_file == str(tmp_path / "metrics.rank1.csv")
    with pytest.raises(ValueError, match="DDP"):
        PlotMatrixError().setup(SimpleNamespace(world_size=2, global_rank=0), None, "test")
