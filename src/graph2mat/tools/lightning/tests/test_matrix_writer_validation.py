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
