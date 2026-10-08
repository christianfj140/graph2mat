"""Pytorch_lightning uses callbacks that can be plugged into the main loop.

Callbacks are independent from each other and can be included
into the main loop, which might be training, testing, predicting...
They provide some extra functionality like writing predictions to
disk or progress tracking.

This module implements some callbacks that might be useful in particular
for the matrix learning process.
"""
import csv
import io
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Type, Union

import numpy as np
import pytorch_lightning as pl
import sisl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.core.data.metrics import OrbitalMatrixMetric, _valid_label_rows
from graph2mat.tools.viz import plot_basis_matrix


class MatrixWriter(Callback):
    """Callback to write predicted matrices to disk."""

    def __init__(
        self,
        output_file: str,
        # Only predictions by default: writing every training step is very expensive.
        splits: Union[str, Sequence[str]] = [
            "predict",
        ],
    ):
        """Callback to write produced matrices to disk.

        Parameters
        ----------
        output_file :
            Path to the output file.

            The structures might contain as metadata the path from which they were read.
            In that case, if ``output_file`` is a relative path, it will be relative to
            the directory from which the structure was read.

            If ``output_file`` contains the string "$name$", it will be replaced
            by the name of the directory from which the structure was read.
        splits :
            List of splits for which to write the matrices. Can be any combination of
            "train", "val", "test", "predict".
        """
        super().__init__()

        if splits in ["train", "val", "test", "predict"]:
            splits = [splits]
        elif isinstance(splits, str):
            raise ValueError(f"Invalid value for splits: {splits}")

        self.splits = splits
        self.output_file = output_file
        self.out_is_absolute = Path(output_file).is_absolute()
        self.out_has_suffix = Path(output_file).suffix != ""

    def _matrix_suffix(self, out_matrix: str) -> str:
        return {
            "density_matrix": ".DM",
            "hamiltonian": ".HSX",
            "energy_density_matrix": ".EDM",
        }.get(out_matrix, ".mtx")

    def _get_out_file(self, matrix_data, trainer: "pl.Trainer") -> Path:
        path = Path(matrix_data.metadata.get("path", Path()))
        out_file = Path(self.output_file.replace("$name$", path.parent.name))
        if not self.out_is_absolute:
            out_file = path.parent / out_file

        if self.out_has_suffix:
            return out_file

        suffix = self._matrix_suffix(getattr(trainer.datamodule, "out_matrix", None))
        file_stem = path.parent.name or "matrix"
        return out_file / f"{file_stem}{suffix}"

    def _on_batch_end(
        self,
        split: str,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # Get the data processor that the datamodule uses.
        data_processor: MatrixDataProcessor = trainer.datamodule.data_processor

        # Fully consume the batch before writing. ``yield_from_batch`` performs
        # its strongest edge-label accounting check after yielding the examples,
        # so writing inside the generator loop could leave invalid partial files.
        try:
            matrix_data_items = list(
                data_processor.yield_from_batch(batch, predictions=prediction)
            )
        except ValueError as exc:
            raise ValueError(
                "MatrixWriter refused to write predictions because label "
                "accounting failed before any files were written "
                f"(split={split!r}, batch_idx={batch_idx}, "
                f"dataloader_idx={dataloader_idx}): {exc}"
            ) from exc

        write_items = []
        for matrix_data in matrix_data_items:
            sparse_orbital_matrix = matrix_data.convert_to(
                data_processor.default_out_format,
                matrix_component_policy=getattr(
                    data_processor, "matrix_component_policy", None
                ),
            )

            out_file = self._get_out_file(matrix_data, trainer)
            write_items.append((sparse_orbital_matrix, out_file))

        # Loop through structures in the batch
        for sparse_orbital_matrix, out_file in write_items:
            if not out_file.parent.exists():
                out_file.parent.mkdir(parents=True)

            # And write the matrix to it.
            sparse_orbital_matrix.write(out_file)
            self._write_metadata(
                out_file=out_file,
                sparse_orbital_matrix=sparse_orbital_matrix,
                data_processor=data_processor,
                trainer=trainer,
            )

    def _write_metadata(
        self,
        out_file: Path,
        sparse_orbital_matrix,
        data_processor: MatrixDataProcessor,
        trainer: "pl.Trainer",
    ) -> None:
        if getattr(trainer.datamodule, "out_matrix", None) != "hamiltonian":
            return

        policy = getattr(data_processor, "matrix_component_policy", None)
        is_polarized = bool(
            getattr(getattr(sparse_orbital_matrix, "spin", None), "is_polarized", False)
        )
        is_orthogonal = bool(getattr(sparse_orbital_matrix, "orthogonal", True))

        metadata = {
            "out_matrix": "hamiltonian",
            "matrix_component_policy": policy,
            "n_matrix_components": getattr(data_processor, "n_matrix_components", None),
            "serialized_spin_polarized": is_polarized,
            "serialized_orthogonal": is_orthogonal,
        }
        if policy in ("h_only", "spin_h_only") and is_orthogonal:
            metadata["note"] = (
                "Hamiltonian-only prediction written without an overlap matrix. "
                "For non-orthogonal reference data this output is not a "
                "self-contained generalized-eigenproblem Hamiltonian+S file."
            )

        metadata_file = out_file.with_suffix(out_file.suffix + ".metadata.json")
        metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "train" in self.splits:
            self._on_batch_end(
                "train", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "val" in self.splits:
            self._on_batch_end(
                "val", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "test" in self.splits:
            self._on_batch_end(
                "test", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "predict" in self.splits:
            self._on_batch_end(
                "predict", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )


class SamplewiseMetricsLogger(Callback):
    """Creates a CSV file with multiple metrics for each sample of the dataset.

    This callback is needed because otherwise the metrics are computed and logged
    on a per-batch basis.

    Each row of the CSV file is a computation of all metrics for a single sample on a given epoch.
    Therefore, the csv file contains the following columns: [sample_name, ...metrics..., split_key, epoch_index]

    Parameters
    ----------
    metrics : Sequence[Type[OrbitalMatrixMetric]]
        List of metrics to compute.
    splits : Sequence[str], optional
        List of splits for which to compute the metrics. Can be any combination of "train", "val", "test".
    output_file : Union[str, Path], optional
        Path to the output CSV file.
    """

    def __init__(
        self,
        metrics: Union[Sequence[Type[OrbitalMatrixMetric]], None] = None,
        splits: Union[str, Sequence[str]] = [
            "train",
            "val",
            "test",
        ],
        output_file: Union[str, Path] = "sample_metrics.csv",
    ):
        super().__init__()

        if splits in ["train", "val", "test"]:
            splits = [splits]
        elif isinstance(splits, str):
            raise ValueError(f"Invalid value for splits: {splits}")

        self.fail_ok = False
        if metrics is None:
            metrics = [
                metric
                for metric in OrbitalMatrixMetric.__subclasses__()
                if not getattr(metric, "requires_model_output", False)
                and not getattr(metric, "requires_overlap_labels", False)
            ]
            self.fail_ok = True

        self.splits = splits
        self.metrics = [metric() for metric in metrics]
        self.output_file = output_file
        self._handles = {}
        self._warned_metric_failures = set()

        self._init_file()

    def on_train_epoch_end(self, trainer, pl_module):
        self.close_file_handle("train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.close_file_handle("val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.close_file_handle("test")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "train" in self.splits:
            self._on_batch_end(
                "train", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "val" in self.splits:
            self._on_batch_end(
                "val", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None
    ):
        if "test" in self.splits:
            self._on_batch_end(
                "test", trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def _on_batch_end(
        self, split, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # Sanity-check batches are not an epoch; with several processes, rank 0 writes.
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        # Get the atomic table, which some metrics might need.
        basis_table: AtomicTableWithEdges = trainer.datamodule.basis_table

        # Compute all the metrics
        metrics = []
        for metric in self.metrics:
            if (
                trainer.datamodule.out_matrix != "density_matrix"
                and metric.__class__.__name__ == "normalized_density_error"
            ):
                metrics.append(np.full(len(batch), np.nan))
                continue
            try:
                metrics.append(
                    metric(
                        nodes_pred=outputs["node_labels"],
                        nodes_ref=batch.point_labels,
                        edges_pred=outputs["edge_labels"],
                        edges_ref=batch.edge_labels,
                        batch=batch,
                        basis_table=basis_table,
                        config_resolved=True,
                        symmetric_matrix=trainer.datamodule.symmetric_matrix,
                    )[0]
                )
            except Exception as e:
                if self.fail_ok:
                    name = metric.__class__.__name__
                    if name not in self._warned_metric_failures:
                        warnings.warn(
                            f"Metric {name} failed; recording NaN. {type(e).__name__}: {e}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._warned_metric_failures.add(name)
                    metrics.append(np.full(len(batch), np.nan))
                else:
                    raise e

        # Concatenate them (if we wish to accumulate them) TODO

        # Get the index of the current epoch
        current_epoch = trainer.current_epoch
        # And the names of the samples that we are going to log
        if "path" in batch.metadata:
            sample_names = [Path(path).parent.name for path in batch.metadata["path"]]
        else:
            sample_names = [f"batch{batch_idx}_{i}" for i in range(len(batch))]

        # Create an iterator that will return the data to be written to the CSV file for each row.
        # That is, first the sample name, then the metrics, and finally the split and the epoch index.
        iterator = (
            [sample_names[i], *data, split, current_epoch]
            for i, data in enumerate(np.array(metrics).T)
        )

        # Write the data
        self._writer(split).writerows(iterator)

    def _init_file(self):
        # Open the file
        with open(self.output_file, "w", newline="") as csv_file:
            # And write the headers, i.e. column names
            metric_names = [metric.__class__.__name__ for metric in self.metrics]
            fieldnames = ["sample_name", *metric_names, "split", "epoch"]

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(fieldnames)

    def _writer(self, split):
        # One handle per split, opened on first use: a validation run in the middle of
        # a training epoch must not close (or leak) the training handle.
        if split not in self._handles:
            fd = open(self.output_file, "a", newline="")
            self._handles[split] = (fd, csv.writer(fd))
        return self._handles[split][1]

    def close_file_handle(self, split):
        handle = self._handles.pop(split, None)
        if handle is not None:
            handle[0].close()


class PlotMatrixError(Callback):
    """Callback to create plots of MAE and RMSE for each entry of matrix.

    It only works if the matrix for all structures has always exactly the
    same shape. This happens for example if the structures are all snapshots
    from a molecular dynamics simulation.

    This callback is then useful to visualize how the errors are distributed
    within the matrix.

    Atom types, matrix components and edge/image ordering must match throughout
    the split; changing MD coordinates is allowed. Missing label rows do not
    contribute to entry counts, and unobserved entries stay blank. Multiple
    components are plotted separately without interpreting errors as a physical
    Hamiltonian, overlap or density matrix.

    Parameters
    ----------
    split :
        Split for which to plot the errors. If None, the callback will be used for both
        validation and test splits.
    show :
        Whether to show the plots or not.
    store_in_logger :
        Whether to store the plots in the logger or not.
    """

    def __init__(
        self,
        split: Literal[None, "val", "test"] = None,
        show: bool = False,
        store_in_logger: bool = True,
    ):
        super().__init__()
        self.split = split
        self.show = show
        self.store_in_logger = store_in_logger

    def _setup(self):
        self._matrix_data = None
        for branch in ("node", "edge"):
            for reduction in ("ae", "se", "count"):
                setattr(self, f"{branch}_running_{reduction}", None)
        self.matrix_count = 0
        self.positions = self.point_types = self.cell = None

    def _on_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        processor = trainer.datamodule.data_processor
        references = processor.yield_from_batch(batch, as_matrix=False)
        predictions = processor.yield_from_batch(
            batch, predictions=outputs, as_matrix=False
        )
        for reference, prediction in zip(references, predictions):
            arrays = reference.numpy_arrays()
            predicted = prediction.numpy_arrays()
            if self._matrix_data is None:
                self._matrix_data = reference.copy(cls=BasisMatrixData)
                self.positions = arrays.positions.copy()
                self.point_types = arrays.point_types.copy()
                self.cell = arrays.cell.copy()
                self.nsc = arrays.nsc.squeeze().copy()
                self.edge_index = arrays.edge_index.copy()
                self.neigh_isc = arrays.neigh_isc.copy()
            else:
                first = self._matrix_data.numpy_arrays()
                for key in ("point_types", "edge_index", "edge_types", "neigh_isc", "nsc"):
                    if not np.array_equal(first[key], arrays[key]):
                        raise ValueError(
                            f"PlotMatrixError requires an identical matrix layout: {key} differs."
                        )
                for key in ("point_labels", "edge_labels"):
                    if first[key].shape != arrays[key].shape:
                        raise ValueError(
                            f"PlotMatrixError requires an identical matrix layout: {key} shape differs."
                        )

            for branch, key in (("node", "point_labels"), ("edge", "edge_labels")):
                target = arrays[key]
                values = predicted[key]
                if values.shape != target.shape:
                    raise ValueError(
                        f"PlotMatrixError prediction/reference shapes differ for {key}."
                    )
                valid = _valid_label_rows(target)
                if target.ndim == 2:
                    valid = valid[:, None]
                error = np.where(valid, values.astype(np.float64) - target, 0.)
                if not np.isfinite(error).all():
                    raise ValueError(
                        f"PlotMatrixError has non-finite predictions for valid {key}."
                    )
                terms = (
                    np.abs(error), np.square(error),
                    np.broadcast_to(valid, error.shape).astype(np.int64),
                )
                for reduction, term in zip(("ae", "se", "count"), terms):
                    name = f"{branch}_running_{reduction}"
                    previous = getattr(self, name)
                    setattr(self, name, term.copy() if previous is None else previous + term)
            self.matrix_count += 1

    def _on_epoch_end(self, trainer, pl_module):
        if self._matrix_data is None:
            return

        # Errors are not physical matrices: do not restore an atomic reference or
        # serialize spin/overlap channels as a single Hamiltonian or density matrix.
        processor = trainer.datamodule.data_processor.copy(sub_point_matrix=False)
        unique_atoms = processor.basis_table.get_sisl_atoms()
        geometry = sisl.Geometry(
            processor.basis_to_cartesian(self.positions),
            atoms=[unique_atoms[k] for k in self.point_types],
            lattice=processor.basis_to_cartesian(self.cell),
        )
        geometry.set_nsc(self.nsc)

        for label, node_total, edge_total in (
            ("MAE", self.node_running_ae, self.edge_running_ae),
            ("RMSE", self.node_running_se, self.edge_running_se),
        ):
            # Entries with no observations stay blank in the existing sparse plot.
            ne = np.divide(
                node_total, self.node_running_count,
                out=np.zeros_like(node_total), where=self.node_running_count > 0,
            )
            ee = np.divide(
                edge_total, self.edge_running_count,
                out=np.zeros_like(edge_total), where=self.edge_running_count > 0,
            )
            if label == "RMSE":
                ne, ee = np.sqrt(ne), np.sqrt(ee)
            components = ne.shape[1] if ne.ndim == 2 else 1
            for component in range(components):
                title = label if components == 1 else f"{label}_component{component}"
                matrix = processor.labels_to(
                    "scipy_csr",
                    data={
                        **self._matrix_data._data,
                        "point_labels": ne[:, component] if ne.ndim == 2 else ne,
                        "edge_labels": ee[:, component] if ee.ndim == 2 else ee,
                    },
                    threshold=None,
                )
                fig = plot_basis_matrix(
                    matrix, configuration=geometry, point_lines=True, basis_lines=True,
                    text=".3f", colorscale="temps",
                ).update_layout(title=title)
                if self.show:
                    fig.show("browser")
                if self.store_in_logger:
                    import PIL.Image

                    img_bytes = fig.to_image(format="png", width=800, height=600)
                    img_buf = io.BytesIO(img_bytes)
                    img = PIL.Image.open(img_buf, formats=["PNG"])
                    img_array = np.array(img)
                    trainer.logger.experiment.add_image(
                        title, img_array, dataformats="HWC", global_step=trainer.global_step
                    )

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.split in [None, "val"]:
            self._setup()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self.split in [None, "val"]:
            self._on_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split in [None, "val"]:
            self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self.split in [None, "test"]:
            self._setup()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self.split in [None, "test"]:
            self._on_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_test_epoch_end(self, trainer, pl_module):
        if self.split in [None, "test"]:
            self._on_epoch_end(trainer, pl_module)
