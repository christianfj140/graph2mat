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
from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Type, Union

import numpy as np
import pytorch_lightning as pl
import sisl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.core.data.metrics import OrbitalMatrixMetric
from graph2mat.core.data.sparse import nodes_and_edges_to_sparse_orbital
from graph2mat.tools.viz import plot_basis_matrix


class MatrixWriter(Callback):
    """Callback to write predicted matrices to disk."""

    def __init__(
        self,
        output_file: str,
        splits: Sequence = [
            "train",
            "val",
            "test",
            "predict",
        ],  # I don't know why, but Sequence[str] breaks the lightning CLI
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

        # Get iterator to loop through matrices in batch
        matrix_iter = data_processor.yield_from_batch(batch, predictions=prediction)

        # Loop through structures in the batch
        for matrix_data in matrix_iter:
            sparse_orbital_matrix = matrix_data.convert_to(
                data_processor.default_out_format,
                matrix_component_policy=getattr(
                    data_processor, "matrix_component_policy", None
                ),
            )

            out_file = self._get_out_file(matrix_data, trainer)

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
        splits: Sequence = [
            "train",
            "val",
            "test",
        ],  # I don't know why, but Sequence[str] breaks the lightning CLI
        output_file: Union[str, Path] = "sample_metrics.csv",
    ):
        super().__init__()

        if splits in ["train", "val", "test"]:
            splits = [splits]
        elif isinstance(splits, str):
            raise ValueError(f"Invalid value for splits: {splits}")

        self.fail_ok = False
        if metrics is None:
            metrics = OrbitalMatrixMetric.__subclasses__()
            self.fail_ok = True

        self.splits = splits
        self.metrics = [metric() for metric in metrics]
        self.output_file = output_file

        self._init_file()

    def on_train_epoch_start(self, trainer, pl_module):
        if "train" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if "val" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if "test" in self.splits:
            self._on_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.splits:
            self._on_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val" in self.splits:
            self._on_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if "test" in self.splits:
            self._on_epoch_end(trainer, pl_module)

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

    def _on_epoch_start(self, trainer, pl_module):
        self.open_file_handle()

    def _on_batch_end(
        self, split, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
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
        self.csv_writer.writerows(iterator)

    def _on_epoch_end(self, trainer, pl_module):
        self.close_file_handle()

    def _init_file(self):
        # Open the file
        with open(self.output_file, "w", newline="") as csv_file:
            # And write the headers, i.e. column names
            metric_names = [metric.__class__.__name__ for metric in self.metrics]
            fieldnames = ["sample_name", *metric_names, "split", "epoch"]

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(fieldnames)

    def open_file_handle(self):
        # Open the file
        self.output_fd = open(self.output_file, "a", newline="")
        self.csv_writer = csv.writer(self.output_fd)

    def close_file_handle(self):
        self.output_fd.close()


class PlotMatrixError(Callback):
    """Callback to create plots of MAE and RMSE for each entry of matrix.

    It only works if the matrix for all structures has always exactly the
    same shape. This happens for example if the structures are all snapshots
    from a molecular dynamics simulation.

    This callback is then useful to visualize how the errors are distributed
    within the matrix.

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
        self.node_running_ae = None
        self.node_running_se = None
        self.edge_running_ae = None
        self.edge_running_se = None
        self.matrix_count = 0
        self.positions = None
        self.point_types = None
        self.cell = None

    def _on_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # And the atomic table, which will help us constructing the matrices from
        # the batched flat arrays.
        basis_table: AtomicTableWithEdges = trainer.datamodule.basis_table
        # Find out whether the model was trained with the imposition that the matrix
        # is symmetric.
        symmetric_matrix = trainer.datamodule.symmetric_matrix

        # Pointer arrays to understand where the data for each structure starts in the batch.
        atom_ptr = batch.ptr.numpy(force=True)
        edge_ptr = np.zeros_like(atom_ptr)
        np.cumsum(batch.n_edges.numpy(force=True), out=edge_ptr[1:])

        # Types for both atoms and edges.
        point_types = batch.point_types.numpy(force=True)
        edge_types = batch.edge_types.numpy(force=True)

        # Get the values for the node blocks and the pointer to the start of each block.
        node_labels = outputs["node_labels"].numpy(force=True)
        node_labels_ptr = basis_table.point_block_pointer(point_types)

        # Get the values for the edge blocks and the pointer to the start of each block.
        edge_index = batch.edge_index.numpy(force=True)
        if symmetric_matrix:
            full_edge_ptr = edge_ptr
            n_edges = batch.n_edges.numpy(force=True)

            edge_ptr = np.zeros_like(atom_ptr)
            np.cumsum((n_edges + 1) // 2, out=edge_ptr[1:])

            edge_slices = [
                slice(start, end, 2)
                for start, end in zip(full_edge_ptr[:-1], full_edge_ptr[1:])
            ]
            selected_edge_types = [edge_types[s] for s in edge_slices]
            selected_edge_indices = [edge_index[:, s] for s in edge_slices]
            if len(selected_edge_types) > 0:
                edge_types = np.concatenate(selected_edge_types)
                edge_index = np.concatenate(selected_edge_indices, axis=1)
            else:
                edge_types = np.empty(0, dtype=edge_types.dtype)
                edge_index = np.empty((2, 0), dtype=edge_index.dtype)

        edge_labels = outputs["edge_labels"].numpy(force=True)
        edge_labels_ptr = basis_table.edge_block_pointer(edge_types)

        # Save the relevant information for use when plotting
        # We only save the information for one configuration and assume the
        # rest are the same
        self.positions = batch.positions.numpy(force=True)[atom_ptr[0] : atom_ptr[1]]
        self.cell = batch.cell.numpy(force=True)[0:3]
        self.nsc = batch.nsc[0].numpy(force=True)
        self.point_types = point_types[atom_ptr[0] : atom_ptr[1]]
        self.node_labels_ptr = node_labels_ptr[atom_ptr[0] : (atom_ptr[1] + 1)]
        self.edge_labels_ptr = edge_labels_ptr[edge_ptr[0] : (edge_ptr[1] + 1)]
        self.edge_index = edge_index[:, edge_ptr[0] : edge_ptr[1]]

        # Calculate errors on the flat arrays
        node_error = node_labels - batch["point_labels"].numpy(force=True)
        edge_error = edge_labels - batch["edge_labels"].numpy(force=True)

        # Some edge labels are np.nan (which means no entry), manually set the error
        # for those to 0.
        edge_error[np.isnan(edge_error)] = 0

        # Sum the errors across the different configurations
        # First get the pointers that split the labels...
        atom_split = atom_ptr[1:-1]
        edge_split = edge_ptr[1:-1]

        edge_labels_split = edge_labels_ptr[edge_split]
        node_labels_split = node_labels_ptr[atom_split]
        np.testing.assert_array_equal(edge_labels_split.shape, node_labels_split.shape)
        # ..and then sum across the configurations
        self.matrix_count += edge_labels_split.shape[0] + 1
        nae, eae, nse, ese = (
            sum(np.split(np.abs(node_error), node_labels_split)),
            sum(np.split(np.abs(edge_error), edge_labels_split)),
            sum(np.split(np.square(node_error), node_labels_split)),
            sum(np.split(np.square(edge_error), edge_labels_split)),
        )
        self.node_running_ae = (
            nae + self.node_running_ae if self.node_running_ae is not None else nae
        )
        self.edge_running_ae = (
            eae + self.edge_running_ae if self.edge_running_ae is not None else eae
        )
        self.node_running_se = (
            nse + self.node_running_se if self.node_running_se is not None else nse
        )
        self.edge_running_se = (
            ese + self.edge_running_se if self.edge_running_se is not None else ese
        )

    def _on_epoch_end(self, trainer, pl_module):
        # Find out which matrix class we should use based on what matrix type the data
        # has been trained on.
        matrix_cls = {
            "density_matrix": sisl.DensityMatrix,
            "energy_density_matrix": sisl.EnergyDensityMatrix,
            "hamiltonian": sisl.Hamiltonian,
            "dynamical_matrix": sisl.DynamicalMatrix,
        }[trainer.datamodule.out_matrix]

        # The basis_table helps us arrange the errors into matrix format
        basis_table: AtomicTableWithEdges = trainer.datamodule.basis_table

        labels = ["MAE", "RMSE"]
        node_errors = [self.node_running_ae, self.node_running_se]
        edge_errors = [self.edge_running_ae, self.edge_running_se]

        assert self.point_types is not None
        for i, (label, node_error, edge_error) in enumerate(
            zip(labels, node_errors, edge_errors)
        ):
            unique_atoms = basis_table.get_sisl_atoms()

            geometry = sisl.Geometry(
                self.positions,
                atoms=[unique_atoms[at_type] for at_type in self.point_types],
                lattice=self.cell,
            )

            geometry.set_nsc(self.nsc)

            assert node_error is not None
            assert edge_error is not None
            if label == "RMSE":
                ne = np.sqrt(node_error / self.matrix_count)
                ee = np.sqrt(edge_error / self.matrix_count)
            else:
                ne = node_error / self.matrix_count
                ee = edge_error / self.matrix_count

            # Convert flat error array to matrix format
            matrix = nodes_and_edges_to_sparse_orbital(
                node_vals=ne,
                edge_vals=ee,
                edge_index=self.edge_index,
                geometry=geometry,
                sp_class=matrix_cls,
                symmetrize_edges=trainer.datamodule.symmetric_matrix,
                matrix_component_policy=getattr(
                    trainer.datamodule, "matrix_component_policy", None
                ),
            ).tocsr()

            # Plot image to figure object
            fig = plot_basis_matrix(
                matrix,
                configuration=geometry,
                point_lines=True,
                basis_lines=True,
                text=".3f",
                colorscale="temps",
            ).update_layout(title=label)

            if self.show:
                fig.show("browser")
            if self.store_in_logger:
                import PIL.Image

                # Convert image to png
                img_bytes = fig.to_image(format="png", width=800, height=600)
                img_buf = io.BytesIO(img_bytes)
                img = PIL.Image.open(img_buf, formats=["PNG"])
                # Convert to numpy array
                img_array = np.array(img)
                # Add to tensorboard
                tensorboard = trainer.logger.experiment
                tensorboard.add_image(
                    label, img_array, dataformats="HWC", global_step=trainer.global_step
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
