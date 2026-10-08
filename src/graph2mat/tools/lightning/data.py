"""Data loading for pytorch_lightning workflows."""
import atexit
import json
import logging
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from torch_geometric.loader.dataloader import DataLoader

from graph2mat import AtomicTableWithEdges, BasisTableWithEdges, MatrixDataProcessor
from graph2mat.bindings.torch import (
    InMemoryData,
    RotatingPoolData,
    TorchBasisMatrixDataset,
)
from graph2mat.bindings.torch.data import TorchBasisMatrixData
from graph2mat.bindings.torch.modules.graph2mat import torch_type_layout
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.core.data.configuration import MatrixComponentPolicy, PhysicsMatrixType
from graph2mat.core.data.node_feats import NodeFeature

from ._helpers import glob, maybe_clean_zip_path, maybe_zip_path


def _cleanup_temporary_root(directory, root, owner_pid):
    """Never remove another process's temporary data after fork."""
    if os.getpid() != owner_pid:
        return
    if isinstance(root, zipfile.Path):
        root.root.close()
    shutil.rmtree(directory, ignore_errors=True)


class TypeLayoutCollate:
    """PyG collate that also attaches the batch's per-type layout (roadmap 6.3).

    The layout (``graph2mat.core.modules.graph2mat.type_layout``) is computed here, in the
    DataLoader workers while the GPU runs, so that ``Graph2Mat.forward`` needs neither
    boolean masks nor host syncs. It is kept outside the PyG store (a plain attribute), so
    ``get_example`` and the rest of the batch API ignore it; the Lightning model moves it to
    the device in ``transfer_batch_to_device``.
    """

    def __init__(self, collate, basis_table):
        self.collate = collate
        self.basis_table = basis_table

    def __call__(self, data_list):
        batch = self.collate(data_list)
        layout = torch_type_layout(batch.point_types.numpy(), batch.edge_types.numpy(), self.basis_table)
        if getattr(batch, "point_labels", None) is not None:
            # Structure of every label element, for the per-structure losses
            # (gauge_invariant_mse): on the GPU it would cost a device -> host copy per step.
            from graph2mat.core.data.metrics import _per_element_structure_ids

            node_struct, edge_struct, n_structures = _per_element_structure_ids(
                batch.point_labels, batch.edge_labels, batch, self.basis_table
            )
            layout.update(node_struct=node_struct, edge_struct=edge_struct, n_structures=n_structures)
            if getattr(batch, "spectral_C", None) is not None:
                from graph2mat.core.data.metrics import spectral_window_layout

                layout.update(spectral_window_layout(batch.edge_labels, batch, self.basis_table))
        object.__setattr__(batch, LAYOUT_ATTR, layout)
        return batch


#: Physics matrices that are real-symmetric (M_ij(R) = M_ji(-R)^T) without spin-orbit.
SYMMETRIC_MATRIX_TYPES = ("density_matrix", "hamiltonian", "energy_density_matrix")


def resolve_symmetric_matrix(
    symmetric_matrix: Optional[bool], out_matrix: Optional[str] = None
) -> bool:
    """``None`` means: symmetric for density matrix, Hamiltonian and energy density matrix."""
    if symmetric_matrix is None:
        return out_matrix in SYMMETRIC_MATRIX_TYPES
    return bool(symmetric_matrix)


def infer_n_matrix_components_from_data_inputs(
    *,
    root_dir: str = ".",
    basis_files: Optional[str] = None,
    no_basis: Optional[dict] = None,
    basis_table: Optional[BasisTableWithEdges] = None,
    out_matrix: Optional[PhysicsMatrixType] = None,
    matrix_component_policy: MatrixComponentPolicy = "h_only",
    symmetric_matrix: Optional[bool] = None,
    sub_point_matrix: bool = True,
    initial_node_feats: str = "OneHotZ",
    train_runs: Optional[Union[str, List]] = None,
    val_runs: Optional[Union[str, List]] = None,
    test_runs: Optional[Union[str, List]] = None,
    runs_json: Optional[str] = None,
) -> Optional[int]:
    """Infer the number of matrix components from the first labeled sample."""
    root = maybe_zip_path(root_dir)

    if basis_table is None:
        if basis_files is None:
            return None
        basis_table = AtomicTableWithEdges.from_basis_glob(
            glob(root, basis_files), no_basis_atoms=no_basis
        )

    if runs_json is not None:
        json_path = Path(runs_json)
        if not json_path.is_absolute():
            json_path = maybe_clean_zip_path(root / json_path)

        with json_path.open("r") as f:
            runs_dict = json.load(f)
    else:
        runs_dict = {}

    data_processor = MatrixDataProcessor(
        basis_table=basis_table,
        out_matrix=out_matrix,
        matrix_component_policy=matrix_component_policy,
        symmetric_matrix=resolve_symmetric_matrix(symmetric_matrix, out_matrix),
        sub_point_matrix=sub_point_matrix,
        n_matrix_components=1,
        node_attr_getters=[
            NodeFeature.registry[k] for k in initial_node_feats.split(" ")
        ],
    )

    for split, runs in (
        ("train", train_runs),
        ("val", val_runs),
        ("test", test_runs),
    ):
        if isinstance(runs, str):
            runs = glob(root, runs)
        elif runs is None and split in runs_dict:
            runs = [maybe_clean_zip_path(root / p) for p in runs_dict[split]]

        first_run = next(iter(runs), None)
        if first_run is None:
            continue

        sample = TorchBasisMatrixData.new(
            first_run, data_processor=data_processor, labels=True
        )
        point_labels = getattr(sample, "point_labels", None)
        edge_labels = getattr(sample, "edge_labels", None)

        if point_labels is not None:
            return point_labels.shape[1] if point_labels.ndim == 2 else 1
        if edge_labels is not None:
            return edge_labels.shape[1] if edge_labels.ndim == 2 else 1

    return None


class MatrixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        out_matrix: Optional[PhysicsMatrixType] = None,
        basis_files: Optional[str] = None,
        no_basis: Optional[dict] = None,
        basis_table: Optional[BasisTableWithEdges] = None,
        root_dir: str = ".",
        train_runs: Optional[str] = None,
        val_runs: Optional[str] = None,
        test_runs: Optional[str] = None,
        predict_structs: Optional[str] = None,
        runs_json: Optional[str] = None,
        symmetric_matrix: Optional[bool] = None,
        sub_point_matrix: bool = True,
        n_matrix_components: int = 1,
        matrix_component_policy: MatrixComponentPolicy = "h_only",
        batch_size: int = 5,
        loader_threads: int = 1,
        copy_root_to_tmp: bool = False,
        store_in_memory: bool = False,
        rotating_pool_size: Optional[int] = None,
        initial_node_feats: str = "OneHotZ",
        spectral_window: Union[bool, str] = False,
        predict_labels: bool = False,
        split_seed: int = 32,
        val_fraction: float = 0.1,
    ):
        """

        Parameters
        ----------
            out_matrix :'density_matrix', 'hamiltonian', 'energy_density_matrix', 'dynamical_matrix'
            basis_files : Union[str, None]
            basis_table : Union[BasisTableWithEdges, None]
            root_dir :
                Path to the directory to use as root for all other paths (in case those other paths
                are relative). This root directory can also be a zip file.

                Setting this argument is useful because it makes the config files/checkpoints easier
                to transfer between different machines, as one then only needs to change the root
                directory which presumably is the directory containiing the dataset.
            train_runs : Optional[str]
            val_runs : Optional[str]
            test_runs : Optional[str]
            predict_structs : Optional[str]
            runs_json: Optional[str]
                Path to json-file with a dictionary where the keys are train/val/test/predict.
                and the dictionary values are list of paths to the run files relative to `root_dir`
                The paths will be overwritten by train_runs/val_runs/test_runs/predict_structs if given.
            symmetric_matrix : bool, optional
                Whether the matrix is symmetric. ``None`` (default) chooses it from
                ``out_matrix``: True for density matrix, Hamiltonian and energy
                density matrix, False otherwise.
            sub_point_matrix : bool
            matrix_component_policy : str
                Component policy for multi-component Hamiltonian matrices.
            batch_size : int
            loader_threads : int
            copy_root_to_tmp: bool
            store_in_memory: bool
                If true, will load the dataset into host memory, otherwise it will be read from disk
            rotating_pool_size: int
                If given, the training data will be continously loaded into a smaller poool of this
                size. The data in the active pool can be used several times before it is swapped out
                with new data. This is useful if the data loading is slow. Note that the notion of
                epochs will not be meaningful when this kind of loading is used.
                This will not affect test/val/predict data.
            spectral_window: bool or str
                Attach to every labelled structure the reference eigenvectors of a window around
                E_F at Gamma (``spectral_C``), read from ``spectral_window.npz`` (array ``C``,
                one row per orbital, the same number of columns in every structure, zero
                columns as padding) next to its run file. Needed by the spectral term of
                ``gauge_invariant_mse`` (``spectral_weight``, roadmap 4.4). A string reads
                that file instead, with any S-orthonormal columns (e.g. the near-null
                directions of S, roadmap 4.7).
            predict_labels: bool
                Also load the labels of the structures to predict: a model with
                ``long_range`` needs their overlap S (geometry only, as in H0).
            split_seed: int
                Seed for the automatic train/validation split (default 32).
            val_fraction: float
                Fraction drawn for validation if no explicit validation data is given.
                Must be in [0, 1); at least one training sample is retained.

        """
        super().__init__()
        if not isinstance(split_seed, (int, np.integer)) or not 0 <= split_seed < 2**32:
            raise ValueError("split_seed must be an integer in [0, 2**32)")
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        self.save_hyperparameters()

        self.root_dir = maybe_zip_path(root_dir)

        self.basis_files = basis_files
        self.no_basis = no_basis
        self.basis_table = basis_table
        self.out_matrix: Optional[PhysicsMatrixType] = out_matrix
        self.symmetric_matrix = resolve_symmetric_matrix(symmetric_matrix, out_matrix)
        # Checkpoints must record the value actually used, not "auto".
        self.hparams["symmetric_matrix"] = self.symmetric_matrix
        self.initial_node_feats = [
            NodeFeature.registry[k] for k in initial_node_feats.split(" ")
        ]

        self.train_runs = train_runs
        self.val_runs = val_runs
        self.test_runs = test_runs
        self.runs_json = runs_json

        self.predict_runs = predict_structs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.sub_point_matrix = sub_point_matrix
        self.n_matrix_components = n_matrix_components
        self.matrix_component_policy = matrix_component_policy

        self.batch_size = batch_size
        self.copy_root_to_tmp = copy_root_to_tmp
        self.store_in_memory = store_in_memory
        self.rotating_pool_size = rotating_pool_size
        self.spectral_window = spectral_window
        self.predict_labels = predict_labels
        self.split_seed = split_seed
        self.val_fraction = val_fraction
        self.prepare_data_per_node = True
        self.tmp_dir = None
        self._tmp_root = None
        self._tmp_pid = None

    def prepare_data(self):
        if not self.copy_root_to_tmp:
            return
        if self.tmp_dir is not None and self._tmp_pid == os.getpid():
            return
        if self.tmp_dir is not None:
            # Cached lazy datasets inherited after fork still point at the parent's copy.
            for split in ("train", "val", "test", "predict"):
                setattr(self, "%s_dataset" % split, None)
            if isinstance(self._tmp_root, zipfile.Path):
                self._tmp_root.root.close()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="graph2mat_"))
        self._tmp_pid = os.getpid()
        self._tmp_root = None
        logging.info("copying %s to %s", self.root_dir, self.tmp_dir)
        try:
            if isinstance(self.root_dir, zipfile.Path):
                destination = self.tmp_dir / Path(self.root_dir.root.filename).name
                shutil.copy2(self.root_dir.root.filename, destination)
                self._tmp_root = zipfile.Path(
                    destination,
                    at=self.root_dir.at if self.root_dir.at != "." else "",
                )
            else:
                shutil.copytree(self.root_dir, self.tmp_dir, dirs_exist_ok=True)
                self._tmp_root = self.tmp_dir
        except BaseException:
            _cleanup_temporary_root(self.tmp_dir, self._tmp_root, self._tmp_pid)
            self.tmp_dir = None
            raise
        atexit.register(_cleanup_temporary_root, self.tmp_dir, self._tmp_root, self._tmp_pid)

    def teardown(self, stage: Optional[str] = None):
        if stage in (None, "fit") and isinstance(self.train_dataset, RotatingPoolData):
            pool = self.train_dataset
            pool.close()
            self.train_dataset = pool.parent_data
        # Lightning tears down every fit/test/predict call. Lazy datasets must remain
        # readable for the next call; None (or process exit) performs final cleanup.
        if stage is None:
            for split in ("train", "val", "test", "predict"):
                dataset = getattr(self, "%s_dataset" % split)
                while isinstance(dataset, torch.utils.data.Subset):
                    dataset = dataset.dataset
                if isinstance(dataset, TorchBasisMatrixDataset):
                    dataset.close()
                setattr(self, "%s_dataset" % split, None)
            if isinstance(self.root_dir, zipfile.Path):
                self.root_dir.root.close()
            if self.tmp_dir is not None:
                logging.info("deleting dir %s", self.tmp_dir)
                _cleanup_temporary_root(self.tmp_dir, self._tmp_root, self._tmp_pid)
            self.tmp_dir = None
            self._tmp_root = None
            self._tmp_pid = None

    def setup(self, stage: Optional[str] = None):
        stage_splits = {
            None: ("train", "val", "test", "predict"),
            "fit": ("train", "val"),
            "validate": ("val",),
            "test": ("test",),
            "predict": ("predict",),
        }
        if stage not in stage_splits:
            raise ValueError(f"Unknown setup stage: {stage!r}")
        requested_splits = stage_splits[stage]

        if isinstance(self.root_dir, zipfile.Path) and self.root_dir.root.fp is None:
            self.root_dir = maybe_zip_path(str(self.root_dir))

        if self.copy_root_to_tmp:
            # prepare_data runs only on local rank zero; each other rank owns its copy.
            # ponytail: per-rank copies avoid shared cleanup races; share if I/O proves costly.
            self.prepare_data()
            root = self._tmp_root
        else:
            root = self.root_dir
        if self.basis_table is None:
            # Read the basis from the basis files provided.
            assert self.basis_files is not None
            self.basis_table = AtomicTableWithEdges.from_basis_glob(
                glob(root, self.basis_files), no_basis_atoms=self.no_basis
            )

        # Load json file with paths for each split
        if self.runs_json is not None:
            json_path = Path(self.runs_json)
            if not json_path.is_absolute():
                json_path = maybe_clean_zip_path(root / json_path)

            f = json_path.open("r")
            runs_dict = json.load(f)
            f.close()
        else:
            runs_dict = {}

        if getattr(self, "data_processor", None) is None:
            self.data_processor = MatrixDataProcessor(
                basis_table=self.basis_table,
                out_matrix=self.out_matrix,
                matrix_component_policy=self.matrix_component_policy,
                symmetric_matrix=self.symmetric_matrix,
                sub_point_matrix=self.sub_point_matrix,
                n_matrix_components=self.n_matrix_components,
                node_attr_getters=self.initial_node_feats,
            )

        needed_splits = list(requested_splits)
        if (
            "val" in requested_splits and self.val_dataset is None
            and self.val_runs is None and "val" not in runs_dict
            and "train" not in needed_splits and self.train_dataset is None
        ):
            # Validation may need the automatic split, but not all train samples in memory.
            needed_splits.append("train")
        # Set the paths for each split
        for split in needed_splits:
            if getattr(self, "%s_dataset" % split) is not None:
                continue
            runs = getattr(self, "%s_runs" % split)
            if isinstance(runs, str):
                # This is a glob pattern
                runs = glob(root, runs)
            # Else use the json file
            elif runs is None and split in runs_dict:
                runs = [maybe_clean_zip_path(root / p) for p in runs_dict[split]]

            if runs is not None:
                # Contruct the dataset
                # For predictions, we don't need to load the labels (actually we don't have them)
                # For the other splits, we need to load the labels (target matrices)
                dataset = TorchBasisMatrixDataset(
                    list(runs),
                    data_processor=self.data_processor,
                    data_cls=TorchBasisMatrixData,
                    load_labels=split != "predict" or self.predict_labels,
                    spectral_window=self.spectral_window if split != "predict" else False,
                )

                setattr(self, "%s_dataset" % split, dataset)

        # Now check if we should take some data out of the training data to use it for
        # validation.
        if "val" in requested_splits and self.val_dataset is None and self.train_dataset is not None:
            if len(self.train_dataset) == 1:
                logging.warning(
                    "There is only one training sample and no validation samples."
                    " Unable to draw a validation set from the training data, therefore"
                    " proceeding without validation dataset."
                )

                train_indices = np.array([0])
                val_indices = np.array([])
            else:
                # Instantiate a random state to sample from data
                rng = np.random.RandomState(self.split_seed)
                # Draw the configured fraction, retaining at least one training sample.
                perms = rng.permutation(len(self.train_dataset))
                num_val = min(math.ceil(len(self.train_dataset) * self.val_fraction),
                              max(len(self.train_dataset)-1, 0))
                val_indices = np.sort(perms[0:num_val]).tolist()
                train_indices = np.sort(perms[num_val:]).tolist()

            self.val_dataset = torch.utils.data.Subset(self.train_dataset, val_indices)
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset, train_indices
            )

        if self.store_in_memory:
            for split in requested_splits:
                dataset = getattr(self, "%s_dataset" % split)
                if dataset is None or isinstance(dataset, (InMemoryData, RotatingPoolData)):
                    continue
                if self.rotating_pool_size and split == "train":
                    logging.warning(
                        "Does not load training data to memory because rotating_pool_size is set"
                    )
                else:
                    logging.debug("Loading dataset split=%s into memory" % split)
                    setattr(self, "%s_dataset" % split, InMemoryData(dataset))

        self._validate_n_matrix_components(requested_splits)

        # Wrap training dataset in rotating pool
        if (
            self.rotating_pool_size and "train" in requested_splits
            and self.train_dataset is not None
            and not isinstance(self.train_dataset, RotatingPoolData)
        ):
            self.train_dataset = RotatingPoolData(
                self.train_dataset, self.rotating_pool_size
            )

    def _infer_dataset_matrix_components(self, splits=("train", "val", "test")) -> Optional[int]:
        """Infer matrix components from the first labeled sample available."""
        for split in splits:
            dataset = getattr(self, "%s_dataset" % split)
            if dataset is None or len(dataset) == 0:
                continue

            sample = dataset[0]
            point_labels = getattr(sample, "point_labels", None)
            if point_labels is None:
                continue

            return point_labels.shape[1] if point_labels.ndim == 2 else 1

        return None

    def _validate_n_matrix_components(self, splits=("train", "val", "test")) -> None:
        inferred = self._infer_dataset_matrix_components(splits)
        if inferred is None:
            return

        if inferred != self.n_matrix_components:
            raise ValueError(
                "n_matrix_components mismatch: datamodule is configured with "
                f"n_matrix_components={self.n_matrix_components}, but loaded labels "
                f"have {inferred} component(s) after matrix_component_policy="
                f"{self.matrix_component_policy!r}. Set "
                f"data.n_matrix_components={inferred}. "
                "If using the Lightning CLI, this value is linked to "
                "model.n_matrix_components."
            )

    def train_dataloader(self):
        assert (
            self.train_dataset is not None
        ), "No training data was provided, please set the ``train_runs`` argument."
        if self.rotating_pool_size:
            assert isinstance(self.train_dataset, RotatingPoolData)
            data_handle = self.train_dataset.get_data_pool()
        else:
            data_handle = self.train_dataset
        return self._with_type_layout(
            DataLoader(
                data_handle,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                **self._loader_kwargs(),
            )
        )

    def _with_type_layout(self, loader):
        """Training and validation batches carry their type layout (symmetric matrices,
        the only case Graph2Mat's fast path covers)."""
        if self.data_processor.symmetric_matrix:
            loader.collate_fn = TypeLayoutCollate(loader.collate_fn, self.data_processor.basis_table)
        return loader

    def _loader_kwargs(self) -> dict:
        """Worker settings shared by all loaders: workers are kept alive between
        epochs (validation included) and batches go to pinned memory for the GPU."""
        workers = self.hparams.loader_threads
        return {
            "num_workers": workers,
            "persistent_workers": bool(workers),
            "pin_memory": torch.cuda.is_available(),
        }

    def val_dataloader(self):
        assert (
            self.val_dataset is not None
        ), "No validation data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return self._with_type_layout(
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                **self._loader_kwargs(),
            )
        )

    def test_dataloader(self):
        assert (
            self.test_dataset is not None
        ), "No test data was provided, please set the ``test_runs`` argument."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def predict_dataloader(self):
        assert (
            self.predict_dataset is not None
        ), "No prediction data was provided, please set the ``predict_structs`` argument."
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )
