"""Data loading for pytorch_lightning workflows."""
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
from graph2mat.core.data.configuration import MatrixComponentPolicy, PhysicsMatrixType
from graph2mat.core.data.node_feats import NodeFeature

from ._helpers import glob, maybe_clean_zip_path, maybe_zip_path


def infer_n_matrix_components_from_data_inputs(
    *,
    root_dir: str = ".",
    basis_files: Optional[str] = None,
    no_basis: Optional[dict] = None,
    basis_table: Optional[BasisTableWithEdges] = None,
    out_matrix: Optional[PhysicsMatrixType] = None,
    matrix_component_policy: MatrixComponentPolicy = "h_only",
    symmetric_matrix: bool = False,
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
        symmetric_matrix=symmetric_matrix,
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
        symmetric_matrix: bool = False,
        sub_point_matrix: bool = True,
        n_matrix_components: int = 1,
        matrix_component_policy: MatrixComponentPolicy = "h_only",
        batch_size: int = 5,
        loader_threads: int = 1,
        copy_root_to_tmp: bool = False,
        store_in_memory: bool = False,
        rotating_pool_size: Optional[int] = None,
        initial_node_feats: str = "OneHotZ",
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
            symmetric_matrix : bool
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

        """
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = maybe_zip_path(root_dir)

        self.basis_files = basis_files
        self.no_basis = no_basis
        self.basis_table = basis_table
        self.out_matrix: Optional[PhysicsMatrixType] = out_matrix
        self.symmetric_matrix = symmetric_matrix
        self.initial_node_feats = [
            NodeFeature.registry[k] for k in initial_node_feats.split(" ")
        ]

        self.train_runs = train_runs
        self.val_runs = val_runs
        self.test_runs = test_runs
        self.runs_json = runs_json

        self.predict_runs = predict_structs
        self.sub_point_matrix = sub_point_matrix
        self.n_matrix_components = n_matrix_components
        self.matrix_component_policy = matrix_component_policy

        self.batch_size = batch_size
        self.copy_root_to_tmp = copy_root_to_tmp
        self.store_in_memory = store_in_memory
        self.rotating_pool_size = rotating_pool_size
        self.prepare_data_per_node = True
        if self.copy_root_to_tmp:
            self.tmp_dir = Path(tempfile.gettempdir()) / "e3nn_matrix"
        else:
            self.tmp_dir = None

    def prepare_data(self):
        if self.copy_root_to_tmp:
            assert self.tmp_dir is not None
            os.makedirs(self.tmp_dir)
            logging.info("copying %s to %s" % (self.root_dir, self.tmp_dir))
            shutil.copytree(self.root_dir, self.tmp_dir, dirs_exist_ok=True)

    def teardown(self, stage: str):
        if self.tmp_dir is not None:
            logging.info("deleting dir %s" % (self.tmp_dir))
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def setup(self, stage: str):
        if self.copy_root_to_tmp:
            assert self.tmp_dir is not None
            root = self.tmp_dir
        else:
            root = self.root_dir
        if self.basis_table is None:
            # Read the basis from the basis files provided.
            assert self.basis_files is not None
            self.basis_table = AtomicTableWithEdges.from_basis_glob(
                glob(root, self.basis_files), no_basis_atoms=self.no_basis
            )

        # Initialize the data.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

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

        self.data_processor = MatrixDataProcessor(
            basis_table=self.basis_table,
            out_matrix=self.out_matrix,
            matrix_component_policy=self.matrix_component_policy,
            symmetric_matrix=self.symmetric_matrix,
            sub_point_matrix=self.sub_point_matrix,
            n_matrix_components=self.n_matrix_components,
            node_attr_getters=self.initial_node_feats,
        )

        # Set the paths for each split
        for split in ["train", "val", "test", "predict"]:
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
                    load_labels=split != "predict",
                )

                if self.store_in_memory:
                    if self.rotating_pool_size and split == "train":
                        logging.warning(
                            "Does not load training data to memory because rotating_pool_size is set"
                        )
                    else:
                        logging.debug("Loading dataset split=%s into memory" % split)
                        dataset = InMemoryData(dataset)

                setattr(self, "%s_dataset" % split, dataset)

        self._validate_n_matrix_components()

        # Now check if we should take some data out of the training data to use it for
        # validation.
        if self.val_dataset is None and self.train_dataset is not None:
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
                rng = np.random.RandomState(32)
                # User didn't specify a validation directory, just randomly draw 10 percent from the training.
                perms = rng.permutation(len(self.train_dataset))
                num_val = int(math.ceil(len(self.train_dataset) / 10))
                val_indices = np.sort(perms[0:num_val]).tolist()
                train_indices = np.sort(perms[num_val:]).tolist()

            self.val_dataset = torch.utils.data.Subset(self.train_dataset, val_indices)
            self.train_dataset = torch.utils.data.Subset(
                self.train_dataset, train_indices
            )

        # Wrap training dataset in rotating pool
        if self.rotating_pool_size:
            self.train_dataset = RotatingPoolData(
                self.train_dataset, self.rotating_pool_size
            )

    def _infer_dataset_matrix_components(self) -> Optional[int]:
        """Infer matrix components from the first labeled sample available."""
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is None:
                continue

            sample = dataset[0]
            point_labels = getattr(sample, "point_labels", None)
            if point_labels is None:
                continue

            return point_labels.shape[1] if point_labels.ndim == 2 else 1

        return None

    def _validate_n_matrix_components(self) -> None:
        inferred = self._infer_dataset_matrix_components()
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
        return DataLoader(
            data_handle,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
            persistent_workers=bool(self.hparams.loader_threads),
        )

    def val_dataloader(self):
        assert (
            self.val_dataset is not None
        ), "No validation data was provided, please set either the ``train_runs`` or the ``val_runs`` argument."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.loader_threads,
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
            num_workers=self.hparams.loader_threads,
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
            num_workers=self.hparams.loader_threads,
        )
