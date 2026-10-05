import logging
import os
import re
import threading
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Generic, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import sisl
import torch.utils.data
from torch import multiprocessing

from graph2mat import BasisConfiguration, MatrixDataProcessor

from .data import TorchBasisMatrixData

__all__ = ["TorchBasisMatrixDataset", "InMemoryData", "RotatingPoolData"]

reading = False


class TorchBasisMatrixDataset(torch.utils.data.Dataset):
    """Stores all configuration info of a dataset.

    Given all of its arguments, it has information to generate all the
    `BasisMatrixTorchData` objects. However, **the objects are created on the fly
    as they are requested**. They are not stored by this class.

    `torch_geometric's` data loader can be used out of the box to load data from
    this dataset.

    Parameters
    ----------
    input_data :
        A list of input data. Each item can be of any kind that is possible to
        convert to the class specified by `data_cls` using the `new` method.
    data_processor :
        A data processor object that is passed to `data_cls.new` to assist with
        the creation of the data objects from the `input_data`.
    data_cls :
        The class of the data objects that will be generated from this dataset.
        Must have a `new` method that takes the `input_data` and `data_processor`
        as arguments to create a new object. The `new` method also receives a
        `labels` argument specifying whether matrix labels should be loaded
        or not for the configurations.
    load_labels :
        Whether to load the matrix labels or not.

    See Also
    --------
    InMemoryData
        A wrapper for a dataset that loads all data into memory.
    RotatingPoolData
        A wrapper for a dataset that continously loads data into a smaller pool.

    Examples
    --------

    .. code-block:: python

        from graph2mat import BasisConfiguration, MatrixDataProcessor
        from graph2mat.bindings.torch import

        # Initialize basis configurations (substitute ... by appropriate arguments)
        config_1 = BasisConfiguration(...)
        config_2 = BasisConfiguration(...)

        # Initialize data processor (substitute ... by appropriate arguments)
        processor = MatrixDataProcessor(...)

        # Initialize dataset
        dataset = TorchBasisMatrixDataset([config_1, config_2], processor)

        # Import the loader class from torch_geometric
        from torch_geometric.loader import DataLoader

        # Create a data loader from this dataset
        loader = DataLoader(dataset, batch_size=2)

    """

    def __init__(
        self,
        input_data: Sequence[Union[BasisConfiguration, Path, str, sisl.Geometry]],
        data_processor: MatrixDataProcessor,
        data_cls: Type[TorchBasisMatrixData] = TorchBasisMatrixData,
        load_labels: bool = True,
        spectral_window: Union[bool, str] = False,
    ):
        self.input_data = input_data
        self.data_processor = data_processor
        self.data_cls = data_cls
        self.load_labels = load_labels
        self.spectral_window = spectral_window

        self._root_pid = os.getpid()
        self._worker_pid = None

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index: int) -> TorchBasisMatrixData:
        item = self.input_data[index]

        self._fix_multiprocessing_zipfile(item)

        data = self.data_cls.new(
            item, data_processor=self.data_processor, labels=self.load_labels
        )
        if str(item).endswith(".fdf"):
            # Total charge of the structure (SIESTA NetCharge, 0 by default), for models whose
            # charges depend on it (long_range="qeq", roadmap 8.4).
            match = re.search(r"(?im)^\s*NetCharge\s+(\S+)", Path(str(item)).read_text(errors="ignore"))
            data.total_charge = torch.tensor([float(match.group(1)) if match else 0.0], dtype=torch.get_default_dtype())
        if self.spectral_window:
            # Projection columns of the spectral loss (MatrixDataModule.spectral_window).
            name = self.spectral_window if isinstance(self.spectral_window, str) else "spectral_window.npz"
            with np.load(Path(str(item)).parent / name) as f:
                data.spectral_C = torch.as_tensor(f["C"], dtype=torch.get_default_dtype())
            if data.spectral_C.shape[0] != self._n_orbitals(data):
                raise ValueError(f"{name} of {item} does not match its number of orbitals.")
        return data

    def _n_orbitals(self, data) -> int:
        types = data.point_types.numpy()
        return int(self.data_processor.basis_table.basis_size[types].sum())

    def _fix_multiprocessing_zipfile(self, item):
        """Fix issues with zipfiles when using multiprocessing.

        The buffered reader of zipfile for some reason can't be shared
        between processes. This function checks if
            1. The item is a zipfile.Path
            2. We are in a different process than the one that created the zipfile

        If so, it substitutes the buffered reader of the zipfile with a new one.
        This is done only once because (we assume) the zipfile is shared between
        all zipfile.Path.

        An alternative would be to substitute the zipfile with a new one, but then
        you'd have to do it for each item, because they all have a reference to the
        original zipfile.

        By just substituting the buffered reader, the zipfile also keeps the
        table used to access files fast. If we substituted the zipfile, it would
        need to re-read the table, which takes some time if the file is large.
        """

        if self._worker_pid is not None:
            # We already dealt with multiprocessing
            return item

        # Get the id of the process we are in, and store it so that we
        # can later skip if it is already saved.
        self._worker_pid = os.getpid()

        if isinstance(item, zipfile.Path) and self._worker_pid != self._root_pid:
            # Init a zipfile so that it inits a new buffered reader
            aux_zip = zipfile.ZipFile(item.root.filename)
            # Set the buffered reader to the new one
            item.root.fp = aux_zip.fp
            # Unlink the buffered reader from the auxiliary zipfile
            # so that it doesn't get closed on garbage collection
            aux_zip.fp = None

        return item


class InMemoryData(torch.utils.data.Dataset):
    """Wrapper for a dataset that loads all data into memory.

    Parameters
    ----------
    dataset:
        The dataset to wrap.
    size:
        If not None, it truncates the dataset to the given size.
    """

    def __init__(
        self, dataset: TorchBasisMatrixDataset, size: Optional[int] = None, **kwargs
    ):
        super().__init__(**kwargs)
        size = size or len(dataset)
        self.data_objects = [dataset[i] for i in range(size)]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index: int) -> TorchBasisMatrixData:
        return self.data_objects[index]


class SimpleCounter:
    def __init__(self):
        self.reset()

    def inc(self):
        self.count += 1

    def reset(self):
        self.count = 0

    def get_count(self):
        return self.count


def _rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)).tolist():
            queue.put(dataset[index])


def _transfer_thread(
    queue: multiprocessing.Queue, datalist: list, counter: SimpleCounter
):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()
            counter.inc()


class RotatingPoolData(torch.utils.data.Dataset):
    """Wrapper for a dataset that continously loads data into a smaller pool.

    The data loading is performed in a separate process and is assumed to be IO bound.

    Parameters
    ----------
    dataset:
        The dataset to wrap.
    pool_size:
        The size of the pool to keep in memory.
    """

    def __init__(self, dataset: TorchBasisMatrixDataset, pool_size: int, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        self.counter = SimpleCounter()
        self.manager = multiprocessing.Manager()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        data_list = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.data_pool = self.manager.list(data_list)
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=_rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
            daemon=True,
        )
        self.transfer_thread = threading.Thread(
            target=_transfer_thread,
            args=(self.loader_queue, self.data_pool, self.counter),
            daemon=True,
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index: int) -> TorchBasisMatrixData:
        return self.data_pool[index]

    def get_data_pool(self):
        """
        Get the minimal dataset handle object for transfering to dataloader workers

        Returns
        -------
            Multiprocessing proxy data object

        """
        return self.data_pool
