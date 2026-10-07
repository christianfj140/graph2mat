import logging
import os
import re
import threading
import zipfile
from copy import deepcopy
from pathlib import Path
from queue import Empty
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
        self._worker_zips = {}

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index: int) -> TorchBasisMatrixData:
        item = self.input_data[index]

        item = self._fix_multiprocessing_zipfile(item)

        data = self.data_cls.new(
            item, data_processor=self.data_processor, labels=self.load_labels
        )
        if str(item).endswith(".fdf"):
            # Total charge of the structure (SIESTA NetCharge, 0 by default), for models whose
            # charges depend on it (long_range="qeq", roadmap 8.4).
            path = item if isinstance(item, zipfile.Path) else Path(item)
            match = re.search(r"(?im)^\s*NetCharge\s+(\S+)", path.read_text(errors="ignore"))
            data.total_charge = torch.tensor([float(match.group(1)) if match else 0.0], dtype=torch.get_default_dtype())
        if self.spectral_window:
            # Projection columns of the spectral loss (MatrixDataModule.spectral_window).
            name = self.spectral_window if isinstance(self.spectral_window, str) else "spectral_window.npz"
            path = item if isinstance(item, zipfile.Path) else Path(item)
            with (path.parent / name).open("rb") as stream, np.load(stream) as f:
                data.spectral_C = torch.as_tensor(f["C"], dtype=torch.get_default_dtype())
            if data.spectral_C.shape[0] != self._n_orbitals(data):
                raise ValueError(f"{name} of {item} does not match its number of orbitals.")
        return data

    def _n_orbitals(self, data) -> int:
        types = data.point_types.numpy()
        return int(self.data_processor.basis_table.basis_size[types].sum())

    def _fix_multiprocessing_zipfile(self, item):
        """Give each child process its own reader for every file-backed ZIP."""
        if not isinstance(item, zipfile.Path) or os.getpid() == self._root_pid:
            return item
        if self._worker_pid != os.getpid():
            self.close()
            self._worker_pid = os.getpid()
        filename = item.root.filename
        if filename is None:
            # Memory-backed archives have no shared OS file offset to reopen.
            return item
        if filename not in self._worker_zips:
            self._worker_zips[filename] = zipfile.ZipFile(filename)
        return zipfile.Path(self._worker_zips[filename], at=item.at)

    def close(self):
        """Close readers owned by this dataset, never caller-owned archives."""
        for archive in self._worker_zips.values():
            archive.close()
        self._worker_zips.clear()


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
    queue: multiprocessing.Queue, datalist: list, counter: SimpleCounter,
    stop: threading.Event,
):
    index = 0
    while not stop.is_set():
        try:
            data = queue.get(timeout=0.1)
        except Empty:
            continue
        datalist[index] = data
        counter.inc()
        index = (index + 1) % len(datalist)


class RotatingPoolData(torch.utils.data.Dataset):
    """Wrapper for a dataset that continously loads data into a smaller pool.

    The data loading is performed in a separate process and is assumed to be IO bound.

    Parameters
    ----------
    dataset:
        The dataset to wrap.
    pool_size:
        The size of the pool to keep in memory.
    seed:
        Seed for sampling. None draws a seed from NumPy's global seeded RNG.
        Sampling order is reproducible, but asynchronous replacement timing is not.
    """

    def __init__(self, dataset: TorchBasisMatrixDataset, pool_size: int, seed=None, **kwargs):
        super().__init__(**kwargs)
        if pool_size < 1 or len(dataset) == 0:
            raise ValueError("RotatingPoolData needs a positive pool_size and nonempty dataset")
        self.pool_size = pool_size
        self.parent_data = dataset
        if seed is None:
            seed = int(np.random.randint(0, 2**32))
        self.rng = np.random.default_rng(seed)
        self.counter = SimpleCounter()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        data_list = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.manager = multiprocessing.Manager()
        self.data_pool = self.manager.list(data_list)
        self.loader_queue = multiprocessing.Queue(2)
        self._stop_transfer = threading.Event()
        self._closed = False

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=_rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
            daemon=True,
        )
        self.transfer_thread = threading.Thread(
            target=_transfer_thread,
            args=(self.loader_queue, self.data_pool, self.counter, self._stop_transfer),
            daemon=True,
        )
        try:
            self.loader_process.start()
            self.transfer_thread.start()
        except BaseException:
            self.close()
            raise

    def close(self):
        """Stop the owned thread, reader, queue and manager; safe to call twice."""
        if self._closed:
            return
        self._stop_transfer.set()
        # Keep the tensor sender alive while the thread finishes unpickling its item.
        if self.transfer_thread.ident is not None:
            self.transfer_thread.join(timeout=5)
            if self.transfer_thread.is_alive():
                raise RuntimeError("RotatingPoolData transfer thread did not stop")
        if self.loader_process.pid is not None:
            self.loader_process.terminate()
            self.loader_process.join(timeout=5)
        self.loader_queue.cancel_join_thread()
        self.loader_queue.close()
        self.manager.shutdown()
        self._closed = True

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
