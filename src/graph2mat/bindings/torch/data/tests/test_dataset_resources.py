"""ZIP ownership and lifecycle regressions for lazy datasets."""

import io
import multiprocessing
import os
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph2mat import BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixDataset


class _TextSample:
    """Minimal supported data_cls, reading real paths rather than mocking I/O."""

    @classmethod
    def new(cls, item, *, data_processor, labels):
        return SimpleNamespace(
            text=item.read_text(), point_types=torch.tensor([0, 0]),
            reader_fd=item.root.fp.fileno() if isinstance(item, zipfile.Path) else None,
        )


def _read_child(dataset, connection):
    try:
        source_fds = [item.root.fp.fileno() for item in dataset.input_data]
        samples = [dataset[i] for i in range(len(dataset))]
        read_fds = [sample.reader_fd for sample in samples]
        dataset.close()
        readers_closed = []
        for descriptor in read_fds:
            try:
                os.fstat(descriptor)
            except OSError:
                readers_closed.append(True)
            else:
                readers_closed.append(False)
        connection.send({
            "texts": [sample.text for sample in samples],
            "source_fds": source_fds,
            "read_fds": read_fds, "readers_closed": readers_closed,
            "source_texts_after_close": [item.read_text() for item in dataset.input_data],
            "source_fds_after": [item.root.fp.fileno() for item in dataset.input_data],
        })
    except Exception as error:
        connection.send({"error": repr(error)})
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
        connection.close()


@pytest.mark.parametrize("read_before_fork", [False, True])
def test_fork_reopens_every_zip_without_mutating_source(tmp_path, read_before_fork):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This regression targets inherited file offsets after fork")
    archives = []
    try:
        paths = []
        for index in range(2):
            filename = tmp_path / f"{index}.zip"
            with zipfile.ZipFile(filename, "w") as archive:
                archive.writestr("sample.XV", f"sample-{index}")
            archive = zipfile.ZipFile(filename)
            archives.append(archive)
            paths.append(zipfile.Path(archive, "sample.XV"))
        dataset = TorchBasisMatrixDataset(paths, None, data_cls=_TextSample)
        if read_before_fork:
            assert dataset[0].text == "sample-0"
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_read_child, args=(dataset, sender))
        process.start()
        sender.close()
        try:
            assert receiver.poll(10), "ZIP reader child did not respond"
            result = receiver.recv()
            assert "error" not in result, result
            assert result["texts"] == ["sample-0", "sample-1"]
            assert result["read_fds"] != result["source_fds"]
            assert all(a != b for a, b in zip(result["read_fds"], result["source_fds"]))
            assert result["source_fds_after"] == result["source_fds"]
            assert all(result["readers_closed"])
            assert result["source_texts_after_close"] == ["sample-0", "sample-1"]
            process.join(10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
            receiver.close()
        assert dataset[1].text == "sample-1"
    finally:
        for archive in archives:
            archive.close()


def test_zip_fdf_loads_charge_and_spectral_sidecar(tmp_path):
    filename = tmp_path / "data.zip"
    columns = np.array([[1., 0.], [0., 1.]])
    buffer = io.BytesIO()
    np.savez(buffer, C=columns)
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr("run/sample.fdf", "NetCharge -1.5\n")
        archive.writestr("run/spectral_window.npz", buffer.getvalue())
    processor = MatrixDataProcessor(
        basis_table=BasisTableWithEdges([PointBasis("A", R=2, basis=[1])]),
        sub_point_matrix=False,
    )
    with zipfile.ZipFile(filename) as archive:
        dataset = TorchBasisMatrixDataset(
            [zipfile.Path(archive, "run/sample.fdf")], processor,
            data_cls=_TextSample, spectral_window=True,
        )
        sample = dataset[0]
        assert sample.total_charge.item() == -1.5
        np.testing.assert_array_equal(sample.spectral_C.numpy(), columns)
