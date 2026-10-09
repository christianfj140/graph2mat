"""Cached CPU tensors and metadata have independent per-access ownership."""

import io
import multiprocessing
import pickle
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from graph2mat import AtomicTableWithEdges, BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import InMemoryData, TorchBasisMatrixData, TorchBasisMatrixDataset
from graph2mat.bindings.torch.data.tests.test_dataset_resources import _TextSample
from graph2mat.core.data.matrices.basis_matrix import BasisMatrix


def dataset_for(components=1, symmetric=True, preprocessed=False, config_cls=BasisConfiguration):
    table = BasisTableWithEdges([
        PointBasis("A", R=1.2, basis="1x0e+1x1o+1x2e"),
        PointBasis("B", R=1.2, basis="1x0e+1x1o+1x3o"),
    ])
    counts = np.array([9, 11])
    blocks = {(i, j, sc): np.full((counts[i], counts[j], components), 1. + i + 3 * j + sc)
              for i in range(2) for j in range(2) for sc in range(3)}
    if components == 1:
        blocks = {key: value[:, :, 0] for key, value in blocks.items()}
    config = config_cls(
        point_types=["A", "B"], positions=np.array([[0., 0., 0.], [1.2, .2, 0.]]),
        basis=table, cell=np.diag([3., 20., 20.]), pbc=(True, False, False),
        matrix=BasisMatrix(blocks, np.array([3, 1, 1]), counts),
        metadata={"custom": {"array": np.array([3., 4.]), "nested": [torch.tensor([5.])]}}
    )
    processor = MatrixDataProcessor(basis_table=table, sub_point_matrix=False,
                                   symmetric_matrix=symmetric, n_matrix_components=components)
    item = TorchBasisMatrixData.from_config(config, processor) if preprocessed else config
    return TorchBasisMatrixDataset([item, item], processor)


def assert_same_sample(actual, expected):
    assert type(actual) is type(expected)
    assert actual.keys() == expected.keys()
    for key in expected.keys():
        if torch.is_tensor(expected[key]):
            assert actual[key].dtype == expected[key].dtype
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
    for key in ("array",):
        np.testing.assert_array_equal(actual.metadata["custom"][key], expected.metadata["custom"][key])
    torch.testing.assert_close(actual.metadata["custom"]["nested"][0], expected.metadata["custom"]["nested"][0])


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("preprocessed", [False, True])
def test_cache_returns_independent_periodic_samples(components, symmetric, preprocessed):
    dataset = dataset_for(components, symmetric, preprocessed)
    expected = deepcopy(dataset[0])
    cache = InMemoryData(dataset)
    assert_same_sample(cache[0], expected)
    changed = cache[0]
    changed.point_labels.add_(7)
    changed.edge_labels.mul_(3)
    changed.positions.add_(1)
    changed.metadata["custom"]["array"][:] = -9
    changed.metadata["custom"]["nested"][0].add_(11)
    assert_same_sample(cache[0], expected)
    assert_same_sample(dataset[0], expected)


@pytest.mark.parametrize("components", [1, 2])
def test_cache_snapshot_does_not_alias_preprocessed_source(components):
    dataset = dataset_for(components, preprocessed=True)
    expected = deepcopy(dataset[0])
    cache = InMemoryData(dataset)
    dataset[0].point_labels.add_(7)
    dataset[0].metadata["custom"]["array"][0] = -9
    assert_same_sample(cache[0], expected)


def test_cache_deeply_copies_non_tensor_metadata():
    cache = InMemoryData(dataset_for())
    cache[0].metadata["custom"]["array"][0] = -9
    cache[0].metadata["custom"]["nested"][0].add_(11)
    assert cache[0].metadata["custom"]["array"][0] == 3
    assert cache[0].metadata["custom"]["nested"][0].item() == 5


def test_device_transfer_cannot_move_cached_snapshot():
    cache = InMemoryData(dataset_for())
    expected = deepcopy(cache[0])
    moved = cache[0].to("meta")
    assert moved.point_labels.device.type == "meta"
    assert cache[0].point_labels.device.type == "cpu"
    assert_same_sample(cache[0], expected)


def test_cache_preserves_actual_atomic_reference_for_reconstruction():
    import sisl
    from graph2mat.core.data.matrices.physics.density_matrix import DensityMatrix

    table = AtomicTableWithEdges([sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])])
    table.point_matrix = [np.array([[7.]])]
    processor = MatrixDataProcessor(basis_table=table, sub_point_matrix=True)
    config = BasisConfiguration(point_types=table.types, positions=np.zeros((1, 3)), basis=table,
        cell=np.eye(3) * 10, pbc=(False, False, False),
        matrix=DensityMatrix({(0, 0, 0): np.array([[10.]])}, np.ones(3, dtype=int), np.array([1])))
    dataset = TorchBasisMatrixDataset([config], processor)
    expected = dataset[0]
    sample = InMemoryData(dataset)[0]
    torch.testing.assert_close(sample.point_labels, expected.point_labels, rtol=0, atol=0)
    np.testing.assert_array_equal(sample.metadata["data_processor"].basis_table.point_matrix[0], [[7.]])
    reconstructed = sample.convert_to("scipy_csr")
    np.testing.assert_array_equal(reconstructed.toarray(), [[10.]])


def test_explicit_zero_cache_size_is_empty():
    cache = InMemoryData(dataset_for(), size=0)
    assert len(cache) == 0


@pytest.mark.parametrize("size", [-1, 3])
def test_invalid_size_is_rejected_before_loading(size, monkeypatch):
    dataset = dataset_for()
    def forbidden(*args):
        pytest.fail("invalid size must fail before loading")
    monkeypatch.setattr(type(dataset), "__getitem__", forbidden)
    with pytest.raises(ValueError, match="size"):
        InMemoryData(dataset, size=size)


@pytest.mark.parametrize("size,count", [(None, 2), (1, 1)])
def test_cache_size_preserves_valid_truncation(size, count):
    cache = InMemoryData(dataset_for(), size=size)
    assert len(cache) == count


class NoRepresentationConfiguration(BasisConfiguration):
    def __repr__(self):
        pytest.fail("in-memory configuration must not be formatted for an FDF suffix check")


def test_lazy_configuration_never_formats_arrays_to_test_path_suffix():
    assert dataset_for(config_cls=NoRepresentationConfiguration)[0].num_nodes == 2


class MutatingAccess(torch.utils.data.Dataset):
    def __init__(self, cache):
        self.cache = cache
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, index):
        sample = self.cache[index]
        sample.point_labels.add_(.25)
        sample.metadata["custom"]["array"][0] += 1
        return sample


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("context", [None, "spawn", "fork"])
def test_worker_transform_is_applied_once_on_each_access(components, context):
    if context is not None and context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{context} not available")
    cache = InMemoryData(dataset_for(components))
    expected = deepcopy(cache[0])
    kwargs = {} if context is None else {
        "num_workers": 2, "multiprocessing_context": context, "persistent_workers": True}
    loader = DataLoader(MutatingAccess(cache), batch_size=1, **kwargs)
    for _ in range(2):
        for batch in loader:
            torch.testing.assert_close(batch.point_labels, expected.point_labels + .25, rtol=0, atol=0)
            assert batch.metadata["custom"]["array"][0][0] == 4
    assert_same_sample(cache[0], expected)
    del loader


def test_zip_snapshot_keeps_charge_spectral_columns_without_reader_ownership(tmp_path):
    filename = tmp_path / "cached.zip"
    columns = np.eye(2)
    buffer = io.BytesIO()
    np.savez(buffer, C=columns)
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr("sample.fdf", "NetCharge -1.5\n")
        archive.writestr("spectral_window.npz", buffer.getvalue())
    processor = MatrixDataProcessor(basis_table=BasisTableWithEdges([PointBasis("A", R=2, basis=[1])]),
                                   sub_point_matrix=False)
    with zipfile.ZipFile(filename) as archive:
        dataset = TorchBasisMatrixDataset([zipfile.Path(archive, "sample.fdf")], processor,
                                         data_cls=_TextSample, spectral_window=True)
        cache = InMemoryData(dataset)
        dataset.close()
        assert zipfile.Path(archive, "sample.fdf").read_text() == "NetCharge -1.5\n"
    restored = pickle.loads(pickle.dumps(cache))
    sample = restored[0]
    assert sample.total_charge.item() == -1.5
    np.testing.assert_array_equal(sample.spectral_C.numpy(), columns)
    sample.spectral_C.add_(2)
    np.testing.assert_array_equal(restored[0].spectral_C.numpy(), columns)
