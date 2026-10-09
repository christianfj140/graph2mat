"""Stage-aware loading and reuse of real data (DATA_PIPE-12)."""

import numpy as np
import pytest
import sisl

from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
from graph2mat.bindings.torch import InMemoryData
from graph2mat.tools.lightning import MatrixDataModule


def _inputs():
    basis = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    config = BasisConfiguration(
        point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
        basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
        matrix=np.ones((2, 2)),
    )
    return basis, config


@pytest.mark.parametrize("stage,requested", [
    ("fit", ("train", "val")), ("validate", ("val",)),
    ("test", ("test",)), ("predict", ("predict",)),
])
def test_setup_only_reads_requested_splits(stage, requested, tmp_path):
    basis, config = _inputs()
    runs = {
        split: [config] if split in requested else [tmp_path / (split + "-unavailable.XV")]
        for split in ("train", "val", "test", "predict")
    }
    module = MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=runs["train"], val_runs=runs["val"], test_runs=runs["test"],
        predict_structs=runs["predict"], store_in_memory=True, loader_threads=0,
    )
    module.setup(stage)
    for split in runs:
        dataset = getattr(module, split + "_dataset", None)
        if split in requested:
            assert len(dataset) == 1
            assert dataset[0].num_nodes == 2
        else:
            assert dataset is None


def test_setup_reuses_datasets_across_stages():
    basis, config = _inputs()
    module = MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=[config, config], val_runs=[config], test_runs=[config],
        store_in_memory=True, loader_threads=0,
    )
    module.setup("fit")
    train, val = module.train_dataset, module.val_dataset
    module.setup("fit")
    assert module.train_dataset is train and module.val_dataset is val
    assert module.test_dataset is None
    module.setup("validate")
    assert module.train_dataset is train and module.val_dataset is val
    module.setup("test")
    assert module.train_dataset is train and module.val_dataset is val
    assert module.test_dataset[0].num_nodes == 2


def test_validate_without_explicit_val_uses_training_split():
    basis, config = _inputs()
    module = MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=[config]*10, store_in_memory=True, loader_threads=0,
    )
    module.setup("validate")
    assert len(module.val_dataset) == 1
    assert module.val_dataset[0].num_nodes == 2
    assert module.test_dataset is None and module.predict_dataset is None


def test_fit_materializes_cached_training_split_after_validate():
    basis, config = _inputs()
    module = MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=[config]*10, store_in_memory=True, loader_threads=0,
    )
    module.setup("validate")
    assert not isinstance(module.train_dataset, InMemoryData)
    validation = module.val_dataset
    module.setup("fit")
    assert isinstance(module.train_dataset, InMemoryData)
    assert len(module.train_dataset) == 9
    assert module.train_dataset[0].num_nodes == 2
    training = module.train_dataset
    module.setup("fit")
    assert module.train_dataset is training
    assert module.val_dataset is validation


def test_setup_rejects_unknown_stage():
    basis, _ = _inputs()
    module = MatrixDataModule(basis_table=basis, sub_point_matrix=False)
    with pytest.raises(ValueError, match="stage"):
        module.setup("unknown")


def test_setup_skips_component_inference_for_empty_dataset():
    basis, _ = _inputs()
    module = MatrixDataModule(
        basis_table=basis, sub_point_matrix=False, test_runs=[], store_in_memory=True,
    )
    module.setup("test")
    assert len(module.test_dataset) == 0


def test_store_in_memory_keeps_train_and_validation_snapshots_on_cpu():
    basis, config = _inputs()
    module = MatrixDataModule(basis_table=basis, sub_point_matrix=False, train_runs=[config] * 10,
                              store_in_memory=True, loader_threads=0)
    module.setup("fit")
    train, val = module.train_dataset, module.val_dataset
    expected = train[0].point_labels.clone()
    train[0].point_labels.add_(7)
    val[0].to("meta")
    assert train[0].point_labels.equal(expected)
    assert val[0].point_labels.device.type == "cpu"
    module.setup("fit")
    assert module.train_dataset is train and module.val_dataset is val
    module.teardown(None)


@pytest.mark.parametrize("archived", [False, True])
def test_cached_native_geometry_survives_owned_temporary_root_cleanup(tmp_path, archived):
    import zipfile
    import torch

    source = tmp_path / "source"
    source.mkdir()
    sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=[1, 1], lattice=[10] * 3).write(source / "sample.XV")
    root = source
    if archived:
        root = tmp_path / "source.zip"
        with zipfile.ZipFile(root, "w") as archive:
            archive.write(source / "sample.XV", "sample.XV")
    module = MatrixDataModule(root_dir=str(root), copy_root_to_tmp=True, predict_structs="*.XV",
        basis_table=BasisTableWithEdges([PointBasis(1, R=2, basis=[1])]),
        sub_point_matrix=False, store_in_memory=True, loader_threads=0)
    try:
        module.setup("predict")
        snapshot = module.predict_dataset
        expected = snapshot[0].positions.clone()
        snapshot[0].positions.add_(9)
        temporary = module.tmp_dir
        module.teardown(None)
        assert not temporary.exists()
        assert root.exists()
        torch.testing.assert_close(snapshot[0].positions, expected, rtol=0, atol=0)
        assert snapshot[0].num_nodes == 2
        assert snapshot[0].positions.device.type == "cpu"
    finally:
        module.teardown(None)
