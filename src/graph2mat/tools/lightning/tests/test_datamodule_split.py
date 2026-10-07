"""Reproducible, configurable splits without changing the historical defaults."""

import numpy as np
import pytest

from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
from graph2mat.tools.lightning import MatrixDataModule


def _module(count=20, **options):
    basis = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    configs = [BasisConfiguration(
        point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
        basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
        matrix=np.full((2, 2), float(index+1)),
    ) for index in range(count)]
    return MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=configs, loader_threads=0, **options,
    )


def _sample_ids(dataset):
    return [int(sample.point_labels[0].item())-1 for sample in dataset]


def test_default_split_preserves_seed32_and_ten_percent():
    module = _module()
    module.setup("fit")
    assert _sample_ids(module.val_dataset) == [8, 15]
    assert sorted(_sample_ids(module.train_dataset)+[8, 15]) == list(range(20))


def test_explicit_empty_validation_runs_disables_automatic_split():
    module = _module(val_runs=[], val_fraction=0.99)
    module.setup("fit")
    assert _sample_ids(module.train_dataset) == list(range(20))
    assert len(module.val_dataset) == 0


def test_split_seed_and_fraction_control_real_sample_partition():
    module = _module(split_seed=7, val_fraction=0.25)
    module.setup("fit")
    train, val = _sample_ids(module.train_dataset), _sample_ids(module.val_dataset)
    assert val == [1, 2, 5, 11, 17]
    assert len(train) == 15
    assert set(train).isdisjoint(val)
    assert sorted(train+val) == list(range(20))
    assert module.hparams.split_seed == 7 and module.hparams.val_fraction == 0.25

    repeated = _module(split_seed=7, val_fraction=0.25)
    repeated.setup("validate")
    repeated.setup("fit")
    assert _sample_ids(repeated.train_dataset) == train
    assert _sample_ids(repeated.val_dataset) == val

    changed = _module(split_seed=8, val_fraction=0.25)
    changed.setup("fit")
    assert _sample_ids(changed.val_dataset) != val


@pytest.mark.parametrize("count,fraction,train_size,val_size", [
    (0, 0.1, 0, 0), (1, 0.1, 1, 0), (2, 0.99, 1, 1), (20, 0., 20, 0),
])
def test_automatic_split_retains_training_samples(count, fraction, train_size, val_size):
    module = _module(count=count, val_fraction=fraction)
    module.setup("fit")
    assert len(module.train_dataset) == train_size
    assert len(module.val_dataset) == val_size


@pytest.mark.parametrize("fraction", [-0.1, 1., 1.1, float("nan"), float("inf")])
def test_invalid_validation_fraction_rejected_before_loading(fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        _module(val_fraction=fraction)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5])
def test_invalid_split_seed_rejected_before_loading(seed):
    with pytest.raises(ValueError, match="split_seed"):
        _module(split_seed=seed)
