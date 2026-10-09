"""Cost budgets must preserve samples, declared padding and actual epoch counts."""

import multiprocessing

import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning import data as data_module


def cost_sampler(costs, budget, batch_size=5, sampler=None, **kwargs):
    assert hasattr(data_module, "_CostBatchSampler"), "cost sampler not implemented"
    return data_module._CostBatchSampler(sampler or SequentialSampler(costs), batch_size=batch_size,
                                         drop_last=False, costs=costs, max_batch_cost=budget, **kwargs)


@pytest.mark.parametrize("costs,budget,size,want", [
    ([2, 3, 7, 1, 12, 2, 0], 8, 5, [[0, 1], [2, 3], [4], [5, 6]]),
    ([1, 1, 1, 1, 1], 10, 2, [[0, 1], [2, 3], [4]]),
    ([0, 0, 0], 1, 2, [[0, 1], [2]]),
    ([], 1, 2, []),
])
def test_budget_size_oversize_and_last_preserve_exact_coverage(costs, budget, size, want):
    sampler = cost_sampler(costs, budget, size)
    assert len(sampler) == len(sampler) == len(want)
    assert list(sampler) == want
    assert [index for batch in want for index in batch] == list(range(len(costs)))


@pytest.mark.parametrize("budget", [0, -1, True, False, 2.0, float("inf"), float("nan")])
def test_datamodule_invalid_budget_rejected_before_io(budget):
    with pytest.raises(ValueError, match="max_batch_cost"):
        MatrixDataModule(root_dir="/unavailable", max_batch_cost=budget)


def test_cost_batching_rejects_dynamic_pool_before_io():
    with pytest.raises(ValueError, match="rotating_pool"):
        MatrixDataModule(root_dir="/unavailable", max_batch_cost=10, rotating_pool_size=2)


def test_random_len_does_not_reshuffle_reserved_plan():
    costs = [1, 2, 3, 8, 1, 2, 1]
    def make():
        return cost_sampler(costs, 8, sampler=RandomSampler(costs, generator=torch.Generator().manual_seed(21)))
    sampler, other = make(), make()
    length = len(sampler)
    assert len(sampler) == length
    first = list(iter(sampler))
    assert len(first) == length
    assert first == list(iter(other))
    second = list(iter(sampler))
    assert second == list(iter(other)) and first != second
    for batches in (first, second):
        assert sorted(index for batch in batches for index in batch) == list(range(len(costs)))


@pytest.mark.parametrize("epoch", [0, 1])
def test_distributed_batches_equal_steps_and_declare_padding(epoch):
    costs = [10, 1, 10, 1, 10, 1, 10]
    batches, paddings = [], []
    for rank in (0, 1):
        source = DistributedSampler(costs, num_replicas=2, rank=rank, shuffle=False, drop_last=False)
        source.set_epoch(epoch)
        sampler = cost_sampler(costs, 10, sampler=source)
        length = len(sampler)
        actual = list(iter(sampler))
        assert len(actual) == length == 4
        batches.append(actual)
        paddings.append(sampler.padding_batches)
    assert batches[0] == [[0], [2], [4], [6]]
    assert batches[1] == [[1, 3, 5], [0], [1, 3, 5], [0]]
    assert paddings == [0, 2]
    assert set(index for rank in batches for batch in rank for index in batch) == set(range(7))


def test_distributed_epoch_change_invalidates_cached_plan():
    costs = [1, 3, 2, 8, 1, 1, 9, 2, 1]
    source = DistributedSampler(costs, num_replicas=2, rank=0, shuffle=True, seed=12)
    sampler = cost_sampler(costs, 8, sampler=source)
    len(sampler)
    source.set_epoch(1)
    actual = list(iter(sampler))
    assert actual == list(iter(cost_sampler(costs, 8, sampler=source)))
    source.set_epoch(2)
    assert actual != list(iter(sampler))


def test_custom_distributed_sampler_rejected_in_opt_in():
    class CustomSampler(DistributedSampler):
        pass
    with pytest.raises(ValueError, match="sampler"):
        cost_sampler([1, 2], 5, sampler=CustomSampler([1, 2], num_replicas=1, rank=0))


def fixtures():
    table = BasisTableWithEdges([PointBasis("A", R=1, basis=[1]), PointBasis("B", R=1, basis=[2])])
    configs = [BasisConfiguration(
        point_types=kinds, positions=np.array([[j * .7, 0., 0.] for j in range(len(kinds))]),
        basis=table, cell=np.eye(3) * 20, pbc=(False, False, False),
        matrix=np.eye(sum(1 if kind == "A" else 2 for kind in kinds)),
        metadata={"id": index}) for index, kinds in enumerate((["A"], ["B"], ["A", "B"], ["A"] * 4, ["B"]))]
    return table, configs


@pytest.mark.parametrize("store", [False, True])
def test_datamodule_uses_split_costs_and_leaves_evaluation_fixed(store):
    table, configs = fixtures()
    module = MatrixDataModule(basis_table=table, train_runs=configs, val_runs=[configs[0]],
        test_runs=configs, predict_structs=configs, sub_point_matrix=False, symmetric_matrix=True,
        loader_threads=0, batch_size=3, store_in_memory=store, max_batch_cost=8)
    module.setup(None)
    loader = module.train_dataloader()
    assert loader.batch_sampler.costs == (1, 4, 9, 14, 4)
    batches = list(loader)
    assert sorted(int(index) for batch in batches for index in batch.metadata["id"]) == list(range(5))
    assert max(batch.num_graphs for batch in batches) <= 3
    assert [batch.num_graphs for batch in module.val_dataloader()] == [1]
    assert [batch.num_graphs for batch in module.test_dataloader()] == [3, 2]
    assert [batch.num_graphs for batch in module.predict_dataloader()] == [3, 2]
    assert module.hparams.max_batch_cost == 8
    module.teardown(None)


@pytest.mark.parametrize("context", [None, "spawn", "fork"])
def test_worker_epochs_keep_cost_batches_and_sample_ownership(context):
    if context and context not in multiprocessing.get_all_start_methods():
        pytest.skip("multiprocessing context unavailable")
    table, configs = fixtures()
    module = MatrixDataModule(basis_table=table, train_runs=configs, val_runs=[], loader_threads=0,
        sub_point_matrix=False, store_in_memory=True, batch_size=3, max_batch_cost=8)
    module.setup("fit")
    loader = module.train_dataloader()
    if context:
        # Reuse the actual production batch sampler/collate with an explicit test worker context.
        loader = torch.utils.data.DataLoader(loader.dataset, batch_sampler=loader.batch_sampler,
            collate_fn=loader.collate_fn, num_workers=2, persistent_workers=True, multiprocessing_context=context)
    try:
        for _ in range(2):
            batches = list(loader)
            assert sorted(int(index) for batch in batches for index in batch.metadata["id"]) == list(range(5))
            batches[0].point_labels.add_(99)
        assert module.train_dataset[0].point_labels.tolist() == [1.]
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
        module.teardown(None)


@pytest.mark.parametrize("staged", [False, True])
def test_cost_batching_pair_loss_rejected_before_backbone(monkeypatch, staged):
    import pytorch_lightning as pl
    from torch_geometric.data import Batch
    from graph2mat.core.data import metrics
    from graph2mat.tools.lightning.tests.test_pair_epoch_metrics import pair_model, pair_samples

    model = pair_model()
    module = MatrixDataModule(basis_table=model.basis_table, max_batch_cost=10)
    trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.datamodule = module
    model._trainer = trainer
    if staged:
        model.loss_fn = metrics.elementwise_mse()
        model._apply_stage_loss({"loss": "pair_difference_mse"})
    def forbidden(*args, **kwargs):
        raise AssertionError("backbone ran before incompatible pair loss was rejected")
    monkeypatch.setattr(model.model, "forward", forbidden)
    with pytest.raises(ValueError, match="max_batch_cost.*pair_difference_mse"):
        model.training_step(Batch.from_data_list(pair_samples()[:2]), 0)


def test_default_pair_training_unchanged():
    from torch_geometric.data import Batch
    from graph2mat.tools.lightning.tests.test_pair_epoch_metrics import pair_model, pair_samples
    model = pair_model()
    assert float(model.training_step(Batch.from_data_list(pair_samples()[:2]), 0)["loss"]) == pytest.approx(5.)


def test_default_loader_remains_fixed_without_cost_prepass():
    table, configs = fixtures()
    module = MatrixDataModule(basis_table=table, train_runs=configs, val_runs=[],
        sub_point_matrix=False, batch_size=3, loader_threads=0)
    module.setup("fit")
    loader = module.train_dataloader()
    assert type(loader.batch_sampler) is torch.utils.data.BatchSampler
    assert isinstance(loader.sampler, RandomSampler)
    assert [batch.num_graphs for batch in loader] == [3, 2]
    assert module._batch_cost_cache is None
    assert module.hparams.max_batch_cost is None
    module.teardown(None)


def test_cost_prepass_counts_ghost_zero_independently_of_labels():
    from graph2mat import MatrixDataProcessor
    from graph2mat.bindings.torch import TorchBasisMatrixData
    table = BasisTableWithEdges([PointBasis("A", R=1, basis=[1]), PointBasis("ghost", R=1, basis=[])])
    config = BasisConfiguration(point_types=["ghost"], positions=np.zeros((1, 3)),
        basis=table, cell=np.eye(3) * 20, pbc=(False, False, False))
    sample = TorchBasisMatrixData.from_config(config, MatrixDataProcessor(basis_table=table, sub_point_matrix=False))
    sample.point_labels, sample.edge_labels = torch.empty(0), torch.empty(0)
    module = MatrixDataModule(basis_table=table, train_runs=[sample], val_runs=[],
        sub_point_matrix=False, max_batch_cost=np.int64(1), loader_threads=0)
    module.setup("fit")
    assert module.train_dataloader().batch_sampler.costs == (0,)
    assert [batch.num_graphs for batch in module.train_dataloader()] == [1]
    module.teardown(None)


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_cost_partitions_preserve_real_mace_predictions(components, symmetric):
    from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs

    model, _, samples = training_inputs(components=components, symmetric=symmetric)
    module = MatrixDataModule(basis_table=model.basis_table, n_matrix_components=components,
        train_runs=samples, val_runs=[], symmetric_matrix=symmetric, sub_point_matrix=False,
        max_batch_cost=1, batch_size=2, loader_threads=0)
    module.setup("fit")
    loader = module.train_dataloader()
    model.eval()
    full = module._with_type_layout(
        data_module.DataLoader(samples, batch_size=2)).collate_fn(samples)
    with torch.no_grad():
        expected = model(full)
        actual = {}
        for indices in cost_sampler(loader.batch_sampler.costs, 1):
            batch = loader.collate_fn([samples[index] for index in indices])
            out = model(batch)
            for key in ("node_labels", "edge_labels"):
                actual.setdefault(key, []).append(out[key])
        for key, values in actual.items():
            torch.testing.assert_close(torch.cat(values), expected[key], rtol=1e-10, atol=1e-10)
    assert full.num_graphs == 2
    module.teardown(None)


def test_cost_prepass_cache_reused_then_invalidated_with_dataset_and_teardown():
    table, configs = fixtures()
    module = MatrixDataModule(basis_table=table, train_runs=configs, val_fraction=.4,
        sub_point_matrix=False, max_batch_cost=8, loader_threads=0)
    module.setup("fit")
    original = module.train_dataset
    assert len(original) == 3
    class CountingDataset(torch.utils.data.Dataset):
        def __init__(self, values):
            self.values, self.reads = values, 0
        def __len__(self):
            return len(self.values)
        def __getitem__(self, index):
            self.reads += 1
            return self.values[index]
    dataset = CountingDataset([original[index] for index in range(len(original))])
    module.train_dataset = dataset
    first = module.train_dataloader()
    assert dataset.reads == 3
    assert first.batch_sampler.costs == tuple((1, 4, 9, 14, 4)[index] for index in original.indices)
    assert module.train_dataloader().batch_sampler.costs == first.batch_sampler.costs
    assert dataset.reads == 3
    module.train_dataset = CountingDataset([dataset[0]])
    assert len(module.train_dataloader().batch_sampler.costs) == 1
    assert module.train_dataset.reads == 1
    module.teardown(None)
    assert module._batch_cost_cache is None


@pytest.mark.parametrize("components", [1, 2])
def test_cost_batches_preserve_coefficient_targets(components):
    from graph2mat.bindings.torch.data.coefficient_targets import COEFFICIENT_TARGETS_ATTR, CoefficientMatrixMSEProjector
    from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs
    model, _, samples = training_inputs(components=components)
    module = MatrixDataModule(basis_table=model.basis_table, n_matrix_components=components,
        train_runs=samples, val_runs=[], symmetric_matrix=True, sub_point_matrix=False,
        max_batch_cost=1, batch_size=2, loader_threads=0)
    module.setup("fit")
    module.set_coefficient_mse_projector(CoefficientMatrixMSEProjector(model.model.matrix_readouts))
    batches = list(module.train_dataloader())
    assert [batch.num_graphs for batch in batches] == [1, 1]
    for batch in batches:
        assert not isinstance(getattr(batch, COEFFICIENT_TARGETS_ATTR), str)
        assert set(model.training_step(batch, 0)) == {"loss"}
    module.teardown(None)


def test_native_zip_cost_prepass_and_owned_temporary_cleanup(tmp_path):
    import zipfile
    import sisl
    root = tmp_path / "inputs.zip"
    geometry = sisl.Geometry([[0., 0., 0.], [.5, 0., 0.]], atoms=[1, 1], lattice=[10] * 3)
    xv = tmp_path / "sample.XV"
    geometry.write(xv)
    with zipfile.ZipFile(root, "w") as archive:
        archive.write(xv, "sample.XV")
    module = MatrixDataModule(root_dir=str(root), copy_root_to_tmp=True,
        basis_table=BasisTableWithEdges([PointBasis(1, R=1, basis=[1])]), predict_structs="*.XV",
        sub_point_matrix=False, loader_threads=0, max_batch_cost=4)
    try:
        module.setup("predict")
        module.train_dataset = module.predict_dataset
        temporary = module.tmp_dir
        assert module.train_dataloader().batch_sampler.costs == (4,)
        assert next(iter(module.train_dataloader())).num_nodes == 2
        module.teardown(None)
        assert not temporary.exists() and root.exists()
        with zipfile.ZipFile(root) as archive:
            assert archive.read("sample.XV")
    finally:
        module.teardown(None)
