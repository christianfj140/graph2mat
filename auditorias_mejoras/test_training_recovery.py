"""Recovery must retain complete, compatible state even when publication fails."""
import hashlib
import importlib
import inspect
import multiprocessing
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


FINGERPRINT = {"seed": 7}
FINGERPRINT_SHA = hashlib.sha256(b'{"seed":7}').hexdigest()


def recovery():
    assert Path(__file__).with_name("training_recovery.py").is_file(), "RecoveryStore is not implemented"
    return importlib.import_module("training_recovery")


def boundary(epoch=1):
    return dict(next_epoch=epoch, global_step=4 * epoch, samples_committed=12 * epoch,
                plan_sha256="a" * 64, epoch_complete=True)


class SavedTrainer:
    """Disk-writing double of save_checkpoint; payload mirrors inspected Lightning state."""
    def __init__(self):
        self.payload = {
            "pytorch-lightning_version": "2.6.1", "state_dict": {"weight": torch.tensor([1.])},
            "optimizer_states": [{"state": {0: {"step": torch.tensor(4.),
                "exp_avg": torch.tensor([.1]), "exp_avg_sq": torch.tensor([.01])}},
                "param_groups": [{"params": [0], "lr": .001}]}],
            "lr_schedulers": [{"last_epoch": 1, "T_max": 4}],
            "callbacks": {"best": {"best_model_score": torch.tensor(.1), "best_model_path": ""}},
            "loops": {"fit_loop": {}},
            "g2m_recovery": {"fingerprint_sha256": FINGERPRINT_SHA, "seed": 7,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(), "cuda": []}},
        }
        self.advance(boundary())

    def advance(self, state):
        self.payload["epoch"] = state["next_epoch"] - 1
        self.payload["global_step"] = state["global_step"]
        self.payload["g2m_recovery"].update(state)
        epoch = state["next_epoch"]
        self.payload["loops"]["fit_loop"].update({
            "epoch_progress": {"current": dict(ready=epoch, started=epoch,
                processed=epoch, completed=epoch - 1)},
            "epoch_loop.batch_progress": {"current": dict(ready=4, started=4,
                processed=4, completed=4), "is_last_batch": True}})

    def save_checkpoint(self, path):
        torch.save(self.payload, path)


def inputs(tmp_path):
    api = recovery()
    store = api.RecoveryStore(tmp_path / "job", FINGERPRINT)
    return api, store, SavedTrainer()


def test_store_preserves_all_training_state_and_provenance(tmp_path):
    _, store, trainer = inputs(tmp_path)
    generation = store.publish(trainer, boundary=boundary(), best_path=None)
    receipt = store.latest()
    assert Path(receipt["checkpoint"]).parent == generation
    assert receipt["fingerprint"] == {"seed": 7}
    assert receipt["next_epoch"] == 1 and receipt["samples_committed"] == 12
    state = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
    assert set(trainer.payload) <= set(state)
    assert state["optimizer_states"][0]["state"][0]["exp_avg"].tolist() == pytest.approx([.1])
    assert state["lr_schedulers"][0]["T_max"] == 4
    assert receipt["best"] is None and store.restore_best(receipt) is None


def test_corrupt_latest_falls_back(tmp_path):
    _, store, trainer = inputs(tmp_path)
    first = store.publish(trainer, boundary=boundary(), best_path=None)
    trainer.advance(boundary(2))
    second = store.publish(trainer, boundary=boundary(2), best_path=None)
    (second / "state.ckpt").write_bytes(b"truncated")
    receipt = store.latest()
    assert Path(receipt["checkpoint"]).parent == first
    assert receipt["fallback"] and receipt["global_step"] == 4


@pytest.mark.parametrize("index", [None, "truncated"])
def test_published_generation_survives_missing_or_invalid_index(tmp_path, index):
    _, store, trainer = inputs(tmp_path)
    generation = store.publish(trainer, boundary=boundary(), best_path=None)
    path = store.root / "recovery/latest.json"
    if index is None:
        path.unlink()
    else:
        path.write_text(index)
    assert Path(store.latest()["checkpoint"]).parent == generation


def test_empty_store_is_not_same_as_lost_checkpoints(tmp_path):
    _, store, trainer = inputs(tmp_path)
    assert store.latest() is None
    generation = store.publish(trainer, boundary=boundary(), best_path=None)
    (generation / "state.ckpt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="valid"):
        store.latest()


def test_fingerprint_change_blocks_resume_before_loading_pickle(tmp_path, monkeypatch):
    api, store, trainer = inputs(tmp_path)
    store.publish(trainer, boundary=boundary(), best_path=None)
    changed = api.RecoveryStore(store.root, {"seed": 8})
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint was deserialized")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="fingerprint"):
        changed.latest()


@pytest.mark.parametrize("field", ["optimizer_states", "lr_schedulers", "callbacks", "loops", "g2m_recovery"])
def test_incomplete_checkpoint_never_replaces_previous_generation(tmp_path, field):
    _, store, trainer = inputs(tmp_path)
    first = store.publish(trainer, boundary=boundary(), best_path=None)
    del trainer.payload[field]
    with pytest.raises(ValueError, match="checkpoint"):
        store.publish(trainer, boundary=boundary(), best_path=None)
    assert Path(store.latest()["checkpoint"]).parent == first


@pytest.mark.parametrize("broken", ["incomplete_epoch", "epoch", "step", "progress", "fingerprint", "nan"])
def test_contradictory_boundary_never_publishes(tmp_path, broken):
    _, store, trainer = inputs(tmp_path)
    state = boundary()
    if broken == "incomplete_epoch":
        state["epoch_complete"] = False
    elif broken == "epoch":
        trainer.payload["epoch"] = 1
    elif broken == "step":
        trainer.payload["global_step"] = 5
    elif broken == "progress":
        trainer.payload["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"] = False
    elif broken == "fingerprint":
        trainer.payload["g2m_recovery"]["fingerprint_sha256"] = "b" * 64
    else:
        trainer.payload["state_dict"]["weight"][0] = float("nan")
    with pytest.raises(ValueError):
        store.publish(trainer, boundary=state, best_path=None)
    assert store.latest() is None


@pytest.mark.parametrize("phase", ["checkpoint", "fsync", "generation", "index"])
def test_atomic_publication_failure_keeps_a_valid_generation(tmp_path, monkeypatch, phase):
    api, store, trainer = inputs(tmp_path)
    first = store.publish(trainer, boundary=boundary(), best_path=None)
    trainer.advance(boundary(2))
    replace, fsync, save = os.replace, os.fsync, trainer.save_checkpoint
    def fail_save(path):
        Path(path).write_bytes(b"partial")
        raise OSError("injected save failure")
    def fail_fsync(fd):
        raise OSError("injected fsync failure")
    def fail_replace(source, target):
        target = Path(target)
        if (phase == "index" and target.name == "latest.json") or (
                phase == "generation" and target.name.startswith("generation-")):
            raise OSError("injected rename failure")
        return replace(source, target)
    with monkeypatch.context() as patch:
        if phase == "checkpoint":
            patch.setattr(trainer, "save_checkpoint", fail_save)
        elif phase == "fsync":
            patch.setattr(os, "fsync", fail_fsync)
        else:
            patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError):
            store.publish(trainer, boundary=boundary(2), best_path=None)
    assert os.replace is replace and os.fsync is fsync and trainer.save_checkpoint == save
    receipt = store.latest()
    assert receipt["next_epoch"] == (2 if phase == "index" else 1)
    assert (first / "state.ckpt").is_file()


def test_retention_keeps_two_valid_generations_and_ignores_unowned_paths(tmp_path):
    _, store, trainer = inputs(tmp_path)
    sentinel = store.root / "recovery/user-notes.txt"
    sentinel.write_text("user data")
    generations = []
    for epoch in (1, 2, 3):
        trainer.advance(boundary(epoch))
        generations.append(store.publish(trainer, boundary=boundary(epoch), best_path=None))
    assert not generations[0].exists()
    assert all(path.is_dir() for path in generations[1:])
    assert sentinel.read_text() == "user data"
    (generations[-1] / "state.ckpt").write_bytes(b"corrupt")
    assert store.latest()["next_epoch"] == 2


def test_symlink_root_and_recovery_directory_cannot_write_outside_job(tmp_path):
    api = recovery()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="path|symlink"):
        api.RecoveryStore(tmp_path / "linked/child", FINGERPRINT)
    assert list(outside.iterdir()) == []
    _, store, trainer = inputs(tmp_path)
    # Store can create its empty recovery directory; replace it only if empty.
    directory = store.root / "recovery"
    directory.rmdir()
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="path|symlink"):
        store.publish(trainer, boundary=boundary(), best_path=None)
    assert list(outside.iterdir()) == []


def test_symlink_checkpoint_is_rejected_without_loading_external_pickle(tmp_path, monkeypatch):
    _, store, trainer = inputs(tmp_path)
    generation = store.publish(trainer, boundary=boundary(), best_path=None)
    checkpoint = generation / "state.ckpt"
    external = tmp_path / "foreign.ckpt"
    checkpoint.replace(external)
    checkpoint.symlink_to(external)
    def forbidden(*args, **kwargs):
        pytest.fail("external pickle was loaded")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="path|symlink"):
        store.latest()
    assert external.is_file()


def test_best_snapshot_survives_deletion_and_is_not_silently_overwritten(tmp_path):
    _, store, trainer = inputs(tmp_path)
    best = store.root / "best/early.ckpt"
    best.parent.mkdir()
    torch.save({"state_dict": {"weight": torch.tensor([.5])}}, best)
    original = best.read_bytes()
    trainer.payload["callbacks"]["best"]["best_model_path"] = str(best)
    store.publish(trainer, boundary=boundary(), best_path=best)
    best.unlink()
    receipt = store.latest()
    assert store.restore_best(receipt) == best
    assert best.read_bytes() == original
    best.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="best"):
        store.restore_best(receipt)
    assert best.read_bytes() == b"unexpected"


def test_external_or_symlink_best_cannot_be_copied_into_generation(tmp_path):
    _, store, trainer = inputs(tmp_path)
    foreign = tmp_path / "foreign.ckpt"
    foreign.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="path|own"):
        store.publish(trainer, boundary=boundary(), best_path=foreign)
    link = store.root / "best.ckpt"
    link.symlink_to(foreign)
    with pytest.raises(ValueError, match="path|symlink"):
        store.publish(trainer, boundary=boundary(), best_path=link)
    assert foreign.read_bytes() == b"foreign"


def test_second_store_cannot_acquire_writer_lock(tmp_path):
    api, store, _ = inputs(tmp_path)
    other = api.RecoveryStore(store.root, FINGERPRINT)
    with store.lock():
        with pytest.raises(RuntimeError, match="writer|lock"):
            with other.lock():
                pytest.fail("two writers acquired same store")


def _forked_lock(store, connection):
    try:
        with store.lock():
            connection.send("acquired")
    except RuntimeError:
        connection.send("rejected")
    finally:
        connection.close()


def test_fork_cannot_borrow_parent_writer_lock(tmp_path):
    _, store, _ = inputs(tmp_path)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    with store.lock():
        child = context.Process(target=_forked_lock, args=(store, sender))
        child.start()
        sender.close()
        try:
            assert receiver.poll(5), "fork child failed to return lock result"
            assert receiver.recv() == "rejected"
        finally:
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join()
            receiver.close()
        assert child.exitcode == 0


def paired_module(mode, workers=0, epoch=0):
    api = recovery()
    assert hasattr(api, "RecoveryDataModule"), "recovery loaders are not implemented"
    from bench_batching_coste import inputs as batching_inputs
    base = batching_inputs("cost")
    options = dict(base.hparams)
    options.update(store_in_memory=True, loader_threads=workers,
                   max_batch_cost=base.max_batch_cost if mode == "cost" else None)
    torch.manual_seed(7)
    module = api.RecoveryDataModule(**options)
    module.setup("fit")
    for index, data in enumerate(module.train_dataset.data_objects):
        data.sample_id = torch.tensor([index])
    module.trainer = SimpleNamespace(current_epoch=epoch, world_size=1,
                                    reload_dataloaders_every_n_epochs=0)
    base.teardown(None)
    return module


@pytest.mark.parametrize("workers", [0, 4])
@pytest.mark.parametrize("epoch", [0, 1, 2])
def test_loader_coverage_order_and_prefetch_are_paired(workers, epoch):
    api = recovery()
    fixed, cost = paired_module("fixed", workers, epoch), paired_module("cost", workers, epoch)
    loaders = [module.train_dataloader() for module in (fixed, cost)]
    try:
        plans = [api.loader_plan(loader) for loader in loaders]
        flatten = lambda plan: [index for batch in plan for index in batch]
        assert flatten(plans[0]) == flatten(plans[1])
        assert sorted(flatten(plans[0])) == list(range(12))
        assert len(plans[1]) > len(plans[0])
        for loader, plan in zip(loaders, plans):
            rng = torch.get_rng_state().clone()
            iterator = iter(loader)
            first = next(iterator)
            assert api.loader_plan(loader) == plan  # Workers have already prefetched.
            actual = first.sample_id.tolist()
            for batch in iterator:
                actual.extend(batch.sample_id.tolist())
            assert actual == flatten(plan)
            assert len(loader) == len(plan)
            assert api.loader_plan(loader) == plan
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            if workers:
                iterator._shutdown_workers()
    finally:
        fixed.teardown(None)
        cost.teardown(None)


@pytest.mark.parametrize("mode", ["fixed", "cost"])
def test_recreated_loader_uses_same_epoch_order_not_constructor_rng(mode):
    api = recovery()
    continuous = paired_module(mode)
    first = api.loader_plan(continuous.train_dataloader())
    continuous.trainer.current_epoch = 2
    second = api.loader_plan(continuous.train_dataloader())
    reconstructed = paired_module(mode, epoch=2)
    torch.rand(100)  # Setup/constructor work may differ after a process restart.
    assert api.loader_plan(reconstructed.train_dataloader()) == second
    assert first != second
    continuous.teardown(None)
    reconstructed.teardown(None)


def test_recovery_loader_preserves_constructor_and_collation_contract():
    api = recovery()
    module = paired_module("fixed")
    from graph2mat.tools.lightning import MatrixDataModule
    assert inspect.signature(api.RecoveryDataModule) == inspect.signature(MatrixDataModule)
    loader = module.train_dataloader()
    indices = api.loader_plan(loader)[0]
    expected = loader.collate_fn([module.train_dataset[index] for index in indices])
    actual = next(iter(loader))
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
    assert module.trainer.reload_dataloaders_every_n_epochs == 1


@pytest.mark.parametrize("mode", ["fixed", "cost"])
def test_epoch_reloading_does_not_retain_pinned_persistent_worker_pools(mode):
    module = paired_module(mode, workers=4)
    try:
        for loader in (module.train_dataloader(), module.val_dataloader()):
            assert loader.num_workers == 4
            assert not loader.persistent_workers
    finally:
        module.teardown(None)
    module.teardown(None)


@pytest.mark.parametrize("workers", [0, 4])
def test_fixed_epoch_plan_survives_iterator_reopening_after_prefetch(workers):
    api = recovery()
    module = paired_module("fixed", workers=workers, epoch=1)
    loader = module.train_dataloader()
    expected = [index for indices in api.loader_plan(loader) for index in indices]
    iterator = iter(loader)
    assert next(iterator).sample_id.tolist() == expected[:5]
    restarted = iter(loader)
    try:
        actual = [value for batch in restarted for value in batch.sample_id.tolist()]
        assert actual == expected
    finally:
        if workers:
            restarted._shutdown_workers()
        module.teardown(None)


@pytest.mark.parametrize("broken", ["lazy", "rotating", "distributed", "custom_dataset"])
def test_recovery_loader_rejects_uncertified_data_before_iteration(broken):
    module = paired_module("fixed")
    if broken == "lazy":
        module.store_in_memory = False
    elif broken == "rotating":
        module.rotating_pool_size = 1
    elif broken == "distributed":
        module.trainer.world_size = 2
    else:
        class Stochastic(torch.utils.data.Dataset):
            def __len__(self):
                return 12
            def __getitem__(self, index):
                pytest.fail("uncertified dataset was read")
        module.train_dataset = Stochastic()
    with pytest.raises(ValueError, match="recover|cache|deterministic|single"):
        module.train_dataloader()
    module.rotating_pool_size = None
    module.teardown(None)


def test_rng_roundtrip():
    api = recovery()
    assert hasattr(api, "capture_rng"), "recovery RNG capture is not implemented"
    state = api.capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    api.restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


def test_rng_capture_does_not_initialize_an_unused_gpu(monkeypatch):
    api = recovery()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden():
        pytest.fail("CPU recovery initialized a GPU that it did not own")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    assert api.capture_rng()["cuda"] == []


def assert_state_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            assert_state_equal(value, target)
    else:
        assert actual == expected


class ControlledInterruption(Exception):
    pass


def toy_fit(root, *, checkpoint=None, interrupt_step=None, epochs=4, validation_every=1,
            max_steps=-1, scheduler="cosine", accumulate=1, validation_curve=None):
    api = recovery()
    assert hasattr(api, "RecoveryCallback"), "recovery callback is not implemented"
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(.25),
                                             torch.nn.Linear(3, 1)).double()
            self.trace = []

        def on_train_start(self):
            # Every restart consumes preparatory RNG; recovery must undo only this extra work.
            random.random(), np.random.random(), torch.rand(5)

        def on_train_epoch_start(self):
            # This is legitimate per-epoch work, which must NOT be undone by resume.
            random.random(), np.random.random(), torch.rand(2)

        def training_step(self, batch, batch_idx):
            identifiers = batch.sample_id
            x = identifiers.to(dtype=torch.float64)[:, None] / 12.
            prediction = self.layer(x)
            loss = (prediction - x.square()).square().mean() * (1 + random.random() + np.random.random())
            self.trace.append((identifiers.tolist(), float(loss.detach())))
            return loss

        def validation_step(self, batch, batch_idx):
            score = .1 + self.current_epoch if validation_curve is None else validation_curve[self.current_epoch]
            self.log("val_loss", torch.tensor(score), batch_size=batch.num_graphs)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=.01)
            schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                        if scheduler == "cosine" else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "monitor": "val_loss"}}

    class Interrupt(pl.Callback):
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            if interrupt_step is not None and trainer.global_step == interrupt_step:
                raise ControlledInterruption

    module = paired_module("fixed")
    module.trainer = None
    module.batch_size = 3
    module.val_dataset = [module.train_dataset[index] for index in (0, 1)]
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = Toy()
    store = api.RecoveryStore(root, {"seed": 7, "epochs": epochs})
    if checkpoint is not None:
        store.restore_best(store.latest())
    best = ModelCheckpoint(dirpath=root / "best", monitor="val_loss", mode="min", save_top_k=1)
    recover = api.RecoveryCallback(store, 7, [str(index) for index in range(12)])
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, max_steps=max_steps,
        accumulate_grad_batches=accumulate, logger=False, enable_model_summary=False,
        enable_progress_bar=False, num_sanity_val_steps=1, check_val_every_n_epoch=validation_every,
        callbacks=[Interrupt(), best, recover])
    try:
        trainer.fit(model, datamodule=module, ckpt_path=checkpoint, weights_only=False)
    finally:
        module.teardown(None)
    return SimpleNamespace(model=model, trainer=trainer, store=store, best=best, recover=recover)


@pytest.mark.parametrize("validation_every", [1, 5])
def test_callback_continuity_preserves_optimizer_scheduler_rng_and_early_best(tmp_path, validation_every):
    api = recovery()
    assert hasattr(api, "RecoveryCallback"), "recovery callback is not implemented"
    epochs = 4 if validation_every == 1 else 6
    full = toy_fit(tmp_path / "full", epochs=epochs, validation_every=validation_every)
    root = tmp_path / "resumed"
    with pytest.raises(ControlledInterruption):
        toy_fit(root, interrupt_step=6, epochs=epochs, validation_every=validation_every)
    store = api.RecoveryStore(root, {"seed": 7, "epochs": epochs})
    first = store.latest()
    assert first["next_epoch"] == 1 and first["global_step"] == 4
    resumed = toy_fit(root, checkpoint=first["checkpoint"], epochs=epochs, validation_every=validation_every)
    expected = torch.load(full.store.latest()["checkpoint"], map_location="cpu", weights_only=False)
    actual = torch.load(resumed.store.latest()["checkpoint"], map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers"):
        assert_state_equal(actual[key], expected[key])
    assert_state_equal(actual["g2m_recovery"]["rng"], expected["g2m_recovery"]["rng"])
    assert resumed.model.trace == full.model.trace[4:]
    assert resumed.trainer.global_step == full.trainer.global_step == 4 * epochs
    assert resumed.best.best_model_score == full.best.best_model_score
    assert resumed.best.current_score == full.best.current_score
    assert Path(resumed.best.best_model_path).is_file()
    best_epoch = "epoch=0" if validation_every == 1 else "epoch=4"
    assert best_epoch in resumed.best.best_model_path
    assert actual["g2m_recovery"]["samples_committed"] == epochs * 12


def test_callback_does_not_publish_partial_epoch(tmp_path):
    api = recovery()
    assert hasattr(api, "RecoveryCallback"), "recovery callback is not implemented"
    run = toy_fit(tmp_path / "partial", max_steps=2)
    assert run.trainer.global_step == 2
    assert run.store.latest() is None


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf")])
def test_nonfinite_validation_cannot_commit_over_an_earlier_finite_best(tmp_path, bad_score):
    api = recovery()
    root = tmp_path / "invalid-validation"
    with pytest.raises(ValueError, match="nonfinite.*validation|validation.*nonfinite"):
        toy_fit(root, epochs=2, validation_curve=(.1, bad_score))
    receipt = api.RecoveryStore(root, {"seed": 7, "epochs": 2}).latest()
    assert receipt["next_epoch"] == 1 and receipt["samples_committed"] == 12
    assert Path(receipt["best"]["path"]).is_file()


@pytest.mark.parametrize("options", [{"scheduler": "plateau"}, {"accumulate": 2}])
def test_callback_rejects_uncertified_frontier_before_training(tmp_path, options):
    api = recovery()
    assert hasattr(api, "RecoveryCallback"), "recovery callback is not implemented"
    with pytest.raises(ValueError, match="recover|cosine|accumul"):
        toy_fit(tmp_path / "invalid", **options)
    assert api.RecoveryStore(tmp_path / "invalid", {"seed": 7, "epochs": 4}).latest() is None


def test_callback_repeated_interruptions_recover_only_confirmed_epochs(tmp_path):
    api = recovery()
    control = toy_fit(tmp_path / "control")
    root = tmp_path / "twice"
    store = api.RecoveryStore(root, {"seed": 7, "epochs": 4})
    with pytest.raises(ControlledInterruption):
        toy_fit(root, interrupt_step=6)
    first = store.latest()
    assert first["global_step"] == 4
    with pytest.raises(ControlledInterruption):
        toy_fit(root, checkpoint=first["checkpoint"], interrupt_step=11)
    second = store.latest()
    assert second["global_step"] == 8 and second["samples_committed"] == 24
    resumed = toy_fit(root, checkpoint=second["checkpoint"])
    assert_state_equal(resumed.model.state_dict(), control.model.state_dict())
    assert_state_equal(resumed.trainer.optimizers[0].state_dict(), control.trainer.optimizers[0].state_dict())
    assert resumed.store.latest()["samples_committed"] == 48
    assert Path(resumed.best.best_model_path).is_file()


def test_callback_restores_deleted_early_best_after_uncommitted_improvement(tmp_path, monkeypatch):
    api = recovery()
    root = tmp_path / "rollback"
    curve = (1., .1, 2., 3.)
    publish = api.RecoveryStore.publish
    def fail_second_epoch(self, trainer, *, boundary, best_path):
        if boundary["next_epoch"] == 2:
            raise OSError("save failed after a new scientific best")
        return publish(self, trainer, boundary=boundary, best_path=best_path)
    with monkeypatch.context() as patch:
        patch.setattr(api.RecoveryStore, "publish", fail_second_epoch)
        with pytest.raises(OSError, match="new scientific best"):
            toy_fit(root, validation_curve=curve)
    store = api.RecoveryStore(root, {"seed": 7, "epochs": 4})
    receipt = store.latest()
    assert receipt["next_epoch"] == 1
    early_best = Path(receipt["best"]["path"])
    assert not early_best.exists()  # Native top-k removed it before recovery failed.
    original = (Path(receipt["generation"]) / "best.ckpt").read_bytes()
    assert store.restore_best(receipt) == early_best
    assert early_best.read_bytes() == original
    resumed = toy_fit(root, checkpoint=receipt["checkpoint"], validation_curve=curve)
    assert resumed.best.best_model_score == torch.tensor(.1)
    assert "epoch=1" in resumed.best.best_model_path


def test_callback_first_epoch_interrupt_has_no_false_committed_state(tmp_path):
    api = recovery()
    root = tmp_path / "first"
    with pytest.raises(ControlledInterruption):
        toy_fit(root, interrupt_step=2)
    assert api.RecoveryStore(root, {"seed": 7, "epochs": 4}).latest() is None
    restarted = toy_fit(root)
    control = toy_fit(tmp_path / "control")
    assert_state_equal(restarted.model.state_dict(), control.model.state_dict())
