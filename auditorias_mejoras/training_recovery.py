"""Opt-in experiment recovery; published generations never depend on a live writer."""
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import random
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

import graph2mat  # Register optional-module safe globals before Lightning/e3nn imports.
from graph2mat.bindings.torch import InMemoryData
from graph2mat.tools.lightning.data import MatrixDataModule, _CostBatchSampler
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from run_readout_comparison import sha256


class UnsafeRecoveryPath(ValueError):
    """An owned checkpoint operation would cross its filesystem boundary."""


class IncompatibleRecovery(ValueError):
    """An intact generation belongs to another experiment configuration."""


def fsync_path(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fingerprint_sha256(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class RecoveryStore:
    """Two complete generations, each pinning its scientifically selected best.

    The publisher lock must not be held by a supervising parent during child fit.
    """
    _generation = re.compile(r"generation-\d{8}-\d{12}-\d+")
    _boundary_fields = ("next_epoch", "global_step", "samples_committed", "plan_sha256", "epoch_complete")

    def __init__(self, root, fingerprint):
        self.root = Path(root).absolute()
        self._path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = self._path(self.root / "recovery")
        self.directory.mkdir(exist_ok=True)
        self.fingerprint = json.loads(json.dumps(fingerprint, allow_nan=False))
        self.fingerprint_sha256 = fingerprint_sha256(self.fingerprint)
        self._lock_handle = None
        self._lock_owner = None

    def _path(self, path):
        path = Path(path).absolute()
        if path != self.root and self.root not in path.parents:
            raise UnsafeRecoveryPath("path must be owned by recovery job: " + str(path))
        if path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
            raise UnsafeRecoveryPath("symlink/noncanonical recovery path: " + str(path))
        return path

    def _best_path(self, path):
        path = self._path(path)
        if self.root / "best" not in path.parents or path.suffix != ".ckpt":
            raise UnsafeRecoveryPath("best path must belong to own best checkpoint directory")
        return path

    @contextmanager
    def lock(self):
        if self._lock_handle is not None:
            if self._lock_owner != os.getpid():
                raise RuntimeError("fork child cannot borrow its parent's writer lock")
            yield
            return
        path = self._path(self.directory / "writer.lock")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("recovery writer lock is already held") from error
            self._lock_handle = descriptor
            self._lock_owner = os.getpid()
            yield
        finally:
            self._lock_handle = None
            self._lock_owner = None
            os.close(descriptor)

    def _atomic_json(self, path, value):
        path = self._path(path)
        temporary = self._path(path.with_name("." + path.name + "." + uuid.uuid4().hex))
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_path(path.parent)

    def _copy(self, source, target):
        source, target = self._path(source), self._path(target)
        with source.open("rb") as original, target.open("xb") as copied:
            shutil.copyfileobj(original, copied)
            copied.flush()
            os.fsync(copied.fileno())

    def _validate_boundary(self, boundary):
        if boundary.get("epoch_complete") is not True:
            raise ValueError("checkpoint boundary is not a complete epoch")
        for key in ("next_epoch", "global_step", "samples_committed"):
            if type(boundary.get(key)) is not int or boundary[key] < 1:
                raise ValueError("invalid checkpoint boundary: " + key)
        if not re.fullmatch(r"[0-9a-f]{64}", boundary.get("plan_sha256", "")):
            raise ValueError("invalid checkpoint plan fingerprint")

    def _validate_checkpoint(self, path, boundary):
        state = torch.load(self._path(path), map_location="cpu", weights_only=False)
        required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "loops",
                    "epoch", "global_step", "g2m_recovery"}
        if not isinstance(state, dict) or not required <= state.keys():
            raise ValueError("incomplete recovery checkpoint")
        metadata = state["g2m_recovery"]
        if metadata.get("fingerprint_sha256") != self.fingerprint_sha256:
            raise IncompatibleRecovery("checkpoint fingerprint mismatch")
        if any(metadata.get(key) != boundary[key] for key in self._boundary_fields):
            raise ValueError("checkpoint disagrees with committed boundary")
        if state["epoch"] != boundary["next_epoch"] - 1 or state["global_step"] != boundary["global_step"]:
            raise ValueError("checkpoint epoch/step disagrees with boundary")
        if not all(state[key] for key in ("optimizer_states", "lr_schedulers", "callbacks", "loops")):
            raise ValueError("checkpoint training state missing")
        rng = metadata.get("rng", {})
        if not {"python", "numpy", "torch", "cuda"} <= rng.keys():
            raise ValueError("checkpoint RNG state missing")
        fit = state["loops"]["fit_loop"]
        current = fit["epoch_progress"]["current"]
        batches = fit["epoch_loop.batch_progress"]
        if any(current[key] != boundary["next_epoch"] for key in ("ready", "started", "processed")):
            raise ValueError("checkpoint epoch progress is incomplete")
        if not batches["is_last_batch"] or len(set(batches["current"].values())) != 1:
            raise ValueError("checkpoint batch progress is incomplete")
        if any(not torch.isfinite(value).all() for value in state["state_dict"].values()):
            raise ValueError("checkpoint model state is nonfinite")
        return state

    def _generations(self):
        self._path(self.directory)
        return sorted((self._path(p) for p in self.directory.iterdir()
                       if self._generation.fullmatch(p.name)), reverse=True)

    def _read_generation(self, generation):
        generation = self._path(generation)
        if generation.parent != self.directory or not self._generation.fullmatch(generation.name):
            raise ValueError("invalid checkpoint generation")
        receipt = json.loads(self._path(generation / "receipt.json").read_text())
        if receipt["format_version"] != 1:
            raise ValueError("unsupported checkpoint receipt")
        if (receipt["fingerprint_sha256"] != self.fingerprint_sha256
                or receipt["fingerprint"] != self.fingerprint):
            raise IncompatibleRecovery("recovery fingerprint mismatch")
        self._validate_boundary(receipt)
        expected = {"state.ckpt", "best.ckpt"} if receipt["best"] is not None else {"state.ckpt"}
        if set(receipt["files"]) != expected:
            raise ValueError("checkpoint receipt files mismatch")
        for name, identity in receipt["files"].items():
            path = self._path(generation / name)
            if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
                raise ValueError("checkpoint integrity/hash mismatch: " + name)
        state = self._validate_checkpoint(generation / "state.ckpt", receipt)
        if receipt["best"] is not None:
            best = self._best_path(receipt["best"]["path"])
            if receipt["best"]["sha256"] != receipt["files"]["best.ckpt"]["sha256"]:
                raise ValueError("best checkpoint identity mismatch")
            if not any(callback.get("best_model_path") == str(best)
                       for callback in state["callbacks"].values() if isinstance(callback, dict)):
                raise ValueError("checkpoint selection does not match protected best")
        return {**receipt, "checkpoint": str(generation / "state.ckpt"), "generation": str(generation)}

    def latest(self):
        self._path(self.directory / "latest.json")
        generations, errors = self._generations(), []
        for generation in generations:
            try:
                return {**self._read_generation(generation), "fallback": errors}
            except (UnsafeRecoveryPath, IncompatibleRecovery):
                raise
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as error:
                errors.append({"generation": generation.name, "reason": str(error)})
        if generations:
            raise ValueError("no valid recovery generation: " + str(errors))
        return None

    def publish(self, trainer, *, boundary, best_path):
        self._validate_boundary(boundary)
        with self.lock():
            # Validate the old state before trusting this directory for another publication.
            self.latest()
            best = self._best_path(best_path) if best_path is not None else None
            temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=self._path(self.directory)))
            checkpoint = temporary / "state.ckpt"
            trainer.save_checkpoint(checkpoint)
            self._validate_checkpoint(checkpoint, boundary)
            fsync_path(checkpoint)
            files = {"state.ckpt": {"bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)}}
            selected = None
            if best is not None:
                self._copy(best, temporary / "best.ckpt")
                files["best.ckpt"] = {"bytes": best.stat().st_size, "sha256": sha256(temporary / "best.ckpt")}
                selected = {"path": str(best), "sha256": files["best.ckpt"]["sha256"]}
            receipt = {"format_version": 1, "fingerprint": self.fingerprint,
                       "fingerprint_sha256": self.fingerprint_sha256,
                       **{key: boundary[key] for key in self._boundary_fields},
                       "files": files, "best": selected}
            self._atomic_json(temporary / "receipt.json", receipt)
            generation = self.directory / (f"generation-{boundary['next_epoch']:08d}-"
                                          f"{boundary['global_step']:012d}-{time.time_ns()}")
            # Validate selection before publishing, using the same reader as resume.
            state = self._validate_checkpoint(checkpoint, boundary)
            if best is not None and not any(callback.get("best_model_path") == str(best)
                    for callback in state["callbacks"].values() if isinstance(callback, dict)):
                raise ValueError("checkpoint selection does not match protected best")
            os.replace(temporary, self._path(generation))
            fsync_path(self.directory)
            self._atomic_json(self.directory / "latest.json", {"generation": generation.name})
            self._prune()
            return generation

    def _prune(self):
        valid = []
        for generation in self._generations():
            try:
                valid.append(self._read_generation(generation))
            except (UnsafeRecoveryPath, IncompatibleRecovery):
                raise
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError):
                continue
        pinned = {receipt["best"]["path"] for receipt in valid[:2] if receipt["best"] is not None}
        for receipt in valid[2:]:
            generation = self._path(receipt["generation"])
            children = list(generation.iterdir())
            if any(child.name not in {"state.ckpt", "best.ckpt", "receipt.json"} for child in children):
                raise ValueError("unknown files in recovery generation; refusing deletion")
            for child in children:
                self._path(child).unlink()
            generation.rmdir()
            best = receipt["best"]
            if best is not None and best["path"] not in pinned:
                path = self._best_path(best["path"])
                if path.exists() and sha256(path) == best["sha256"]:
                    path.unlink()
        fsync_path(self.directory)

    def restore_best(self, receipt):
        validated = self._read_generation(Path(receipt["generation"]))
        best = validated["best"]
        if best is None:
            return None
        path = self._best_path(best["path"])
        if path.exists():
            if sha256(path) != best["sha256"]:
                raise ValueError("existing best checkpoint is unexpected; refusing overwrite")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
        self._copy(Path(validated["generation"]) / "best.ckpt", temporary)
        os.replace(temporary, path)
        fsync_path(path.parent)
        return path


def loader_plan(loader):
    """Read a frozen epoch plan, including after workers consume sampler prefetch."""
    if not hasattr(loader, "_recovery_plan"):
        raise ValueError("loader has no certified recovery plan")
    return [list(batch) for batch in loader._recovery_plan]


class _FrozenFixedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, plan):
        super().__init__(sampler, batch_size, drop_last)
        self.plan = tuple(tuple(batch) for batch in plan)

    def __iter__(self):
        yield from self.plan

    def __len__(self):
        return len(self.plan)


class RecoveryDataModule(MatrixDataModule):
    """Experiment-only seed/epoch order, independent of model and worker RNG."""

    __signature__ = inspect.signature(MatrixDataModule)

    def __init__(self, *args, **kwargs):
        # Do not wrap this signature: Lightning reads actual __init__ frame locals.
        self.recovery_seed = torch.initial_seed()
        super().__init__(*args, **kwargs)

    def _certify_dataset(self, dataset):
        if not self.store_in_memory or self.rotating_pool_size:
            raise ValueError("recovery requires cached deterministic data, without rotating pools")
        if self.trainer is not None and self.trainer.world_size != 1:
            raise ValueError("recovery requires a single device")
        while type(dataset) is torch.utils.data.Subset:
            dataset = dataset.dataset
        if type(dataset) not in (InMemoryData, list, tuple):
            raise ValueError("recovery requires deterministic cached snapshots, not custom datasets")

    def _loader_kwargs(self):
        # Every epoch owns a new loader; pinned persistent pools retain atexit resources.
        return {**super()._loader_kwargs(), "persistent_workers": False}

    def train_dataloader(self):
        self._certify_dataset(self.train_dataset)
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        if self.trainer is not None:
            self.trainer.reload_dataloaders_every_n_epochs = 1
        loader = super().train_dataloader()
        batches = loader.batch_sampler
        if type(batches.sampler) is not torch.utils.data.RandomSampler:
            raise ValueError("recovery requires the standard random sampler")
        batches.sampler.generator = torch.Generator().manual_seed((self.recovery_seed + epoch) % 2**64)
        loader.generator = torch.Generator().manual_seed((self.recovery_seed + epoch + 2**32) % 2**64)
        if type(batches) is _CostBatchSampler:
            # Preserve the native plan even for unbound preflight loaders (epoch=None).
            batches._retain_plan = True
            plan = batches._batches()
        elif type(batches) is torch.utils.data.BatchSampler:
            generator = torch.Generator().set_state(batches.sampler.generator.get_state())
            sampler = torch.utils.data.RandomSampler(loader.dataset, generator=generator)
            plan = list(torch.utils.data.BatchSampler(sampler, batches.batch_size, batches.drop_last))
            frozen = _FrozenFixedBatchSampler(batches.sampler, batches.batch_size, batches.drop_last, plan)
            replacement = DataLoader(loader.dataset, batch_sampler=frozen, generator=loader.generator,
                                     **self._loader_kwargs())
            replacement.collate_fn = loader.collate_fn
            loader = replacement
        else:
            raise ValueError("recovery requires a standard fixed/cost batch sampler")
        loader._recovery_plan = tuple(tuple(batch) for batch in plan)
        return loader

    def val_dataloader(self):
        self._certify_dataset(self.val_dataset)
        loader = super().val_dataloader()
        loader.generator = torch.Generator().manual_seed((self.recovery_seed + 2**33) % 2**64)
        return loader


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class RecoveryCallback(ModelCheckpoint):
    """Commit after best selection and cosine update; never save a partial epoch."""
    def __init__(self, store, seed, sample_ids):
        super().__init__(dirpath=store.directory, monitor=None, save_top_k=0,
                         save_last=False, save_on_train_epoch_end=True)
        self.store, self.seed, self.sample_ids = store, seed, tuple(sample_ids)
        self.samples_committed = 0
        self.samples_observed = 0
        self.save_seconds = 0.
        self._pending = None
        self._publishing = False
        self._best_callback = None
        self._plan, self._batch_count = [], 0
        self._boundary = None

    def on_fit_start(self, trainer, model):
        if (trainer.world_size != 1 or trainer.accumulate_grad_batches != 1
                or not model.automatic_optimization):
            raise ValueError("recovery requires single-device automatic optimization without accumulation")
        if len(trainer.optimizers) != 1 or type(trainer.optimizers[0]) is not torch.optim.Adam:
            raise ValueError("recovery currently certifies one Adam optimizer")
        schedules = trainer.lr_scheduler_configs
        if (len(schedules) != 1 or type(schedules[0].scheduler) is not torch.optim.lr_scheduler.CosineAnnealingLR
                or schedules[0].interval != "epoch" or schedules[0].frequency != 1):
            raise ValueError("recovery requires an epoch cosine scheduler")
        if not isinstance(trainer.datamodule, RecoveryDataModule) or trainer.datamodule.recovery_seed != self.seed:
            raise ValueError("recovery datamodule seed mismatch")
        if len(self.sample_ids) != len(trainer.datamodule.train_dataset) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("recovery sample identity/coverage mismatch")
        selected = [callback for callback in trainer.callbacks if isinstance(callback, ModelCheckpoint)
                    and callback is not self and callback.monitor == "val_loss" and callback.mode == "min"]
        if len(selected) != 1:
            raise ValueError("recovery requires one scientific best-val_loss callback")
        self._best_callback = selected[0]
        if Path(self._best_callback.dirpath) != self.store.root / "best":
            raise ValueError("recovery scientific best directory must be stable within job")
        if trainer.callbacks.index(self._best_callback) > trainer.callbacks.index(self):
            raise ValueError("recovery must run after scientific checkpoint selection")

    def on_train_epoch_start(self, trainer, model):
        # on_train_start (including the module hook) repeats on resume, not every epoch.
        if self._pending is not None:
            if trainer.current_epoch != self._pending["next_epoch"]:
                raise ValueError("recovery next epoch does not match native Lightning progress")
            restore_rng(self._pending["rng"])
            self._pending = None
        self._plan = loader_plan(trainer.train_dataloader)
        flattened = [index for batch in self._plan for index in batch]
        if sorted(flattened) != list(range(len(self.sample_ids))):
            raise ValueError("recovery epoch plan must cover each sample exactly once")
        self._batch_count = 0
        self.samples_observed = self.samples_committed

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if batch_idx != self._batch_count or batch_idx >= len(self._plan):
            raise ValueError("recovery batch progress does not match epoch plan")
        if batch.num_graphs != len(self._plan[batch_idx]):
            raise ValueError("recovery batch size does not match epoch plan")
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None or not torch.isfinite(loss).all():
            raise ValueError("nonfinite/missing recovery training loss")
        self._batch_count += 1
        self.samples_observed += batch.num_graphs

    def on_validation_end(self, trainer, model):
        score = trainer.callback_metrics.get("val_loss")
        if score is not None and not torch.isfinite(torch.as_tensor(score)).all():
            raise ValueError("nonfinite validation loss; recovery epoch not committed")

    def on_train_epoch_end(self, trainer, model):
        if self._batch_count != len(self._plan):
            return
        next_epoch = trainer.current_epoch + 1
        if trainer.lr_scheduler_configs[0].scheduler.last_epoch != next_epoch:
            raise ValueError("recovery boundary precedes cosine scheduler update")
        self.samples_committed = self.samples_observed
        self._boundary = dict(next_epoch=next_epoch, global_step=trainer.global_step,
            samples_committed=self.samples_committed, epoch_complete=True,
            plan_sha256=fingerprint_sha256({"epoch": trainer.current_epoch,
                "plan": [[self.sample_ids[index] for index in batch] for batch in self._plan]}))
        self._publishing = True
        started = time.monotonic()
        try:
            self.store.publish(trainer, boundary=self._boundary,
                best_path=Path(self._best_callback.best_model_path) if self._best_callback.best_model_path else None)
        finally:
            self.save_seconds += time.monotonic() - started
            self._publishing = False

    def on_save_checkpoint(self, trainer, model, checkpoint):
        if self._publishing:
            checkpoint["g2m_recovery"] = {**self._boundary, "seed": self.seed,
                "fingerprint_sha256": self.store.fingerprint_sha256,
                "sample_ids_sha256": fingerprint_sha256(self.sample_ids), "rng": capture_rng()}

    def on_load_checkpoint(self, trainer, model, checkpoint):
        metadata = checkpoint.get("g2m_recovery")
        if (not metadata or metadata.get("fingerprint_sha256") != self.store.fingerprint_sha256
                or metadata.get("seed") != self.seed
                or metadata.get("sample_ids_sha256") != fingerprint_sha256(self.sample_ids)):
            raise ValueError("recovery checkpoint provenance/seed mismatch")
        self._pending = metadata
        self.samples_committed = metadata["samples_committed"]
        self.samples_observed = self.samples_committed
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.monitor == "val_loss":
                # Lightning serializes current_score but does not restore it itself.
                callback.current_score = checkpoint["callbacks"][callback.state_key]["current_score"]

    def state_dict(self):
        return {**super().state_dict(), "samples_committed": self.samples_committed}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        samples = state_dict["samples_committed"]
        if self._pending is not None and samples != self._pending["samples_committed"]:
            raise ValueError("recovery callback progress disagrees with checkpoint")
        self.samples_committed = samples
