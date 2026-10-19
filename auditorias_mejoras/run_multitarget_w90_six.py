"""Run exactly the six authorized direct w90 fits, without modifying autorun.

python auditorias_mejoras/run_multitarget_w90_six.py prepare --out <new-directory>
python auditorias_mejoras/run_multitarget_w90_six.py supervise --out <directory>
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

try:
    from auditorias_mejoras import pilot_multitarget_w90 as pilot
except ModuleNotFoundError:
    import pilot_multitarget_w90 as pilot

REPO = pilot.REPO
V2 = REPO / "auditorias_mejoras/piloto_multitarget_w90_20261019_v2"
JOBS = [dict(id=f"{arm}_s{seed}", arm=arm, seed=seed) for seed in range(3) for arm in ("h", "joint")]


def atomic_state(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def verify_artifacts(out, hashes):
    for name, digest in hashes.items():
        if pilot.sha(out / name) != digest:
            raise ValueError(f"campaign artifact changed: {name}")


def verify_campaign(out):
    campaign = json.loads((out / "campaign.json").read_text())
    if campaign["jobs"] != JOBS:
        raise ValueError("authorized six-job list changed")
    for name, digest in campaign["code"].items():
        if pilot.sha(REPO / name) != digest:
            raise ValueError(f"campaign code changed: {name}")
    for name, digest in campaign["certificate"].items():
        if pilot.sha(V2 / name) != digest:
            raise ValueError(f"pilot certificate changed: {name}")
    pilot.verify_sources(json.loads((out / "manifest.json").read_text()))
    verify_artifacts(out, campaign["hashes"])
    return campaign


def prepare(out):
    import torch
    from torch_geometric.data import Batch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    out.mkdir()  # Never adopt an interrupted or previous campaign.
    manifest = json.loads((V2 / "manifest.json").read_text())
    pilot.verify_sources(manifest)
    old_summary = json.loads((V2 / "summary.json").read_text())
    if old_summary["status"] != "complete":
        raise ValueError("v2 pilot was not complete")
    certificate = json.loads((V2 / "preflight.json").read_text())
    if certificate["status"] != "complete" or certificate["structures"] != 1498:
        raise ValueError("reference preflight incomplete")
    del certificate
    pilot.write_json(out / "manifest.json", manifest)
    table = pilot.certified_table(manifest)
    paths = [manifest["paths"][s] for s in manifest["split_ids"]["train"][:2]]
    joint_data = TorchBasisMatrixDataset(paths, pilot.arm_processor(table, "joint"))
    h_data = TorchBasisMatrixDataset(paths, pilot.arm_processor(table, "h"))
    collate = TypeLayoutCollate(Batch.from_data_list, table)
    jb, hb = (collate([dataset[0], dataset[1]]) for dataset in (joint_data, h_data))
    hashes = {"manifest.json": pilot.sha(out / "manifest.json")}
    checks = {}
    for seed in range(3):
        torch.manual_seed(seed)
        h = pilot.make_arm("h", table)
        joint = pilot.make_arm("joint", table, h.model.mace.state_dict())
        checks[str(seed)] = pilot.check_pair((h, joint), jb, hb)
        pilot.write_json(out / f"order-s{seed}.json", pilot.batch_order(manifest["split_ids"]["train"], seed, 4000))
        hashes[f"order-s{seed}.json"] = pilot.sha(out / f"order-s{seed}.json")
        for arm, model in (("h", h), ("joint", joint)):
            job_id = f"{arm}_s{seed}"
            (out / job_id).mkdir()
            with (out / f"initial-{job_id}.pt").open("xb") as stream:
                torch.save(model.state_dict(), stream)
            optimizer = model.configure_optimizers()
            names = {id(p): name for name, p in model.named_parameters()}
            config = dict(seed=seed, model=pilot.serializable(dict(model.hparams)),
                optimizer=dict(defaults=pilot.serializable(optimizer.defaults), groups=[
                    {**{k: pilot.serializable(v) for k, v in group.items() if k != "params"},
                     "params": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups]),
                scheduler=dict(T_max=4000, eta_min=1e-5, interval="step"),
                data=dict(batch_size=32, drop_last=False, loader_threads=0, pin_memory=True,
                    store_in_memory=True, sub_point_matrix=False, symmetric_matrix=True),
                precision="32-true", tf32=False, updates=4000, val_every=200,
                monitor="val_loss" if arm == "h" else "val/target/hamiltonian/loss")
            pilot.write_json(out / f"config-{job_id}.json", config)
            for name in (f"initial-{job_id}.pt", f"config-{job_id}.json"):
                hashes[name] = pilot.sha(out / name)
        del h, joint, optimizer
    own = [Path(__file__), REPO / "auditorias_mejoras/test_run_multitarget_w90_six.py"]
    campaign = dict(jobs=JOBS, hashes=hashes, checks=checks,
        code={str(p.relative_to(REPO)): pilot.sha(p) for p in own},
        certificate={name: pilot.sha(V2 / name) for name in ("manifest.json", "preflight.json", "kpoints.json", "summary.json")},
        safety=dict(total_seconds=5400, per_job_seconds=1200, poll_seconds=.5,
            start_temperature_below=75, abort_temperature=80, abort_free_fraction=.25),
        limitations=manifest["warnings"], authorization="User Lanzalos, 2026-10-19: six fits only; no automatic retry")
    pilot.write_json(out / "campaign.json", campaign)
    verify_campaign(out)
    return campaign


def fit_model(model, train, val, out, arm, order, device, *, updates=4000, validate_every=200):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    # Pilot deliberately disables logging; restore the real Lightning method.
    model.log = MethodType(pl.LightningModule.log, model)
    original_optimizer = model.configure_optimizers

    def configure(module):
        optimizer = original_optimizer()
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=pilot.make_scheduler(optimizer), interval="step"))

    model.configure_optimizers = MethodType(configure, model)
    monitor = "val_loss" if arm == "h" else "val/target/hamiltonian/loss"
    checkpoint = ModelCheckpoint(dirpath=out / "checkpoints", monitor=monitor, mode="min", save_top_k=1,
        filename="best-u{step:04d}", auto_insert_metric_name=False, save_on_train_epoch_end=False)

    class FrozenData(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(dict(out_matrix="hamiltonian", sub_point_matrix=False,
                symmetric_matrix=True, n_matrix_components=2 if arm == "h" else 4,
                matrix_component_policy="h_and_overlap" if arm == "h" else "h_only",
                matrix_targets=None if arm == "h" else list(pilot.TARGETS),
                matrix_target_spin=None if arm == "h" else 1))

        def train_dataloader(self):
            return train

        def val_dataloader(self):
            return val

    class Trace(pl.Callback):
        def __init__(self, stream):
            self.stream, self.validation, self.samples = stream, [], 0

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if batch_idx % 38 == 0:
                pilot.reset_metrics(module)
            self.ids = list(batch.sample_id)
            if self.ids != order[batch_idx]:
                raise ValueError(f"sample order mismatch at update {batch_idx + 1}")
            self.lr = trainer.optimizers[0].param_groups[0]["lr"]
            if device == "cuda":
                torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_before_backward(self, trainer, module, loss):
            pilot.require_finite([loss], "training loss")

        def on_after_backward(self, trainer, module):
            pilot.require_finite([p.grad for p in module.parameters() if p.grad is not None], "gradients")

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if device == "cuda":
                torch.cuda.synchronize()
            self.samples += batch.num_graphs
            row = dict(update=trainer.global_step, ids=self.ids, samples=batch.num_graphs,
                loss=float(outputs["loss"].detach()), lr=self.lr, wall_s=time.perf_counter() - self.started)
            expected_lr = 1e-5 + (.0018 - 1e-5) * (1 + math.cos(math.pi * batch_idx / 4000)) / 2
            if trainer.global_step != batch_idx + 1 or not math.isclose(self.lr, expected_lr, rel_tol=1e-8):
                raise ValueError("optimizer update or scheduler mismatch")
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.stream.flush()
            atomic_state(out / "progress.json", dict(status="running", update=trainer.global_step,
                samples=self.samples, time=time.time(), validation=self.validation))

        def on_validation_end(self, trainer, module):
            value = float(trainer.callback_metrics[monitor])
            if not math.isfinite(value):
                raise ValueError("nonfinite H validation metric")
            row = dict(update=trainer.global_step, h_gauge_mse=value,
                       total_loss=float(trainer.callback_metrics["val_loss"]))
            if not math.isfinite(row["total_loss"]):
                raise ValueError("nonfinite total validation metric")
            self.validation.append(row)
            atomic_state(out / "validation.json", self.validation)

    started = time.perf_counter()
    with (out / "updates.jsonl").open("x") as stream:
        trace = Trace(stream)
        trainer = pl.Trainer(accelerator="gpu" if device == "cuda" else "cpu", devices=1,
            precision="32-true", max_steps=updates, max_epochs=1, val_check_interval=validate_every,
            check_val_every_n_epoch=1, num_sanity_val_steps=0, enable_progress_bar=False,
            enable_model_summary=False, logger=CSVLogger(str(out), name="metrics", version=0),
            log_every_n_steps=20, callbacks=[trace, checkpoint], default_root_dir=out)
        trainer.fit(model, datamodule=FrozenData())
        if trainer.global_step != updates or [r["update"] for r in trace.validation] != list(range(validate_every, updates + 1, validate_every)):
            raise ValueError("incomplete update/validation budget")
        pilot.require_finite(list(model.parameters()), "final parameters")
        trainer.save_checkpoint(out / "last.ckpt")
    if not checkpoint.best_model_path:
        raise ValueError("missing best checkpoint")
    result = dict(updates=trainer.global_step, samples=trace.samples, monitor=monitor,
        validation=trace.validation, best_h_gauge_mse=float(checkpoint.best_model_score),
        best_checkpoint=checkpoint.best_model_path, best_sha256=pilot.sha(checkpoint.best_model_path),
        last_sha256=pilot.sha(out / "last.ckpt"), fit_seconds=time.perf_counter() - started)
    if device == "cuda":
        result["peaks_mib"] = dict(allocated=torch.cuda.max_memory_allocated() / 2**20,
                                  reserved=torch.cuda.max_memory_reserved() / 2**20)
    return result


def worker(out, job_id):
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from graph2mat.bindings.torch import TorchBasisMatrixDataset, InMemoryData
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    started = time.perf_counter()
    campaign = verify_campaign(out)
    job = next(job for job in campaign["jobs"] if job["id"] == job_id)
    arm, seed = job["arm"], job["seed"]
    torch.manual_seed(seed)
    manifest = json.loads((out / "manifest.json").read_text())
    table = pilot.certified_table(manifest)
    processor = pilot.arm_processor(table, arm)
    ids = manifest["split_ids"]
    datasets = {}
    for split in ("train", "validation"):
        for sample_id in ids[split]:
            pilot.checked_candidates(Path(manifest["paths"][sample_id]).parent, manifest)
        datasets[split] = InMemoryData(TorchBasisMatrixDataset(
            [pilot.check_fdf(manifest["paths"][s]) for s in ids[split]], processor))
        for sample, sample_id in zip(datasets[split].data_objects, ids[split]):
            sample.sample_id = sample_id
    order = json.loads((out / f"order-s{seed}.json").read_text())
    indices = {sample_id: i for i, sample_id in enumerate(ids["train"])}
    collate = TypeLayoutCollate(Batch.from_data_list, table)
    train = DataLoader(datasets["train"], batch_sampler=[[indices[s] for s in batch] for batch in order],
        collate_fn=collate, num_workers=0, pin_memory=True)
    val = DataLoader(datasets["validation"], batch_size=32, collate_fn=collate, num_workers=0, pin_memory=True)
    model = pilot.make_arm(arm, table)
    model.load_state_dict(torch.load(out / f"initial-{job_id}.pt", map_location="cpu", weights_only=True), strict=True)
    setup_s = time.perf_counter() - started
    result = fit_model(model, train, val, out / job_id, arm, order, "cuda")
    result.update(job=job, setup_s=setup_s, total_s=time.perf_counter() - started,
        order_sha256=pilot.sha(out / f"order-s{seed}.json"), initial_sha256=pilot.sha(out / f"initial-{job_id}.pt"))
    verify_campaign(out)
    pilot.write_json(out / job_id / "result.json", result)
    atomic_state(out / job_id / "progress.json", dict(status="complete", update=4000, time=time.time()))


def supervise_job(out, job, stream, deadline):
    now = time.monotonic()
    deadline = min(deadline, now + 1200)
    state, pids, uuid = pilot.query_gpu(timeout=min(2., max(.001, (deadline - now) / 2)))
    reason = pilot.guard_reason(state, pids, set(), time.monotonic(), deadline)
    if reason or state["temperature.gpu"] >= 75:
        raise RuntimeError(reason or "initial temperature >=75C")
    fraction = pilot.memory_fraction(state["memory.total"], state["memory.free"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (out / job["id"] / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
            "--job", job["id"], "--fraction", str(fraction)], env=env, cwd=REPO,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            atomic_state(out / job["id"] / "process.json", dict(pid=proc.pid, time=time.time(), fraction=fraction))
            while proc.poll() is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise RuntimeError("deadline")
                state, pids, current_uuid = pilot.query_gpu(timeout=min(2., left / 2))
                if current_uuid != uuid:
                    raise RuntimeError("GPU identity changed")
                host = pilot.process_tree_memory(proc.pid)
                owned = {proc.pid}
                for pid in host["pids"]:
                    try:
                        if os.getpgid(pid) == proc.pid:
                            owned.add(pid)
                    except ProcessLookupError:
                        pass
                reason = pilot.guard_reason(state, pids, owned, time.monotonic(), deadline)
                state.update(time=time.time(), job=job["id"], host=host, compute_pids=sorted(pids))
                stream.write(json.dumps(state, allow_nan=False) + "\n")
                stream.flush()
                if reason:
                    raise RuntimeError(reason)
                time.sleep(min(.5, max(0., deadline - time.monotonic())))
            if proc.returncode:
                raise RuntimeError(f"{job['id']} worker exit {proc.returncode}")
        finally:
            pilot.terminate_group(proc)
    for state in pilot.wait_context_exit(proc.pid, uuid, deadline):
        state["job"] = job["id"]
        stream.write(json.dumps(state, allow_nan=False) + "\n")
        stream.flush()
    result = json.loads((out / job["id"] / "result.json").read_text())
    if result["updates"] != 4000 or result["job"] != job:
        raise ValueError("incomplete/mismatched worker result")
    return result


def supervise(out):
    pilot.write_json(out / "started.json", dict(time=time.time(), pid=os.getpid()))
    state = dict(status="preparing", pid=os.getpid(), time=time.time(), jobs={}, results={}, reason=None)
    current = None
    try:
        campaign = json.loads((out / "campaign.json").read_text())
        state["jobs"] = {job["id"]: "pending" for job in campaign["jobs"]}
        atomic_state(out / "state.json", state)
        verify_campaign(out)
        deadline = time.monotonic() + 5400
        with (out / "telemetry.jsonl").open("x") as stream:
            for job in campaign["jobs"]:
                current = job["id"]
                state.update(status="running", active_job=current, time=time.time())
                state["jobs"][current] = "running"
                atomic_state(out / "state.json", state)
                state["results"][current] = supervise_job(out, job, stream, deadline)
                state["jobs"][current] = "complete"
                atomic_state(out / "state.json", state)
        verify_campaign(out)
        state["status"] = "complete"
    except Exception as error:
        state.update(status="aborted", reason=f"{type(error).__name__}: {error}")
        if current is not None:
            state["jobs"][current] = "failed"
    state.update(active_job=None, time=time.time())
    atomic_state(out / "state.json", state)
    pilot.write_json(out / "summary.json", state)
    print(json.dumps(dict(status=state["status"], jobs=state["jobs"], reason=state["reason"])), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "supervise", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job", choices=[job["id"] for job in JOBS])
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != REPO / "auditorias_mejoras":
        parser.error("output must be a direct child of graph2mat/auditorias_mejoras")
    import torch
    import graph2mat
    if Path(graph2mat.__file__).resolve() != REPO / "src/graph2mat/__init__.py":
        raise RuntimeError("wrong graph2mat checkout")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.command == "prepare":
        prepare(out)
    elif args.command == "supervise":
        sys.exit(0 if supervise(out)["status"] == "complete" else 2)
    else:
        if args.job is None or args.fraction is None or not 0 < args.fraction <= .55:
            parser.error("worker needs job and safe allocator fraction")
        torch.cuda.set_per_process_memory_fraction(args.fraction)
        worker(out, args.job)


if __name__ == "__main__":
    main()
