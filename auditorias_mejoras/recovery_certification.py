"""Synthetic recovery certification in fresh children; never a scientific campaign."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import bench_pipeline_gpu as pipeline
from run_bilayer_edge_updates import environment
from run_readout_comparison import REPO, gpu_processes, process_tree, sha256, write_json


class ControlledStop(Exception):
    pass


def child(args):
    pipeline.compiled_extensions()
    import graph2mat
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from graph2mat.bindings.torch import InMemoryData
    from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs
    from readout_comparison_callback import tensor_hash
    import training_recovery as recovery

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        if os.environ.get("G2M_GPU_GUARDED") != "1" or args.fraction is None:
            raise RuntimeError("CUDA certification child requires its owned supervisor")
        torch.cuda.set_per_process_memory_fraction(args.fraction)
    else:
        torch.use_deterministic_algorithms(True)
    random.seed(13)
    np.random.seed(13)
    dtype = torch.float64 if args.device == "cpu" else torch.float32
    mace, _, samples = training_inputs(dtype=dtype, coefficient_training=False,
        readout_fp32=args.precision != "32-true", torch_compile=args.compiled)

    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(.25),
                                              torch.nn.Linear(3, 1)).to(dtype=dtype)

        def training_step(self, batch, batch_idx):
            x = batch.sample_id.to(dtype=dtype)[:, None] / 12.
            return (self.layers(x) - x.square()).square().mean() * (1 + random.random() + np.random.random())

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", torch.tensor(.1 + self.current_epoch, device=self.device),
                     batch_size=batch.num_graphs)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=.01)

    model = mace if args.case == "mace" else Toy()
    module = recovery.RecoveryDataModule(basis_table=mace.basis_table, train_runs=[], val_runs=[],
        symmetric_matrix=True, sub_point_matrix=False, store_in_memory=True, batch_size=3,
        loader_threads=args.workers, n_matrix_components=1)
    module.setup("fit")
    module.train_dataset = InMemoryData(samples * 6)
    module.val_dataset = InMemoryData(samples)
    for index, sample in enumerate(module.train_dataset.data_objects):
        sample.sample_id = torch.tensor([index])
    for index, sample in enumerate(module.val_dataset.data_objects):
        sample.sample_id = torch.tensor([index])
    sizes = module.basis_table.basis_size.astype(np.int64)
    costs = []
    for sample in module.train_dataset:
        orbitals = sizes[sample.point_types.numpy()]
        edges = sample.edge_index.numpy()
        costs.append(int(np.square(orbitals).sum() + (orbitals[edges[0]] * orbitals[edges[1]]).sum()))
    module.max_batch_cost = max(costs) if args.mode == "cost" else None
    module.hparams["max_batch_cost"] = module.max_batch_cost
    fingerprint = {"case": args.case, "mode": args.mode, "workers": args.workers,
        "device": args.device, "precision": args.precision, "dtype": str(dtype), "epochs": 4,
        "seed": 13, "max_batch_cost": module.max_batch_cost, "negative_no_rng": args.negative_no_rng,
        "torch_compile": args.compiled, "torch_compile_mode": None,
        "initial_parameters": tensor_hash(model.state_dict().items()),
        "sources": {str(path): sha256(path) for path in (Path(__file__).resolve(),
                    Path(recovery.__file__).resolve(), REPO / "src/graph2mat/tools/lightning/data.py")},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "pytorch-lightning", "e3nn")}}
    store = recovery.RecoveryStore(args.root, fingerprint)
    receipt = store.latest()
    resume = args.phase in ("interrupt2", "finish")
    if resume != (receipt is not None):
        raise ValueError("certification phase does not match committed state")
    if receipt is not None:
        store.restore_best(receipt)
    native_optimizer = model.configure_optimizers
    def configure():
        optimizer = native_optimizer()
        return {"optimizer": optimizer, "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)}
    model.configure_optimizers = configure
    native_start, native_epoch_start = model.on_train_start, model.on_train_epoch_start
    def prepare():
        native_start()
        random.random(), np.random.random(), torch.rand(17 if resume else 3)
        if args.device == "cuda":
            torch.rand(17 if resume else 3, device="cuda")
    def epoch_start():
        native_epoch_start()
        random.random(), np.random.random(), torch.rand(2)
    model.on_train_start, model.on_train_epoch_start = prepare, epoch_start
    if args.negative_no_rng:
        recovery.restore_rng = lambda state: None
    attempt = args.root / ("attempt-" + args.phase)
    attempt.mkdir()
    rows = []
    stop_epoch = {"interrupt1": 1, "interrupt2": 2}.get(args.phase)

    class Trace(pl.Callback):
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            rows.append({"step": trainer.global_step, "epoch": trainer.current_epoch,
                         "sample_ids": batch.sample_id.tolist(), "loss": float(loss.detach().cpu())})
            write_json(attempt / "trace.json", rows)
            if trainer.current_epoch == stop_epoch and batch_idx == 1:
                write_json(attempt / "interrupt-ready.json", {"step": trainer.global_step, "pid": os.getpid()})
                if args.device == "cuda":
                    # The parent triggers our real guard; no foreign process is manipulated.
                    while not trainer.received_sigterm:
                        time.sleep(.05)
                raise ControlledStop

    best = ModelCheckpoint(dirpath=args.root / "best", monitor="val_loss", mode="min", save_top_k=1)
    callback = recovery.RecoveryCallback(store, 13, [str(index) for index in range(12)])
    trainer = pl.Trainer(accelerator="cpu" if args.device == "cpu" else "gpu", devices=1,
        precision=args.precision, max_epochs=4, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=1, callbacks=[Trace(), best, callback])
    started, status = time.monotonic(), "complete"
    try:
        trainer.fit(model, datamodule=module, ckpt_path=None if receipt is None else receipt["checkpoint"],
                    weights_only=False)
    except ControlledStop:
        status = "interrupted"
    finally:
        module.teardown(None)
    committed = store.latest()
    selected = store.restore_best(committed)
    best_state = torch.load(selected, map_location="cpu", weights_only=False)
    summary = {"status": status, "pid": os.getpid(), "checkpoint": committed["checkpoint"],
        "next_epoch": committed["next_epoch"], "global_step": committed["global_step"],
        "parameter_sha256": tensor_hash(model.state_dict().items()),
        "best_tensor_sha256": tensor_hash(best_state["state_dict"].items()),
        "seconds": time.monotonic() - started, "checkpoint_seconds": callback.save_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None}
    write_json(attempt / "summary.json", summary)
    return 75 if status == "interrupted" else 0


def compare(actual, expected, path, tolerance):
    import numpy as np
    import torch
    if torch.is_tensor(expected):
        numerical = (expected.is_floating_point() and path.rsplit(".", 1)[-1] != "step"
                     and path.startswith(("state_dict", "optimizer", "loss", "callbacks")))
        rtol, atol = tolerance if numerical else (0, 0)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda message: path + ": " + message)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], path + "." + str(key), tolerance)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (value, target) in enumerate(zip(actual, expected)):
            compare(value, target, path + "." + str(index), tolerance)
    else:
        assert actual == expected, path + f": {actual!r} != {expected!r}"


def normalize_paths(value, root):
    if isinstance(value, str):
        return value.replace(str(root), "$JOB")
    if isinstance(value, dict):
        return {normalize_paths(key, root): normalize_paths(item, root) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize_paths(item, root) for item in value]
    return value


def run_child(args, root, phase):
    root.mkdir(exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--device", args.device,
        "--precision", args.precision, "--case", args.case, "--mode", args.mode,
        "--workers", str(args.workers), "--phase", phase]
    if args.negative_no_rng:
        command.append("--negative-no-rng")
    if args.compiled:
        command.append("--compiled")
    log = root / (phase + ".log")
    release = 0.
    if args.device == "cpu":
        with log.open("w") as output:
            proc = subprocess.Popen(command, cwd=REPO, env=environment(cuda=""), stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = proc.wait(timeout=120)
            finally:
                for value in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, value)
                    except ProcessLookupError:
                        pass
                    if value == signal.SIGTERM:
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                proc.wait()
    else:
        if gpu_processes(process_tree(os.getpid())):
            raise RuntimeError("foreign GPU active; certification not launched")
        original = pipeline.gpu_state
        state = original()
        fraction = pipeline.memory_fraction(state["memory.total"], state["memory.free"])
        command += ["--fraction", str(fraction)]
        interrupted_at = None
        started = time.monotonic()
        def controlled_state():
            nonlocal interrupted_at
            if time.monotonic() - started > (120 if args.compiled else 60):
                raise RuntimeError("bounded certification child exceeded wall-clock budget")
            if gpu_processes(process_tree(os.getpid())):
                raise RuntimeError("foreign GPU appeared; certification yields without foreign signals")
            if (root / ("attempt-" + phase) / "interrupt-ready.json").exists():
                interrupted_at = time.monotonic()
                raise ControlledStop
            return original()
        pipeline.gpu_state = controlled_state
        try:
            status = pipeline.guarded_command(command, log, root / (phase + ".gpu.json"),
                {**environment(), "G2M_GPU_GUARDED": "1"})
        except ControlledStop:
            release = time.monotonic() - interrupted_at
            status = 75
        finally:
            pipeline.gpu_state = original
    expected = 75 if phase.startswith("interrupt") else 0
    if status != expected:
        raise RuntimeError(f"certification child status={status}, expected={expected}; see {log}")
    attempt = root / ("attempt-" + phase)
    summary = json.loads((attempt / "summary.json").read_text())
    if summary["status"] != ("interrupted" if expected == 75 else "complete"):
        raise RuntimeError("child status/summary mismatch")
    return {**summary, "trace": json.loads((attempt / "trace.json").read_text()),
            "supervised": args.device == "cuda", "release_seconds": release}


def certify(args):
    args.root.mkdir(parents=True, exist_ok=True)
    if any(args.root.iterdir()):
        raise ValueError("certification requires a new empty output root")
    # Fixed before controls/resumes; ceilings follow existing MACE CUDA certifications.
    tolerance = (0., 0.) if args.device == "cpu" else (
        (5e-6, 5e-6) if args.precision == "32-true" else (1e-2, 1e-3))
    report = {"status": "running", "device": args.device, "precision": args.precision,
              "case": args.case, "mode": args.mode, "workers": args.workers,
              "compiled": args.compiled,
              "tolerance": tolerance, "children": [], "control_repeat_checked": False,
              "scaler_checked": False}
    write_json(args.root / "report.json", report)
    control = run_child(args, args.root / "control", "full")
    report["children"].append({key: value for key, value in control.items() if key != "trace"})
    pipeline.compiled_extensions()
    import graph2mat
    import torch
    full = torch.load(control["checkpoint"], map_location="cpu", weights_only=False)

    def compare_checkpoints(actual, expected, actual_root, expected_root):
        for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops"):
            compare(actual[key], expected[key], key, tolerance)
        compare(normalize_paths(actual["callbacks"], actual_root),
                normalize_paths(expected["callbacks"], expected_root), "callbacks", tolerance)
        compare(actual["g2m_recovery"], expected["g2m_recovery"], "recovery", (0, 0))
        for key in ("epoch", "global_step"):
            compare(actual[key], expected[key], key, (0, 0))
        precision_keys = set(expected) - {"epoch", "global_step", "pytorch-lightning_version", "state_dict", "loops",
            "callbacks", "optimizer_states", "lr_schedulers", "hyper_parameters", "hparams_name", "basis_table",
            "version", "training_stages", "active_training_stage", "provenance", "g2m_recovery",
            "datamodule_hyper_parameters", "datamodule_hparams_name"}
        for key in precision_keys:
            compare(actual[key], expected[key], key, (0, 0))
        if args.precision == "16-mixed":
            assert any("MixedPrecision" in key or "amp_scaling" in key for key in precision_keys), "scaler missing"
            report["scaler_checked"] = True

    if args.device == "cuda":
        repeated = run_child(args, args.root / "repeat", "full")
        report["children"].append({key: value for key, value in repeated.items() if key != "trace"})
        repeat_state = torch.load(repeated["checkpoint"], map_location="cpu", weights_only=False)
        compare_checkpoints(repeat_state, full, args.root / "repeat", args.root / "control")
        report["control_repeat_checked"] = True
    admitted, discarded = [], []
    for phase in ("interrupt1", "interrupt2", "finish"):
        result = run_child(args, args.root / "resumed", phase)
        report["children"].append({key: value for key, value in result.items() if key != "trace"})
        admitted.extend(row for row in result["trace"] if row["step"] <= result["global_step"])
        discarded.extend(row for row in result["trace"] if row["step"] > result["global_step"])
    write_json(args.root / "admitted_trace.json", {"control": control["trace"], "resumed": admitted})
    write_json(args.root / "discarded_trace.json", discarded)
    resumed = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    report.update(control_checkpoint=control["checkpoint"], resumed_checkpoint=result["checkpoint"],
        next_epoch=result["next_epoch"], admitted_samples=sum(len(row["sample_ids"]) for row in admitted),
        repeated_samples=sum(len(row["sample_ids"]) for row in discarded),
        parameter_sha256_control=control["parameter_sha256"], parameter_sha256_resumed=result["parameter_sha256"],
        best_tensor_sha256_control=control["best_tensor_sha256"], best_tensor_sha256_resumed=result["best_tensor_sha256"])
    try:
        compare_checkpoints(resumed, full, args.root / "resumed", args.root / "control")
        assert len(admitted) == len(control["trace"])
        for value, expected in zip(admitted, control["trace"]):
            assert value["sample_ids"] == expected["sample_ids"] and value["step"] == expected["step"]
            compare(torch.tensor(value["loss"], dtype=torch.float64),
                    torch.tensor(expected["loss"], dtype=torch.float64), "loss", tolerance)
    except AssertionError as error:
        report.update(status="failed", failure_stage="comparison", error=str(error))
        write_json(args.root / "report.json", report)
        print(str(error), file=sys.stderr)
        return 1
    report["status"] = "passed"
    write_json(args.root / "report.json", report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true")
    parser.add_argument("--case", choices=("toy", "mace"), default="mace")
    parser.add_argument("--mode", choices=("fixed", "cost"), default="fixed")
    parser.add_argument("--workers", type=int, choices=(0, 4), default=0)
    parser.add_argument("--phase", choices=("full", "interrupt1", "interrupt2", "finish"))
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--negative-no-rng", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    args = parser.parse_args()
    args.root = args.root.absolute()
    raise SystemExit(child(args) if args.phase is not None else certify(args))


if __name__ == "__main__":
    main()
