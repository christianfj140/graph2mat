"""Six approved 7.2 runs, frozen historical recipes; no reference training or defaults.

prepare -> pilot (both materials) -> enqueue; existing autorun executes run.
All new outputs stay in Graph2Mat. External datasets/checkpoints are read-only.
"""
import argparse
import copy
import glob
import inspect
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

import bench_pipeline_gpu as pipeline
from run_readout_comparison import (LEGACY, REPO, SCRIPTS, PREVIOUS, append_queue,
                                   gpu_processes, process_tree, sha256, verify_files, write_json)

SYSTEMS = {
    "tbg": ("bilayer", "n474__ts{seed}__g2mv2_fp32_lin_gaugeh_delta", 474, 60),
    "grhbn_bl": ("grhbn_bl", "grhbn_bl75__ts{seed}__g2mv2_fp32_lin_gaugeh_kl_delta_u16k", 66, 9),
}
QUEUE = PREVIOUS / "g2m_improvements/autorun/queue.jsonl"


def owned(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError("output must belong to campaign: " + str(path))
    return path


def recipe(original, effective_model, source, output):
    cfg = copy.deepcopy(original)
    cfg["model"] = {**effective_model, **cfg["model"], "edge_update_layers": 1}
    # AtomicTableWithEdges expects a relative basis glob under root_dir.
    cfg["data"]["root_dir"] = str(source)
    cfg["trainer"]["default_root_dir"] = str(output)
    cfg["trainer"]["enable_progress_bar"] = False
    cfg["trainer"]["logger"]["init_args"].update(save_dir=str(output / "logs"), name="edge1")
    return cfg


def manifest_files(manifest):
    files = set()
    for row in manifest["rows"]:
        for name, path in row.get("artifact_paths", {}).items():
            if not path:
                continue
            expected = row.get("artifact_sha256", {}).get(name)
            path = Path(path)
            if not path.is_absolute():
                path = LEGACY / path
            if not expected or not path.is_file() or sha256(path) != expected:
                raise ValueError("manifest artifact hash missing/mismatch: " + str(path))
            files.add(path.resolve())
    return files


def live_dependencies(jobs, state):
    done, failed = set(state["done"]), set(state["failed"])
    pending = {j["id"]: j for j in jobs if j["id"] not in done | failed}
    reachable = set(done)
    while True:
        more = {key for key, job in pending.items() if set(job.get("after", [])) <= reachable}
        if more <= reachable:
            return [j["id"] for j in jobs if j["id"] in reachable - done]
        reachable |= more


def paired_summary(rows):
    summary = {}
    for material in SYSTEMS:
        group = sorted((r for r in rows if r["material"] == material), key=lambda r: r["seed"])
        if not group:
            continue
        seeds = [r["seed"] for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seed")
        result = {"seeds": seeds, "status": "complete" if seeds == [0, 1, 2] else "partial"}
        for population in ("id", "ood"):
            for metric in group[0]["baseline"][population]:
                off = [r["baseline"][population][metric] for r in group]
                on = [r["candidate"][population][metric] for r in group]
                if not all(math.isfinite(v) and v > 0 for v in off + on):
                    raise ValueError("nonfinite/nonpositive primary metric")
                ratios = [b / a for a, b in zip(off, on)]
                result[population + "/" + metric] = {
                    "baseline_mean": statistics.mean(off), "candidate_mean": statistics.mean(on),
                    "ratio_of_means": statistics.mean(on) / statistics.mean(off),
                    "paired_ratios": ratios, "paired_min": min(ratios), "paired_max": max(ratios)}
        summary[material] = result
    return summary


def modules():
    pipeline.compiled_extensions()
    import graph2mat  # Must precede e3nn imports.
    assert Path(graph2mat.__file__).resolve() == REPO / "src/graph2mat/__init__.py"
    sys.path.insert(0, str(SCRIPTS))
    import g2m_bilayer_readouts  # Registers historical uncompiled hBN recipe.
    import g2m_delta as delta
    import g2m_ood_eval as ood
    return g2m_bilayer_readouts.bilayer, delta, ood


def check_progress(path, started, now, pilot):
    last = path.stat().st_mtime if path.exists() else started
    if now - last > (300 if pilot else 900):
        raise RuntimeError("fit has no progress; aborting owned group")


def prepare(root):
    if root.exists() or root.parent != REPO / "auditorias_mejoras" or not root.name.startswith("experimentos_edge_updates_bilayer_"):
        raise ValueError("new exclusive Graph2Mat campaign directory required")
    b, delta, ood = modules()
    import torch
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    accepted = set(inspect.signature(LitMACEMatrixModel.__init__).parameters) - {
        "self", "root_dir", "basis_files", "basis_table", "no_basis", "symmetric_matrix",
        "n_matrix_components", "initial_node_feats"}
    files = set((REPO / "src").rglob("*.py")) | {Path(__file__),
        REPO / "auditorias_mejoras/bench_pipeline_gpu.py", REPO / "auditorias_mejoras/run_readout_comparison.py"}
    files |= set(SCRIPTS.glob("*.py")) | set((LEGACY / "shared").glob("*.py"))
    files |= set((LEGACY / "scripts/torch_serialization_compat").glob("*.py"))
    files.update(Path(sys.modules[name].__file__) for name in (
        "graph2mat.core.data._sparse", "graph2mat.core.modules._labels_resort"))
    records, configurations = [], {}
    for material, (folder, template, n_train, n_val) in SYSTEMS.items():
        b.configure(material)
        dataset = json.loads(b.DATASET.read_text())
        selected = {**dataset, "rows": [r for r in dataset["rows"] if r["split"] in ("train", "validation")]}
        files |= manifest_files(selected)
        files |= manifest_files(json.loads(ood.TARGETS[b.TARGET].read_text()))
        files.update((b.DATASET, ood.TARGETS[b.TARGET]))
        train, val = b.samples("train"), b.samples("validation")
        target = ood.target_samples(b.TARGET)
        if (len(train), len(val)) != (n_train, n_val):
            raise ValueError("historical sample counts changed")
        delta_inputs = {}
        for split, samples in (("train_samples", train), ("validation_samples", val)):
            delta_inputs[split] = {}
            for sample in samples:
                matrix = delta.ROOT / "delta" / delta._key(sample) / sample.reference_matrix.name
                fdf = matrix.parent / "RUN.fdf"
                if not matrix.is_file() or not fdf.is_file():
                    raise FileNotFoundError("existing delta label required: " + str(matrix))
                delta_inputs[split][sample.sample_id] = str(matrix.parent.resolve())
                files.update((matrix, fdf, delta.h0_path(sample)))
        for sample in target:
            files.update((sample.run_fdf, sample.reference_matrix, delta.h0_path(sample)))
        for seed in (0, 1, 2):
            source = PREVIOUS / "runs" / folder / template.format(seed=seed)
            original = yaml.safe_load((source / "config.yaml").read_text())
            result = json.loads((source / "result.json").read_text())
            done = json.loads((source / "train_done.json").read_text())
            checkpoint = Path(done["checkpoint"])
            if checkpoint != Path(result["checkpoint"]) or result["training_seed"] != seed or not result.get("delta"):
                raise ValueError("reference checkpoint/seed/target provenance mismatch")
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            hp = ck["hyper_parameters"]
            if hp.get("edge_update_layers", 0) != 0 or ck["epoch"] >= original["trainer"]["max_epochs"]:
                raise ValueError("reference architecture/selection mismatch")
            # Resolve aliases by identity; historical checkpoint stores public short paths.
            for key, expected in original["model"].items():
                actual = hp[key]
                if actual != expected and not (isinstance(actual, str) and isinstance(expected, str)
                        and pipeline.resolve(actual) is pipeline.resolve(expected)):
                    raise ValueError("reference effective model differs: " + key)
            for split, expected in delta_inputs.items():
                links = {p.name: str(p.resolve(strict=True)) for p in (source / split).iterdir()}
                if links != expected:
                    raise ValueError("historical split/label links changed: " + str(source / split))
            bases = [Path(p).resolve() for p in glob.glob(str(source / original["data"]["basis_files"]))]
            if not bases:
                raise FileNotFoundError("reference basis missing")
            files.update(bases)
            files.update((source / "config.yaml", source / "result.json", source / "train_done.json", checkpoint))
            key = f"{material}_s{seed}"
            output = owned(root, root / key)
            cfg = recipe(original, {k: v for k, v in hp.items() if k in accepted}, source, output)
            configurations[key] = cfg
            records.append({"key": key, "material": material, "seed": seed, "source": str(source),
                "baseline_checkpoint": str(checkpoint), "baseline_train_seconds": done["seconds"],
                "target": b.TARGET, "n_train": n_train, "n_val": n_val,
                "baseline_hparams": hp, "baseline_datamodule_hparams": ck["datamodule_hyper_parameters"]})
    hashes = {str(path.resolve()): sha256(path) for path in sorted(files)}
    root.mkdir()
    for key, cfg in configurations.items():
        output = owned(root, root / key)
        output.mkdir()
        (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        hashes[str(output / "config.yaml")] = sha256(output / "config.yaml")
    write_json(root / "manifest.json", {"jobs": records, "hashes": hashes,
        "approval": "Apruebo 7.2 en ambas bicapas, seis trabajos (recomendado)",
        "scientific_change": "edge_update_layers: 0 -> 1; all saved model settings frozen",
        "selection": "best val_loss; same schedule, no OOD tuning",
        "primary_metrics": ["H_MAE_gauge_meV", "band_rmse_aligned_meV"],
        "limits": "3 seeds exploratory; reused OOD is not independent; no default promotion or speed claim",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "e3nn", "mace-torch", "pytorch-lightning", "numpy", "sisl")},
        "gpu": pipeline.gpu_state(),
        "created": time.time()})


def environment(cuda="0"):
    return {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO / "auditorias_mejoras/probe_site"),
        str(REPO / "src"), str(REPO / "auditorias_mejoras"))), "CUDA_VISIBLE_DEVICES": cuda,
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "PATH": "/usr/bin:/bin"}


class ForeignGPU(RuntimeError):
    pass


def guarded_fit(config, output, pilot=False):
    # Existing guard cleans only its owned group on every exit, including foreign-GPU yield.
    original = pipeline.gpu_state
    def exclusive_state():
        if gpu_processes(process_tree(os.getpid())):
            raise ForeignGPU("yielding to foreign compute; no foreign signals")
        check_progress(attempt / "progress.json", started_wall, time.time(), pilot)
        return original()
    while True:
        while gpu_processes(process_tree(os.getpid())):
            print("WAIT foreign GPU", flush=True)
            time.sleep(30)
        state = original()
        fraction = pipeline.memory_fraction(state["memory.total"], state["memory.free"])
        attempt = owned(output.parent, output / ("attempt_" + str(time.time_ns())))
        attempt.mkdir()
        started_wall = time.time()
        pipeline.gpu_state = exclusive_state
        try:
            command = [sys.executable, str(Path(__file__)), "fit", "--root", str(output),
                       "--config", str(config), "--fraction", str(fraction), "--attempt", str(attempt)]
            if pilot:
                command.append("--pilot")
            start = time.monotonic()
            status = pipeline.guarded_command(command, attempt / "train.log", attempt / "telemetry.json", environment())
            if status:
                raise RuntimeError(f"fit failed (rc={status}); see " + str(attempt / "train.log"))
            info = json.loads((attempt / "fit.json").read_text())
            info["seconds"] = time.monotonic() - start
            info["telemetry"] = str(attempt / "telemetry.json")
            return info
        except ForeignGPU as error:
            write_json(attempt / "yield.json", {"reason": str(error), "time": time.time()})
        finally:
            pipeline.gpu_state = original


def fit(args):
    # Timed all-thread dumps crashed this environment; external watchdog handles stalls.
    pipeline.compiled_extensions()
    import graph2mat
    import torch
    from pytorch_lightning import Callback
    from graph2mat.tools.lightning import OrbitalMatrixCLI, MatrixDataModule, SaveConfigSkipBasisTableCallback
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    torch.cuda.set_per_process_memory_fraction(args.fraction)
    class CheckEdges(Callback):
        def __init__(self):
            self.active, self.steps = False, 0
            self.train_seconds, self.val_seconds = [], []

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.batch_start = time.monotonic()
            if self.steps < 2:
                print("PREFLIGHT batch", self.steps, flush=True)

        def on_after_backward(self, trainer, model):
            self.steps += 1
            if self.steps <= 2:
                print("PREFLIGHT backward", self.steps, flush=True)
            if self.steps > 40:
                return
            edges = [(name, p) for name, p in model.named_parameters() if any(
                marker in name for marker in ("update_node_linears", "update_tp_weights", "update_linears"))]
            if not edges or any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in edges):
                raise RuntimeError("edge update gradients absent/nonfinite")
            self.active |= any(bool(p.grad.abs().max() > 0) for _, p in edges)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            self.train_seconds.append(time.monotonic() - self.batch_start)
            if self.steps % 10 == 0:
                write_json(args.attempt / "progress.json", {"steps": self.steps, "time": time.time()})

        def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
            self.val_start = time.monotonic()

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            self.val_seconds.append(time.monotonic() - self.val_start)
    cfg = yaml.safe_load(args.config.read_text())
    cfg["trainer"]["default_root_dir"] = str(args.attempt)
    cfg["trainer"]["logger"]["init_args"]["save_dir"] = str(args.attempt / "logs")
    if args.pilot:
        cfg["trainer"].update(max_steps=40, max_epochs=40, limit_train_batches=1,
            limit_val_batches=1, num_sanity_val_steps=0, enable_checkpointing=False, callbacks=[])
    checker = CheckEdges()
    cli = OrbitalMatrixCLI(LitMACEMatrixModel, MatrixDataModule, args={"fit": cfg},
        trainer_defaults={"callbacks": [checker]}, save_config_callback=SaveConfigSkipBasisTableCallback)
    if not checker.active:
        raise RuntimeError("edge updates never active")
    value = cli.trainer.callback_metrics.get("val_loss")
    if value is None or not torch.isfinite(value):
        raise RuntimeError("validation loss absent/nonfinite")
    if args.pilot:
        checkpoint = str(args.attempt / "pilot.ckpt")
        cli.trainer.save_checkpoint(checkpoint)
    else:
        checkpoint = cli.trainer.checkpoint_callback.best_model_path
    if not args.pilot and not Path(checkpoint).is_file():
        raise RuntimeError("best val_loss checkpoint missing")
    write_json(args.attempt / "fit.json", {"checkpoint": checkpoint, "val_loss": float(value),
        "steps": checker.steps, "edge_updates_active": checker.active,
        "train_batch_seconds_median": statistics.median(checker.train_seconds[10:]),
        "val_batch_seconds_median": statistics.median(checker.val_seconds[10:]),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})


def evaluate(root, job, checkpoint, label, limit=None):
    b, delta, ood = modules()
    b.configure(job["material"])
    directory = owned(root, root / job["key"] / label)
    directory.mkdir(exist_ok=True)
    cached = directory / "metrics.json"
    if cached.exists():
        return json.loads(cached.read_text())
    groups = {"id": b.samples("validation"), "ood": ood.target_samples(job["target"])}
    seen = {"registry_" + r["sample_id"] for r in json.loads(ood.TARGETS["registry"].read_text())["rows"]
            if r["registry_seen_in_training"]} if job["target"] == "registry" else set()
    results = {}
    for population, samples in groups.items():
        if limit:
            samples = [s for s in samples if population != "ood" or s.sample_id not in seen][:limit]
        output = owned(root, directory / population)
        output.mkdir(parents=True, exist_ok=True)
        manifest = output / "manifest.csv"
        b.s4.write_val_manifest(manifest, samples)
        bootstrap = ("import sys, runpy; import bench_pipeline_gpu as p; p.compiled_extensions(); "
                     "import graph2mat; runpy.run_path(sys.argv.pop(1), run_name='__main__')")
        command = [sys.executable, "-c", bootstrap, str(SCRIPTS / "predict_model_on_dataset.py"),
            "--checkpoint", str(checkpoint), "--train-method", "md", "--test-set", "bilayer_edge_updates",
            "--test-manifest", str(manifest), "--output-dir", str(output),
            "--basis-files", b.s4.DEFAULT_BASIS_GLOB, "--matrix-component-policy", "h_only",
            "--n-matrix-components", "1", "--accelerator", "cpu", "--no-store-in-memory"]
        with (output / "predict.log").open("w") as log:
            subprocess.run(command, cwd=LEGACY, stdout=log, stderr=subprocess.STDOUT,
                           env=environment(cuda=""), check=True)
        predictions = output / "predicted_hamiltonians"
        delta.add_h0(predictions, samples)
        rows = [ood.metrics(sample, predictions) for sample in samples]
        b.c.write_csv(output / "per_structure_metrics.csv", rows)
        if population == "ood":
            rows = [row for row in rows if row["sample_id"] not in seen]
        if not rows:
            raise ValueError("empty evaluation population")
        # Auxiliary intralayer/interlayer metrics retained in per-structure CSVs.
        results[population] = {key: statistics.mean(row[key] for row in rows)
            for key in ("H_MAE_gauge_meV", "band_rmse_aligned_meV")}
        if not all(math.isfinite(v) for v in results[population].values()):
            raise ValueError("nonfinite evaluation")
    write_json(cached, results)
    return results


def run(root, key):
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(manifest["hashes"])
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    output = owned(root, root / key)
    if not (output / "train_done.json").exists():
        info = guarded_fit(output / "config.yaml", output)
        write_json(output / "train_done.json", info)
    info = json.loads((output / "train_done.json").read_text())
    # CPU predictions: no unbudgeted subprocess CUDA allocation through legacy helper.
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, str(Path(__file__)), "eval", "--root", str(root), "--key", key],
                   env=env, check=True)
    verify_files(manifest["hashes"])
    results = [json.loads(p.read_text()) for p in root.glob("*_s*/result.json")]
    write_json(root / "summary.json", paired_summary(results))
    print("DONE", key, info["seconds"], flush=True)


def eval_job(root, key):
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(manifest["hashes"])
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    output = owned(root, root / key)
    info = json.loads((output / "train_done.json").read_text())
    candidate = evaluate(root, job, Path(info["checkpoint"]), "candidate_eval")
    baseline = evaluate(root, job, Path(job["baseline_checkpoint"]), "baseline_eval")
    write_json(output / "result.json", {"material": job["material"], "seed": job["seed"],
        "candidate": candidate, "baseline": baseline, "training": info})


def pilot(root):
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(manifest["hashes"])
    results = {}
    for material in SYSTEMS:
        output = owned(root, root / (material + "_pilot"))
        output.mkdir(exist_ok=True)
        info = guarded_fit(root / (material + "_s0/config.yaml"), output, pilot=True)
        job = next(j for j in manifest["jobs"] if j["key"] == material + "_s0")
        previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            for label, checkpoint in (("pilot_candidate_eval", info["checkpoint"]),
                                      ("pilot_baseline_eval", job["baseline_checkpoint"])):
                evaluate(root, job, Path(checkpoint), label, limit=1)
        finally:
            if previous_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        info["prediction_smoke_passed"] = True
        write_json(output / "passed.json", info)
        results[material] = info
    write_json(root / "pilots.json", results)


def enqueue(root):
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(manifest["hashes"])
    pilots = json.loads((root / "pilots.json").read_text())
    if set(pilots) != set(SYSTEMS) or not all(p["edge_updates_active"] and p["steps"] == 40
            and p["prediction_smoke_passed"] for p in pilots.values()):
        raise ValueError("both 40-step pilots required")
    jobs = [json.loads(line) for line in QUEUE.read_text().splitlines() if line.strip()]
    state = json.loads(QUEUE.with_name("state.json").read_text())
    previous = live_dependencies(jobs, state)
    receipt = {"old_bytes": QUEUE.stat().st_size, "old_sha256": sha256(QUEUE), "jobs": []}
    for seed in (0, 1, 2):
        for material in SYSTEMS:
            key = f"{material}_s{seed}"
            job_id = f"{root.name}:{key}"
            job = {"id": job_id, "script": str(Path(__file__)), "args": ["run", "--root", str(root),
                "--key", key], "vram_mib": math.ceil(pilots[material]["peak_reserved_mib"] + 1024),
                "min_free_vram_frac": .30, "after": previous}
            append_queue(QUEUE, job)
            receipt["jobs"].append(job)
            write_json(root / "queue_receipt.json", receipt)
            previous = [job_id]
    prefix = QUEUE.read_bytes()[:receipt["old_bytes"]]
    import hashlib
    if hashlib.sha256(prefix).hexdigest() != receipt["old_sha256"]:
        raise RuntimeError("queue prior bytes changed; investigate external writer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "pilot", "enqueue", "run", "fit", "eval"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "fit":
        fit(args)
    elif args.action == "run":
        run(root, args.key)
    elif args.action == "eval":
        eval_job(root, args.key)
    else:
        {"prepare": prepare, "pilot": pilot, "enqueue": enqueue}[args.action](root)


if __name__ == "__main__":
    main()
