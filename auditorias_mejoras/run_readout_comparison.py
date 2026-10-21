"""Approved w90/6x6 campaign: frozen recipes, idle-GPU queue, no shared-code edits."""

import argparse
import copy
import csv
import fcntl
import glob
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import yaml


REPO = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get("MDVS_ROOT", REPO.parent / "MD_vs_AtomicDisplacement"))
SCRIPTS = LEGACY / "Comparison/scripts"
PREVIOUS = LEGACY / "Comparison/results/dataset_design_curves_v1"
VARIANTS = {"baseline": "E3nnLinearEdgeBlock", "low_rank": "E3nnLowRankEdgeBlock",
            "sk_residual": "E3nnSlaterKosterEdgeBlock"}
RECIPES = {"w90": "random_cartesian__3D__k2__r0.120__res64__scaled_seed0__N64__ts{seed}__g2mv2_fp32poly6_lin_kl_gaugeh_delta",
           "6x6": "sobol_sparse__1D_in__R0.12__d64__N8__ts{seed}__g2mv2_fp32_lin_steps16k_gaugeh_kl_delta_compile"}
INPUT_SUFFIXES = {".fdf", ".tshs", ".hsx", ".xml", ".ion", ".npz", ".xv", ".nc", ".struct", ".psf", ".psml"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def verify_files(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError("frozen input/source changed: " + path)


def recipe(original, variant, run, shared, phase="train", accelerator="gpu"):
    cfg = copy.deepcopy(original)
    if cfg["data"].get("basis_files"):
        basis = Path(cfg["data"].get("root_dir", ".")) / cfg["data"]["basis_files"]
        cfg["data"].update(root_dir=str(run), basis_files=os.path.relpath(basis, run),
                           train_runs="train_samples/*/RUN.fdf", val_runs="validation_samples/*/RUN.fdf")
    cfg["model"].update(edge_block_readout="graph2mat.bindings.e3nn."+VARIANTS[variant],
        edge_block_readout_kwargs=None if variant == "baseline" else {"rank": 4},
        torch_compile=False, gradient_checkpointing=False)
    cfg["trainer"].update(precision="32-true", accelerator=accelerator, devices=1,
        default_root_dir=str(run), enable_progress_bar=False)
    cfg["trainer"]["logger"] = {"class_path": "CSVLogger", "init_args": {"save_dir": str(run), "name": "metrics"}}
    cfg["trainer"]["callbacks"] = [cb for cb in cfg["trainer"].get("callbacks", [])
        if cb["class_path"] != "readout_comparison_callback.ComparisonCallback"]
    if phase != "train":
        steps = 40 if phase == "probe" else 2
        cfg["trainer"].update(max_steps=steps, max_epochs=steps, limit_train_batches=1,
            check_val_every_n_epoch=1, num_sanity_val_steps=0)
        if phase == "probe":
            cfg["trainer"].update(limit_val_batches=0, enable_checkpointing=False)
            cfg["trainer"]["callbacks"] = []
            cfg["data"]["train_runs"] = ("probe_samples/*/RUN.fdf" if "root_dir" in cfg["data"]
                                         else str(run / "probe_samples/*/RUN.fdf"))
        else:
            cfg["trainer"].update(limit_val_batches=1)
            cfg["trainer"]["callbacks"] = [{"class_path": "ModelCheckpoint",
                "init_args": {"save_last": True, "monitor": "val_loss", "mode": "min", "save_top_k": 1}}]
    cfg["trainer"]["callbacks"].append({"class_path": "readout_comparison_callback.ComparisonCallback",
        "init_args": {"shared_path": str(shared), "output_path": str(run/"telemetry.json"),
        "seed": cfg["seed_everything"], "warmup": 10 if phase == "probe" else 0,
        "probe": phase == "probe"}})
    return cfg


def append_queue(path, job):
    # Local Linux O_APPEND + one write keeps concurrent appenders' existing bytes intact.
    with Path(path).open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        original = handle.read()
        if original and not original.endswith(b"\n"):
            raise ValueError("incomplete queue tail; refuse to modify")
        existing = [json.loads(line) for line in original.splitlines() if line.strip()]
        matches = [entry for entry in existing if entry["id"] == job["id"]]
        if matches:
            if matches != [job]:
                raise ValueError("queue id collision")
            return False
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            payload = (json.dumps(job, sort_keys=True)+"\n").encode()
            if os.write(fd, payload) != len(payload):
                raise OSError("incomplete queue append")
            os.fsync(fd)
        finally:
            os.close(fd)
    return True


def foreign_pids(text, own):
    pids = set()
    for line in text.strip().splitlines():
        pids.add(int(line.split(",")[0].strip()))
    return pids - own


def process_tree(pid):
    own = {pid}
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            pass
    while True:
        extended = own | {child for child, parent in parents.items() if parent in own}
        if extended == own:
            break
        own = extended
    return own


def gpu_processes(own=None):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                     "--format=csv,noheader,nounits"], text=True)
    return foreign_pids(output, own or set())


def idle_gpu():
    while gpu_processes():
        print("WAIT GPU occupied; no foreign signals", flush=True)
        time.sleep(30)


def legacy_modules():
    import graph2mat
    sys.path.insert(0, str(SCRIPTS))
    import dataset_design_curves_v1 as c
    import g2m_delta as delta
    import g2m_ood_eval as ood
    return c, delta, ood


def prepare(root):
    if root.exists():
        raise FileExistsError("campaign output already exists: " + str(root))
    c, delta, ood = legacy_modules()
    import g2m_improvements_benchmark as old
    manifest = {"id": root.name, "root": str(root), "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "hashes": {}, "jobs": [],
        "evaluation": {}, "versions": {name: importlib.metadata.version(name) for name in
            ("torch", "e3nn", "mace-torch", "pytorch-lightning", "numpy", "sisl")}}
    files = set(REPO.glob("src/**/*.py")) | {Path(__file__), REPO/"auditorias_mejoras/readout_comparison_callback.py"}
    final = c.read_json(PREVIOUS/"final_test_manifest.json")
    files.add(PREVIOUS/"final_test_manifest.json")
    for system in RECIPES:
        groups = {"validation": c.dev_samples(system),
                  "test": [c.sample_from_dict(row) for row in final["systems"][system]["samples"]]}
        groups.update({target: ood.target_samples(target) for target in ("registry", "ab_md", "moire")})
        manifest["evaluation"][system] = {key: [c.sample_to_dict(s) for s in samples] for key, samples in groups.items()}
        for samples in groups.values():
            for sample in samples:
                files.update((sample.run_fdf, sample.reference_matrix, delta.h0_path(sample)))
                files.update(p for p in sample.run_fdf.parent.iterdir()
                             if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)
        for seed in (0, 1, 2):
            source = PREVIOUS/"runs"/system/RECIPES[system].format(seed=seed)
            original = yaml.safe_load((source/"config.yaml").read_text())
            files.add(source/"config.yaml")
            bases = sorted(Path(p).resolve() for p in glob.glob(str(source/original["data"]["basis_files"])))
            if not bases:
                raise FileNotFoundError("basis files missing for " + str(source))
            files.update(bases)
            original["data"]["basis_files"] = str(bases[0].parent/"*.ion.xml")
            links = {split: {p.name: str(p.resolve(strict=True)) for p in sorted((source/split).iterdir())}
                     for split in ("train_samples", "validation_samples")}
            expected = 64 if system == "w90" else 8
            if len(links["train_samples"]) != expected:
                raise ValueError("unexpected training size")
            old_job = (old.jobs("fp32poly6_lin_kl_gaugeh_delta", [seed]) if system == "w90"
                       else old.jobs_6x6("fp32_lin_steps16k_gaugeh_kl_delta_compile", [seed]))[0]
            if set(links["train_samples"]) != {r["sample_id"] for r in old_job["train"]}:
                raise ValueError("historical train split changed")
            hashes = [{r["hash"] for r in old_job["train"]}]
            for key in ("validation", "test"):
                hashes.append({c.geometry_record(system, s.run_fdf)["hash"] for s in groups[key]})
            if any(a & b for index, a in enumerate(hashes) for b in hashes[index+1:]):
                raise ValueError("train/validation/test geometry leakage")
            if set(links["validation_samples"]) != {s.sample_id for s in groups["validation"]}:
                raise ValueError("historical validation split changed")
            for paths in links.values():
                for path in paths.values():
                    files.update(p for p in Path(path).iterdir() if p.is_file() and
                                 p.suffix.lower() in INPUT_SUFFIXES)
            shared = root/"shared"/f"{system}-seed{seed}.pt"
            for variant in VARIANTS:
                run = root/"runs"/system/f"{variant}-seed{seed}"
                cfg = recipe(original, variant, run, shared)
                manifest["jobs"].append({"system": system, "seed": seed, "variant": variant,
                    "source": str(source), "run": str(run), "config": cfg, "links": links})
    files.update(Path(m.__file__) for m in tuple(sys.modules.values())
                 if getattr(m, "__file__", None) and str(getattr(m, "__file__")).startswith(str(LEGACY))
                 and Path(m.__file__).suffix == ".py")
    files.add(SCRIPTS/"predict_model_on_dataset.py")
    files.update(LEGACY/path for path in (
        "scripts/torch_serialization_compat/torch_safe_globals.py",
        "scripts/torch_serialization_compat/sitecustomize.py",
        "shared/graph2mat_material_config.py", "shared/material_bundle.py", "shared/material_presets.py"))
    for path in sorted(files):
        manifest["hashes"][str(path)] = sha256(path)
    root.mkdir(parents=True)
    write_json(root/"manifest.json", manifest)
    for job in manifest["jobs"]:
        materialize(job, Path(job["run"]), job["config"])
    return manifest


def materialize(job, run, cfg):
    run.mkdir(parents=True, exist_ok=False)
    for split, links in job["links"].items():
        (run/split).mkdir()
        for name, path in links.items():
            (run/split/name).symlink_to(path, target_is_directory=True)
    if cfg["trainer"]["callbacks"][-1]["init_args"]["probe"]:
        (run/"probe_samples").mkdir()
        for name, path in list(job["links"]["train_samples"].items())[:cfg["data"]["batch_size"]]:
            (run/"probe_samples"/name).symlink_to(path, target_is_directory=True)
    (run/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def child_fit(run, hashes, probe=False):
    idle_gpu()
    verify_files(hashes)
    env = {**os.environ, "PYTHONPATH": str(REPO/"src")+os.pathsep+str(REPO/"auditorias_mejoras"),
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    with (run/"train.log").open("w") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__)), "fit", "--root", str(run),
            *(["--probe"] if probe else [])], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            while child.poll() is None:
                time.sleep(2)
                if child.poll() is not None:
                    break
                if gpu_processes(process_tree(child.pid)):
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    print("YIELD interrupted own attempt; foreign GPU process arrived", flush=True)
                    return False
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
    if child.returncode:
        raise RuntimeError(f"training exit{child.returncode}; see {run/'train.log'}")
    return True


def execute(manifest, job, phase, label):
    root = Path(manifest["root"])
    marker = root/"completed"/(label+".json")
    if marker.exists():
        return json.loads(marker.read_text())
    while True:
        verify_files(manifest["hashes"])
        parent = root/"attempts"/label
        parent.mkdir(parents=True, exist_ok=True)
        run = parent/f"attempt-{len(list(parent.iterdir()))+1:03d}"
        shared = root/"shared"/f"{job['system']}-seed{job['seed']}.pt"
        cfg = recipe(job["config"], job["variant"], run, shared, phase)
        materialize(job, run, cfg)
        print("START", label, run, flush=True)
        started = time.monotonic()
        if not child_fit(run, manifest["hashes"], phase == "probe"):
            continue
        verify_files(manifest["hashes"])
        telemetry = json.loads((run/"telemetry.json").read_text())
        checkpoints = sorted(run.rglob("best*.ckpt"))
        if phase == "train" and len(checkpoints) != 1:
            raise ValueError("expected one best-validation checkpoint")
        if phase == "probe" and (telemetry["steps"] != 40 or len(telemetry["raw_ms"]["step"]) != 30):
            raise ValueError("probe did not execute warmup10+measured30")
        proof = {"run": str(run), "seconds": time.monotonic()-started, "telemetry": telemetry,
                 "checkpoint": str(checkpoints[0]) if checkpoints else None}
        write_json(marker, proof)
        print("DONE", label, flush=True)
        return proof


def evaluate(manifest, job, checkpoint):
    c, delta, ood = legacy_modules()
    root = Path(manifest["root"])/"evaluation"/f"{job['system']}-{job['variant']}-seed{job['seed']}"
    for target, rows in manifest["evaluation"][job["system"]].items():
        out = root/target
        if (out/"per_structure_metrics.csv").exists():
            continue
        verify_files(manifest["hashes"])
        samples = [c.sample_from_dict(row) for row in rows]
        # CPU evaluation leaves the occupied GPU untouched; prediction artifacts are owned.
        predicted = c.predict(Path(checkpoint), samples, out, "cpu")
        for sample in samples:
            h0 = delta.h0_path(sample)
            if not h0.is_file():
                raise FileNotFoundError("cached H0 missing; SIESTA is never launched")
            matrix = predicted/sample.sample_id/"ML_prediction.HSX"
            delta._combine(delta._read(matrix), delta._read(h0), 1.0, matrix)
        metrics = []
        for sample in samples:
            row = c.structure_metrics(sample, predicted)
            if target in ("registry", "ab_md", "moire"):
                row.update(ood.metrics(sample, predicted))
            else:
                row.update(c.band_dos_metrics(sample, predicted/sample.sample_id/"ML_prediction.HSX", job["system"]))
            metrics.append(row)
        tmp = out/"per_structure_metrics.csv.tmp"
        verify_files(manifest["hashes"])
        c.write_csv(tmp, metrics)
        tmp.replace(out/"per_structure_metrics.csv")


def run_campaign(root):
    with (root/"campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((root/"manifest.json").read_text())
        verify_files(manifest["hashes"])
        jobs = manifest["jobs"]
        for job in jobs:
            if job["seed"] == 0:
                execute(manifest, job, "smoke", f"smoke-{job['system']}-{job['variant']}")
        for system in RECIPES:
            selected = [j for j in jobs if j["system"] == system and j["seed"] == 0]
            for index in range(12):
                order = selected[index % 3:]+selected[:index % 3]
                if (index//3) % 2:
                    order.reverse()
                for job in order:
                    execute(manifest, job, "probe", f"probe-{system}-{index:02d}-{job['variant']}")
        summarize(root, include_precision=False)
        for job in jobs:
            execute(manifest, job, "train", f"train-{job['system']}-{job['variant']}-seed{job['seed']}")
        for job in jobs:
            proof = json.loads((root/"completed"/f"train-{job['system']}-{job['variant']}-seed{job['seed']}.json").read_text())
            evaluate(manifest, job, proof["checkpoint"])
        summarize(root)
        print("done campaign", flush=True)


def summarize(root, include_precision=True):
    probes = {}
    for path in (root/"completed").glob("probe-*.json"):
        value = json.loads(path.read_text())["telemetry"]
        _, system, index, variant = path.stem.split("-", 3)
        record = probes.setdefault(system, {}).setdefault(variant, {"rounds": []})
        record["rounds"].append({"round": int(index), **value})
    for variants in probes.values():
        for record in variants.values():
            raw = [statistics.median(r["raw_ms"]["step"]) for r in record["rounds"]]
            median = statistics.median(raw)
            record.update(step_median_ms=median, step_mad_ms=statistics.median(abs(x-median) for x in raw))
        baseline = {r["round"]: r for r in variants["baseline"]["rounds"]}
        for record in variants.values():
            ratios = {key: [] for key in ("forward", "backward", "step", "peak_allocated_bytes", "peak_reserved_bytes")}
            for row in sorted(record["rounds"], key=lambda r: r["round"]):
                reference = baseline[row["round"]]
                if any(row[key] != reference[key] for key in ("batch_hash", "initial_hashes")):
                    raise ValueError("unpaired probe batch or initialization")
                for key in ratios:
                    a, b = ((statistics.median(r["raw_ms"][key]) for r in (row, reference))
                            if key in row["raw_ms"] else (row[key], reference[key]))
                    ratios[key].append(a/b if a is not None and b else None)
            record["paired_ratios_vs_baseline"] = ratios
    write_json(root/"probe_summary.json", probes)
    if not include_precision or not (root/"evaluation").exists():
        return
    import numpy as np
    science = {}
    for system in RECIPES:
        values = {}
        for variant in VARIANTS:
            values[variant] = {}
            for target in ("test", "registry", "ab_md", "moire"):
                by_seed = []
                for seed in (0, 1, 2):
                    path = root/"evaluation"/f"{system}-{variant}-seed{seed}"/target/"per_structure_metrics.csv"
                    if not path.exists():
                        raise ValueError("incomplete evaluation: " + str(path))
                    with path.open() as handle:
                        rows = list(csv.DictReader(handle))
                    if not rows:
                        raise ValueError("empty evaluation")
                    names = [key for key in rows[0] if key.endswith("meV")]
                    means = {key: statistics.mean(float(row[key]) for row in rows) for key in names}
                    if not math.isfinite(means["H_MAE_gauge_meV"]):
                        raise ValueError("nonfinite primary metric")
                    by_seed.append(means)
                values[variant][target] = by_seed
        science[system] = {}
        for variant in ("low_rank", "sk_residual"):
            ratios = [a["H_MAE_gauge_meV"]/b["H_MAE_gauge_meV"]
                      if b["H_MAE_gauge_meV"] > 0 else None
                      for a, b in zip(values[variant]["test"], values["baseline"]["test"])]
            defined = all(r is not None and r > 0 for r in ratios)
            geo = math.exp(statistics.mean(math.log(r) for r in ratios)) if defined else None
            # Preregistered paired bootstrap: 10000 resamples of the three seed pairs.
            rng = np.random.default_rng(20261010)
            ci = np.quantile(np.exp(np.log(ratios)[rng.integers(0, 3, (10000, 3))].mean(axis=1)),
                             [.025, .975]).tolist() if defined else None
            regressions = []
            for target in ("registry", "ab_md", "moire"):
                for metric in ("H_MAE_gauge_meV", "band_rmse_aligned_meV"):
                    for a, b in zip(values[variant][target], values["baseline"][target]):
                        if (metric not in a or metric not in b or not math.isfinite(a[metric])
                                or not math.isfinite(b[metric]) or a[metric] > 1.05*b[metric]):
                            regressions.append(target+":"+metric)
            science[system][variant] = {"primary_ratios_per_seed": ratios,
                "primary_ratio_geomean": geo, "primary_ratio_ci95": ci,
                "ood_regressions": sorted(set(regressions)), "metrics_per_seed": values[variant],
                "baseline_metrics_per_seed": values["baseline"],
                "promotion": bool(defined and geo < 1 and ci[1] < 1 and not regressions)}
        science[system]["sk_vs_low_rank_primary_ratios"] = [
            a["H_MAE_gauge_meV"]/b["H_MAE_gauge_meV"] if b["H_MAE_gauge_meV"] else None
            for a, b in zip(values["sk_residual"]["test"], values["low_rank"]["test"])]
    write_json(root/"precision_summary.json", science)


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["PYTHONPATH"] = str(REPO/"src")+os.pathsep+str(REPO/"auditorias_mejoras")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "enqueue", "fit", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root)
        print(json.dumps({"jobs": len(result["jobs"]), "frozen_files": len(result["hashes"]), "root": str(root)}))
    elif args.action == "enqueue":
        manifest = json.loads((root/"manifest.json").read_text())
        verify_files(manifest["hashes"])
        job = {"id": "readout:"+root.name, "script": str(Path(__file__).resolve()),
               "args": ["run", "--root", str(root)], "vram_mib": 22000}
        print("enqueued" if append_queue(PREVIOUS/"g2m_improvements/autorun/queue.jsonl", job) else "already queued")
    elif args.action == "fit":
        import graph2mat
        from graph2mat.tools.lightning.cli import OrbitalMatrixCLI
        from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
        from readout_comparison_callback import FrozenProbeDataModule, PairedDataModule
        OrbitalMatrixCLI(LitMACEMatrixModel, FrozenProbeDataModule if args.probe else PairedDataModule,
            args=["fit", "-c", str(root/"config.yaml")])
    elif args.action == "run":
        run_campaign(root)
    else:
        summarize(root)


if __name__ == "__main__":
    main()
