"""Three approved 4.7 fits against frozen fixed baselines; no default changes."""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

import run_h2oclu_batching_science as science
from h2oclu_batching_metrics import owned_json as write_json
from h2oclu_nullspace_callback import projection_columns
from run_bilayer_edge_updates import environment, live_dependencies
from run_readout_comparison import append_queue, sha256, verify_files

recovery = science.recovery
BASELINE = recovery.REPO/"auditorias_mejoras/experimentos_batching_h2oclu_20261016_science_v1"
BASELINE_SHA = "da7a0e3a1268cb9504f2ede4dd0b3cfe1df263294cb5ff21324949e26ac0ddca"
CALLBACK = "h2oclu_nullspace_callback.NullspaceTrainingCheckpoint"


def read_baselines(root):
    root = recovery.own_output(root)
    if sha256(root/"manifest.json") != BASELINE_SHA:
        raise ValueError("approved baseline manifest changed")
    manifest = science.read_campaign(root)
    files, records = dict(manifest["files"]), []
    files[str(root/"manifest.json")] = BASELINE_SHA
    for job in manifest["jobs"]:
        if job["arm"] != "fixed":
            continue
        native, cfg = recovery.read_job(Path(job["config"]), Path(job["root"]))
        result = science.validated_result(manifest, job)
        store = recovery.recovery_api().RecoveryStore(Path(job["root"]), native["fingerprint"])
        receipt = store.latest()
        for name in receipt["files"]:
            path = Path(receipt["generation"])/name
            files[str(path)] = sha256(path)
        paths = [Path(job["root"])/name for name in ("result.json", "evaluation_receipt.json")]
        paths += [Path(result["evaluation"]["checkpoint"])]
        for row in result["evaluation"]["rows"]:
            if sha256(row["matrix"]) != row["matrix_sha256"]:
                raise ValueError("baseline physical matrix changed")
            paths.append(Path(row["matrix"]))
        for path in paths:
            files[str(path)] = sha256(path)
        records.append(dict(seed=job["seed"], root=job["root"], native=native, config=cfg,
                            result=result, result_path=str(Path(job["root"])/"result.json")))
    if sorted(r["seed"] for r in records) != [0, 1, 2]:
        raise ValueError("three fixed baseline seeds required")
    return dict(rows=manifest["rows"], files=files, versions=manifest["versions"],
                records=sorted(records, key=lambda r: r["seed"]), source_root=str(root), manifest_sha256=BASELINE_SHA)


def check_recipe(cfg, seed):
    if (cfg["seed_everything"] != seed or cfg["data"]["batch_size"] != 8
            or cfg["data"].get("max_batch_cost") is not None or cfg["trainer"]["max_epochs"] != 2000
            or cfg["trainer"]["check_val_every_n_epoch"] != 5
            or cfg["model"]["loss"] != "graph2mat.metrics.gauge_invariant_mse"
            or cfg["model"].get("loss_kwargs") or cfg["model"].get("training_stages")
            or cfg["lr_scheduler"] != dict(class_path="torch.optim.lr_scheduler.CosineAnnealingLR",
                                          init_args=dict(T_max=2000, eta_min=1e-5))):
        raise ValueError("baseline differs from approved fixed recipe")


def overlay(root, baseline):
    """Only reference-S derived columns; all original matrix/geometry bytes stay put."""
    import sisl
    rows = [r for r in baseline["rows"] if r["split"] in ("train", "validation")]
    if len(rows) != 36 or len({r["sample_id"] for r in rows}) != 36:
        raise ValueError("32train/4validation unique inputs required")
    overlaps = {}
    for r in rows:
        reference = sisl.get_sile(r["reference_tshs_path"]).read_hamiltonian()
        S = reference.tocsr(reference.S_idx).toarray()
        overlaps[r["sample_id"]] = S
    padded = max(int((np.linalg.eigvalsh((S+S.T)/2) < 1e-2).sum()) for S in overlaps.values())
    columns = {key: projection_columns(S, padded) for key, S in overlaps.items()}
    prepared = []
    for r in rows:
        folder = recovery.own_output(root/"data"/r["sample_id"])
        if root not in folder.parents:
            raise ValueError("sample path escaped campaign")
        folder.mkdir(parents=True)
        label = Path(r["delta"])
        source = label.with_name("RUN.fdf")
        # Both FDF and direct-matrix historical inputs are supported without rewriting.
        inputs = {label.name: label, "RUN.fdf": source}
        for suffix in (".ion.xml", ".psf"):
            for asset in Path(r["reference_tshs_path"]).parent.glob("*"+suffix):
                inputs[asset.name] = asset
        for name, path in inputs.items():
            (folder/name).symlink_to(path.resolve(strict=True))
        path = folder/"spectral_null.npz"
        np.savez(path, C=columns[r["sample_id"]])
        prepared.append(dict(sample_id=r["sample_id"], split=r["split"], folder=str(folder),
            input=str(folder/label.name), fdf=str(folder/"RUN.fdf"), projection=str(path),
            columns=int((np.linalg.norm(columns[r["sample_id"]], axis=0) > 0).sum()), padded_columns=padded))
    return prepared


def overlay_splits(record, rows):
    original = recovery.input_resolution(record["config"])
    by_target = {}
    for p in rows:
        by_target[Path(p["input"]).resolve()] = p
        by_target[Path(p["fdf"]).resolve()] = p
    splits = {}
    for argument, split in (("train_runs", "train"), ("val_runs", "val")):
        paths = []
        for item in original[argument]:
            target = Path(item["target"])
            if target not in by_target:
                raise ValueError("overlay input resolution changed")
            p = by_target[target]
            paths.append(p["fdf"] if target.suffix == ".fdf" else p["input"])
        splits[split] = paths
    if [by_target[Path(x["target"])]["sample_id"] for x in original["train_runs"]] != record["native"]["sample_ids"]:
        raise ValueError("baseline sample order differs from IDs")
    return splits


def candidate_config(record, root, rows):
    cfg = copy.deepcopy(record["config"])
    runs = root.parent/"data/runs.json"
    if json.loads(runs.read_text()) != overlay_splits(record, rows):
        raise ValueError("overlay ordered split paths changed")
    cfg["data"].update(train_runs=None, val_runs=None, runs_json=str(runs))
    cfg["data"]["spectral_window"] = "spectral_null.npz"
    cfg["model"]["loss_kwargs"] = {"spectral_weight": 1e-4}
    callbacks = cfg["trainer"].get("callbacks", [])
    if len(callbacks) != 1 or callbacks[0]["init_args"].get("monitor") != "val_loss":
        raise ValueError("one baseline best-val_loss callback required")
    callbacks[0]["class_path"] = CALLBACK
    callbacks[0]["init_args"]["spectral_weight"] = 1e-4
    cfg["trainer"]["default_root_dir"] = str(root)
    return cfg


def check_overlay(old_cfg, cfg):
    import torch
    api = recovery.recovery_api()
    before = api.RecoveryDataModule(**old_cfg["data"])
    after = api.RecoveryDataModule(**cfg["data"])
    checked = 0
    try:
        before.setup("fit")
        after.setup("fit")
        for split in ("train_dataset", "val_dataset"):
            old, new = getattr(before, split), getattr(after, split)
            if len(old) != len(new):
                raise ValueError("overlay sample coverage changed")
            for a, b in zip(old, new):
                x = {k: v for k, v in a.to_dict().items() if isinstance(v, torch.Tensor)}
                y = {k: v for k, v in b.to_dict().items() if isinstance(v, torch.Tensor) and k != "spectral_C"}
                if x.keys() != y.keys() or any(not torch.equal(x[k], y[k]) for k in x):
                    raise ValueError("overlay changed native graph/labels/tensors")
                checked += 1
    finally:
        before.teardown(None)
        after.teardown(None)
    return dict(samples_checked=checked, comparison="all native tensor fields except added spectral_C, torch.equal")


def prepare(root, baseline_root=BASELINE, approved=False):
    if not approved:
        raise ValueError("explicit authorization of three 4.7 fits required")
    root = recovery.own_output(root)
    if root.exists():
        raise ValueError("new exclusive experiment directory required")
    baseline = read_baselines(baseline_root)
    verify_files(baseline["files"])
    recovery.recovery_api()
    import torch
    from readout_comparison_callback import tensor_hash
    for record in baseline["records"]:
        check_recipe(record["config"], record["seed"])
        native = record["native"]
        if tensor_hash(torch.load(native["initial_state"], map_location="cpu", weights_only=True).items()) != native["initial_sha256"]:
            raise ValueError("baseline initialization changed")
    root.mkdir(parents=True)
    rows = overlay(root, baseline)
    write_json(root/"data/runs.json", overlay_splits(baseline["records"][0], rows))
    configs = [candidate_config(r, root/f"null_s{r['seed']}", rows) for r in baseline["records"]]
    equivalent = check_overlay(baseline["records"][0]["config"], configs[0])
    files = {**baseline["files"], **{str(p): sha256(p) for p in (
        root/"data/runs.json",
        Path(__file__).resolve(), Path(__file__).with_name("h2oclu_nullspace_callback.py").resolve(),
        Path(science.__file__).resolve(), Path(__file__).with_name("h2oclu_batching_metrics.py").resolve())}}
    for row in rows:
        for path in Path(row["folder"]).iterdir():
            if path.is_file():
                files[str(path)] = sha256(path)
    baselines, jobs = [], []
    for record, cfg in zip(baseline["records"], configs):
        seed, native = record["seed"], record["native"]
        files[record["result_path"]] = sha256(record["result_path"])
        job = recovery.make_job(root/f"null_s{seed}", cfg, files, Path(native["initial_state"]),
                                native["initial_sha256"], native["sample_ids"], baseline["versions"])
        path = Path(job["root"])/"job.json"
        state = json.loads(path.read_text())
        state.update(technical_pilot=False, scientific_authorized=True, approved_samples=64000,
                     authorization="Apruebo plan 2026-10-17-h2oclu-nullspace-validation")
        write_json(path, state)
        jobs.append(dict(job, key=f"null_s{seed}", arm="null", seed=seed))
        baselines.append(dict(seed=seed, root=record["root"], result=record["result_path"],
                              initial_sha256=native["initial_sha256"], config=record["config"]))
    for job in jobs:
        for path in (Path(job["config"]), Path(job["root"])/"job.json"):
            files[str(path)] = sha256(path)
    result = dict(jobs=jobs, baselines=baselines, files=files, versions=baseline["versions"], rows=baseline["rows"],
        projection_rows=rows, overlay_equivalence=equivalent, approved_fit_count=3,
        scientific_authorized=True, weight=1e-4, selection="basic gauge val_loss/min, every5, spectral train only",
        baseline_root=baseline["source_root"], baseline_manifest_sha256=baseline["manifest_sha256"],
        limitations="3 seeds; previously observed OOD and incomplete lineage; exploratory; no default promotion")
    write_json(root/"manifest.json", result)
    read_campaign(root)
    return result


def read_campaign(root):
    root = recovery.own_output(root)
    m = json.loads((root/"manifest.json").read_text())
    receipt_path = recovery.own_output(root/"queue_receipt.json")
    if receipt_path.exists() and json.loads(receipt_path.read_text())["manifest_sha256"] != sha256(root/"manifest.json"):
        raise ValueError("submitted manifest changed")
    if (not m["scientific_authorized"] or m["approved_fit_count"] != 3 or m["weight"] != 1e-4
            or [j["key"] for j in m["jobs"]] != ["null_s0", "null_s1", "null_s2"]
            or [b["seed"] for b in m["baselines"]] != [0, 1, 2]):
        raise ValueError("campaign differs from three approved candidates")
    verify_files(m["files"])
    source_path = Path(m["baseline_root"])/"manifest.json"
    if (sha256(source_path) != m["baseline_manifest_sha256"]
            or json.loads(source_path.read_text())["rows"] != m["rows"]):
        raise ValueError("baseline manifest/rows metadata changed")
    for name, version in m["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen software version changed")
    for job, baseline in zip(m["jobs"], m["baselines"]):
        if Path(job["root"]) != root/job["key"] or job["seed"] != baseline["seed"]:
            raise ValueError("job/root/baseline identity changed")
        native, cfg = recovery.read_job(Path(job["config"]), Path(job["root"]))
        expected = candidate_config(dict(config=baseline["config"], native={"sample_ids": native["sample_ids"]}),
                                    Path(job["root"]), m["projection_rows"])
        if (native["max_epochs"] != 2000 or len(native["sample_ids"]) != 32
                or native["initial_sha256"] != baseline["initial_sha256"] or native["technical_pilot"]
                or not native["scientific_authorized"] or cfg["model"]["loss_kwargs"] != {"spectral_weight": 1e-4}
                or cfg["data"]["spectral_window"] != "spectral_null.npz"
                or cfg["trainer"]["callbacks"][0]["class_path"] != CALLBACK or cfg != expected):
            raise ValueError("candidate treatment/budget changed")
    return m


def enqueue(root, queue=science.QUEUE):
    root = recovery.own_output(root)
    m = read_campaign(root)
    queue, path = Path(queue), recovery.own_output(root/"queue_receipt.json")
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt["queue"] != str(queue) or receipt["manifest_sha256"] != sha256(root/"manifest.json"):
            raise ValueError("queue receipt provenance mismatch")
    else:
        previous = queue.read_bytes()
        entries = [json.loads(line) for line in previous.splitlines() if line.strip()]
        dependencies = live_dependencies(entries, json.loads(queue.with_name("state.json").read_text()))
        jobs = []
        for job in m["jobs"]:
            identifier = f"{root.name}:{job['key']}"
            jobs.append(dict(id=identifier, script=str(Path(__file__).resolve()),
                args=["run", "--root", str(root), "--key", job["key"]], after=dependencies,
                vram_mib=4096, min_free_vram_frac=.30))
            dependencies = [identifier]
        receipt = dict(queue=str(queue), old_bytes=len(previous), old_sha256=hashlib.sha256(previous).hexdigest(),
            manifest_sha256=sha256(root/"manifest.json"), jobs=jobs, complete=False, appended_ids=[])
        write_json(path, receipt)
    if len(receipt["jobs"]) != 3:
        raise ValueError("three queued jobs required")
    for item, job in zip(receipt["jobs"], m["jobs"]):
        if (item["id"] != f"{root.name}:{job['key']}" or item["script"] != str(Path(__file__).resolve())
                or item["args"] != ["run", "--root", str(root), "--key", job["key"]]
                or item["vram_mib"] != 4096 or item["min_free_vram_frac"] != .30):
            raise ValueError("queued candidate identity/command changed")
    for job in receipt["jobs"]:
        if hashlib.sha256(queue.read_bytes()[:receipt["old_bytes"]]).hexdigest() != receipt["old_sha256"]:
            raise ValueError("previous queue prefix changed")
        append_queue(queue, job)
        if job["id"] not in receipt["appended_ids"]:
            receipt["appended_ids"].append(job["id"])
        write_json(path, receipt)
    receipt["complete"] = True
    write_json(path, receipt)
    return receipt


def checked_result(m, job):
    r = science.validated_result(m, job)
    for row in r["evaluation"]["rows"]:
        matrix = recovery.own_output(row["matrix"])
        if Path(job["root"])/"evaluation" not in matrix.parents or sha256(matrix) != row["matrix_sha256"]:
            raise ValueError("candidate matrix changed/outside evaluation")
    return r


def evaluation_paths(directory):
    return [recovery.own_output(directory/name)
            for name in ("result.json", "evaluation_receipt.json", "evaluation")]


def evaluate(root, key):
    from h2oclu_batching_metrics import evaluate_checkpoint
    m = read_campaign(root)
    job = next(j for j in m["jobs"] if j["key"] == key)
    directory = Path(job["root"])
    result_path, proof_path, output = evaluation_paths(directory)
    if result_path.exists() and proof_path.exists():
        return checked_result(m, job)
    if any(p.exists() for p in (result_path, proof_path, output)):
        raise ValueError("incomplete existing evaluation retained; no overwrite authorized")
    native, _ = recovery.read_job(Path(job["config"]), directory)
    store = recovery.recovery_api().RecoveryStore(directory, native["fingerprint"])
    receipt = store.latest()
    if receipt is None or receipt["next_epoch"] != 2000 or receipt["samples_committed"] != 64000:
        raise ValueError("complete training budget required before scientific evaluation")
    checkpoint = store.restore_best(receipt)
    fit = json.loads((directory/"fit.json").read_text())
    if fit["checkpoint"] != str(checkpoint) or fit["recovery_checkpoint"] != receipt["checkpoint"]:
        raise ValueError("fit/recovery selection mismatch")
    rows = [r for r in m["rows"] if r["split"] in ("validation", "test")]
    output.mkdir()
    evaluation = evaluate_checkpoint(checkpoint, rows, output)
    verify_files(m["files"])
    write_json(result_path, dict(arm="null", seed=job["seed"], training=fit, evaluation=evaluation))
    write_json(proof_path, dict(checkpoint=str(checkpoint),
        checkpoint_sha256=receipt["best"]["sha256"], result_sha256=sha256(result_path)))
    return checked_result(m, job)


def summarize(root):
    m = read_campaign(root)
    candidates = {j["seed"]: checked_result(m, j) for j in m["jobs"] if (Path(j["root"])/"result.json").exists()}
    baselines = {b["seed"]: json.loads(Path(b["result"]).read_text()) for b in m["baselines"]}
    summary = dict(completed=len(candidates), approved=3, limitations=m["limitations"], metrics={})
    for split in ("validation", "test"):
        for metric in science.PRIMARY:
            seeds = sorted(candidates)
            fixed, null, paired_seeds, excluded = [], [], [], []
            for seed in seeds:
                pair, failures = [], []
                for arm, result in (("fixed", baselines[seed]), ("null", candidates[seed])):
                    rows = [r for r in result["evaluation"]["rows"] if r["split"] == split]
                    values = [r[metric] if metric == "H_MAE_gauge_meV" or r["spectral_valid"] else None for r in rows]
                    for row, value in zip(rows, values):
                        if value is None or not np.isfinite(value):
                            failures.append(dict(arm=arm, sample_id=row["sample_id"],
                                reason=row.get("spectral_failure", "missing/nonfinite metric")))
                    pair.append(float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None)
                if None not in pair:
                    fixed.append(pair[0]); null.append(pair[1])
                    paired_seeds.append(seed)
                else:
                    excluded.append(dict(seed=seed, failures=failures))
            record = dict(paired_count=len(fixed), paired_seeds=paired_seeds, excluded_pairs=excluded,
                          fixed_per_seed=fixed, null_per_seed=null)
            if fixed:
                record.update(fixed_mean=float(np.mean(fixed)), null_mean=float(np.mean(null)),
                    ratio_of_means=float(np.mean(null)/np.mean(fixed)) if np.mean(fixed) > 0 else None,
                    paired_ratios=[b/a if a > 0 else None for a, b in zip(fixed, null)])
            if len(fixed) == 3 and all(a > 0 for a in fixed):
                ratios = [np.mean([null[i] for i in indices])/np.mean([fixed[i] for i in indices])
                          for indices in itertools.product(range(3), repeat=3)]
                record["paired_bootstrap_ratio_ci95"] = np.quantile(ratios, [.025, .975]).tolist()
            summary["metrics"][split+"/"+metric] = record
    write_json(Path(root)/"summary.json", summary)
    return summary


def run(root, key):
    m = read_campaign(root)
    job = next(j for j in m["jobs"] if j["key"] == key)
    if any(p.exists() for p in evaluation_paths(Path(job["root"]))):
        evaluate(root, key)
        summarize(root)
        print("DONE", key, "verified existing evaluation", flush=True)
        return
    fit = recovery.guarded_fit(Path(job["config"]), Path(job["root"]))
    print("FIT_COMPLETE", key, fit["samples_committed"], flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "eval", "--root", str(root), "--key", key],
                   env=environment(cuda=""), check=True)
    summarize(root)
    print("DONE", key, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "enqueue", "run", "eval", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--approved-three", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, approved=args.approved_three)
    elif args.action == "enqueue":
        enqueue(args.root)
    elif args.action == "run":
        run(args.root, args.key)
    elif args.action == "eval":
        evaluate(args.root, args.key)
    else:
        summarize(args.root)
