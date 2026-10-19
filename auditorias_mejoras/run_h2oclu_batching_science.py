"""Exactly six approved h2oclu fits; certified recovery source remains untouched."""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_cost_batching_comparison as recovery
from h2oclu_batching_metrics import owned_json
from run_bilayer_edge_updates import environment, live_dependencies
from run_readout_comparison import PREVIOUS, append_queue, sha256, verify_files

# Retain the familiar helper name, but reject existing destination/temp symlinks.
write_json = owned_json

QUEUE = PREVIOUS / "g2m_improvements/autorun/queue.jsonl"
ORDER = [("fixed",0),("cost",0),("cost",1),("fixed",1),("fixed",2),("cost",2)]
PRIMARY = ("H_MAE_gauge_meV", "levels_rmse_aligned_meV", "gap_error_meV")


def prepare(root, pilot_root, approved=False):
    if not approved:
        raise ValueError("explicit authorization of these six scientific fits required")
    root, pilot_root = recovery.own_output(root), recovery.own_output(pilot_root)
    if root.exists():
        raise ValueError("new exclusive science directory required")
    manifest_path, pilot_path = pilot_root / "manifest.json", pilot_root / "pilot.json"
    pilot = json.loads(manifest_path.read_text())
    selection = json.loads(pilot_path.read_text())["selected"]
    verify_files(pilot["files"])
    if selection != dict(budget=326283, reduced_both_rounds=True) or pilot["split_counts"] != dict(train=32,validation=4,test=3):
        raise ValueError("approved pilot budget/split mismatch")
    versions = {**pilot["versions"], "scipy": importlib.metadata.version("scipy")}
    for name,version in versions.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen software version changed: "+name)
    pilot_job = pilot_root / "pilot-r0-fixed/job.json"
    expected_resolution = json.loads(pilot_job.read_text())["input_resolution"]
    if (len(pilot["rows"]) != 39 or len(pilot["sample_ids"]) != 32
            or len(set(pilot["sample_ids"])) != 32):
        raise ValueError("frozen split coverage mismatch")
    by_id = {r["sample_id"]:r for r in pilot["rows"]}
    if len(by_id) != 39:
        raise ValueError("duplicate frozen split identity")
    recovery.recovery_api()
    import torch
    from readout_comparison_callback import tensor_hash
    configurations = {}
    for arm,seed in ORDER:
        initial = pilot["initial_states"][str(seed)]
        initial_path = recovery.own_output(initial["path"])
        if (initial_path != pilot_root / f"initial-s{seed}.pt"
                or initial["fixed_sha256"] != initial["cost_sha256"]
                or tensor_hash(torch.load(initial_path,map_location="cpu",weights_only=True).items()) != initial["fixed_sha256"]):
            raise ValueError("changed/unpaired initial model state")
        cfg = copy.deepcopy(pilot["recipes"][str(seed)])
        if (cfg["seed_everything"] != seed or cfg["trainer"]["max_epochs"] != 2000
                or cfg["trainer"]["check_val_every_n_epoch"] != 5 or cfg["data"]["batch_size"] != 8
                or cfg["lr_scheduler"] != dict(class_path="torch.optim.lr_scheduler.CosineAnnealingLR",init_args=dict(T_max=2000,eta_min=1e-5))):
            raise ValueError("approved scientific recipe mismatch")
        resolution = recovery.input_resolution(cfg)
        for argument in ("basis_files","train_runs","val_runs"):
            if ([p["target"] for p in resolution.get(argument,[])] !=
                    [p["target"] for p in expected_resolution.get(argument,[])]):
                raise ValueError("pilot-frozen input resolution/order changed: "+argument)
        train_targets = [Path(p["target"]) for p in resolution["train_runs"]]
        for target,identifier in zip(train_targets,pilot["sample_ids"]):
            row = by_id[identifier]
            expected = Path(row["delta"])
            if row["split"] != "train" or (target.parent != expected.parent if target.suffix==".fdf" else target!=expected):
                raise ValueError("ordered training split identity changed")
        val_targets = [Path(p["target"]) for p in resolution["val_runs"]]
        expected_val = {Path(r["delta"]).parent if val_targets[0].suffix==".fdf" else Path(r["delta"])
                        for r in pilot["rows"] if r["split"]=="validation"}
        if len(val_targets)!=4 or {p.parent if p.suffix==".fdf" else p for p in val_targets} != expected_val:
            raise ValueError("validation split identity changed")
        configurations[arm,seed] = cfg
    files = {**pilot["files"], **{str(p): sha256(p) for p in (manifest_path,pilot_path,pilot_job,Path(__file__).resolve(),
        Path(__file__).with_name("h2oclu_batching_metrics.py").resolve())}}
    files.update({p["path"]:sha256(p["path"]) for p in pilot["initial_states"].values()})
    root.mkdir(parents=True)
    jobs = []
    for arm,seed in ORDER:
        initial = pilot["initial_states"][str(seed)]
        cfg = configurations[arm,seed]
        cfg["data"]["max_batch_cost"] = None if arm == "fixed" else 326283
        key = f"{arm}_s{seed}"
        cfg["trainer"]["default_root_dir"] = str(root / key)
        job = recovery.make_job(root / key, cfg, files, Path(initial["path"]), initial["fixed_sha256"],
                                pilot["sample_ids"], versions)
        job_path = root / key / "job.json"
        native = json.loads(job_path.read_text())
        native.update(technical_pilot=False, scientific_authorized=True,
                      authorization="Autorizo esos 6 trabajos", approved_samples=64000)
        write_json(job_path, native)
        jobs.append(dict(job, key=key, arm=arm, seed=seed))
    for job in jobs:
        for path in (Path(job["config"]), Path(job["root"]) / "job.json"):
            files[str(path)] = sha256(path)
    result = dict(jobs=jobs, files=files, versions=versions, rows=pilot["rows"], scientific_authorized=True,
        approval="Autorizo esos 6 trabajos", approved_fit_count=6, created=time.time(),
        pilot_root=str(pilot_root), selection=selection,
        protocol="propuesta_batching_h2oclu_20261016.md; 2000 epochs/64000 exposures, explicit cosine2000, best val_loss every5",
        limitations="validation used for selection; previously observed OOD; 3 seeds exploratory; no default promotion")
    write_json(root / "manifest.json", result)
    read_campaign(root)
    return result


def read_campaign(root):
    root = recovery.own_output(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (not manifest.get("scientific_authorized") or manifest.get("approval") != "Autorizo esos 6 trabajos"
            or len(manifest["jobs"]) != 6 or [(j["arm"],j["seed"]) for j in manifest["jobs"]] != ORDER):
        raise ValueError("campaign differs from six authorized fits")
    verify_files(manifest["files"])
    for job in manifest["jobs"]:
        if Path(job["root"]) != root / job["key"]:
            raise ValueError("job output outside campaign")
        native,cfg = recovery.read_job(Path(job["config"]), Path(job["root"]))
        if (native["max_epochs"] != 2000 or len(native["sample_ids"]) != 32
                or not native["scientific_authorized"] or native["technical_pilot"]
                or cfg["seed_everything"] != job["seed"]
                or cfg["data"]["max_batch_cost"] != (None if job["arm"] == "fixed" else 326283)):
            raise ValueError("job differs from authorized scientific recipe")
    return manifest


def enqueue(root, queue=QUEUE):
    root = recovery.own_output(root)
    manifest = read_campaign(root)
    queue = Path(queue)
    receipt_path = root / "queue_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["queue"] != str(queue) or receipt["manifest_sha256"] != sha256(root / "manifest.json"):
            raise ValueError("queue receipt provenance mismatch")
    else:
        previous_bytes = queue.read_bytes()
        entries = [json.loads(line) for line in previous_bytes.splitlines() if line.strip()]
        state = json.loads(queue.with_name("state.json").read_text())
        dependencies = live_dependencies(entries,state)
        jobs = []
        for row in manifest["jobs"]:
            job_id = f"{root.name}:{row['key']}"
            jobs.append(dict(id=job_id, script=str(Path(__file__).resolve()),
                args=["run","--root",str(root),"--key",row["key"]], after=dependencies,
                vram_mib=3000, min_free_vram_frac=.30))
            dependencies = [job_id]
        receipt = dict(queue=str(queue), old_bytes=len(previous_bytes), old_sha256=hashlib.sha256(previous_bytes).hexdigest(),
            manifest_sha256=sha256(root / "manifest.json"), jobs=jobs, complete=False, appended_ids=[])
        write_json(receipt_path,receipt)  # Recovery of partial submission uses exactly these dependencies.
    for job in receipt["jobs"]:
        prefix = queue.read_bytes()[:receipt["old_bytes"]]
        if hashlib.sha256(prefix).hexdigest() != receipt["old_sha256"]:
            raise RuntimeError("queue prior bytes changed; refuse append")
        append_queue(queue,job)
        if job["id"] not in receipt["appended_ids"]:
            receipt["appended_ids"].append(job["id"])
        write_json(receipt_path,receipt)
    receipt["complete"] = True
    write_json(receipt_path,receipt)
    return receipt


def evaluate(root, key):
    from h2oclu_batching_metrics import evaluate_checkpoint
    manifest = read_campaign(root)
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    directory = Path(job["root"])
    native,_ = recovery.read_job(Path(job["config"]), directory)
    store = recovery.recovery_api().RecoveryStore(directory,native["fingerprint"])
    receipt = store.latest()
    if receipt is None or receipt["next_epoch"] != 2000 or receipt["samples_committed"] != 64000:
        raise ValueError("evaluation requires verified completed training")
    checkpoint = store.restore_best(receipt)
    fit = json.loads((directory / "fit.json").read_text())
    if fit["checkpoint"] != str(checkpoint) or fit["recovery_checkpoint"] != receipt["checkpoint"]:
        raise ValueError("selected checkpoint provenance mismatch")
    rows = [r for r in manifest["rows"] if r["split"] in ("validation","test")]
    if len(rows) != 7:
        raise ValueError("four ID and three OOD structures required")
    evaluation = evaluate_checkpoint(checkpoint,rows,directory / "evaluation")
    verify_files(manifest["files"])
    write_json(directory / "result.json", dict(arm=job["arm"],seed=job["seed"],training=fit,evaluation=evaluation))
    write_json(directory / "evaluation_receipt.json",dict(checkpoint=str(checkpoint),checkpoint_sha256=receipt["best"]["sha256"],
        result_sha256=sha256(directory/"result.json")))


def validated_result(manifest,job):
    directory = recovery.own_output(job["root"])
    result_path = recovery.own_output(directory/"result.json")
    proof = json.loads(recovery.own_output(directory/"evaluation_receipt.json").read_text())
    if proof["result_sha256"] != sha256(result_path):
        raise ValueError("result hash changed after evaluation")
    result = json.loads(result_path.read_text())
    native,_ = recovery.read_job(Path(job["config"]),directory)
    store = recovery.recovery_api().RecoveryStore(directory,native["fingerprint"])
    receipt = store.latest()
    if (receipt is None or receipt["next_epoch"] != 2000 or receipt["samples_committed"] != 64000
            or receipt["best"] is None):
        raise ValueError("result training budget/selection not verified")
    checkpoint = str(store.restore_best(receipt))
    evaluation,fit = result["evaluation"],result["training"]
    if (result["arm"] != job["arm"] or result["seed"] != job["seed"]
            or evaluation["checkpoint"] != checkpoint or evaluation["checkpoint_sha256"] != receipt["best"]["sha256"]
            or proof["checkpoint"] != checkpoint or proof["checkpoint_sha256"] != receipt["best"]["sha256"]
            or fit["checkpoint"] != checkpoint or fit["recovery_checkpoint"] != receipt["checkpoint"]
            or fit["samples_committed"] != 64000):
        raise ValueError("result identity/checkpoint mismatch")
    expected = {(r["sample_id"],r["split"]) for r in manifest["rows"] if r["split"] in ("validation","test")}
    observed = [(r["sample_id"],r["split"]) for r in evaluation["rows"]]
    if len(observed)!=7 or len(set(observed))!=7 or set(observed)!=expected:
        raise ValueError("result structure coverage mismatch")
    return result


def summarize(root):
    manifest = read_campaign(root)
    results = {(j["arm"],j["seed"]): validated_result(manifest,j)
               for j in manifest["jobs"] if (Path(j["root"]) / "result.json").exists()}
    summary = dict(completed=len(results), approved=6, limitations=manifest["limitations"], metrics={})
    for split,label in (("validation","id_selection"),("test","ood_exploratory")):
        for metric in PRIMARY:
            seeds = [s for s in range(3) if ("fixed",s) in results and ("cost",s) in results]
            fixed, cost = [], []
            for seed in seeds:
                pair = []
                for arm in ("fixed","cost"):
                    population = [r for r in results[arm,seed]["evaluation"]["rows"] if r["split"] == split]
                    values = [r[metric] if metric=="H_MAE_gauge_meV" or r["spectral_valid"] else None for r in population]
                    pair.append(float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None)
                if None not in pair:
                    fixed.append(pair[0]); cost.append(pair[1])
            record = dict(paired_count=len(fixed), fixed_per_seed=fixed, cost_per_seed=cost)
            if fixed:
                record.update(fixed_mean=float(np.mean(fixed)),cost_mean=float(np.mean(cost)),
                    paired_ratios=[b/a if a>0 else None for a,b in zip(fixed,cost)],
                    ratio_of_means=float(np.mean(cost)/np.mean(fixed)) if np.mean(fixed)>0 else None)
            if len(fixed) == 3 and all(v>0 for v in fixed):
                ratios = [np.mean([cost[i] for i in indices])/np.mean([fixed[i] for i in indices])
                          for indices in itertools.product(range(3),repeat=3)]
                record["paired_bootstrap_ratio_ci95"] = np.quantile(ratios,[.025,.975]).tolist()
                record["bootstrap_scope"] = "exact 27 ordered paired resamples of 3 seeds; low power"
            summary["metrics"][label+"/"+metric] = record
    write_json(Path(root) / "summary.json",summary)
    return summary


def run(root, key):
    manifest = read_campaign(root)
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    fit = recovery.guarded_fit(Path(job["config"]),Path(job["root"]))
    print("FIT_COMPLETE",key,fit["samples_committed"],flush=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),"eval","--root",str(root),"--key",key],
                   env=environment(cuda=""),check=True)
    summarize(root)
    print("DONE",key,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","enqueue","run","eval","summary"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pilot-root",type=Path)
    parser.add_argument("--approved-six",action="store_true")
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root,args.pilot_root,approved=args.approved_six)
    elif args.action == "enqueue":
        enqueue(args.root)
    elif args.action == "summary":
        summarize(args.root)
    else:
        {"run":run,"eval":evaluate}[args.action](args.root,args.key)


if __name__ == "__main__":
    main()
