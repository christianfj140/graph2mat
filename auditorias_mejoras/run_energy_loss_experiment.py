#!/usr/bin/env python3
"""Roadmap 4.5 accuracy experiment: the energy-consistency loss on an H model (GPU queue).

Recipe: 6x6 Sobol N8, gauge loss, fp32, linear edge readout, 16k steps, the historical
non-Delta recipe ``..._gaugeh_kl`` (Delta samples carry no TSDE, which the term needs).
Arms: energy_weight 0 (control) and the WEIGHTS below; seeds 0-2 with paired backbone and
on-site initialization (ComparisonCallback). Up to three trainings run at once under autorun's
guard rails (stock ``main fit`` processes it can pause; OOM requeues the job). The last checkpoint is evaluated in every arm,
because val_loss contains the term being tested. Evaluation (CPU) as in the readout
campaign, without H0: validation, test, registry, AB-MD, moire.

Usage: run_energy_loss_experiment.py {prepare,enqueue,run,summary} --root DIR
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_readout_comparison as rc  # noqa: E402
import run_speed_probe as sp  # noqa: E402

SOURCE = "runs/6x6/sobol_sparse__1D_in__R0.12__d64__N8__ts{seed}__g2mv2_fp32_lin_steps16k_gaugeh_kl"
# Set from the CPU calibration (term ~10 % and ~100 % of the H loss at initialization).
WEIGHTS = {"control": 0.0}
SEEDS = (0, 1, 2)


def prepare(root, weights):
    campaign = json.loads((sp.CAMPAIGN / "manifest.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in SEEDS:
        source = rc.PREVIOUS / SOURCE.format(seed=seed)
        original = yaml.safe_load((source / "config.yaml").read_text())
        bases = sorted(Path(p).resolve() for p in rc.glob.glob(str(source / original["data"]["basis_files"])))
        if not bases:
            raise FileNotFoundError("basis files missing for " + str(source))
        original["data"]["basis_files"] = str(bases[0].parent / "*.ion.xml")
        links = {split: {p.name: str(p.resolve(strict=True)) for p in sorted((source / split).iterdir())}
                 for split in ("train_samples", "validation_samples")}
        for arm, weight in weights.items():
            run = root / "runs" / f"{arm}-seed{seed}"
            cfg = rc.recipe(original, "baseline", run, root / "shared" / f"seed{seed}.pt")
            cfg["data"]["physical_labels"] = ["energy"]
            cfg["model"]["loss_kwargs"] = {"energy_weight": weight}
            cfg["trainer"]["callbacks"] = [{"class_path": "ModelCheckpoint", "init_args": {"save_last": True,
                "save_top_k": 0}}] + [cb for cb in cfg["trainer"]["callbacks"] if cb["class_path"] != "ModelCheckpoint"]
            jobs.append({"arm": arm, "seed": seed, "run": str(run), "config": cfg, "links": links, "source": str(source)})
    rc.write_json(root / "plan.json", {"jobs": jobs, "weights": weights,
                                       "evaluation": campaign["evaluation"]["6x6"]})


PARALLEL, VRAM_PER_TRAINING_MIB = 3, 7000


def _parallel(root):
    """PARALLEL minus one per CUDA OOM recorded in this experiment (3 -> 2 -> 1): autorun requeues
    an OOM job, and the retry must not repeat the same load."""
    path = root / "oom_count"
    return max(1, PARALLEL - (int(path.read_text()) if path.exists() else 0))


def _room(n_running, parallel=PARALLEL):
    """autorun's guard rails for starting a training (CPU launch limit, GPU away from its pause
    temperature, 20 % of the VRAM + margin free after this one)."""
    sys.path.insert(0, str(rc.SCRIPTS))
    import g2m_autorun as autorun

    if n_running >= parallel:
        return False
    gpu_temp, used, total = autorun.gpu()
    return (autorun.cpu_temp() < autorun.LIMIT_C and gpu_temp < autorun.GPU_PAUSE_C - 5
            and total - used - VRAM_PER_TRAINING_MIB - autorun.VRAM_MARGIN_MIB >= autorun.MIN_FREE_VRAM_FRAC * total)


def start(root, job):
    """Launch one training with the stock CLI (``... main fit``): autorun counts and can pause it
    (SIGSTOP) like any training. Not exclusive: other GPU work may share the card."""
    parent = root / "attempts" / f"{job['arm']}-seed{job['seed']}"
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / f"attempt-{len(list(parent.iterdir())) + 1:03d}"
    # As the campaign's execute(): re-target the job's config (paths, logger, telemetry) to this attempt.
    rc.materialize(job, run, rc.recipe(job["config"], "baseline", run, root / "shared" / f"seed{job['seed']}.pt"))
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", sp.SOURCES["cpu"], HERE)))}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    log = (run / "train.log").open("w")
    proc = subprocess.Popen([str(Path(sys.executable).parent / "graph2mat"), "models", "mace", "main", "fit",
                             "-c", "config.yaml"], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print("START", run, flush=True)
    return proc, run


def train_all(root, jobs):
    """Up to PARALLEL trainings at once, each started only when _room() allows it. A failure
    (e.g. CUDA OOM) stops the others and exits with the train.log path, so autorun requeues
    the job on OOM; completed trainings are kept."""
    done_marker = lambda job: root / "completed" / f"train-{job['arm']}-seed{job['seed']}.json"
    # Arm-major order: two arms of one seed never start together before its shared init exists.
    pending = sorted((j for j in jobs if not done_marker(j).exists()), key=lambda j: (j["arm"] != "control", j["arm"], j["seed"]))
    running, parallel = {}, _parallel(root)
    print("PARALLEL", parallel, flush=True)
    while pending or running:
        busy_seeds = {job["seed"] for job, _ in running.values()}
        startable = next((j for j in pending if j["seed"] not in busy_seeds
                          or (root / "shared" / f"seed{j['seed']}.pt").exists()), None)
        if startable is not None and (not running or _room(len(running), parallel)):
            pending.remove(startable)
            proc, run = start(root, startable)
            running[proc.pid] = (startable, (proc, run))
            time.sleep(60)  # let it allocate before judging free VRAM again
            continue
        for pid, (job, (proc, run)) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[pid]
            if proc.returncode:
                for _, (other, _) in running.values():
                    os.killpg(other.pid, signal.SIGTERM)
                if (proc.returncode in (-9, 137)  # kernel OOM killer, as autorun counts it
                        or "out of memory" in (run / "train.log").read_text(errors="ignore").lower()):
                    count = root / "oom_count"
                    count.write_text(str((int(count.read_text()) if count.exists() else 0) + 1))
                raise RuntimeError(f"training exit {proc.returncode}; see {run / 'train.log'}")
            checkpoints = sorted(run.rglob("last*.ckpt"))
            if len(checkpoints) != 1:
                raise ValueError("expected one last checkpoint in " + str(run))
            rc.write_json(done_marker(job), {"run": str(run), "checkpoint": str(checkpoints[0])})
            print("DONE", run, flush=True)
        time.sleep(10)
    return {(j["arm"], j["seed"]): json.loads(done_marker(j).read_text()) for j in jobs}


def evaluate(root, job, checkpoint, evaluation):
    c, _, ood = rc.legacy_modules()
    base = root / "evaluation" / f"{job['arm']}-seed{job['seed']}"
    for target, rows in evaluation.items():
        out = base / target
        if (out / "per_structure_metrics.csv").exists():
            continue
        samples = [c.sample_from_dict(row) for row in rows]
        predicted = c.predict(Path(checkpoint), samples, out, "cpu")
        metrics = []
        for sample in samples:
            row = c.structure_metrics(sample, predicted)
            if target in ("registry", "ab_md", "moire"):
                row.update(ood.metrics(sample, predicted))
            else:
                row.update(c.band_dos_metrics(sample, predicted / sample.sample_id / "ML_prediction.HSX", "6x6"))
            metrics.append(row)
        c.write_csv(out / "per_structure_metrics.csv.tmp", metrics)
        (out / "per_structure_metrics.csv.tmp").replace(out / "per_structure_metrics.csv")


def run(root):
    # Both jobs wait only for the campaign; started together, each would yield to the other's
    # GPU processes forever. Run after the speed probe has finished (done or failed).
    state = rc.PREVIOUS / "g2m_improvements/autorun/state.json"
    probe = "speedprobe:experimentos_speed_probe_20261010_v1"
    while not any(probe in json.loads(state.read_text()).get(key, []) for key in ("done", "failed")):
        print("WAIT speed probe", flush=True)
        time.sleep(60)
    plan = json.loads((root / "plan.json").read_text())
    proofs = train_all(root, plan["jobs"])
    for job in plan["jobs"]:
        evaluate(root, job, proofs[job["arm"], job["seed"]]["checkpoint"], plan["evaluation"])
    summarize(root)


def summarize(root):
    import csv

    plan = json.loads((root / "plan.json").read_text())
    table = {}
    for job in plan["jobs"]:
        for target in plan["evaluation"]:
            path = root / "evaluation" / f"{job['arm']}-seed{job['seed']}" / target / "per_structure_metrics.csv"
            if not path.exists():
                continue
            rows = list(csv.DictReader(path.open()))
            for key in rows[0]:
                try:
                    values = [float(r[key]) for r in rows]
                except ValueError:
                    continue
                table.setdefault(target, {}).setdefault(key, {}).setdefault(job["arm"], {})[job["seed"]] = statistics.mean(values)
    summary = {}
    for target, keys in table.items():
        for key, arms in keys.items():
            control = arms.get("control", {})
            for arm, seeds in arms.items():
                paired = [seeds[s] / control[s] for s in seeds if s in control and control[s]]
                summary.setdefault(target, {}).setdefault(key, {})[arm] = {
                    "mean": statistics.mean(seeds.values()),
                    "ratio_vs_control_geomean": (statistics.geometric_mean(paired) if paired and all(p > 0 for p in paired)
                                                 else None)}
    rc.write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=1)[:4000])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "enqueue", "run", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weight", type=float, action="append", default=[], help="energy_weight arms (prepare)")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, {**WEIGHTS, **{f"energy{w:g}": w for w in args.weight}})
    elif args.action == "enqueue":
        job = {"id": "energyloss:" + root.name, "script": str(Path(__file__).resolve()),
               "args": ["run", "--root", str(root)], "vram_mib": 22000,
               "after": ["readout:" + sp.CAMPAIGN.name]}
        print("enqueued" if rc.append_queue(rc.PREVIOUS / "g2m_improvements/autorun/queue.jsonl", job)
              else "already queued")
    elif args.action == "run":
        run(root)
    else:
        summarize(root)


if __name__ == "__main__":
    main()
