#!/usr/bin/env python3
"""GPU speed/VRAM probe of the CPU-verified options (roadmap 2.2, 2.6, 2.8).

Same recipes, probe batch and telemetry as the readout campaign (baseline readout,
seed 0, w90 and 6x6; 10 warmup + 30 measured steps in fresh processes, rotating order):

- pre26: code at 4444ff7 (graph2mat-wt-base), i.e. without the 2.6 mask skip;
- current: g2m-v2-cpu code, defaults (lean_mace is already the default);
- no_lean: lean_mace=False (2.2, the old full MACE);
- simplify: simplify_node_irreps=True (2.8).

5.3 (coefficient_training) is left out: with the gauge loss of these recipes it falls back
to the matrix path (checked in a CPU smoke), so it would time the same computation.
Not measured here: 6.1/6.2 (data loading, outside the timed hooks) and 6.4 (changes the
batches, so per-step times are not comparable).

Usage: run_speed_probe.py {prepare,enqueue,run,summary} --root DIR
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


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_readout_comparison as rc  # noqa: E402

CAMPAIGN = Path("/home/christian/repositorios/graph2mat/auditorias_mejoras/experimentos_readout_w90_6x6_20261010_v1")
SOURCES = {"base": Path("/home/christian/repositorios/graph2mat-wt-base/src"), "cpu": HERE.parent / "src"}
VARIANTS = {"pre26": ("base", {}), "current": ("cpu", {}),
            "no_lean": ("cpu", {"lean_mace": False}), "simplify": ("cpu", {"simplify_node_irreps": True})}
ROUNDS = 6


def prepare(root):
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text())
    jobs = [j for j in manifest["jobs"] if j["seed"] == 0 and j["variant"] == "baseline"]
    root.mkdir(parents=True, exist_ok=False)
    rc.write_json(root / "plan.json", {"jobs": jobs, "variants": VARIANTS, "rounds": ROUNDS,
                                       "sources": {k: str(v) for k, v in SOURCES.items()}})


def child(run, source):
    """Fresh fit process; yields (kills itself) if another GPU process appears."""
    rc.idle_gpu()
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           # probe_site drops the editable finder so ``source`` is the graph2mat imported
           "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", source, HERE)))}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    with (run / "train.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, str(HERE / "run_readout_comparison.py"), "fit", "--root", str(run),
                                 "--probe"], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(2)
            if proc.poll() is None and rc.gpu_processes(rc.process_tree(proc.pid)):
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
                return False
    if proc.returncode:
        raise RuntimeError(f"probe exit {proc.returncode}; see {run / 'train.log'}")
    return True


def run_probes(root):
    plan = json.loads((root / "plan.json").read_text())
    names = list(plan["variants"])
    for job in plan["jobs"]:
        for index in range(plan["rounds"]):
            order = names[index % len(names):] + names[:index % len(names)]
            for name in (reversed(order) if (index // len(names)) % 2 else order):
                label = f"{job['system']}-{index:02d}-{name}"
                done = root / "completed" / f"{label}.json"
                if done.exists():
                    continue
                source, overrides = plan["variants"][name]
                while True:
                    parent = root / "attempts" / label
                    parent.mkdir(parents=True, exist_ok=True)
                    run = parent / f"attempt-{len(list(parent.iterdir())) + 1:03d}"
                    # Own shared init per variant: their backbones differ (no pairing needed for timing).
                    shared = root / "shared" / f"{job['system']}-{name}.pt"
                    cfg = rc.recipe(job["config"], "baseline", run, shared, "probe")
                    cfg["model"].update(overrides)
                    rc.materialize(job, run, cfg)
                    print("START", label, flush=True)
                    if child(run, plan["sources"][source]):
                        break
                    print("YIELD", label, flush=True)
                telemetry = json.loads((run / "telemetry.json").read_text())
                rc.write_json(done, {"run": str(run), "telemetry": telemetry})
                print("DONE", label, flush=True)
    summarize(root)


def summarize(root):
    rows = {}
    for path in sorted((root / "completed").glob("*.json")):
        system, _, name = path.stem.split("-", 2)
        value = json.loads(path.read_text())
        entry = rows.setdefault(system, {}).setdefault(name, {"step_ms": [], "peak_mib": []})
        entry["step_ms"].append(statistics.median(value["telemetry"]["raw_ms"]["step"]))
        entry["peak_mib"].append(value["telemetry"]["peak_allocated_bytes"] / 2**20)
    summary = {}
    for system, variants in rows.items():
        ref = statistics.median(variants["current"]["step_ms"]) if "current" in variants else None
        for name, e in variants.items():
            step = statistics.median(e["step_ms"])
            summary.setdefault(system, {})[name] = {
                "rounds": len(e["step_ms"]), "step_ms_median": step,
                "step_ms_mad": statistics.median(abs(x - step) for x in e["step_ms"]),
                "peak_allocated_mib_median": statistics.median(e["peak_mib"]),
                "step_vs_current": step / ref if ref else None}
    rc.write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "enqueue", "run", "summary"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "enqueue":
        job = {"id": "speedprobe:" + root.name, "script": str(Path(__file__).resolve()),
               "args": ["run", "--root", str(root)], "vram_mib": 22000,
               "after": ["readout:" + CAMPAIGN.name]}
        print("enqueued" if rc.append_queue(rc.PREVIOUS / "g2m_improvements/autorun/queue.jsonl", job)
              else "already queued")
    elif args.action == "run":
        run_probes(root)
    else:
        summarize(root)


if __name__ == "__main__":
    main()
