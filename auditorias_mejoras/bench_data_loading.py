#!/usr/bin/env python3
"""Roadmap 6.1/6.2: end-to-end CPU data loading on the real w90 and 6x6 campaign data.

Times, in fresh processes alternating code trees, MatrixDataModule.setup("fit") (reading
and preprocessing every train/validation sample) and one pass over the train and the
validation dataloaders (collate and type layout, workers off). Trees: f1bb647 (before
6.1/6.2/6.4), 4444ff7 (g2m-v2 head) and g2m-v2-cpu. Same data, configs and threads; the first round is warm-up.

Usage: bench_data_loading.py run --out DIR [--rounds 4]
       bench_data_loading.py one --run DIR      (internal: one measurement)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREES = {"f1bb647": Path("/home/christian/repositorios/graph2mat-wt-f1bb/src"),
         "4444ff7": Path("/home/christian/repositorios/graph2mat-wt-base/src"), "current": HERE.parent / "src"}


def one(run):
    import yaml
    from graph2mat.tools.lightning.data import MatrixDataModule

    data = yaml.safe_load((run / "config.yaml").read_text())["data"]
    data["loader_threads"] = 0
    t0 = time.perf_counter()
    module = MatrixDataModule(**data)
    module.setup("fit")
    t1 = time.perf_counter()
    n_train = sum(1 for _ in module.train_dataloader())
    t2 = time.perf_counter()
    n_val = sum(1 for _ in module.val_dataloader())
    t3 = time.perf_counter()
    print(json.dumps({"setup_s": t1 - t0, "train_epoch_s": t2 - t1, "val_epoch_s": t3 - t2,
                      "train_batches": n_train, "val_batches": n_val}))


def run(out, rounds):
    sys.path.insert(0, str(HERE))
    import run_readout_comparison as rc
    import run_speed_probe as sp

    manifest = json.loads((sp.CAMPAIGN / "manifest.json").read_text())
    results = {}
    for job in (j for j in manifest["jobs"] if j["seed"] == 0 and j["variant"] == "baseline"):
        run_dir = out / job["system"]
        if not run_dir.exists():
            rc.materialize(job, run_dir, rc.recipe(job["config"], "baseline", run_dir, out / "unused.pt", accelerator="cpu"))
        for index in range(rounds):
            names = list(TREES)
            for tree in names[index % 3:] + names[:index % 3]:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                       "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", TREES[tree], HERE)))}
                line = subprocess.run([sys.executable, __file__, "one", "--run", str(run_dir)], env=env,
                                      cwd=run_dir, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
                if index:  # round 0 warms the page cache
                    results.setdefault(job["system"], {}).setdefault(tree, []).append(json.loads(line))
                print(job["system"], index, tree, line, flush=True)
    summary = {system: {tree: {key: statistics.median(r[key] for r in rows) for key in rows[0]}
                        for tree, rows in trees.items()} for system, trees in results.items()}
    (out / "summary.json").write_text(json.dumps({"rounds": results, "median": summary}, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("run", "one"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if args.action == "one":
        one(args.run.resolve())
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        run(args.out.resolve(), args.rounds)


if __name__ == "__main__":
    main()
