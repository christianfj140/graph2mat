#!/usr/bin/env python3
"""Cheap check: constant learning rate of a historical CLI recipe vs cosine.

Until 2026-10-21 the CLI dropped ``lr_scheduler`` when the config had ``optimizer: null``,
so affected CLI recipes trained at constant ``optim_lr`` (see the roadmap). Dedicated
runners that installed a scheduler explicitly are outside that finding. This probe
retrains one historical recipe (w90 graphene, ``fp32_lin``, N=64, from an existing run
directory) with both schedules, same data, same epochs, paired by seed, and compares the
best validation loss. Test data are not touched.

    run_lr_schedule_probe.py run --arm {constant,cosine} --seed 0 [--epochs 1000]
    run_lr_schedule_probe.py summary
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MDVS = Path(os.environ.get("MDVS_ROOT", HERE.parent.parent / "MD_vs_AtomicDisplacement"))
TEMPLATE = (MDVS / "Comparison/results/dataset_design_curves_v1/runs/w90/"
            "random_cartesian__3D__k2__r0.120__res64__scaled_seed0__N64__ts0__g2mv2_fp32_lin")
OUT = HERE / "experimentos_lr_schedule_w90_20261021"


def run(arm: str, seed: int, epochs: int, accelerator: str) -> Path:
    work = OUT / f"{arm}__s{seed}__e{epochs}"
    if (work / "result.json").exists():
        return work / "result.json"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for split in ("train_samples", "validation_samples"):
        os.symlink(TEMPLATE / split, work / split)
    config = yaml.safe_load((TEMPLATE / "config.yaml").read_text())
    config["seed_everything"] = seed
    # basis_files is a glob relative to the run directory: link the template's basis folder.
    basis = Path(os.path.normpath(TEMPLATE / Path(config["data"]["basis_files"]).parent))
    os.symlink(basis, work / "basis")
    config["data"]["basis_files"] = "basis/" + Path(config["data"]["basis_files"]).name
    config["trainer"].update(accelerator=accelerator, max_epochs=epochs,
                             logger={"class_path": "CSVLogger", "init_args": {"save_dir": "logs", "name": "run"}})
    if arm == "cosine":
        config["lr_scheduler"] = {"class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
                                  "init_args": {"T_max": epochs, "eta_min": 1.0e-5}}
    else:
        config.pop("lr_scheduler", None)
    (work / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with (work / "train.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "graph2mat.tools.cli.cli", "models", "mace", "main", "fit",
                        "-c", "config.yaml"], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch

    (best,) = sorted(work.rglob("best-*.ckpt"))
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    callback = next(v for k, v in ckpt["callbacks"].items() if "ModelCheckpoint" in k)
    result = {"arm": arm, "seed": seed, "epochs": epochs, "template": TEMPLATE.name,
              "best_val_loss": float(callback["best_model_score"]), "best_epoch": int(ckpt["epoch"]),
              "lr_at_best": [g["lr"] for g in ckpt["optimizer_states"][0]["param_groups"]][0],
              "n_lr_schedulers_in_ckpt": len(ckpt.get("lr_schedulers", []))}
    (work / "result.json").write_text(json.dumps(result, indent=1))
    return work / "result.json"


def summary() -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/result.json"))]
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        losses = {r["seed"]: r["best_val_loss"] for r in rows if r["arm"] == arm}
        out[arm] = {"n": len(losses), "best_val_loss": losses}
    if {"constant", "cosine"} <= set(out):
        seeds = sorted(set(out["constant"]["best_val_loss"]) & set(out["cosine"]["best_val_loss"]))
        ratios = [out["cosine"]["best_val_loss"][s] / out["constant"]["best_val_loss"][s] for s in seeds]
        out["paired_ratio_cosine_over_constant"] = dict(zip(seeds, ratios))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["constant", "cosine"], required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--epochs", type=int, default=1000)
    r.add_argument("--accelerator", default="gpu")
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.action == "run":
        print("done", run(args.arm, args.seed, args.epochs, args.accelerator), flush=True)
    else:
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
