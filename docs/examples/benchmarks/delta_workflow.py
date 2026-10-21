#!/usr/bin/env python3
"""Delta-learning workflow of the g2m-v2 benchmarks, on your own SIESTA runs.

Layout expected (one converged SIESTA run per directory, with RUN.fdf, the label
<SystemLabel>.TSHS and its pseudopotentials, here or in a parent directory)::

    runs/train/<sample>/      runs/validation/<sample>/      runs/test/<sample>/

1. Prepare: H0 of every run and the H - H0 delta samples (+ spectral windows)::

       python delta_workflow.py prepare runs --siesta "siesta"      # or $SIESTA

   writes h0/<split>/<sample>/<SystemLabel>.TSHS and delta/<split>/<sample>/.

2. Train on the delta samples (delta_recipe.yaml points to delta/train, delta/validation)::

       graph2mat models mace main fit -c delta_recipe.yaml

3. Predict H - H0 for the test structures::

       graph2mat models mace main predict -c delta_recipe.yaml --ckpt_path <best.ckpt> \
           --data.predict_structs "runs/test/*/RUN.fdf" \
           --trainer.callbacks+ MatrixWriter --trainer.callbacks.output_file ML_prediction.HSX

4. Add H0 back and evaluate against the SIESTA labels::

       python delta_workflow.py evaluate runs --out metrics.json [--hexagonal]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from graph2mat.tools.analysis.metrics import band_dos_metrics, gamma_metrics
from graph2mat.tools.siesta.delta import add_h0, compute_h0, write_delta_sample
from graph2mat.tools.siesta.runner import system_label
from graph2mat.tools.spectral_targets import WINDOW_FILE, spectral_window_columns, write_projection_files

SPLITS = ("train", "validation", "test")


def _runs(root: Path, split: str):
    return sorted(p for p in (root / split).glob("*") if (p / "RUN.fdf").exists())


def _label(run: Path) -> Path:
    return run / f"{system_label(run / 'RUN.fdf')}.TSHS"


def prepare(root: Path, siesta, spectral: bool) -> None:
    for split in SPLITS:
        deltas, labels = [], []
        for run in _runs(root, split):
            h0 = compute_h0(run, Path("h0") / split / run.name, reference_matrix=_label(run), siesta=siesta)
            if split != "test":  # test structures are predicted, never trained on
                write_delta_sample(run, _label(run), h0, Path("delta") / split / run.name)
                deltas.append(Path("delta") / split / run.name)
                labels.append(_label(run))
            print(split, run.name, "H0 ready", flush=True)
        if spectral and deltas:
            # Windows of the full label H, written next to the delta labels (roadmap 4.4).
            write_projection_files([spectral_window_columns(m) for m in labels], deltas, WINDOW_FILE)


def evaluate(root: Path, out: Path, hexagonal: bool) -> None:
    rows = {}
    for run in _runs(root, "test"):
        h = add_h0(run / "ML_prediction.HSX", Path("h0") / "test" / run.name / _label(run).name,
                   run / "ML_prediction_H.HSX")
        row = gamma_metrics(h, _label(run))
        if hexagonal:
            row.update(band_dos_metrics(h, _label(run), grid=(4, 4, 1)))
        rows[run.name] = row
    keys = next(iter(rows.values())).keys()
    summary = {k: float(np.nanmean([r[k] for r in rows.values()])) for k in keys}
    out.write_text(json.dumps({"mean": summary, "per_structure": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--siesta", default=None, help="SIESTA command (default: $SIESTA or siesta on the PATH)")
    p.add_argument("--spectral-window", action="store_true", help="also write spectral_window.npz (4.4)")
    e = sub.add_parser("evaluate")
    e.add_argument("root", type=Path)
    e.add_argument("--out", type=Path, default=Path("metrics.json"))
    e.add_argument("--hexagonal", action="store_true", help="also band RMSE on G-K-M-G and DOS")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.siesta, args.spectral_window)
    else:
        evaluate(args.root, args.out, args.hexagonal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
