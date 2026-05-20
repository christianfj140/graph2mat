#!/usr/bin/env python3
"""Compare default, Hamiltonian, and dense H2O coefficient readouts.

This diagnostic is intentionally outside production code. It loads the existing
one-sample H-only H2O checkpoint, keeps the MACE body as the feature source, and
compares local readout controllability for:

- the checkpoint's default readout,
- the experimental scalar-conditioned Hamiltonian readout,
- the diagnostic dense readout.

The swapped readouts are not trained by this script; direct label errors are
smoke metrics, while the useful outputs are coefficient/local rank and input
feature-distance comparisons.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


GRAPH2MAT_ROOT = Path(__file__).resolve().parents[5]
H2O_DIAGNOSTIC_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONTEXT_DIR = H2O_DIAGNOSTIC_ROOT / "local_context_probe"
JOINT_RANK_DIR = H2O_DIAGNOSTIC_ROOT / "concrete_joint_readout_rank"
DEFAULT_PIPELINE_ROOT = Path(
    "/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement"
)
DEFAULT_SOURCE_WORKSPACE = (
    DEFAULT_PIPELINE_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_overfit_h_only_after_yield_fix"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare_imports(pipeline_root: Path) -> None:
    for path in (
        pipeline_root / "scripts" / "torch_serialization_compat",
        pipeline_root / "Comparison" / "scripts",
        LOCAL_CONTEXT_DIR,
        JOINT_RANK_DIR,
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True).strip()
    except Exception:
        return None


def latest_checkpoint(training_dir: Path) -> Path:
    candidates = sorted(
        (training_dir / "logs" / "one_sample_h_only").glob(
            "version_*/checkpoints/best-*.ckpt"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found under {training_dir}")
    return candidates[-1]


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return sanitize(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def build_datamodule_kwargs(
    data: dict[str, Any],
    matrix_datamodule_cls: type,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "out_matrix": data["out_matrix"],
        "basis_files": data["basis_files"],
        "train_runs": data["train_runs"],
        "val_runs": data.get("val_runs", data["train_runs"]),
        "test_runs": data.get("test_runs", data["train_runs"]),
        "symmetric_matrix": bool(data["symmetric_matrix"]),
        "sub_point_matrix": bool(data.get("sub_point_matrix", False)),
        "batch_size": int(data.get("batch_size", 1)),
        "store_in_memory": bool(data.get("store_in_memory", True)),
    }
    signature = inspect.signature(matrix_datamodule_cls).parameters
    if "n_matrix_components" in signature:
        kwargs["n_matrix_components"] = int(data["n_matrix_components"])
    if "matrix_component_policy" in signature:
        kwargs["matrix_component_policy"] = data["matrix_component_policy"]
    if "loader_threads" in signature and data.get("loader_threads") is not None:
        kwargs["loader_threads"] = int(data["loader_threads"])
    return kwargs


def label_metrics(output: dict[str, torch.Tensor], batch: Any) -> dict[str, float]:
    node_error = output["node_labels"].detach() - batch["point_labels"].detach()
    edge_error = output["edge_labels"].detach() - batch["edge_labels"].detach()
    return {
        "node_mae": float(node_error.abs().mean().item()),
        "edge_mae": float(edge_error.abs().mean().item()),
        "label_mae": float(
            torch.cat([node_error.reshape(-1), edge_error.reshape(-1)])
            .abs()
            .mean()
            .item()
        ),
    }


def reconstruction_mae(matrix_model: torch.nn.Module, datamodule: Any, batch: Any):
    output = matrix_model(batch)
    processor = datamodule.data_processor
    prediction = processor.matrix_from_data(
        batch,
        predictions={
            "node_labels": output["node_labels"],
            "edge_labels": output["edge_labels"],
        },
        out_format="numpy",
    )
    reference = processor.matrix_from_data(
        batch,
        predictions={
            "node_labels": batch["point_labels"],
            "edge_labels": batch["edge_labels"],
        },
        out_format="numpy",
    )
    return float(np.mean(np.abs(prediction - reference)))


def edge_hidden_irreps_from_readout(readout: Any):
    preprocessing_edges = getattr(readout, "preprocessing_edges", None)
    irreps_out = getattr(preprocessing_edges, "irreps_out", None)
    if isinstance(irreps_out, tuple):
        return irreps_out[1]
    return None


def class_or_none(module: Any):
    return None if module is None else module.__class__


def build_swapped_model(
    base_model: torch.nn.Module,
    node_operation: type,
    edge_operation: type,
    use_context: bool,
    context_kwargs: dict[str, Any],
):
    from graph2mat.models.mace import MatrixMACE

    readout = base_model.matrix_readouts
    return MatrixMACE(
        mace=copy.deepcopy(base_model.mace),
        readout_per_interaction=False,
        graph2mat_cls=readout.__class__,
        hamiltonian_local_context=use_context,
        hamiltonian_local_context_kwargs=context_kwargs if use_context else None,
        unique_basis=readout.basis_table.basis,
        edge_hidden_irreps=edge_hidden_irreps_from_readout(readout),
        symmetric=readout.symmetric,
        preprocessing_nodes=class_or_none(readout.preprocessing_nodes),
        preprocessing_edges=class_or_none(readout.preprocessing_edges),
        preprocessing_edges_reuse_nodes=readout.preprocessing_edges_reuse_nodes,
        node_operation=node_operation,
        edge_operation=edge_operation,
        basis_grouping=readout.basis_grouping,
        n_matrix_components=readout.n_matrix_components,
    )


def rank_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "model_variant": row["model_variant"],
                "operation": row["operation"],
                "kind": row["kind"],
                "module_key": row["module_key"],
                "joint_output_dim": row["joint_output_dim"],
                "local_input_dim": row["local_input_dim"],
                "parameter_rank_1e-8": row["parameter_jacobian"]["rank_abs_tol"][
                    "1e-08"
                ],
                "input_rank_1e-8": row["input_jacobian"]["rank_abs_tol"]["1e-08"],
                "combined_rank_1e-8": row["combined_parameter_input_jacobian"][
                    "rank_abs_tol"
                ]["1e-08"],
                "parameter_count": row["parameter_count"],
                "operation_class": row["operation_class"],
            }
        )
    return flat_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument(
        "--source-workspace",
        type=Path,
        default=DEFAULT_SOURCE_WORKSPACE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--context-num-radial", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--no-vectorize", action="store_true")
    args = parser.parse_args()

    pipeline_root = args.pipeline_root.resolve()
    prepare_imports(pipeline_root)

    from graph2mat.bindings.e3nn import (
        E3nnDiagnosticDenseEdgeBlock,
        E3nnDiagnosticDenseNodeBlock,
        E3nnHamiltonianEdgeBlock,
        E3nnHamiltonianNodeBlock,
    )
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from hamiltonian_local_context_probe import analyze_readout

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_workspace = args.source_workspace.resolve()
    training_dir = source_workspace / "training"
    config_path = training_dir / "config.yaml"
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else latest_checkpoint(training_dir)
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])
    dtype = getattr(torch, args.dtype)
    context_kwargs = {"num_radial": args.context_num_radial}

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data, MatrixDataModule))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        lit_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        lit_model.eval()
        default_model = lit_model.model
        hamiltonian_model = build_swapped_model(
            default_model,
            node_operation=E3nnHamiltonianNodeBlock,
            edge_operation=E3nnHamiltonianEdgeBlock,
            use_context=True,
            context_kwargs=context_kwargs,
        )
        dense_model = build_swapped_model(
            default_model,
            node_operation=E3nnDiagnosticDenseNodeBlock,
            edge_operation=E3nnDiagnosticDenseEdgeBlock,
            use_context=True,
            context_kwargs=context_kwargs,
        )

        variants = {
            "default": default_model,
            "hamiltonian_readout": hamiltonian_model.eval(),
            "diagnostic_dense_readout": dense_model.eval(),
        }

        label_error = {}
        h_reconstruction_mae = {}
        rank_rows = []
        distance_rows = []
        for name, model in variants.items():
            with torch.no_grad():
                label_error[name] = label_metrics(model(batch), batch)
                h_reconstruction_mae[name] = reconstruction_mae(
                    model, datamodule, batch
                )
            ranks, distances = analyze_readout(
                name,
                model,
                batch,
                dtype=dtype,
                vectorize=not args.no_vectorize,
            )
            rank_rows.extend(ranks)
            distance_rows.extend(distances)

    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    rank_csv = output_dir / "hamiltonian_readout_rank_by_operation.csv"
    feature_csv = output_dir / "hamiltonian_readout_feature_distances.csv"
    json_path = output_dir / "hamiltonian_readout_probe_results.json"

    write_csv(
        rank_csv,
        rank_csv_rows(rank_rows),
        [
            "model_variant",
            "operation",
            "kind",
            "module_key",
            "joint_output_dim",
            "local_input_dim",
            "parameter_rank_1e-8",
            "input_rank_1e-8",
            "combined_rank_1e-8",
            "parameter_count",
            "operation_class",
        ],
    )
    write_csv(
        feature_csv,
        distance_rows,
        [
            "model_variant",
            "operation",
            "kind",
            "module_key",
            "sample_i",
            "sample_j",
            "label_i",
            "label_j",
            "euclidean",
            "cosine",
            "max_abs_delta",
            "near_identical",
            "note",
        ],
    )
    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "command": command,
        "context_kwargs": context_kwargs,
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "direct_one_sample_label_error": label_error,
        "h_reconstruction_mae": h_reconstruction_mae,
        "joint_readout_rank": rank_rows,
        "feature_distances": distance_rows,
        "output_files": {
            "json": str(json_path),
            "rank_csv": str(rank_csv),
            "feature_distances_csv": str(feature_csv),
        },
        "repository_context": {
            "graph2mat_branch": command_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GRAPH2MAT_ROOT
            ),
            "graph2mat_commit": command_output(
                ["git", "rev-parse", "HEAD"], cwd=GRAPH2MAT_ROOT
            ),
            "pipeline_branch": command_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pipeline_root
            ),
            "pipeline_commit": command_output(
                ["git", "rev-parse", "HEAD"], cwd=pipeline_root
            ),
        },
    }
    json_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {rank_csv}")
    print(f"Wrote {feature_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
