#!/usr/bin/env python3
"""Compare default and Hamiltonian-local-context readout inputs on H2O.

This is a diagnostic-only script. It loads the existing one-sample H-only H2O
checkpoint, keeps the trained MACE body as the source of node features, and
compares the default readout input geometry with a freshly initialized
context-enabled readout. The context-enabled readout is intentionally not
treated as a trained model; its purpose here is to measure whether the local
readout inputs become less collapsed for H-O operations.
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
from mace.modules.utils import get_edge_vectors_and_lengths


GRAPH2MAT_ROOT = Path(__file__).resolve().parents[5]
DIAGNOSTIC_ROOT = Path(__file__).resolve().parents[1]
JOINT_RANK_DIAGNOSTIC_DIR = DIAGNOSTIC_ROOT / "concrete_joint_readout_rank"
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
        JOINT_RANK_DIAGNOSTIC_DIR,
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


def reconstruction_mae(
    matrix_model: torch.nn.Module,
    datamodule: Any,
    batch: Any,
) -> float:
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


def make_readout_inputs(matrix_model: torch.nn.Module, batch: Any):
    mace_out = matrix_model.mace(batch, compute_force=False)
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=batch["positions"],
        edge_index=batch["edge_index"],
        shifts=batch["shifts"],
    )
    edge_attrs = matrix_model.mace.spherical_harmonics(vectors)
    edge_feats = matrix_model.mace.radial_embedding(
        lengths,
        batch["node_attrs"],
        batch["edge_index"],
        matrix_model.mace.atomic_numbers,
    )
    if isinstance(edge_feats, tuple):
        edge_feats = edge_feats[0]

    node_feats = mace_out["node_feats"]
    context = getattr(matrix_model, "hamiltonian_local_context", None)
    if context is not None:
        context_output = context(
            data=batch,
            edge_vectors=vectors,
            edge_lengths=lengths,
        )
        node_feats = torch.cat([node_feats, context_output.node], dim=-1)
        edge_feats = torch.cat([edge_feats, context_output.edge], dim=-1)

    data_for_readout = copy.copy(batch)
    data_for_readout["edge_attrs"] = edge_attrs
    data_for_readout["edge_feats"] = edge_feats

    readout = matrix_model.matrix_readouts
    if readout.preprocessing_edges is None:
        return data_for_readout, node_feats, None

    preprocessing_out = readout.preprocessing_edges(
        data=data_for_readout,
        node_feats=node_feats,
    )
    if not isinstance(preprocessing_out, tuple):
        return data_for_readout, node_feats, None

    return data_for_readout, node_feats, preprocessing_out[1]


def edge_hidden_irreps_from_readout(readout: Any):
    preprocessing_edges = getattr(readout, "preprocessing_edges", None)
    irreps_out = getattr(preprocessing_edges, "irreps_out", None)
    if isinstance(irreps_out, tuple):
        return irreps_out[1]
    return None


def readout_class_or_none(module: Any):
    return None if module is None else module.__class__


def build_context_model(base_model: torch.nn.Module, context_kwargs: dict[str, Any]):
    from graph2mat.models.mace import MatrixMACE

    readout = base_model.matrix_readouts
    return MatrixMACE(
        mace=copy.deepcopy(base_model.mace),
        readout_per_interaction=False,
        graph2mat_cls=readout.__class__,
        hamiltonian_local_context=True,
        hamiltonian_local_context_kwargs=context_kwargs,
        unique_basis=readout.basis_table.basis,
        edge_hidden_irreps=edge_hidden_irreps_from_readout(readout),
        symmetric=readout.symmetric,
        preprocessing_nodes=readout_class_or_none(readout.preprocessing_nodes),
        preprocessing_edges=readout_class_or_none(readout.preprocessing_edges),
        preprocessing_edges_reuse_nodes=readout.preprocessing_edges_reuse_nodes,
        node_operation=readout.node_operation_cls,
        edge_operation=readout.edge_operation_cls,
        basis_grouping=readout.basis_grouping,
        n_matrix_components=readout.n_matrix_components,
    )


def analyze_readout(
    label: str,
    matrix_model: torch.nn.Module,
    batch: Any,
    dtype: torch.dtype,
    vectorize: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from concrete_joint_readout_rank import analyze_record, operation_records

    _, node_feats, edge_messages = make_readout_inputs(matrix_model, batch)
    if edge_messages is None:
        raise RuntimeError("Expected edge preprocessing to produce edge messages.")

    records = operation_records(
        readout=matrix_model.matrix_readouts,
        batch=batch,
        node_feats=node_feats,
        edge_messages=edge_messages,
    )
    results = []
    distances = []
    for record in records:
        result, rows = analyze_record(record, dtype=dtype, vectorize=vectorize)
        result["model_variant"] = label
        for row in rows:
            row["model_variant"] = label
        results.append(result)
        distances.extend(rows)
    return results, distances


def rank_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows = []
    for row in rows:
        parameter_rank = row["parameter_jacobian"]["rank_abs_tol"]
        input_rank = row["input_jacobian"]["rank_abs_tol"]
        combined_rank = row["combined_parameter_input_jacobian"]["rank_abs_tol"]
        flat_rows.append(
            {
                "model_variant": row["model_variant"],
                "operation": row["operation"],
                "kind": row["kind"],
                "module_key": row["module_key"],
                "joint_output_dim": row["joint_output_dim"],
                "local_input_dim": row["local_input_dim"],
                "parameter_rank_1e-8": parameter_rank["1e-08"],
                "input_rank_1e-8": input_rank["1e-08"],
                "combined_rank_1e-8": combined_rank["1e-08"],
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

    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

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

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data, MatrixDataModule))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        lit_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        lit_model.eval()
        default_model = lit_model.model
        context_model = build_context_model(
            default_model,
            context_kwargs={"num_radial": args.context_num_radial},
        )
        context_model.eval()

        with torch.no_grad():
            default_output = default_model(batch)
            context_output = context_model(batch)

        default_metrics = label_metrics(default_output, batch)
        context_metrics = label_metrics(context_output, batch)
        default_reconstruction_mae = reconstruction_mae(
            default_model, datamodule, batch
        )
        context_reconstruction_mae = reconstruction_mae(
            context_model, datamodule, batch
        )

        default_rank, default_distances = analyze_readout(
            "default",
            default_model,
            batch,
            dtype=dtype,
            vectorize=not args.no_vectorize,
        )
        context_rank, context_distances = analyze_readout(
            "hamiltonian_local_context",
            context_model,
            batch,
            dtype=dtype,
            vectorize=not args.no_vectorize,
        )

    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    rank_rows = default_rank + context_rank
    distance_rows = default_distances + context_distances
    rank_csv = output_dir / "local_context_joint_rank_by_operation.csv"
    feature_csv = output_dir / "local_context_feature_distances.csv"
    json_path = output_dir / "local_context_probe_results.json"

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
        "context_kwargs": {"num_radial": args.context_num_radial},
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "direct_one_sample_label_error": {
            "default": default_metrics,
            "hamiltonian_local_context": context_metrics,
        },
        "h_reconstruction_mae": {
            "default": default_reconstruction_mae,
            "hamiltonian_local_context": context_reconstruction_mae,
        },
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
