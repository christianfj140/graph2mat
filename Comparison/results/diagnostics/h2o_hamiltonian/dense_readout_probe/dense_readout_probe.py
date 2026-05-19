#!/usr/bin/env python3
"""Dense diagnostic readout probe for one-sample H2O H-only overfit.

This script is diagnostic-only. It loads the existing one-sample H-only
checkpoint, swaps the Graph2Mat readout operations for the opt-in dense
diagnostic readouts, keeps the checkpoint MACE features fixed, and tests
whether the dense coefficient head can fit the independent H2O coefficients.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


GRAPH2MAT_ROOT = Path(__file__).resolve().parents[5]
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
FULL_MODEL_DIAGNOSTIC_DIR = Path(__file__).resolve().parents[1] / "full_model_jacobian_rank"
TOLERANCES = (1e-5, 1e-6, 1e-7, 1e-8, 1e-10)


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
        FULL_MODEL_DIAGNOSTIC_DIR,
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


def reference_h_path(source_workspace: Path) -> Path:
    candidates = sorted((source_workspace / "dataset" / "samples").glob("*/siesta.TSHS"))
    if not candidates:
        raise FileNotFoundError(
            f"No reference siesta.TSHS found under {source_workspace / 'dataset' / 'samples'}"
        )
    return candidates[0]


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


def concatenate_coeffs(coeffs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for _, value in sorted(coeffs.items())])


def coefficient_mse(predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [
        torch.mean((predicted[key] - target_value.to(predicted[key])) ** 2)
        for key, target_value in target.items()
    ]
    return torch.stack(losses).mean()


def coefficient_mae(predicted: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [
        torch.mean(torch.abs(predicted[key] - target_value.to(predicted[key])))
        for key, target_value in target.items()
    ]
    return torch.stack(losses).mean()


def dense_input_matrix(inputs: dict[str, Any]) -> torch.Tensor:
    pieces = []
    for value in inputs.values():
        if isinstance(value, tuple):
            pieces.extend(value)
        else:
            pieces.append(value)
    return torch.cat(pieces, dim=-1)


def fit_final_layer_lstsq(
    *,
    operation: torch.nn.Module,
    inputs: dict[str, Any],
    target: torch.Tensor,
) -> dict[str, Any]:
    if not hasattr(operation, "mlp") or len(operation.mlp) != 3:
        return {"status": "skipped", "reason": "operation_is_not_diagnostic_dense_mlp"}

    first = operation.mlp[0]
    activation = operation.mlp[1]
    final = operation.mlp[2]
    x = dense_input_matrix(inputs).detach()
    target = target.detach().to(x)

    with torch.no_grad():
        hidden = activation(first(x))
        design = torch.cat(
            [hidden, torch.ones(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)],
            dim=1,
        )
        solution = torch.linalg.lstsq(
            design.double(),
            target.double(),
        ).solution.to(final.weight)
        final.weight.copy_(solution[:-1].T)
        final.bias.copy_(solution[-1])

    rank = int(torch.linalg.matrix_rank(design.double(), tol=1e-10).item())
    return {
        "status": "fit",
        "design_shape": list(design.shape),
        "design_rank_1e_10": rank,
        "target_shape": list(target.shape),
    }


def maybe_lstsq_initialize_dense_heads(
    *,
    readout: torch.nn.Module,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    target_coeffs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    from diagnose_h2o_coefficient_space_readout import edge_operation_inputs

    results: dict[str, Any] = {}
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    for op_index, operation in enumerate(readout.self_interactions):
        op_key = f"node:{op_index}"
        if operation is None or op_key not in target_coeffs:
            continue
        mask = graph_node_types == op_index
        results[op_key] = fit_final_layer_lstsq(
            operation=operation.operation,
            inputs={"node_feats": node_feats[mask]},
            target=target_coeffs[op_key],
        )

    for module_key, operation in readout.interactions.items():
        op_key = f"edge:{module_key}"
        if operation is None or op_key not in target_coeffs:
            continue
        if getattr(operation, "symm_transpose", False):
            results[op_key] = {
                "status": "skipped",
                "reason": "symm_transpose_uses_forward_backward_block_average",
            }
            continue
        inputs = edge_operation_inputs(readout, batch, node_feats, edge_messages, module_key)
        results[op_key] = fit_final_layer_lstsq(
            operation=operation.operation,
            inputs=inputs,
            target=target_coeffs[op_key],
        )
    return results


def dense_readout_operation_parameters(readout: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for operation in readout.self_interactions:
        if operation is not None:
            params.extend(operation.operation.parameters())
    for operation in readout.interactions.values():
        if operation is not None:
            params.extend(operation.operation.parameters())
    return params


def train_dense_readout_coefficients(
    *,
    readout: torch.nn.Module,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    target_coeffs: dict[str, torch.Tensor],
    schedule: list[float],
    steps_per_stage: int,
) -> dict[str, Any]:
    from diagnose_h2o_coefficient_space_readout import (
        compare_coefficients,
        predicted_coefficients,
    )

    for parameter in readout.parameters():
        parameter.requires_grad_(False)
    params = dense_readout_operation_parameters(readout)
    for parameter in params:
        parameter.requires_grad_(True)

    readout.train()
    node_feats = node_feats.detach()
    edge_messages = edge_messages.detach()
    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_mse = math.inf
    start = time.perf_counter()

    for stage, lr in enumerate(schedule):
        optimizer = torch.optim.Adam(params, lr=lr)
        stage_best_mae = math.inf
        stage_best_mse = math.inf
        for _ in range(steps_per_stage):
            optimizer.zero_grad()
            pred_coeffs = predicted_coefficients(
                readout,
                data_for_readout,
                batch,
                node_feats,
                edge_messages=edge_messages,
            )
            loss = coefficient_mse(pred_coeffs, target_coeffs)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mae_value = float(coefficient_mae(pred_coeffs, target_coeffs).item())
                mse_value = float(loss.item())
            best_mae = min(best_mae, mae_value)
            best_mse = min(best_mse, mse_value)
            stage_best_mae = min(stage_best_mae, mae_value)
            stage_best_mse = min(stage_best_mse, mse_value)

        with torch.no_grad():
            pred_coeffs = predicted_coefficients(
                readout,
                data_for_readout,
                batch,
                node_feats,
                edge_messages=edge_messages,
            )
            history.append(
                {
                    "stage": int(stage),
                    "learning_rate": float(lr),
                    "steps": int(steps_per_stage),
                    "stage_best_mae": float(stage_best_mae),
                    "stage_best_mse": float(stage_best_mse),
                    "stage_final_mae": float(coefficient_mae(pred_coeffs, target_coeffs).item()),
                    "stage_final_mse": float(coefficient_mse(pred_coeffs, target_coeffs).item()),
                }
            )

    with torch.no_grad():
        pred_coeffs = predicted_coefficients(
            readout,
            data_for_readout,
            batch,
            node_feats,
            edge_messages=edge_messages,
        )

    return {
        "schedule": schedule,
        "steps_per_stage": int(steps_per_stage),
        "total_steps": int(steps_per_stage * len(schedule)),
        "runtime_sec": float(time.perf_counter() - start),
        "best_coefficient_mae": float(best_mae),
        "best_coefficient_mse": float(best_mse),
        "final_coefficient_mae": float(coefficient_mae(pred_coeffs, target_coeffs).item()),
        "final_coefficient_mse": float(coefficient_mse(pred_coeffs, target_coeffs).item()),
        "coefficient_errors": compare_coefficients(pred_coeffs, target_coeffs),
        "history": history,
    }


def compare_tensors(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    pred = prediction.detach().cpu().double().reshape(-1)
    ref = reference.detach().cpu().double().reshape(-1)
    result: dict[str, Any] = {
        "prediction_shape": list(prediction.shape),
        "reference_shape": list(reference.shape),
        "shape_match": tuple(prediction.shape) == tuple(reference.shape),
        "n_values": int(ref.numel()),
    }
    if pred.numel() != ref.numel():
        result.update({"mae": None, "rmse": None, "max_abs": None})
        return result
    delta = pred - ref
    abs_delta = delta.abs()
    result.update(
        {
            "mae": float(abs_delta.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
            "max_abs": float(abs_delta.max().item()),
        }
    )
    return result


def h_reconstruction_metrics(
    *,
    processor: Any,
    batch: Any,
    output: dict[str, torch.Tensor],
    reference_h: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    from diagnose_h2o_readout_bottleneck import compare_h, dense_h

    predictions = {
        "node_labels": output["node_labels"].detach().cpu(),
        "edge_labels": output["edge_labels"].detach().cpu(),
    }
    matrices = list(processor.yield_from_batch(batch, predictions=predictions, as_matrix=True))
    if len(matrices) != 1:
        return {"status": "error", "reason": f"expected_one_matrix_got_{len(matrices)}"}
    return compare_h(reference_h, dense_h(matrices[0]), threshold)


def rank_summary(jacobian: torch.Tensor) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(jacobian.detach().cpu().double())
    singular_values = torch.sort(singular_values, descending=True).values
    max_sv = float(singular_values[0].item()) if singular_values.numel() else 0.0
    ranks = {
        f"{tol:.0e}": int((singular_values > tol).sum().item())
        for tol in TOLERANCES
    }
    effective = singular_values[singular_values > 1e-8]
    min_effective = float(effective[-1].item()) if effective.numel() else 0.0
    condition = None if min_effective == 0.0 else max_sv / min_effective
    return {
        "shape": list(jacobian.shape),
        "rank_abs_tol": ranks,
        "singular_value_max": max_sv,
        "singular_value_min_effective_1e_8": min_effective,
        "condition_estimate_1e_8": condition,
        "singular_values_top10": singular_values[:10].tolist(),
        "singular_values_bottom10": singular_values[-10:].tolist(),
    }


def jacobian_wrt_parameters(
    output: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    rows = []
    flat_output = output.reshape(-1)
    for row_index in range(flat_output.numel()):
        grads = torch.autograd.grad(
            flat_output[row_index],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = []
        for param, grad in zip(params, grads):
            if grad is None:
                pieces.append(torch.zeros_like(param).reshape(-1))
            else:
                pieces.append(grad.reshape(-1))
        rows.append(torch.cat(pieces).detach().cpu().double())
    return torch.stack(rows)


def dense_operation_rank_rows(
    *,
    readout: torch.nn.Module,
    data_for_readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    target_coeffs: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    from diagnose_h2o_coefficient_space_readout import operation_coeff_vector

    rows: list[dict[str, Any]] = []
    for op_key in sorted(target_coeffs):
        if op_key.startswith("node:"):
            module = readout.self_interactions[int(op_key.split(":", 1)[1])]
        else:
            module = readout.interactions[op_key.split(":", 1)[1]]
        params = [param for param in module.operation.parameters() if param.requires_grad]
        output = operation_coeff_vector(
            readout,
            data_for_readout,
            batch,
            node_feats,
            edge_messages,
            op_key,
        )
        jacobian = jacobian_wrt_parameters(output, params)
        summary = rank_summary(jacobian)
        output_dim = int(output.numel())
        rows.append(
            {
                "op_key": op_key,
                "operation_class": type(module.operation).__name__,
                "output_dim": output_dim,
                "parameter_count": int(sum(param.numel() for param in params)),
                "jacobian": summary,
                "full_rank_1e_8": bool(summary["rank_abs_tol"]["1e-08"] == output_dim),
                "exact": True,
            }
        )
        print(
            f"[rank] {op_key}: "
            f"{summary['rank_abs_tol']['1e-08']}/{output_dim} @1e-8"
        )
    return rows


def rank_csv_rows(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in rank_rows:
        summary = row["jacobian"]
        ranks = summary["rank_abs_tol"]
        rows.append(
            {
                "op_key": row["op_key"],
                "operation_class": row["operation_class"],
                "output_dim": row["output_dim"],
                "parameter_count": row["parameter_count"],
                "jacobian_shape": summary["shape"],
                "rank_1e-5": ranks["1e-05"],
                "rank_1e-6": ranks["1e-06"],
                "rank_1e-7": ranks["1e-07"],
                "rank_1e-8": ranks["1e-08"],
                "rank_1e-10": ranks["1e-10"],
                "singular_value_max": summary["singular_value_max"],
                "singular_value_min_effective_1e_8": summary[
                    "singular_value_min_effective_1e_8"
                ],
                "condition_estimate_1e_8": summary["condition_estimate_1e_8"],
                "full_rank_1e_8": row["full_rank_1e_8"],
                "exact": row["exact"],
            }
        )
    return rows


def error_csv_rows(coefficient_errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "op_key": row.get("op_key"),
            "status": row.get("status"),
            "prediction_shape": row.get("prediction_shape"),
            "target_shape": row.get("target_shape"),
            "n_coefficients": row.get("n_coefficients"),
            "mae": row.get("mae"),
            "rmse": row.get("rmse"),
            "max_abs": row.get("max_abs"),
        }
        for row in coefficient_errors
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def final_verdict(rank_rows: list[dict[str, Any]], coeff_output_dim: int, h_metrics: dict[str, Any]) -> dict[str, str]:
    rank_total = sum(row["jacobian"]["rank_abs_tol"]["1e-08"] for row in rank_rows)
    if rank_total == coeff_output_dim and h_metrics.get("comparable") and h_metrics.get("mae_eV", 1.0) <= 1e-5:
        return {
            "status": "A_DENSE_READOUT_MEMORIZES",
            "summary": (
                "The dense diagnostic readout reaches full independent coefficient "
                "rank and reconstructs the one-sample Hamiltonian to numerical accuracy."
            ),
        }
    if rank_total == coeff_output_dim:
        return {
            "status": "B_DENSE_READOUT_FULL_RANK_BUT_NOT_OPTIMIZED",
            "summary": (
                "The dense diagnostic readout is full-rank in independent coefficient "
                "space, but this run did not reach numerical-zero H reconstruction."
            ),
        }
    return {
        "status": "C_DENSE_READOUT_STILL_RANK_LIMITED",
        "summary": (
            "The dense diagnostic readout did not reach full independent coefficient "
            f"rank ({rank_total}/{coeff_output_dim})."
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    training = payload["dense_training"]
    h_metrics = payload["h_reconstruction"]
    lines = [
        "# Dense Diagnostic Readout Probe",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']['status']}**: {payload['verdict']['summary']}",
        "",
        "## Readout API",
        "",
        "- node readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock`",
        "- edge readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock`",
        "- default readouts are unchanged; this script passes the dense classes through `node_block_readout` and `edge_block_readout`.",
        "",
        "## Fit Summary",
        "",
        f"- coefficient output dim: `{payload['coefficient_output_dim']}`",
        f"- dense readout operation-rank sum @1e-8: `{payload['dense_readout_rank_sum_1e_8']}`",
        f"- final coefficient MAE: `{training['final_coefficient_mae']}`",
        f"- final coefficient MSE: `{training['final_coefficient_mse']}`",
        f"- node label MAE: `{payload['direct_node']['mae']}`",
        f"- edge label MAE: `{payload['direct_edge']['mae']}`",
        f"- H MAE: `{h_metrics.get('mae_meV')}` meV",
        f"- H RMSE: `{h_metrics.get('rmse_meV')}` meV",
        f"- H max abs: `{h_metrics.get('max_abs_eV')}` eV",
        "",
        "## Exact Dense Readout Rank By Operation",
        "",
        "| op | output dim | params | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | condition @1e-8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rank_csv_rows(payload["dense_rank_by_operation"]):
        condition = row["condition_estimate_1e_8"]
        condition_text = "" if condition is None else f"{condition:.6g}"
        lines.append(
            f"| `{row['op_key']}` | {row['output_dim']} | {row['parameter_count']} | "
            f"{row['rank_1e-5']} | {row['rank_1e-6']} | {row['rank_1e-7']} | "
            f"{row['rank_1e-8']} | {row['rank_1e-10']} | "
            f"{row['singular_value_max']:.6g} | {condition_text} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- old coefficient-space all-trainable rank @1e-8: `{payload['old_full_model_rank_1e_8']}` / `{payload['coefficient_output_dim']}`",
            f"- old fixed-feature readout-block rank @1e-8: `{payload['old_readout_blocks_rank_1e_8']}` / `{payload['coefficient_output_dim']}`",
            (
                "- dense readout rank is computed operation-by-operation with disjoint "
                "operation parameters, so the summed rank is an exact coefficient-space "
                "rank lower bound for all trainable parameters."
            ),
            "",
            "## Training Details",
            "",
            f"- final-layer least-squares initialization: `{payload['lstsq_initialization']}`",
            f"- optimizer schedule: `{training['schedule']}`",
            f"- steps per stage: `{training['steps_per_stage']}`",
            f"- total steps: `{training['total_steps']}`",
            f"- runtime: `{training['runtime_sec']}` seconds",
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- rank CSV: `{payload['output_files']['rank_csv']}`",
            f"- coefficient errors CSV: `{payload['output_files']['errors_csv']}`",
            "",
            "## Reproduction Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
            "## Repository Context",
            "",
            f"- Graph2Mat branch: `{payload['repository_context']['graph2mat_branch']}`",
            f"- Graph2Mat commit: `{payload['repository_context']['graph2mat_commit']}`",
            f"- Pipeline branch: `{payload['repository_context']['pipeline_branch']}`",
            f"- Pipeline commit: `{payload['repository_context']['pipeline_commit']}`",
            "",
            "## Limitations",
            "",
            "- This is a diagnostic, non-equivariant coefficient readout probe.",
            "- MACE node features and edge preprocessing outputs are fixed from the checkpoint during coefficient fitting.",
            "- Full all-parameter dense Jacobian is not materialized because full readout-block rank already proves the all-parameter rank lower bound is full.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--learning-rate-schedule", default="0.01,0.003,0.001,0.0003")
    parser.add_argument("--steps-per-stage", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--skip-lstsq-init", action="store_true")
    args = parser.parse_args()

    pipeline_root = args.pipeline_root.resolve()
    prepare_imports(pipeline_root)

    from diagnose_h2o_coefficient_space_readout import (
        compare_coefficients,
        predicted_coefficients,
        target_coefficients_and_reconstructed_labels,
    )
    from diagnose_h2o_readout_bottleneck import make_readout_inputs
    from evaluate_hamiltonian_metrics import read_matrix
    from full_model_jacobian_rank import build_datamodule_kwargs
    from graph2mat.bindings.e3nn import (
        E3nnDiagnosticDenseEdgeBlock,
        E3nnDiagnosticDenseNodeBlock,
    )
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_workspace = args.source_workspace.resolve()
    training_dir = source_workspace / "training"
    config_path = training_dir / "config.yaml"
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else latest_checkpoint(training_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(config["data"])
    schedule = [float(item) for item in args.learning_rate_schedule.split(",") if item.strip()]
    reference_path = reference_h_path(source_workspace)
    reference_h = np.asarray(read_matrix(reference_path).hamiltonian.toarray(), dtype=float)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data, MatrixDataModule))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        processor = datamodule.data_processor

        dense_lit = LitMACEMatrixModel.load_from_checkpoint(
            str(checkpoint_path),
            node_block_readout=E3nnDiagnosticDenseNodeBlock,
            edge_block_readout=E3nnDiagnosticDenseEdgeBlock,
            strict=False,
        )
        dense_lit.eval()
        dense_model = dense_lit.model
        dense_readout = dense_model.matrix_readouts

        with torch.no_grad():
            data_for_readout, node_feats, edge_messages = make_readout_inputs(dense_lit, batch)
            target_pack = target_coefficients_and_reconstructed_labels(
                dense_readout,
                processor,
                batch,
            )
            target_coeffs = {**target_pack["node_coeffs"], **target_pack["edge_coeffs"]}

        lstsq_results = {}
        if not args.skip_lstsq_init:
            lstsq_results = maybe_lstsq_initialize_dense_heads(
                readout=dense_readout,
                batch=batch,
                node_feats=node_feats,
                edge_messages=edge_messages,
                target_coeffs=target_coeffs,
            )

        dense_training = train_dense_readout_coefficients(
            readout=dense_readout,
            data_for_readout=data_for_readout,
            batch=batch,
            node_feats=node_feats,
            edge_messages=edge_messages,
            target_coeffs=target_coeffs,
            schedule=schedule,
            steps_per_stage=args.steps_per_stage,
        )

        with torch.no_grad():
            pred_coeffs = predicted_coefficients(
                dense_readout,
                data_for_readout,
                batch,
                node_feats,
                edge_messages=edge_messages,
            )
            node_labels, edge_labels = dense_readout(
                data=data_for_readout,
                node_feats=node_feats,
            )
            final_output = {"node_labels": node_labels, "edge_labels": edge_labels}
            direct_node = compare_tensors(node_labels, batch.point_labels)
            direct_edge = compare_tensors(edge_labels, batch.edge_labels)
            h_metrics = h_reconstruction_metrics(
                processor=processor,
                batch=batch,
                output=final_output,
                reference_h=reference_h,
                threshold=args.threshold,
            )
            final_coeff_errors = compare_coefficients(pred_coeffs, target_coeffs)
            coeff_output_dim = int(concatenate_coeffs(target_coeffs).numel())

        dense_rank_rows = dense_operation_rank_rows(
            readout=dense_readout,
            data_for_readout=data_for_readout,
            batch=batch,
            node_feats=node_feats,
            edge_messages=edge_messages,
            target_coeffs=target_coeffs,
        )

    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    old_rank_csv = GRAPH2MAT_ROOT / "Comparison" / "results" / "diagnostics" / "h2o_hamiltonian" / "full_model_jacobian_rank" / "coefficient_space_full_model_rank.csv"
    old_full_model_rank = None
    old_readout_blocks_rank = None
    if old_rank_csv.exists():
        with old_rank_csv.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("parameter_group") == "all_trainable":
                    old_full_model_rank = int(row["rank_1e-8"])
                if row.get("parameter_group") == "readout_blocks":
                    old_readout_blocks_rank = int(row["rank_1e-8"])

    dense_rank_sum = sum(row["jacobian"]["rank_abs_tol"]["1e-08"] for row in dense_rank_rows)
    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "reference_path": str(reference_path),
        "command": command,
        "readout_classes": {
            "node_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock",
            "edge_block_readout": "graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock",
        },
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "coefficient_output_dim": coeff_output_dim,
        "dense_readout_rank_sum_1e_8": dense_rank_sum,
        "old_full_model_rank_1e_8": old_full_model_rank,
        "old_readout_blocks_rank_1e_8": old_readout_blocks_rank,
        "lstsq_initialization": lstsq_results,
        "dense_training": {
            **dense_training,
            "coefficient_errors": final_coeff_errors,
        },
        "direct_node": direct_node,
        "direct_edge": direct_edge,
        "h_reconstruction": h_metrics,
        "dense_rank_by_operation": dense_rank_rows,
        "repository_context": {
            "graph2mat_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "graph2mat_commit": command_output(["git", "rev-parse", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "pipeline_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pipeline_root),
            "pipeline_commit": command_output(["git", "rev-parse", "HEAD"], cwd=pipeline_root),
        },
    }
    payload["verdict"] = final_verdict(dense_rank_rows, coeff_output_dim, h_metrics)

    json_path = output_dir / "dense_readout_probe_results.json"
    rank_csv = output_dir / "dense_readout_rank_by_operation.csv"
    errors_csv = output_dir / "dense_readout_coefficient_errors.csv"
    report_path = output_dir / "report.md"
    payload["output_files"] = {
        "json": str(json_path),
        "rank_csv": str(rank_csv),
        "errors_csv": str(errors_csv),
        "report": str(report_path),
    }

    write_csv(
        rank_csv,
        rank_csv_rows(dense_rank_rows),
        [
            "op_key",
            "operation_class",
            "output_dim",
            "parameter_count",
            "jacobian_shape",
            "rank_1e-5",
            "rank_1e-6",
            "rank_1e-7",
            "rank_1e-8",
            "rank_1e-10",
            "singular_value_max",
            "singular_value_min_effective_1e_8",
            "condition_estimate_1e_8",
            "full_rank_1e_8",
            "exact",
        ],
    )
    write_csv(
        errors_csv,
        error_csv_rows(final_coeff_errors),
        [
            "op_key",
            "status",
            "prediction_shape",
            "target_shape",
            "n_coefficients",
            "mae",
            "rmse",
            "max_abs",
        ],
    )
    json_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    write_report(report_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {rank_csv}")
    print(f"Wrote {errors_csv}")
    print(f"Wrote {report_path}")
    print(f"Verdict: {payload['verdict']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
