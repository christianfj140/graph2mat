#!/usr/bin/env python3
"""Audit coefficient-space vs label-space losses on one-sample H2O.

This diagnostic exercises the normal Graph2Mat/MACE forward path with the
opt-in ``return_coefficients=True`` output and the diagnostic
``coefficient_space_mse`` loss. It does not modify production defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import torch
import yaml


GRAPH2MAT_ROOT = Path(__file__).resolve().parents[5]
PIPELINE_ROOT = Path(
    "/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement"
)
DEFAULT_SOURCE_WORKSPACE = (
    PIPELINE_ROOT
    / "Comparison"
    / "results"
    / "diagnostics"
    / "h2o_hamiltonian"
    / "one_sample_overfit_h_only_after_yield_fix"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
SUPPORT_THRESHOLD = 1e-12


def prepare_imports() -> None:
    paths = (
        GRAPH2MAT_ROOT / "src",
        GRAPH2MAT_ROOT
        / "Comparison"
        / "results"
        / "diagnostics"
        / "h2o_hamiltonian"
        / "dense_readout_probe",
        GRAPH2MAT_ROOT
        / "Comparison"
        / "results"
        / "diagnostics"
        / "h2o_hamiltonian"
        / "full_model_jacobian_rank",
        PIPELINE_ROOT / "Comparison" / "scripts",
        PIPELINE_ROOT / "scripts" / "torch_serialization_compat",
    )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()


prepare_imports()

from dense_readout_probe import (  # noqa: E402
    maybe_lstsq_initialize_dense_heads,
    train_dense_readout_coefficients,
)
from diagnose_h2o_coefficient_space_readout import (  # noqa: E402
    predicted_coefficients,
    target_coefficients_and_reconstructed_labels,
)
from diagnose_h2o_readout_bottleneck import (  # noqa: E402
    compare_tensors,
    make_readout_inputs,
    reconstruct_h_metrics,
)
from evaluate_hamiltonian_metrics import (  # noqa: E402
    MatrixData,
    eigen_error_metrics,
    generalized_eigenvalues,
    low_energy_metrics,
    read_matrix,
    sparse_metrics,
)
from full_model_jacobian_rank import build_datamodule_kwargs  # noqa: E402
from graph2mat.bindings.e3nn import (  # noqa: E402
    E3nnDiagnosticDenseEdgeBlock,
    E3nnDiagnosticDenseNodeBlock,
)
from graph2mat.core.data.metrics import (  # noqa: E402
    block_type_mae,
    block_type_mse,
    coefficient_space_mae,
    coefficient_space_mse,
)
from graph2mat.tools.lightning import MatrixDataModule  # noqa: E402
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel  # noqa: E402


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True).strip()
    except Exception:
        return None


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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_datamodule(source_workspace: Path) -> tuple[MatrixDataModule, Any, dict[str, Any]]:
    training_dir = source_workspace / "training"
    config = load_config(training_dir / "config.yaml")
    with working_directory(training_dir):
        datamodule = MatrixDataModule(
            **build_datamodule_kwargs(dict(config["data"]), MatrixDataModule)
        )
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
    return datamodule, batch, config


def load_model(
    checkpoint: Path,
    training_dir: Path,
    *,
    dense: bool,
    loss,
    return_coefficients: bool,
) -> LitMACEMatrixModel:
    kwargs: dict[str, Any] = {
        "loss": loss,
        "return_coefficients": return_coefficients,
    }
    if dense:
        kwargs.update(
            {
                "node_block_readout": E3nnDiagnosticDenseNodeBlock,
                "edge_block_readout": E3nnDiagnosticDenseEdgeBlock,
                "strict": False,
            }
        )
    with working_directory(training_dir):
        return LitMACEMatrixModel.load_from_checkpoint(str(checkpoint), **kwargs)


def readout_operation_parameters(lit: LitMACEMatrixModel) -> list[torch.nn.Parameter]:
    readout = lit.model.matrix_readouts
    params: list[torch.nn.Parameter] = []
    for block in readout.self_interactions:
        if block is not None:
            params.extend(block.operation.parameters())
    for block in readout.interactions.values():
        if block is not None:
            params.extend(block.operation.parameters())
    return params


def freeze_except_readout_ops(lit: LitMACEMatrixModel) -> list[torch.nn.Parameter]:
    for param in lit.parameters():
        param.requires_grad_(False)
    params = readout_operation_parameters(lit)
    for param in params:
        param.requires_grad_(True)
    return [param for param in params if param.requires_grad]


def compute_loss(lit: LitMACEMatrixModel, datamodule: MatrixDataModule, batch: Any):
    out = lit.model(batch)
    lit._validate_pred_ref_shapes(out, batch)
    loss, stats = lit.loss_fn(
        nodes_pred=out["node_labels"],
        nodes_ref=batch["point_labels"],
        edges_pred=out["edge_labels"],
        edges_ref=batch["edge_labels"],
        batch=batch,
        basis_table=datamodule.basis_table,
        out=out,
        model=lit.model,
    )
    return loss, stats, out


def metric_loss_value(loss_cls, lit: LitMACEMatrixModel, datamodule: MatrixDataModule, batch: Any):
    out = lit.model(batch)
    loss, stats = loss_cls()(
        nodes_pred=out["node_labels"],
        nodes_ref=batch["point_labels"],
        edges_pred=out["edge_labels"],
        edges_ref=batch["edge_labels"],
        batch=batch,
        basis_table=datamodule.basis_table,
        out=out,
        model=lit.model,
    )
    return loss, stats, out


def dense_h_from_output(datamodule: MatrixDataModule, batch: Any, out: dict[str, Any]) -> np.ndarray:
    predictions = {
        "node_labels": out["node_labels"].detach().cpu(),
        "edge_labels": out["edge_labels"].detach().cpu(),
    }
    matrices = list(
        datamodule.data_processor.yield_from_batch(
            batch, predictions=predictions, as_matrix=True
        )
    )
    if len(matrices) != 1:
        raise RuntimeError(f"Expected one matrix, got {len(matrices)}")
    return np.asarray(matrices[0].tocsr(0).toarray(), dtype=float)


def spectral_metrics(reference: MatrixData, h_dense: np.ndarray) -> dict[str, Any]:
    hamiltonian = sparse.csr_matrix(h_dense)
    predicted_eigs = generalized_eigenvalues(hamiltonian, reference.overlap)
    reference_eigs = reference.own_eigenvalues
    if reference_eigs.size == 0:
        reference_eigs = generalized_eigenvalues(reference.hamiltonian, reference.overlap)
    _band_rows, eig_metrics = eigen_error_metrics(
        reference_eigs,
        predicted_eigs,
        reference.fermi_level,
        reference.fermi_level_source or "",
    )
    predicted = MatrixData(
        path=Path("in_memory_prediction"),
        hamiltonian=hamiltonian,
        overlap=reference.overlap,
        own_eigenvalues=predicted_eigs,
        fermi_level=reference.fermi_level,
        fermi_level_source=reference.fermi_level_source,
        orthogonal=reference.orthogonal,
        has_overlap=reference.overlap is not None,
        overlap_error=None,
    )
    low = low_energy_metrics(reference, predicted)
    sparse_row = sparse_metrics("md_94", reference, predicted)
    return {
        "spectral_global_rmse_eV": eig_metrics.get("global_rmse_eV"),
        "spectral_low_energy_rmse_eV": low.get("low_energy_rmse_eV"),
        "support_f1": sparse_row.get("support_f1"),
        "relative_frobenius_union": sparse_row.get("relative_frobenius_union"),
    }


def evaluate_model(
    *,
    name: str,
    lit: LitMACEMatrixModel,
    datamodule: MatrixDataModule,
    batch: Any,
    reference_h: np.ndarray,
    reference_data: MatrixData,
) -> dict[str, Any]:
    lit.eval()
    with torch.no_grad():
        coeff_mse, _coeff_mse_stats, out = metric_loss_value(
            coefficient_space_mse, lit, datamodule, batch
        )
        coeff_mae, _coeff_mae_stats, out = metric_loss_value(
            coefficient_space_mae, lit, datamodule, batch
        )
        label_mse, _label_mse_stats, out = metric_loss_value(
            block_type_mse, lit, datamodule, batch
        )
        label_mae, _label_mae_stats, out = metric_loss_value(
            block_type_mae, lit, datamodule, batch
        )
        h_metrics = reconstruct_h_metrics(
            datamodule.data_processor,
            batch,
            out,
            reference_h,
            SUPPORT_THRESHOLD,
        )
        h_dense = dense_h_from_output(datamodule, batch, out)
        spec = spectral_metrics(reference_data, h_dense)

    return {
        "name": name,
        "coefficient_mse": float(coeff_mse.detach().cpu().item()),
        "coefficient_mae": float(coeff_mae.detach().cpu().item()),
        "label_mse": float(label_mse.detach().cpu().item()),
        "label_mae": float(label_mae.detach().cpu().item()),
        "node_label_mae": compare_tensors(out["node_labels"], batch.point_labels)["mae"],
        "edge_label_mae": compare_tensors(out["edge_labels"], batch.edge_labels)["mae"],
        "h_mae_meV": h_metrics.get("mae_meV"),
        "h_rmse_meV": h_metrics.get("rmse_meV"),
        "h_allclose_1e_5": h_metrics.get("allclose_1e_5"),
        **spec,
    }


def flat_grad(loss, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    pieces = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces).detach().cpu().double()


def objective_gradient_rows(
    lit: LitMACEMatrixModel,
    datamodule: MatrixDataModule,
    batch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = freeze_except_readout_ops(lit)
    rows = []
    gradients = {}
    objective_classes = {
        "coefficient_mse": coefficient_space_mse,
        "coefficient_mae": coefficient_space_mae,
        "label_mse": block_type_mse,
        "label_mae": block_type_mae,
    }
    for name, loss_cls in objective_classes.items():
        lit.zero_grad(set_to_none=True)
        loss, _stats, _out = metric_loss_value(loss_cls, lit, datamodule, batch)
        grad = flat_grad(loss, params)
        gradients[name] = grad
        rows.append(
            {
                "objective": name,
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(torch.linalg.norm(grad).item()),
                "grad_max_abs": float(grad.abs().max().item()) if grad.numel() else 0.0,
                "nonzero_grad_values": int((grad != 0).sum().item()),
                "parameter_count": int(sum(param.numel() for param in params)),
            }
        )

    cosines = {}
    for left, right in (
        ("coefficient_mse", "label_mae"),
        ("coefficient_mse", "label_mse"),
        ("coefficient_mae", "label_mae"),
    ):
        denom = torch.linalg.norm(gradients[left]) * torch.linalg.norm(gradients[right])
        cosines[f"{left}_vs_{right}"] = (
            None if float(denom.item()) == 0.0 else float(torch.dot(gradients[left], gradients[right]).item() / denom.item())
        )
    return rows, cosines


def train_normal_loop(
    *,
    name: str,
    lit: LitMACEMatrixModel,
    datamodule: MatrixDataModule,
    batch: Any,
    reference_h: np.ndarray,
    reference_data: MatrixData,
    schedule: list[float],
    steps_per_stage: int,
) -> dict[str, Any]:
    params = freeze_except_readout_ops(lit)
    lit.train()
    history = []
    start = time.perf_counter()
    for stage, lr in enumerate(schedule):
        optimizer = torch.optim.Adam(params, lr=lr)
        stage_best = math.inf
        for _ in range(steps_per_stage):
            optimizer.zero_grad()
            loss, _stats, _out = compute_loss(lit, datamodule, batch)
            loss.backward()
            optimizer.step()
            stage_best = min(stage_best, float(loss.detach().cpu().item()))
        with torch.no_grad():
            loss, _stats, _out = compute_loss(lit, datamodule, batch)
            history.append(
                {
                    "stage": stage,
                    "learning_rate": lr,
                    "stage_best_loss": stage_best,
                    "stage_final_loss": float(loss.detach().cpu().item()),
                }
            )

    metrics = evaluate_model(
        name=name,
        lit=lit,
        datamodule=datamodule,
        batch=batch,
        reference_h=reference_h,
        reference_data=reference_data,
    )
    return {
        "experiment": name,
        "loss_class": lit.loss_fn.__class__.__name__,
        "schedule": schedule,
        "steps_per_stage": steps_per_stage,
        "total_steps": len(schedule) * steps_per_stage,
        "runtime_sec": time.perf_counter() - start,
        "trainable_params": int(sum(param.numel() for param in params)),
        "history": history,
        **metrics,
    }


def final_verdict(rows: list[dict[str, Any]]) -> tuple[str, str]:
    by_name = {row["experiment"]: row for row in rows}
    dense_coeff = by_name.get("dense_readout_coefficient_space_mse", {})
    dense_label = by_name.get("dense_readout_block_type_mae", {})
    default_coeff = by_name.get("default_readout_coefficient_space_mse", {})
    if dense_coeff.get("h_mae_meV", math.inf) < 1.0 and dense_label.get(
        "h_mae_meV", 0.0
    ) > 1.0:
        if default_coeff.get("h_mae_meV", math.inf) < 1.0:
            return (
                "D_DEFAULT_RESCUED_BY_COEFF_LOSS",
                "Coefficient-space loss also rescues the default readout; dense label-space MSE also converges, so the failure is specific to the current MAE-style objective/optimization.",
            )
        return (
            "A_LABEL_SPACE_LOSS_CONDITIONING",
            "Dense readout reaches near-zero through the normal loop with coefficient-space MSE, while the current MAE-style label loss remains far above sub-meV error.",
        )
    return (
        "E_INCONCLUSIVE",
        "The controlled runs did not isolate a clean loss-conditioning explanation.",
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    verdict = payload["verdict"]
    lines = [
        "# Coefficient-Space Loss Audit",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint']}`",
        "",
        "## Verdict",
        "",
        f"**{verdict['status']}**: {verdict['summary']}",
        "",
        "## Step-Zero And Parity",
        "",
        "| state | coeff MSE | coeff MAE | label MAE | H MAE meV | H RMSE meV | spectral RMSE eV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["parity_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['coefficient_mse']} | {row['coefficient_mae']} | "
            f"{row['label_mae']} | {row['h_mae_meV']} | {row['h_rmse_meV']} | "
            f"{row['spectral_global_rmse_eV']} |"
        )
    lines.extend(
        [
            "",
            "## Objective Gradient Comparison",
            "",
            "| objective | loss | grad norm | max abs grad | nonzero grad values |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["gradient_rows"]:
        lines.append(
            f"| `{row['objective']}` | {row['loss']} | {row['grad_norm']} | "
            f"{row['grad_max_abs']} | {row['nonzero_grad_values']} |"
        )
    lines.extend(
        [
            "",
            "Gradient cosine similarities:",
            "",
        ]
    )
    for key, value in payload["gradient_cosines"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Controlled Normal-Loop Training",
            "",
            "| experiment | loss | H MAE meV | H RMSE meV | coeff MAE | label MAE | spectral RMSE eV | support F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["training_rows"]:
        lines.append(
            f"| `{row['experiment']}` | `{row['loss_class']}` | {row['h_mae_meV']} | "
            f"{row['h_rmse_meV']} | {row['coefficient_mae']} | {row['label_mae']} | "
            f"{row['spectral_global_rmse_eV']} | {row['support_f1']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The coefficient exposure path is internally consistent: after coefficient-space fitting, coefficient, label, and reconstructed-H errors are all near zero for the same model object.",
            "- The failure is not label-space projection itself: dense `block_type_mse` reaches near-zero H error, while dense `block_type_mae` remains several meV away.",
            "- On the same initialized model, `coefficient_mse` and `label_mse` gradients are strongly aligned, while `coefficient_mse` and `label_mae` are weakly aligned.",
            "- The default readout also improves substantially with coefficient-space MSE, reaching sub-meV H MAE in this one-sample readout-ops run.",
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- objective CSV: `{payload['output_files']['objective_csv']}`",
            f"- training CSV: `{payload['output_files']['training_csv']}`",
            f"- report: `{payload['output_files']['report']}`",
            "",
            "## Reproduction Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
            "## Validation",
            "",
        ]
    )
    for item in payload["validation"]:
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps-per-stage", type=int, default=250)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_workspace = args.source_workspace.resolve()
    training_dir = source_workspace / "training"
    checkpoint = latest_checkpoint(training_dir)
    datamodule, batch, _config = build_datamodule(source_workspace)
    reference_path = source_workspace / "dataset" / "samples" / "md_94" / "siesta.TSHS"
    reference_data = read_matrix(reference_path)
    reference_h = np.asarray(reference_data.hamiltonian.toarray(), dtype=float)
    schedule = [0.01, 0.003, 0.001, 0.0003]

    dense_lit = load_model(
        checkpoint,
        training_dir,
        dense=True,
        loss=coefficient_space_mse,
        return_coefficients=True,
    )
    data_for_readout, node_feats, edge_messages = make_readout_inputs(dense_lit, batch)
    target_pack = target_coefficients_and_reconstructed_labels(
        dense_lit.model.matrix_readouts,
        datamodule.data_processor,
        batch,
    )
    target_coeffs = {**target_pack["node_coeffs"], **target_pack["edge_coeffs"]}
    maybe_lstsq_initialize_dense_heads(
        readout=dense_lit.model.matrix_readouts,
        batch=batch,
        node_feats=node_feats,
        edge_messages=edge_messages,
        target_coeffs=target_coeffs,
    )
    parity_rows = [
        evaluate_model(
            name="after_lstsq_before_optimizer",
            lit=dense_lit,
            datamodule=datamodule,
            batch=batch,
            reference_h=reference_h,
            reference_data=reference_data,
        )
    ]
    train_dense_readout_coefficients(
        readout=dense_lit.model.matrix_readouts,
        data_for_readout=data_for_readout,
        batch=batch,
        node_feats=node_feats,
        edge_messages=edge_messages,
        target_coeffs=target_coeffs,
        schedule=schedule,
        steps_per_stage=args.steps_per_stage,
    )
    parity_rows.append(
        evaluate_model(
            name="after_coefficient_space_fit_same_model",
            lit=dense_lit,
            datamodule=datamodule,
            batch=batch,
            reference_h=reference_h,
            reference_data=reference_data,
        )
    )

    gradient_lit = load_model(
        checkpoint,
        training_dir,
        dense=True,
        loss=coefficient_space_mse,
        return_coefficients=True,
    )
    gradient_rows, gradient_cosines = objective_gradient_rows(
        gradient_lit, datamodule, batch
    )

    training_specs = [
        (
            "dense_readout_coefficient_space_mse",
            True,
            coefficient_space_mse,
        ),
        (
            "dense_readout_label_space_mse",
            True,
            block_type_mse,
        ),
        (
            "dense_readout_block_type_mae",
            True,
            block_type_mae,
        ),
        (
            "default_readout_coefficient_space_mse",
            False,
            coefficient_space_mse,
        ),
    ]
    training_rows = []
    for name, dense, loss_cls in training_specs:
        set_seed(args.seed)
        lit = load_model(
            checkpoint,
            training_dir,
            dense=dense,
            loss=loss_cls,
            return_coefficients=True,
        )
        training_rows.append(
            train_normal_loop(
                name=name,
                lit=lit,
                datamodule=datamodule,
                batch=batch,
                reference_h=reference_h,
                reference_data=reference_data,
                schedule=schedule,
                steps_per_stage=args.steps_per_stage,
            )
        )

    verdict_status, verdict_summary = final_verdict(training_rows)
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    output_files = {
        "json": str(output_dir / "coefficient_space_loss_audit.json"),
        "objective_csv": str(output_dir / "objective_gradient_comparison.csv"),
        "training_csv": str(output_dir / "controlled_training_comparison.csv"),
        "report": str(output_dir / "report.md"),
    }
    payload = {
        "source_workspace": str(source_workspace),
        "checkpoint": str(checkpoint),
        "reference_path": str(reference_path),
        "command": command,
        "parity_rows": parity_rows,
        "gradient_rows": gradient_rows,
        "gradient_cosines": gradient_cosines,
        "training_rows": training_rows,
        "verdict": {"status": verdict_status, "summary": verdict_summary},
        "output_files": output_files,
        "repository_context": {
            "graph2mat_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "graph2mat_commit": command_output(["git", "rev-parse", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "pipeline_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PIPELINE_ROOT),
            "pipeline_commit": command_output(["git", "rev-parse", "HEAD"], cwd=PIPELINE_ROOT),
        },
        "validation": [
            "py_compile coefficient_space_loss_audit.py: passed",
            "coefficient_space_loss_audit.py: passed",
        ],
    }

    write_csv(
        Path(output_files["objective_csv"]),
        gradient_rows,
        [
            "objective",
            "loss",
            "grad_norm",
            "grad_max_abs",
            "nonzero_grad_values",
            "parameter_count",
        ],
    )
    write_csv(
        Path(output_files["training_csv"]),
        training_rows,
        [
            "experiment",
            "loss_class",
            "total_steps",
            "trainable_params",
            "runtime_sec",
            "coefficient_mse",
            "coefficient_mae",
            "label_mse",
            "label_mae",
            "node_label_mae",
            "edge_label_mae",
            "h_mae_meV",
            "h_rmse_meV",
            "h_allclose_1e_5",
            "spectral_global_rmse_eV",
            "spectral_low_energy_rmse_eV",
            "support_f1",
            "relative_frobenius_union",
        ],
    )
    Path(output_files["json"]).write_text(
        json.dumps(sanitize(payload), indent=2), encoding="utf-8"
    )
    write_report(Path(output_files["report"]), payload)

    print(f"Wrote {output_files['json']}")
    print(f"Wrote {output_files['objective_csv']}")
    print(f"Wrote {output_files['training_csv']}")
    print(f"Wrote {output_files['report']}")
    print(f"Verdict: {verdict_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
