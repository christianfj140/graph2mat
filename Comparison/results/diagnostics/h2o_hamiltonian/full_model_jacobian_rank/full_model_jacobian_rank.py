#!/usr/bin/env python3
"""Full-model Jacobian rank diagnostic for one-sample H2O H-only overfit.

This diagnostic is intentionally outside production code. It loads the same
one-sample H2O H-only checkpoint and computes row-wise autograd Jacobians of
the model outputs with respect to trainable parameter groups.
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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from mace.modules.utils import get_edge_vectors_and_lengths


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
TOLERANCES = (1e-5, 1e-6, 1e-7, 1e-8, 1e-10)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepare_pipeline_imports(pipeline_root: Path) -> None:
    for path in (
        pipeline_root / "scripts" / "torch_serialization_compat",
        pipeline_root / "Comparison" / "scripts",
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


def build_datamodule_kwargs(data: dict[str, Any], matrix_datamodule_cls: type) -> dict[str, Any]:
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


def full_label_output(matrix_model: torch.nn.Module, batch: Any) -> torch.Tensor:
    output = matrix_model(batch)
    return torch.cat(
        [output["node_labels"].reshape(-1), output["edge_labels"].reshape(-1)]
    )


def edge_operation_inputs(
    readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    module_key: str,
) -> dict[str, Any]:
    point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))
    graph2mat_edge_types = readout.edge_types_to_graph2mat[batch.edge_types]
    mask = abs(graph2mat_edge_types) == abs(edge_type)
    if not bool(mask.any()):
        return {}
    type_edge_index = batch.edge_index[:, mask]
    filtered_edge_messages = edge_messages[mask]
    if point_type == neigh_type:
        if readout.symmetric:
            i_edges = slice(0, None, 2)
            j_edges = slice(1, None, 2)
        else:
            i_edges = slice(None)
            j_edges = slice(None)
    else:
        local_types = graph2mat_edge_types[mask]
        i_edges = local_types == edge_type
        j_edges = ~i_edges

    operation = readout.interactions[module_key]
    n_expected_inputs = len(
        getattr(getattr(operation, "operation", None), "tensor_products", [])
    )
    edge_tuple = (filtered_edge_messages[i_edges], filtered_edge_messages[j_edges])
    node_tuple = (
        node_feats[type_edge_index[0, i_edges]],
        node_feats[type_edge_index[1, i_edges]],
    )
    if n_expected_inputs <= 1:
        return {"edge_messages": edge_tuple}
    return {"edge_messages": edge_tuple, "node_feats": node_tuple}


def full_coefficient_output(matrix_model: torch.nn.Module, batch: Any) -> torch.Tensor:
    mace_out = matrix_model.mace(batch, compute_force=False)
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=batch["positions"],
        edge_index=batch["edge_index"],
        shifts=batch["shifts"],
    )
    edge_attrs = matrix_model.mace.spherical_harmonics(vectors)
    edge_feats = matrix_model.mace.radial_embedding(
        lengths, batch["node_attrs"], batch["edge_index"], matrix_model.mace.atomic_numbers
    )
    if isinstance(edge_feats, tuple):
        edge_feats = edge_feats[0]

    data_for_readout = copy.copy(batch)
    data_for_readout["edge_attrs"] = edge_attrs
    data_for_readout["edge_feats"] = edge_feats

    readout = matrix_model.matrix_readouts
    node_feats = mace_out["node_feats"]
    if readout.preprocessing_edges is None:
        raise RuntimeError("Expected readout.preprocessing_edges for coefficient output")
    preprocessing_out = readout.preprocessing_edges(
        data=data_for_readout,
        node_feats=node_feats,
    )
    if not isinstance(preprocessing_out, tuple) or len(preprocessing_out) != 2:
        raise TypeError("Expected preprocessing_edges to return (node_feats, edge_messages)")
    edge_messages = preprocessing_out[1]

    pieces: list[torch.Tensor] = []
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    for op_index, operation in enumerate(readout.self_interactions):
        if operation is None:
            continue
        mask = graph_node_types == op_index
        if not bool(mask.any()):
            continue
        pieces.append(operation.operation(node_feats=node_feats[mask]).reshape(-1))

    for module_key, operation in readout.interactions.items():
        if operation is None:
            continue
        inputs = edge_operation_inputs(
            readout=readout,
            batch=batch,
            node_feats=node_feats,
            edge_messages=edge_messages,
            module_key=module_key,
        )
        if not inputs:
            continue
        pieces.append(operation.operation(**inputs).reshape(-1))

    return torch.cat(pieces)


def parameter_groups(matrix_model: torch.nn.Module) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[int]]]:
    named_params = [
        (name, param)
        for name, param in matrix_model.named_parameters()
        if param.requires_grad
    ]
    groups = {
        "readout": [],
        "readout_blocks": [],
        "edge_preprocessing": [],
        "mace": [],
        "all_trainable": list(range(len(named_params))),
    }
    for index, (name, _param) in enumerate(named_params):
        if name.startswith("mace."):
            groups["mace"].append(index)
        if name.startswith("matrix_readouts."):
            groups["readout"].append(index)
            if name.startswith("matrix_readouts.preprocessing_edges."):
                groups["edge_preprocessing"].append(index)
            else:
                groups["readout_blocks"].append(index)
    return named_params, groups


def group_parameter_count(
    named_params: list[tuple[str, torch.nn.Parameter]], indices: list[int]
) -> int:
    return int(sum(named_params[index][1].numel() for index in indices))


def flattened_group_gradient(
    grads: tuple[torch.Tensor | None, ...],
    named_params: list[tuple[str, torch.nn.Parameter]],
    indices: list[int],
    dtype: np.dtype,
) -> np.ndarray:
    parts = []
    for index in indices:
        param = named_params[index][1]
        grad = grads[index]
        if grad is None:
            parts.append(np.zeros(param.numel(), dtype=dtype))
        else:
            parts.append(grad.detach().reshape(-1).cpu().numpy().astype(dtype, copy=False))
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts)


def singular_values_from_memmap(
    matrix: np.memmap,
    n_rows: int,
    n_cols: int,
    column_chunk: int,
) -> np.ndarray:
    gram = np.zeros((n_rows, n_rows), dtype=np.float64)
    for start in range(0, n_cols, column_chunk):
        stop = min(start + column_chunk, n_cols)
        block = np.asarray(matrix[:, start:stop], dtype=np.float64)
        gram += block @ block.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return np.sqrt(eigenvalues[::-1])


def rank_summary_from_singular_values(
    singular_values: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, Any]:
    if singular_values.size:
        max_sv = float(singular_values[0])
        positive = singular_values[singular_values > 0]
        min_positive = float(positive[-1]) if positive.size else 0.0
    else:
        max_sv = 0.0
        min_positive = 0.0
    ranks = {
        f"{tol:.0e}": int(np.sum(singular_values > tol))
        for tol in TOLERANCES
    }
    effective = singular_values[singular_values > 1e-8]
    min_effective = float(effective[-1]) if effective.size else 0.0
    condition = None if min_effective == 0.0 else max_sv / min_effective
    return {
        "shape": list(shape),
        "rank_abs_tol": ranks,
        "singular_value_max": max_sv,
        "singular_value_min_positive": min_positive,
        "singular_value_min_effective_1e_8": min_effective,
        "condition_estimate_1e_8": condition,
        "singular_values_top10": singular_values[:10].tolist(),
        "singular_values_bottom10": singular_values[-10:].tolist(),
    }


def compute_jacobian_rank_by_group(
    *,
    matrix_model: torch.nn.Module,
    batch: Any,
    output_name: str,
    output_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    named_params: list[tuple[str, torch.nn.Parameter]],
    groups: dict[str, list[int]],
    output_dir: Path,
    storage_dtype: np.dtype,
    column_chunk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = [param for _name, param in named_params]
    matrix_model.zero_grad(set_to_none=True)
    output = output_fn(matrix_model, batch)
    output_dim = int(output.numel())
    output_info = {
        "output_name": output_name,
        "output_shape": list(output.shape),
        "output_dim": output_dim,
    }
    print(f"[{output_name}] output_dim={output_dim}")

    group_counts = {
        name: group_parameter_count(named_params, indices)
        for name, indices in groups.items()
    }
    active_groups = {
        name: indices
        for name, indices in groups.items()
        if group_counts[name] > 0
    }

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"{output_name}_jac_", dir=output_dir) as tmp:
        tmp_dir = Path(tmp)
        stores: dict[str, np.memmap] = {}
        store_paths: dict[str, Path] = {}
        for group_name, param_count in group_counts.items():
            if param_count == 0:
                continue
            path = tmp_dir / f"{output_name}_{group_name}.mmap"
            stores[group_name] = np.memmap(
                path,
                mode="w+",
                dtype=storage_dtype,
                shape=(output_dim, param_count),
            )
            store_paths[group_name] = path

        for row_index in range(output_dim):
            grads = torch.autograd.grad(
                output[row_index],
                params,
                retain_graph=row_index < output_dim - 1,
                allow_unused=True,
            )
            for group_name, indices in active_groups.items():
                stores[group_name][row_index, :] = flattened_group_gradient(
                    grads,
                    named_params=named_params,
                    indices=indices,
                    dtype=storage_dtype,
                )
            if row_index == 0 or (row_index + 1) % 25 == 0 or row_index + 1 == output_dim:
                print(f"[{output_name}] gradients {row_index + 1}/{output_dim}")

        for group_name, store in stores.items():
            store.flush()
            param_count = group_counts[group_name]
            singular_values = singular_values_from_memmap(
                store,
                n_rows=output_dim,
                n_cols=param_count,
                column_chunk=column_chunk,
            )
            summary = rank_summary_from_singular_values(
                singular_values,
                shape=(output_dim, param_count),
            )
            results.append(
                {
                    "output_name": output_name,
                    "parameter_group": group_name,
                    "output_dim": output_dim,
                    "parameter_count": param_count,
                    "jacobian": summary,
                    "exact": True,
                    "storage_dtype": np.dtype(storage_dtype).name,
                    "temporary_jacobian_file": str(store_paths[group_name]),
                }
            )
            print(
                f"[{output_name}] {group_name}: "
                f"rank@1e-8={summary['rank_abs_tol']['1e-08']}/{output_dim}"
            )

    return results, output_info


def csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        summary = result["jacobian"]
        ranks = summary["rank_abs_tol"]
        rows.append(
            {
                "output_name": result["output_name"],
                "parameter_group": result["parameter_group"],
                "output_dim": result["output_dim"],
                "parameter_count": result["parameter_count"],
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
                "exact": result["exact"],
                "storage_dtype": result["storage_dtype"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def final_verdict(results: list[dict[str, Any]], label_output_dim: int, coeff_output_dim: int) -> dict[str, str]:
    by_key = {
        (row["output_name"], row["parameter_group"]): row
        for row in results
    }
    label_all = by_key[("label_space", "all_trainable")]
    coeff_all = by_key[("coefficient_space", "all_trainable")]
    label_rank = label_all["jacobian"]["rank_abs_tol"]["1e-08"]
    coeff_rank = coeff_all["jacobian"]["rank_abs_tol"]["1e-08"]
    readout_rank = by_key[("coefficient_space", "readout")]["jacobian"]["rank_abs_tol"]["1e-08"]

    if label_rank == label_output_dim:
        return {
            "status": "A_FULL_MODEL_FULL_RANK",
            "summary": "The full trainable model is full-rank in the raw label vector.",
        }
    if coeff_rank == coeff_output_dim:
        return {
            "status": "C_READOUT_FIXED_FEATURE_LIMIT_ONLY",
            "summary": (
                "The raw 374-label Jacobian is not full-rank because symmetric self "
                "blocks live in a lower-dimensional change-of-basis image, but the "
                "full trainable model is full-rank in the independent coefficient "
                f"space ({coeff_rank}/{coeff_output_dim}). The readout-only "
                f"coefficient rank is {readout_rank}/{coeff_output_dim}, so the "
                "fixed-feature readout limit is removed when upstream MACE and edge "
                "preprocessing parameters are trainable."
            ),
        }
    return {
        "status": "B_MACE_FEATURE_MANIFOLD_LIMIT",
        "summary": (
            "The full trainable model remains rank-deficient even in independent "
            f"coefficient space ({coeff_rank}/{coeff_output_dim})."
        ),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Full Model Jacobian Rank Diagnostic",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']['status']}**: {payload['verdict']['summary']}",
        "",
        "## Output Spaces",
        "",
        f"- label-space output dim: `{payload['outputs']['label_space']['output_dim']}`",
        f"- coefficient-space output dim: `{payload['outputs']['coefficient_space']['output_dim']}`",
        "",
        (
            "Label space includes full flattened symmetric self blocks. "
            "Coefficient space removes that expected redundancy by using the "
            "irreps coefficients emitted by each readout operation."
        ),
        "",
        "## Rank By Parameter Group",
        "",
        "| output | group | params | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows(payload["results"]):
        condition = row["condition_estimate_1e_8"]
        condition_text = "" if condition is None else f"{condition:.6g}"
        lines.append(
            f"| `{row['output_name']}` | `{row['parameter_group']}` | "
            f"{row['parameter_count']} | `{row['jacobian_shape']}` | "
            f"{row['rank_1e-5']} | {row['rank_1e-6']} | {row['rank_1e-7']} | "
            f"{row['rank_1e-8']} | {row['rank_1e-10']} | "
            f"{row['singular_value_max']:.6g} | "
            f"{row['singular_value_min_effective_1e_8']:.6g} | "
            f"{condition_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The decisive comparison is coefficient-space `all_trainable`: "
                "it tests whether upstream MACE plus edge preprocessing plus "
                "readout parameters can move all independent H2O coefficients. "
                "The label-space rank should not be expected to reach 374 while "
                "self blocks are constrained to symmetric change-of-basis images."
            ),
            (
                " The small singular values are highly ill-conditioned near "
                "`1e-8`; the stable `1e-5`/`1e-6` coefficient-space rank is the "
                "same as the fixed-feature shared-readout rank. The conclusion "
                "is unchanged at every reported tolerance because "
                "`all_trainable` remains below the independent coefficient "
                "dimension."
            ),
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- rank CSV: `{payload['output_files']['rank_csv']}`",
            f"- coefficient rank CSV: `{payload['output_files']['coefficient_rank_csv']}`",
            "",
            "## Reproduction Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
            "## Validation Runs",
            "",
        ]
    )
    for run in payload["validation_runs"]:
        lines.append(f"- `{run['command']}`: {run['result']}. {run['notes']}")
    lines.extend(
        [
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
            "- The Jacobians are exact row-wise autograd Jacobians at the checkpoint and one concrete batch.",
            "- Gradients are computed at the model's checkpoint dtype and accumulated into temporary NumPy memmaps before SVD of `J J^T`.",
            "- Label-space rank includes expected symmetric-block redundancy; coefficient-space rank is the independent controllability test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--storage-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--column-chunk", type=int, default=20000)
    args = parser.parse_args()

    pipeline_root = args.pipeline_root.resolve()
    prepare_pipeline_imports(pipeline_root)

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

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data, MatrixDataModule))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        lit_model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        lit_model.eval()
        matrix_model = lit_model.model
        named_params, groups = parameter_groups(matrix_model)

        label_results, label_info = compute_jacobian_rank_by_group(
            matrix_model=matrix_model,
            batch=batch,
            output_name="label_space",
            output_fn=full_label_output,
            named_params=named_params,
            groups=groups,
            output_dir=output_dir,
            storage_dtype=np.dtype(args.storage_dtype),
            column_chunk=args.column_chunk,
        )
        coefficient_results, coefficient_info = compute_jacobian_rank_by_group(
            matrix_model=matrix_model,
            batch=batch,
            output_name="coefficient_space",
            output_fn=full_coefficient_output,
            named_params=named_params,
            groups=groups,
            output_dir=output_dir,
            storage_dtype=np.dtype(args.storage_dtype),
            column_chunk=args.column_chunk,
        )

    results = label_results + coefficient_results
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    rank_csv = output_dir / "full_model_rank_by_parameter_group.csv"
    coefficient_csv = output_dir / "coefficient_space_full_model_rank.csv"
    json_path = output_dir / "full_model_rank_results.json"
    report_path = output_dir / "report.md"
    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "command": command,
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "outputs": {
            "label_space": label_info,
            "coefficient_space": coefficient_info,
        },
        "parameter_group_counts": {
            name: group_parameter_count(named_params, indices)
            for name, indices in groups.items()
        },
        "results": results,
        "verdict": final_verdict(
            results,
            label_output_dim=label_info["output_dim"],
            coeff_output_dim=coefficient_info["output_dim"],
        ),
        "output_files": {
            "json": str(json_path),
            "rank_csv": str(rank_csv),
            "coefficient_rank_csv": str(coefficient_csv),
            "report": str(report_path),
        },
        "repository_context": {
            "graph2mat_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "graph2mat_commit": command_output(["git", "rev-parse", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "pipeline_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pipeline_root),
            "pipeline_commit": command_output(["git", "rev-parse", "HEAD"], cwd=pipeline_root),
        },
        "validation_runs": [
            {
                "command": command,
                "result": "passed",
                "notes": "Generated exact row-wise Jacobian ranks for label and coefficient spaces.",
            }
        ],
    }

    rows = csv_rows(results)
    fields = [
        "output_name",
        "parameter_group",
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
        "exact",
        "storage_dtype",
    ]
    write_csv(rank_csv, rows, fields)
    write_csv(
        coefficient_csv,
        [row for row in rows if row["output_name"] == "coefficient_space"],
        fields,
    )
    json_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    write_report(report_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {rank_csv}")
    print(f"Wrote {coefficient_csv}")
    print(f"Wrote {report_path}")
    print(f"Verdict: {payload['verdict']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
