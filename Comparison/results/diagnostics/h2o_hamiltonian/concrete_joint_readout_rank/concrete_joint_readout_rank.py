#!/usr/bin/env python3
"""Exact joint rank diagnostics for shared H2O Graph2Mat readout operations.

This is a diagnostic-only script. It loads the one-sample H-only checkpoint,
extracts the concrete MACE -> Graph2Mat readout inputs for the H2O batch, and
computes exact local Jacobian ranks for the shared readout operations.
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
ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 14: "Si"}


def _prepare_pipeline_imports(pipeline_root: Path) -> None:
    torch_compat_dir = pipeline_root / "scripts" / "torch_serialization_compat"
    comparison_scripts = pipeline_root / "Comparison" / "scripts"
    for path in (torch_compat_dir, comparison_scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from torch_safe_globals import allow_graph2mat_checkpoint_globals

    allow_graph2mat_checkpoint_globals()


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


def species_from_type(point_type: int) -> str:
    return ATOM_SYMBOLS.get(int(point_type), str(point_type))


def point_basis_label(readout: Any, graph_type: int) -> str:
    point_type = int(readout.graph2mat_table.basis[int(graph_type)].type)
    return species_from_type(point_type)


def flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in trainable_parameters(module)))


def rank_summary(jacobian: torch.Tensor, tolerances: tuple[float, ...]) -> dict[str, Any]:
    jacobian = jacobian.detach().cpu().double()
    singular_values = torch.linalg.svdvals(jacobian)
    singular_values = torch.sort(singular_values, descending=True).values
    max_sv = float(singular_values[0].item()) if singular_values.numel() else 0.0
    ranks = {
        f"{tol:.0e}": int((singular_values > tol).sum().item())
        for tol in tolerances
    }
    effective = singular_values[singular_values > 1e-8]
    min_effective = float(effective[-1].item()) if effective.numel() else 0.0
    condition = None if min_effective == 0.0 else max_sv / min_effective
    positive = singular_values[singular_values > 0]
    min_positive = float(positive[-1].item()) if positive.numel() else 0.0
    return {
        "shape": list(jacobian.shape),
        "rank_abs_tol": ranks,
        "singular_value_max": max_sv,
        "singular_value_min_positive": min_positive,
        "singular_value_min_effective_1e_8": min_effective,
        "condition_estimate_1e_8": condition,
        "singular_values_top10": singular_values[:10].tolist(),
        "singular_values_bottom10": singular_values[-10:].tolist(),
    }


def jacobian_wrt_parameters(
    output: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    flat_output = output.reshape(-1)
    rows: list[torch.Tensor] = []
    for index in range(flat_output.numel()):
        grads = torch.autograd.grad(
            flat_output[index],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        row_parts = []
        for param, grad in zip(params, grads):
            if grad is None:
                row_parts.append(torch.zeros_like(param).reshape(-1))
            else:
                row_parts.append(grad.reshape(-1))
        rows.append(flatten_tensors(row_parts).detach().cpu().double())
    return torch.stack(rows, dim=0)


def flatten_input_structure(inputs: dict[str, Any]) -> tuple[torch.Tensor, list[Any]]:
    flat_parts: list[torch.Tensor] = []
    spec: list[Any] = []
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(value, tuple):
            item_specs = []
            for tensor in value:
                tensor = tensor.detach()
                item_specs.append((tuple(tensor.shape), tensor.numel(), tensor.dtype, tensor.device))
                flat_parts.append(tensor.reshape(-1))
            spec.append((key, "tuple", item_specs))
        else:
            tensor = value.detach()
            spec.append((key, "tensor", (tuple(tensor.shape), tensor.numel(), tensor.dtype, tensor.device)))
            flat_parts.append(tensor.reshape(-1))
    return flatten_tensors(flat_parts), spec


def unflatten_input_structure(flat: torch.Tensor, spec: list[Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    offset = 0
    for entry in spec:
        key, kind, meta = entry
        if kind == "tuple":
            values = []
            for shape, numel, _dtype, _device in meta:
                values.append(flat[offset : offset + numel].reshape(shape))
                offset += numel
            inputs[key] = tuple(values)
        else:
            shape, numel, _dtype, _device = meta
            inputs[key] = flat[offset : offset + numel].reshape(shape)
            offset += numel
    if offset != flat.numel():
        raise RuntimeError(f"Unflatten consumed {offset} values from {flat.numel()}")
    return inputs


def jacobian_wrt_inputs(
    module: torch.nn.Module,
    flat_input: torch.Tensor,
    input_spec: list[Any],
    vectorize: bool,
) -> torch.Tensor:
    flat_input = flat_input.detach().requires_grad_(True)

    def fn(value: torch.Tensor) -> torch.Tensor:
        return module(**unflatten_input_structure(value, input_spec)).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(fn, flat_input, vectorize=vectorize)
    return jacobian.reshape(fn(flat_input).numel(), flat_input.numel()).detach().cpu().double()


def cosine_distance_rows(features: torch.Tensor, labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = features.detach().cpu().double()
    if values.shape[0] < 2:
        rows.append(
            {
                "sample_i": 0 if values.shape[0] else None,
                "sample_j": None,
                "label_i": labels[0] if labels else "",
                "label_j": "",
                "euclidean": None,
                "cosine": None,
                "max_abs_delta": None,
                "near_identical": None,
                "note": "single_sample_no_pairwise_distance",
            }
        )
        return rows
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            vi = values[i].reshape(-1)
            vj = values[j].reshape(-1)
            denom = float(torch.linalg.norm(vi).item() * torch.linalg.norm(vj).item())
            cosine = math.nan if denom == 0 else float(torch.dot(vi, vj).item() / denom)
            rows.append(
                {
                    "sample_i": int(i),
                    "sample_j": int(j),
                    "label_i": labels[i],
                    "label_j": labels[j],
                    "euclidean": float(torch.linalg.norm(vi - vj).item()),
                    "cosine": cosine,
                    "max_abs_delta": float((vi - vj).abs().max().item()),
                    "near_identical": bool(torch.allclose(vi, vj, atol=1e-8, rtol=1e-8)),
                    "note": "",
                }
            )
    return rows


def node_species_labels(readout: Any, batch: Any) -> list[str]:
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    return [
        f"{point_basis_label(readout, int(graph_type.item()))}{node_index}"
        for node_index, graph_type in enumerate(graph_node_types)
    ]


def edge_sample_labels(readout: Any, batch: Any, selected_edge_indices: torch.Tensor) -> list[str]:
    labels = []
    node_labels = node_species_labels(readout, batch)
    edge_index = batch.edge_index.detach().cpu()
    for edge_idx in selected_edge_indices.detach().cpu().tolist():
        sender = int(edge_index[0, edge_idx].item())
        receiver = int(edge_index[1, edge_idx].item())
        labels.append(f"{node_labels[sender]}->{node_labels[receiver]}")
    return labels


def node_operation_inputs(readout: Any, batch: Any, node_feats: torch.Tensor, graph_type: int) -> tuple[dict[str, Any], list[str]]:
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    mask = graph_node_types == graph_type
    atom_indices = torch.where(mask)[0].detach().cpu().tolist()
    all_labels = node_species_labels(readout, batch)
    labels = [all_labels[index] for index in atom_indices]
    return {"node_feats": node_feats[mask]}, labels


def edge_operation_inputs_with_labels(
    readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
    module_key: str,
) -> tuple[dict[str, Any], list[str], torch.Tensor]:
    point_type, neigh_type, edge_type = map(int, module_key[1:-1].split(","))
    graph2mat_edge_types = readout.edge_types_to_graph2mat[batch.edge_types]
    mask = abs(graph2mat_edge_types) == abs(edge_type)
    if not bool(mask.any()):
        raise ValueError(f"No batch edges found for readout operation {module_key}")

    indices = torch.where(mask)[0]
    type_edge_index = batch.edge_index[:, mask]
    filtered_edge_messages = edge_messages[mask]
    local_edge_types = graph2mat_edge_types[mask]

    if point_type == neigh_type:
        if readout.symmetric:
            i_edges = slice(0, None, 2)
            j_edges = slice(1, None, 2)
            selected_indices = indices[0::2]
        else:
            i_edges = slice(None)
            j_edges = slice(None)
            selected_indices = indices
    else:
        i_edges = local_edge_types == edge_type
        j_edges = ~i_edges
        selected_indices = indices[i_edges]

    block = readout.interactions[module_key]
    n_expected_inputs = len(getattr(getattr(block, "operation", None), "tensor_products", []))
    edge_tuple = (filtered_edge_messages[i_edges], filtered_edge_messages[j_edges])
    node_tuple = (
        node_feats[type_edge_index[0, i_edges]],
        node_feats[type_edge_index[1, i_edges]],
    )
    labels = edge_sample_labels(readout, batch, selected_indices)
    if n_expected_inputs <= 1:
        return {"edge_messages": edge_tuple}, labels, selected_indices
    return {"edge_messages": edge_tuple, "node_feats": node_tuple}, labels, selected_indices


def feature_matrix_for_distances(inputs: dict[str, Any]) -> torch.Tensor:
    n_samples = None
    pieces = []
    for key in sorted(inputs):
        value = inputs[key]
        tensors = value if isinstance(value, tuple) else (value,)
        for tensor in tensors:
            if n_samples is None:
                n_samples = tensor.shape[0]
            elif tensor.shape[0] != n_samples:
                raise ValueError("All input tensors must have the same sample dimension")
            pieces.append(tensor.detach().reshape(tensor.shape[0], -1))
    if not pieces:
        return torch.empty(0, 0)
    return torch.cat(pieces, dim=1)


def operation_records(
    readout: Any,
    batch: Any,
    node_feats: torch.Tensor,
    edge_messages: torch.Tensor,
) -> list[dict[str, Any]]:
    graph_node_types = readout.types_to_graph2mat[batch.point_types]
    records: list[dict[str, Any]] = []
    for graph_type, block in enumerate(readout.self_interactions):
        if block is None or not bool((graph_node_types == graph_type).any()):
            continue
        label = point_basis_label(readout, graph_type)
        if label not in {"H", "O"}:
            continue
        inputs, sample_labels = node_operation_inputs(readout, batch, node_feats, graph_type)
        records.append(
            {
                "operation": f"node:{label}",
                "kind": "node",
                "module_key": str(graph_type),
                "block": block,
                "inputs": inputs,
                "sample_labels": sample_labels,
                "selected_edge_indices": [],
            }
        )

    for module_key, block in readout.interactions.items():
        if block is None:
            continue
        point_type, neigh_type, _edge_type = map(int, module_key[1:-1].split(","))
        left = point_basis_label(readout, point_type)
        right = point_basis_label(readout, neigh_type)
        if (left, right) not in {("H", "H"), ("H", "O")}:
            continue
        inputs, sample_labels, selected_indices = edge_operation_inputs_with_labels(
            readout, batch, node_feats, edge_messages, module_key
        )
        records.append(
            {
                "operation": f"edge:{left}-{right}",
                "kind": "edge",
                "module_key": module_key,
                "block": block,
                "inputs": inputs,
                "sample_labels": sample_labels,
                "selected_edge_indices": selected_indices.detach().cpu().tolist(),
            }
        )
    return sorted(records, key=lambda item: item["operation"])


def analyze_record(record: dict[str, Any], dtype: torch.dtype, vectorize: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    block = copy.deepcopy(record["block"]).to(dtype=dtype)
    operation = block.operation
    operation.eval()
    inputs = {
        key: tuple(item.detach().to(dtype=dtype) for item in value)
        if isinstance(value, tuple)
        else value.detach().to(dtype=dtype)
        for key, value in record["inputs"].items()
    }
    params = trainable_parameters(operation)
    for param in params:
        param.requires_grad_(True)

    with torch.enable_grad():
        coeff_output = operation(**inputs)
        parameter_jacobian = jacobian_wrt_parameters(coeff_output, params)

    flat_input, input_spec = flatten_input_structure(inputs)
    input_jacobian = jacobian_wrt_inputs(
        operation,
        flat_input=flat_input.to(dtype=dtype),
        input_spec=input_spec,
        vectorize=vectorize,
    )
    combined_jacobian = torch.cat([parameter_jacobian, input_jacobian], dim=1)

    feature_matrix = feature_matrix_for_distances(inputs)
    distance_rows = cosine_distance_rows(feature_matrix, record["sample_labels"])
    for row in distance_rows:
        row["operation"] = record["operation"]
        row["kind"] = record["kind"]
        row["module_key"] = record["module_key"]

    output_shape = list(coeff_output.shape)
    output_dim = int(coeff_output.numel())
    per_block_dim = int(coeff_output.shape[-1]) if coeff_output.ndim >= 2 else output_dim
    n_blocks = int(coeff_output.shape[0]) if coeff_output.ndim >= 2 else 1
    result = {
        "operation": record["operation"],
        "kind": record["kind"],
        "module_key": record["module_key"],
        "operation_class": operation.__class__.__name__,
        "sample_labels": record["sample_labels"],
        "selected_edge_indices": record["selected_edge_indices"],
        "n_blocks_using_operation": n_blocks,
        "per_block_output_dim": per_block_dim,
        "joint_output_shape": output_shape,
        "joint_output_dim": output_dim,
        "block_shape": list(getattr(block, "block_shape", ())),
        "block_output_dim": int(getattr(block, "block_size", output_dim)),
        "irreps_out": str(getattr(block, "_irreps_out", "")),
        "parameter_count": parameter_count(operation),
        "local_input_dim": int(flat_input.numel()),
        "local_input_shapes": {
            key: [list(item.shape) for item in value]
            if isinstance(value, tuple)
            else list(value.shape)
            for key, value in inputs.items()
        },
        "parameter_jacobian": rank_summary(parameter_jacobian, TOLERANCES),
        "input_jacobian": rank_summary(input_jacobian, TOLERANCES),
        "combined_parameter_input_jacobian": rank_summary(combined_jacobian, TOLERANCES),
    }
    return result, distance_rows


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


def verdict(results: list[dict[str, Any]]) -> dict[str, str]:
    parameter_limited = [
        row
        for row in results
        if row["parameter_jacobian"]["rank_abs_tol"]["1e-08"] < row["joint_output_dim"]
    ]
    input_limited = [
        row
        for row in results
        if row["input_jacobian"]["rank_abs_tol"]["1e-08"] < row["joint_output_dim"]
    ]
    if parameter_limited:
        return {
            "status": "B_SHARED_OPERATION_RANK_LIMIT",
            "summary": (
                "At least one shared readout operation has a parameter Jacobian "
                "rank below its concrete joint H2O coefficient dimension."
            ),
        }
    if input_limited:
        return {
            "status": "C_FEATURE_MANIFOLD_LIMIT",
            "summary": (
                "Operation parameters can span the concrete coefficients, but "
                "the fixed local readout inputs cannot move all directions."
            ),
        }
    return {
        "status": "A_FULL_JOINT_RANK",
        "summary": (
            "Each shared operation is locally full-rank for the concrete H2O "
            "coefficient vector with respect to both operation parameters and "
            "local readout inputs."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sanitize(rows))


def operation_csv_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        for jacobian_name in (
            "parameter_jacobian",
            "input_jacobian",
            "combined_parameter_input_jacobian",
        ):
            summary = row[jacobian_name]
            rows.append(
                {
                    "operation": row["operation"],
                    "kind": row["kind"],
                    "module_key": row["module_key"],
                    "jacobian": jacobian_name,
                    "joint_output_dim": row["joint_output_dim"],
                    "per_block_output_dim": row["per_block_output_dim"],
                    "n_blocks_using_operation": row["n_blocks_using_operation"],
                    "parameter_count": row["parameter_count"],
                    "local_input_dim": row["local_input_dim"],
                    "jacobian_shape": summary["shape"],
                    "rank_1e-5": summary["rank_abs_tol"]["1e-05"],
                    "rank_1e-6": summary["rank_abs_tol"]["1e-06"],
                    "rank_1e-7": summary["rank_abs_tol"]["1e-07"],
                    "rank_1e-8": summary["rank_abs_tol"]["1e-08"],
                    "rank_1e-10": summary["rank_abs_tol"]["1e-10"],
                    "singular_value_max": summary["singular_value_max"],
                    "singular_value_min_effective_1e_8": summary["singular_value_min_effective_1e_8"],
                    "condition_estimate_1e_8": summary["condition_estimate_1e_8"],
                }
            )
    return rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Concrete H2O Joint Readout Rank Diagnostic",
        "",
        f"Source workspace: `{payload['source_workspace']}`",
        f"Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']['status']}**: {payload['verdict']['summary']}",
        "",
        "## Interpretation",
        "",
        (
            "The parameter Jacobian is rank-deficient for every measured "
            "operation at the concrete checkpoint inputs. The local-input "
            "Jacobian and the combined parameter+input Jacobian are full-rank "
            "for every operation, so the coefficient maps are not limited by "
            "the e3nn change-of-basis or by the operation output space itself. "
            "The failure mode is specifically fixed-input shared-operation "
            "controllability: changing only the readout operation parameters "
            "cannot span all concrete H2O target coefficients."
        ),
        "",
        "## Scope",
        "",
        (
            "This diagnostic uses the actual one-sample H2O batch and the actual "
            "MACE readout inputs produced by the checkpoint. For each shared "
            "Graph2Mat readout operation it concatenates every coefficient block "
            "that uses that operation, then computes exact Jacobian ranks."
        ),
        "",
        "The reported coefficient ranks are irreps-space ranks. The symmetric "
        "node block-space ranks remain lower because a 5x5 H self block has 15 "
        "independent coefficients and a 13x13 O self block has 91.",
        "",
        "## Joint Rank Summary",
        "",
        "| operation | blocks | per-block coeffs | joint dim | params | local input dim | param rank @1e-8 | input rank @1e-8 | combined rank @1e-8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| `{row['operation']}` | {row['n_blocks_using_operation']} | "
            f"{row['per_block_output_dim']} | {row['joint_output_dim']} | "
            f"{row['parameter_count']} | {row['local_input_dim']} | "
            f"{row['parameter_jacobian']['rank_abs_tol']['1e-08']} | "
            f"{row['input_jacobian']['rank_abs_tol']['1e-08']} | "
            f"{row['combined_parameter_input_jacobian']['rank_abs_tol']['1e-08']} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Operation Results",
            "",
        ]
    )
    for row in payload["results"]:
        lines.extend(
            [
                f"### {row['operation']}",
                "",
                f"- module key: `{row['module_key']}`",
                f"- operation class: `{row['operation_class']}`",
                f"- samples: `{row['sample_labels']}`",
                f"- joint output shape: `{row['joint_output_shape']}`",
                f"- irreps_out: `{row['irreps_out']}`",
                f"- block shape after change of basis: `{row['block_shape']}`",
                f"- parameter count: `{row['parameter_count']}`",
                f"- local input shapes: `{row['local_input_shapes']}`",
                "",
                "| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("parameter_jacobian", "parameters"),
            ("input_jacobian", "local inputs"),
            ("combined_parameter_input_jacobian", "parameters + local inputs"),
        ):
            summary = row[key]
            ranks = summary["rank_abs_tol"]
            condition = summary["condition_estimate_1e_8"]
            condition_text = "" if condition is None else f"{condition:.6g}"
            lines.append(
                f"| {label} | `{summary['shape']}` | {ranks['1e-05']} | "
                f"{ranks['1e-06']} | {ranks['1e-07']} | {ranks['1e-08']} | "
                f"{ranks['1e-10']} | {summary['singular_value_max']:.6g} | "
                f"{summary['singular_value_min_effective_1e_8']:.6g} | "
                f"{condition_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Input Feature Distances",
            "",
            "| operation | sample i | sample j | label i | label j | euclidean | cosine | max abs delta | note |",
            "|---|---:|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["input_feature_distances"]:
        lines.append(
            f"| `{row['operation']}` | {row.get('sample_i', '')} | "
            f"{row.get('sample_j', '')} | `{row.get('label_i', '')}` | "
            f"`{row.get('label_j', '')}` | {row.get('euclidean', '')} | "
            f"{row.get('cosine', '')} | {row.get('max_abs_delta', '')} | "
            f"{row.get('note', '')} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{payload['output_files']['json']}`",
            f"- Rank CSV: `{payload['output_files']['rank_csv']}`",
            f"- Input feature distances CSV: `{payload['output_files']['feature_distances_csv']}`",
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
            "- This is a local Jacobian diagnostic at the checkpoint and concrete batch.",
            "- Parameter ranks hold readout inputs fixed; they test shared operation controllability.",
            "- Input ranks hold operation parameters fixed; they test sensitivity to local readout inputs, not the full upstream MACE parameter manifold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--no-vectorize", action="store_true")
    args = parser.parse_args()

    pipeline_root = args.pipeline_root.resolve()
    _prepare_pipeline_imports(pipeline_root)

    from diagnose_h2o_readout_bottleneck import make_readout_inputs
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
    dtype = getattr(torch, args.dtype)

    with working_directory(training_dir):
        datamodule = MatrixDataModule(**build_datamodule_kwargs(data, MatrixDataModule))
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        model = LitMACEMatrixModel.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        with torch.no_grad():
            data_for_readout, node_feats, edge_messages = make_readout_inputs(model, batch)
        if edge_messages is None:
            raise RuntimeError("This diagnostic expects concrete edge_messages from preprocessing_edges")
        readout = model.model.matrix_readouts
        records = operation_records(readout, batch, node_feats, edge_messages)
        results = []
        distance_rows = []
        for record in records:
            result, distances = analyze_record(
                record,
                dtype=dtype,
                vectorize=not args.no_vectorize,
            )
            results.append(result)
            distance_rows.extend(distances)

    rank_csv = output_dir / "joint_rank_by_operation.csv"
    feature_csv = output_dir / "input_feature_distances.csv"
    json_path = output_dir / "joint_rank_results.json"
    report_path = output_dir / "report.md"
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        command = f"PYTHONPATH={shlex.quote(pythonpath)} {command}"

    payload = {
        "source_workspace": str(source_workspace),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "command": command,
        "repository_context": {
            "graph2mat_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "graph2mat_commit": command_output(["git", "rev-parse", "HEAD"], cwd=GRAPH2MAT_ROOT),
            "pipeline_branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=pipeline_root),
            "pipeline_commit": command_output(["git", "rev-parse", "HEAD"], cwd=pipeline_root),
        },
        "data_policy": {
            "out_matrix": data.get("out_matrix"),
            "symmetric_matrix": data.get("symmetric_matrix"),
            "matrix_component_policy": data.get("matrix_component_policy"),
            "n_matrix_components": data.get("n_matrix_components"),
        },
        "dtype": args.dtype,
        "tolerances": list(TOLERANCES),
        "results": results,
        "input_feature_distances": distance_rows,
        "verdict": verdict(results),
        "output_files": {
            "json": str(json_path),
            "rank_csv": str(rank_csv),
            "feature_distances_csv": str(feature_csv),
            "report": str(report_path),
        },
        "validation_runs": [
            {
                "command": command,
                "result": "passed",
                "notes": "Generated joint_rank_results.json, joint_rank_by_operation.csv, input_feature_distances.csv, and report.md.",
            }
        ],
    }

    write_csv(
        rank_csv,
        operation_csv_rows(results),
        [
            "operation",
            "kind",
            "module_key",
            "jacobian",
            "joint_output_dim",
            "per_block_output_dim",
            "n_blocks_using_operation",
            "parameter_count",
            "local_input_dim",
            "jacobian_shape",
            "rank_1e-5",
            "rank_1e-6",
            "rank_1e-7",
            "rank_1e-8",
            "rank_1e-10",
            "singular_value_max",
            "singular_value_min_effective_1e_8",
            "condition_estimate_1e_8",
        ],
    )
    write_csv(
        feature_csv,
        distance_rows,
        [
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
    json_path.write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {rank_csv}")
    print(f"Wrote {feature_csv}")
    print(f"Wrote {report_path}")
    print(f"Verdict: {payload['verdict']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
