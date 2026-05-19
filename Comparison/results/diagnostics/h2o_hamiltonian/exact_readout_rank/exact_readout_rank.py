"""Exact local Jacobian rank diagnostics for Graph2Mat e3nn readouts.

This script is intentionally diagnostic-only. It does not import or modify any
training code paths beyond instantiating the same readout blocks used by
``E3nnGraph2Mat``.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from e3nn import o3

from graph2mat import PointBasis
from graph2mat.bindings.e3nn.modules.edge_operations import E3nnSimpleEdgeBlock
from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock
from graph2mat.bindings.e3nn.modules.node_operations import E3nnSimpleNodeBlock


DEFAULT_TOLERANCES = (1e-5, 1e-6, 1e-7, 1e-8, 1e-10)


@dataclass(frozen=True)
class OperationSpec:
    name: str
    kind: str
    i_basis: PointBasis
    j_basis: PointBasis
    symmetry: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact local Jacobian ranks for H2O-like Graph2Mat e3nn "
            "readout operations."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where JSON and Markdown reports are written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional trained checkpoint path. When provided, the diagnostic "
            "loads the checkpoint's actual MatrixMACE readout blocks in "
            "addition to fresh standalone readout blocks."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for fresh readout initializations and local inputs.",
    )
    parser.add_argument(
        "--h-basis",
        default="2x0e+1x1o",
        help="PointBasis specification for H.",
    )
    parser.add_argument(
        "--o-basis",
        default="4x0e+3x1o",
        help="PointBasis specification for O.",
    )
    parser.add_argument(
        "--node-feats-irreps",
        default="20x0e+20x1o+20x2e",
        help="Input node feature irreps used by node readouts.",
    )
    parser.add_argument(
        "--edge-messages-irreps",
        default="4x0e+4x1o+4x2e",
        help="Input edge-message irreps used by edge readouts.",
    )
    parser.add_argument(
        "--n-matrix-components",
        type=int,
        default=1,
        help="Number of matrix components in the readout blocks.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Autograd/SVD dtype.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for diagnostics.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="Absolute singular-value tolerances used for rank counts.",
    )
    parser.add_argument(
        "--relative-tolerances",
        type=float,
        nargs="+",
        default=[1e-5, 1e-6, 1e-7, 1e-8],
        help="Relative tolerances multiplied by max singular value.",
    )
    parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="Disable torch.autograd.functional.jacobian vectorization.",
    )
    return parser.parse_args()


def _count_parameters(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(val) for val in value]
    return value


def _rank_summary(
    jacobian: torch.Tensor,
    tolerances: list[float],
    relative_tolerances: list[float],
) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(jacobian)
    singular_values = torch.sort(singular_values, descending=True).values

    if singular_values.numel() == 0:
        max_sv = torch.tensor(0.0, dtype=jacobian.dtype, device=jacobian.device)
        min_sv = torch.tensor(0.0, dtype=jacobian.dtype, device=jacobian.device)
        cond = None
    else:
        max_sv = singular_values[0]
        positive = singular_values[singular_values > 0]
        min_sv = positive[-1] if positive.numel() else torch.tensor(0.0)
        cond = None if min_sv.item() == 0 else (max_sv / min_sv).item()

    abs_ranks = {
        f"{tol:.0e}": int((singular_values > tol).sum().item())
        for tol in tolerances
    }
    rel_ranks = {
        f"{rtol:.0e}": int((singular_values > (max_sv * rtol)).sum().item())
        for rtol in relative_tolerances
    }

    return {
        "shape": list(jacobian.shape),
        "rank_abs_tol": abs_ranks,
        "rank_rel_tol": rel_ranks,
        "singular_values_top10": singular_values[:10],
        "singular_values_bottom10": singular_values[-10:],
        "singular_value_max": max_sv.item(),
        "singular_value_min_positive": min_sv.item(),
        "condition_number_estimate": cond,
    }


def _make_block(
    spec: OperationSpec,
    node_feats_irreps: o3.Irreps,
    edge_messages_irreps: o3.Irreps,
    n_matrix_components: int,
    dtype: torch.dtype,
    device: torch.device,
) -> E3nnIrrepsMatrixBlock:
    if spec.kind == "node":
        operation_cls = E3nnSimpleNodeBlock
        irreps = {
            "node_feats_irreps": node_feats_irreps,
            "edge_messages_irreps": None,
        }
    elif spec.kind == "edge":
        operation_cls = E3nnSimpleEdgeBlock
        irreps = {
            "node_feats_irreps": None,
            "edge_messages_irreps": edge_messages_irreps,
        }
    else:
        raise ValueError(f"Unsupported operation kind: {spec.kind!r}")

    block = E3nnIrrepsMatrixBlock(
        i_basis=spec.i_basis,
        j_basis=spec.j_basis,
        symmetry=spec.symmetry,
        operation_cls=operation_cls,
        irreps=irreps,
        n_matrix_components=n_matrix_components,
    )
    return block.to(device=device, dtype=dtype)


def _infer_node_input_dim(
    block: E3nnIrrepsMatrixBlock, fallback_irreps: o3.Irreps
) -> int:
    tsq = getattr(block.operation, "tsq", None)
    if tsq is not None and hasattr(tsq, "irreps_in"):
        return int(tsq.irreps_in.dim)
    return int(fallback_irreps.dim)


def _infer_edge_input_dim(
    block: E3nnIrrepsMatrixBlock, fallback_irreps: o3.Irreps
) -> int:
    tensor_products = getattr(block.operation, "tensor_products", None)
    if tensor_products is not None and len(tensor_products) > 0:
        first_tp = tensor_products[0]
        if hasattr(first_tp, "irreps_in1"):
            return int(first_tp.irreps_in1.dim)
    return int(fallback_irreps.dim)


def _split_edge_input(
    flat_input: torch.Tensor,
    edge_input_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = flat_input[:edge_input_dim].reshape(1, edge_input_dim)
    right = flat_input[edge_input_dim:].reshape(1, edge_input_dim)
    return left, right


def _operation_functions(
    block: E3nnIrrepsMatrixBlock,
    spec: OperationSpec,
    node_feats_irreps: o3.Irreps,
    edge_messages_irreps: o3.Irreps,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    if spec.kind == "node":
        input_dim = _infer_node_input_dim(block, node_feats_irreps)
        base_input = torch.randn(
            input_dim,
            device=block.change_of_basis.device,
            dtype=block.change_of_basis.dtype,
        )

        def irreps_fn(flat_input: torch.Tensor) -> torch.Tensor:
            node_feats = flat_input.reshape(1, input_dim)
            return block.operation(node_feats=node_feats).reshape(-1)

        def block_fn(flat_input: torch.Tensor) -> torch.Tensor:
            node_feats = flat_input.reshape(1, input_dim)
            return block(node_feats=node_feats).reshape(-1)

        return base_input, irreps_fn, block_fn

    edge_input_dim = _infer_edge_input_dim(block, edge_messages_irreps)
    input_dim = edge_input_dim * 2
    base_input = torch.randn(
        input_dim,
        device=block.change_of_basis.device,
        dtype=block.change_of_basis.dtype,
    )

    def irreps_fn(flat_input: torch.Tensor) -> torch.Tensor:
        edge_messages = _split_edge_input(flat_input, edge_input_dim)
        return block.operation(edge_messages=edge_messages).reshape(-1)

    def block_fn(flat_input: torch.Tensor) -> torch.Tensor:
        edge_messages = _split_edge_input(flat_input, edge_input_dim)
        return block(edge_messages=edge_messages).reshape(-1)

    return base_input, irreps_fn, block_fn


def _jacobian(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    vectorize: bool,
) -> torch.Tensor:
    jac = torch.autograd.functional.jacobian(fn, x, vectorize=vectorize)
    return jac.reshape(fn(x).numel(), x.numel()).detach()


def analyze_operation(
    spec: OperationSpec,
    seed: int,
    node_feats_irreps: o3.Irreps,
    edge_messages_irreps: o3.Irreps,
    n_matrix_components: int,
    dtype: torch.dtype,
    device: torch.device,
    tolerances: list[float],
    relative_tolerances: list[float],
    vectorize: bool,
    source: str = "fresh",
    block_override: E3nnIrrepsMatrixBlock | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if block_override is None:
        block = _make_block(
            spec=spec,
            node_feats_irreps=node_feats_irreps,
            edge_messages_irreps=edge_messages_irreps,
            n_matrix_components=n_matrix_components,
            dtype=dtype,
            device=device,
        )
    else:
        block = copy.deepcopy(block_override).to(device=device, dtype=dtype)
    block.eval()

    base_input, irreps_fn, block_fn = _operation_functions(
        block=block,
        spec=spec,
        node_feats_irreps=node_feats_irreps,
        edge_messages_irreps=edge_messages_irreps,
    )
    base_input = base_input.detach().requires_grad_(True)

    with torch.enable_grad():
        irreps_output = irreps_fn(base_input)
        block_output = block_fn(base_input)
        irreps_jac = _jacobian(irreps_fn, base_input, vectorize=vectorize)
        block_jac = _jacobian(block_fn, base_input, vectorize=vectorize)

    return {
        "source": source,
        "seed": seed,
        "operation": spec.name,
        "kind": spec.kind,
        "i_basis_type": spec.i_basis.type,
        "j_basis_type": spec.j_basis.type,
        "i_basis_irreps": str(spec.i_basis.e3nn_irreps),
        "j_basis_irreps": str(spec.j_basis.e3nn_irreps),
        "symmetry": spec.symmetry,
        "n_matrix_components": n_matrix_components,
        "operation_class": block.operation.__class__.__name__,
        "parameter_count": _count_parameters(block),
        "input_dim": int(base_input.numel()),
        "irreps_out": str(block._irreps_out),
        "irreps_output_dim": int(irreps_output.numel()),
        "block_shape": list(block.block_shape),
        "block_output_dim": int(block_output.numel()),
        "irreps_jacobian": _rank_summary(
            irreps_jac, tolerances=tolerances, relative_tolerances=relative_tolerances
        ),
        "block_jacobian": _rank_summary(
            block_jac, tolerances=tolerances, relative_tolerances=relative_tolerances
        ),
    }


def _point_label(point_type: Any) -> str:
    if point_type in (1, "1", "H", "h"):
        return "H"
    if point_type in (8, "8", "O", "o"):
        return "O"
    return str(point_type)


def _checkpoint_blocks(checkpoint: Path, device: torch.device):
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    basis_table = checkpoint_data.get("basis_table")
    if basis_table is None:
        raise ValueError(f"Checkpoint does not contain 'basis_table': {checkpoint}")

    lit_model = LitMACEMatrixModel.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        basis_table=basis_table,
        basis_files=None,
        root_dir=".",
    )
    readout = lit_model.model.matrix_readouts
    if isinstance(readout, torch.nn.ModuleList):
        raise NotImplementedError(
            "readout_per_interaction checkpoints are not supported by this "
            "diagnostic script yet."
        )

    basis = readout.graph2mat_table.basis
    blocks: list[tuple[OperationSpec, E3nnIrrepsMatrixBlock]] = []
    for index, block in enumerate(readout.self_interactions):
        if block is None:
            continue
        label = _point_label(basis[index].type)
        if label not in {"H", "O"}:
            continue
        blocks.append(
            (
                OperationSpec(
                    name=f"node:{label}",
                    kind="node",
                    i_basis=basis[index],
                    j_basis=basis[index],
                    symmetry="ij=ji",
                ),
                block,
            )
        )

    for key, block in readout.interactions.items():
        if block is None:
            continue
        point_type, neigh_type, _edge_type = map(int, key[1:-1].split(","))
        left = _point_label(basis[point_type].type)
        right = _point_label(basis[neigh_type].type)
        if (left, right) not in {("H", "H"), ("H", "O")}:
            continue
        blocks.append(
            (
                OperationSpec(
                    name=f"edge:{left}-{right}",
                    kind="edge",
                    i_basis=basis[point_type],
                    j_basis=basis[neigh_type],
                    symmetry="ij",
                ),
                block,
            )
        )

    return blocks


def _verdict_for(entries: list[dict[str, Any]], space: str = "irreps") -> str:
    key = f"{space}_jacobian"
    dim_key = f"{space}_output_dim"
    ranks = [entry[key]["rank_abs_tol"]["1e-08"] for entry in entries]
    dims = [entry[dim_key] for entry in entries]
    seed_text = f"{len(entries)} local-input seeds"
    if all(rank == dim for rank, dim in zip(ranks, dims)):
        return f"full rank at abs tol 1e-8 in all {seed_text}"
    return (
        "rank deficient at abs tol 1e-8 across local-input seeds: "
        + ", ".join(f"{rank}/{dim}" for rank, dim in zip(ranks, dims))
    )


def _write_markdown_report(
    output_file: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    by_source_operation: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in results:
        key = (entry["source"], entry["operation"])
        by_source_operation.setdefault(key, []).append(entry)

    lines = [
        "# Exact Readout Rank Diagnostic",
        "",
        "## Scope",
        "",
        (
            "This report computes exact local Jacobians with PyTorch autograd "
            "for Graph2Mat e3nn readout blocks. Fresh rows use standalone "
            "blocks; checkpoint rows use the actual readout blocks loaded "
            "from the provided checkpoint. The report includes both the "
            "irreducible operation output and the final block after "
            "`E3nnIrrepsMatrixBlock.change_of_basis`."
        ),
        "",
        "## Reproduction Command",
        "",
        "```bash",
        config["command"],
        "```",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(_as_jsonable(config), indent=2, sort_keys=True),
        "```",
        "",
    ]

    if config["checkpoint"] is None:
        lines.extend(
            [
                "## Trained Checkpoint",
                "",
                (
                    "No checkpoint path was provided to this run, so "
                    "trained-checkpoint ranks were not run. "
                    "The verdict below is exact for fresh random readout "
                    "initializations under the listed basis/irreps."
                ),
                "",
            ]
        )
    elif not Path(config["checkpoint"]).exists():
        lines.extend(
            [
                "## Trained Checkpoint",
                "",
                f"Checkpoint path does not exist: `{config['checkpoint']}`.",
                "",
            ]
        )

    lines.extend(["## Operation Verdicts", ""])
    for (source, operation), entries in by_source_operation.items():
        sample = entries[0]
        lines.extend(
            [
                f"### {source}: {operation}",
                "",
                f"- source: `{source}`",
                f"- kind: `{sample['kind']}`",
                f"- operation class: `{sample['operation_class']}`",
                f"- parameter count: `{sample['parameter_count']}`",
                f"- input dim: `{sample['input_dim']}`",
                f"- irreps output dim: `{sample['irreps_output_dim']}`",
                f"- block output dim: `{sample['block_output_dim']}`",
                f"- irreps-space verdict: {_verdict_for(entries, 'irreps')}",
                f"- block-space verdict: {_verdict_for(entries, 'block')}",
                "",
                "| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for entry in entries:
            irreps_rank = entry["irreps_jacobian"]["rank_abs_tol"]["1e-08"]
            block_rank = entry["block_jacobian"]["rank_abs_tol"]["1e-08"]
            max_sv = entry["irreps_jacobian"]["singular_value_max"]
            min_sv = entry["irreps_jacobian"]["singular_value_min_positive"]
            cond = entry["irreps_jacobian"]["condition_number_estimate"]
            cond_text = "null" if cond is None else f"{cond:.6g}"
            lines.append(
                f"| {entry['seed']} | {irreps_rank}/{entry['irreps_output_dim']} "
                f"| {block_rank}/{entry['block_output_dim']} | {max_sv:.6g} "
                f"| {min_sv:.6g} | {cond_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            (
                "- Fresh rows instantiate standalone readout blocks with random "
                "weights and random local inputs."
            ),
            (
                "- Checkpoint rows use trained readout weights, but still probe "
                "random local readout inputs. They test the local readout "
                "operation itself, not whether MACE feature extraction collapses "
                "to a lower-dimensional manifold on a concrete H2O geometry."
            ),
            (
                "- The irreps-space rank is the most direct test of the learned "
                "readout operation. The block-space rank is additionally bounded "
                "by the change-of-basis image for symmetric self blocks."
            ),
            "",
        ]
    )

    output_file.write_text("\n".join(lines))


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    node_feats_irreps = o3.Irreps(args.node_feats_irreps)
    edge_messages_irreps = o3.Irreps(args.edge_messages_irreps)
    h_basis = PointBasis("H", R=2, basis=args.h_basis)
    o_basis = PointBasis("O", R=2, basis=args.o_basis)

    specs = [
        OperationSpec("node:H", "node", h_basis, h_basis, "ij=ji"),
        OperationSpec("node:O", "node", o_basis, o_basis, "ij=ji"),
        OperationSpec("edge:H-H", "edge", h_basis, h_basis, "ij"),
        OperationSpec("edge:H-O", "edge", h_basis, o_basis, "ij"),
    ]

    config = {
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_exists": None
        if args.checkpoint is None
        else args.checkpoint.exists(),
        "seeds": args.seeds,
        "h_basis": args.h_basis,
        "o_basis": args.o_basis,
        "node_feats_irreps": args.node_feats_irreps,
        "edge_messages_irreps": args.edge_messages_irreps,
        "n_matrix_components": args.n_matrix_components,
        "dtype": args.dtype,
        "device": str(device),
        "tolerances": args.tolerances,
        "relative_tolerances": args.relative_tolerances,
        "vectorize": not args.no_vectorize,
    }

    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for spec in specs:
            results.append(
                analyze_operation(
                    spec=spec,
                    seed=seed,
                    node_feats_irreps=node_feats_irreps,
                    edge_messages_irreps=edge_messages_irreps,
                    n_matrix_components=args.n_matrix_components,
                    dtype=dtype,
                    device=device,
                    tolerances=args.tolerances,
                    relative_tolerances=args.relative_tolerances,
                    vectorize=not args.no_vectorize,
                )
            )

    if args.checkpoint is not None and args.checkpoint.exists():
        checkpoint_blocks = _checkpoint_blocks(args.checkpoint, device=device)
        for seed in args.seeds:
            for spec, block in checkpoint_blocks:
                results.append(
                    analyze_operation(
                        spec=spec,
                        seed=seed,
                        node_feats_irreps=node_feats_irreps,
                        edge_messages_irreps=edge_messages_irreps,
                        n_matrix_components=args.n_matrix_components,
                        dtype=dtype,
                        device=device,
                        tolerances=args.tolerances,
                        relative_tolerances=args.relative_tolerances,
                        vectorize=not args.no_vectorize,
                        source=f"checkpoint:{args.checkpoint.name}",
                        block_override=block,
                    )
                )

    json_file = output_dir / "rank_results.json"
    json_file.write_text(json.dumps(_as_jsonable({"config": config, "results": results}), indent=2))
    _write_markdown_report(output_dir / "report.md", config=config, results=results)

    print(f"Wrote {json_file}")
    print(f"Wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
