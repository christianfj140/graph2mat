"""Paired CPU 7.1 probe on frozen synthetic C2 s/p/d/f data; not a GPU benchmark.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python auditorias_mejoras/bench_readout_rango_bajo.py
No user datasets, scientific accuracy claim, or GPU execution. Timing includes
input cloning and loss; excludes data preparation. Host load is not isolated.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from torch_geometric.data import Batch

from bench_cache_tensores_cpu import dataset_for
from e3nn import o3
from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock, E3nnLowRankEdgeBlock
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


def digest_tensors(items):
    digest = hashlib.sha256()
    for key, value in sorted(items):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameters(module):
    return sum(param.numel() for param in module.parameters())


def summary(values):
    median = statistics.median(values)
    return {
        "median": median,
        "min": min(values),
        "max": max(values),
        "mad": statistics.median(abs(value - median) for value in values),
    }


def measure(models, loss_for, rounds, warmup):
    """Same inputs/work; alternate architectures; forward/backward/Adam separately."""
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=1e-3)
        for name, model in models.items()
    }

    def run(name, phase):
        model, optimizer = models[name], optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        if phase == "backward":
            loss = loss_for(model)
        start = time.perf_counter()
        if phase != "backward":
            loss = loss_for(model)
        if phase != "forward":
            loss.backward()
        if phase == "step":
            optimizer.step()
        elapsed = (time.perf_counter() - start) * 1000
        # Self-check outside the measured region; never accept a fast NaN result.
        assert torch.isfinite(loss.detach())
        if phase != "forward":
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(grad).all() for grad in grads)
            assert any(grad.abs().sum() > 0 for grad in grads)
        return elapsed

    times = {
        name: {phase: [] for phase in ("forward", "backward", "step")}
        for name in models
    }
    for phase in ("forward", "backward", "step"):
        # Keep both architectures live equally; initialize Adam before timed steps.
        for _ in range(warmup):
            for name in models:
                run(name, phase)
        for index in range(rounds):
            for name in list(models) if index % 2 == 0 else list(reversed(models)):
                times[name][phase].append(run(name, phase))
    return {
        name: {phase + "_ms": summary(values) for phase, values in phases.items()}
        for name, phases in times.items()
    }


def probe(points, rank, rounds, warmup):
    dataset = dataset_for(points)
    batch = Batch.from_data_list([dataset[0]])
    batch_hash = digest_tensors(batch.items())
    models = {}
    for name, cls in (
        ("baseline", E3nnLinearEdgeBlock),
        ("candidate", E3nnLowRankEdgeBlock),
    ):
        torch.manual_seed(20261009)
        models[name] = LitMACEMatrixModel(
            basis_table=dataset.data_processor.basis_table,
            hidden_irreps="16x0e+16x1o+16x2e+16x3o",
            edge_hidden_irreps="16x0e+16x1o+16x2e+16x3o",
            num_bessel=4,
            max_ell=3,
            num_interactions=2,
            keep_last_layer_irreps=True,
            correlation=1,
            n_matrix_components=2,
            symmetric_matrix=False,
            edge_block_readout=cls,
            edge_block_readout_kwargs={"rank": rank} if name == "candidate" else None,
        )
    # Preserve identical backbone/on-site weights; only the edge architecture differs.
    models["candidate"].model.mace.load_state_dict(
        models["baseline"].model.mace.state_dict(), strict=True
    )
    models["candidate"].model.matrix_readouts.self_interactions.load_state_dict(
        models["baseline"].model.matrix_readouts.self_interactions.state_dict(),
        strict=True,
    )
    before = {
        name: digest_tensors(model.state_dict().items())
        for name, model in models.items()
    }

    def model_loss(model):
        out = model(batch.clone())
        assert out["node_labels"].shape == batch.point_labels.shape
        assert out["edge_labels"].shape == batch.edge_labels.shape
        return (
            (out["node_labels"] - batch.point_labels).square().sum()
            + (out["edge_labels"] - batch.edge_labels).square().sum()
        ) / (batch.point_labels.numel() + batch.edge_labels.numel())

    count = {
        name: {
            "total": parameters(model),
            "edge_operations": sum(
                parameters(block.operation)
                for block in model.model.matrix_readouts.interactions.values()
            ),
        }
        for name, model in models.items()
    }
    model_times = measure(models, model_loss, rounds, warmup)
    assert digest_tensors(batch.items()) == batch_hash

    torch.manual_seed(20261009)
    inputs, outputs = o3.Irreps("32x0e+32x1o+32x2e"), o3.Irreps("4x0e+4x1o+4x2e")
    ops = {
        "baseline": E3nnLinearEdgeBlock(inputs, outputs),
        "candidate": E3nnLowRankEdgeBlock(inputs, outputs, "4x0e", rank=rank),
    }
    edges = batch.edge_index.shape[1]
    a, b = (torch.randn(edges, inputs.dim) for _ in range(2))
    radial = torch.randn(edges, 4)

    def op_loss(op):
        kwargs = (
            {"edge_feats": (radial, radial)}
            if isinstance(op, E3nnLowRankEdgeBlock)
            else {}
        )
        return op(edge_messages=(a, b), **kwargs).square().mean()

    op_times = measure(ops, op_loss, rounds, warmup)
    return {
        "points": points,
        "directed_edges": edges,
        "batch_tensor_sha256": batch_hash,
        "initial_model_sha256": before,
        "parameters": count,
        "full_model": model_times,
        "operation_parameters": {name: parameters(op) for name, op in ops.items()},
        "operation": op_times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=72)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if (
        args.points < 2
        or args.points % 2
        or min(args.rank, args.rounds, args.warmup) < 1
    ):
        parser.error("positive rank/rounds/warmup and even points >=2 required")
    if torch.cuda.is_available():
        parser.error("CPU-only probe: disable CUDA")
    torch.set_default_dtype(torch.float32)
    root = Path(__file__).resolve().parent.parent
    source = root / "src/graph2mat/bindings/e3nn/modules/edge_operations.py"
    result = {
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "e3nn", "mace-torch", "pytorch-lightning")
        },
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "rank": args.rank,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "measure": probe(args.points, args.rank, args.rounds, args.warmup),
        "limits": "synthetic CPU only; host load not isolated; architectures differ; no VRAM/accuracy conclusion",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
