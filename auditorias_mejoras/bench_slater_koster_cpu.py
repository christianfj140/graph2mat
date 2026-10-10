"""Paired synthetic CPU probe: default / 7.1 / SK+residual, never GPU or accuracy."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

# Some optional-backend imports print diagnostics; stdout is exclusively JSON.
with redirect_stdout(sys.stderr):
    from bench_cache_tensores_cpu import dataset_for
    from bench_readout_rango_bajo import digest_tensors, parameters, summary
    import torch
    from torch_geometric.data import Batch
    from e3nn import o3
    from graph2mat import PointBasis
    from graph2mat.bindings.e3nn import (
        E3nnLinearEdgeBlock, E3nnLowRankEdgeBlock, E3nnSlaterKosterEdgeBlock,
    )
    from graph2mat.bindings.e3nn.modules.matrixblock import E3nnIrrepsMatrixBlock
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


VARIANTS = {"baseline": E3nnLinearEdgeBlock, "low_rank": E3nnLowRankEdgeBlock,
            "sk_residual": E3nnSlaterKosterEdgeBlock}
SEED = 20261009


def model_for(table, variant, rank):
    torch.manual_seed(SEED)
    return LitMACEMatrixModel(basis_table=table,
        hidden_irreps="16x0e+16x1o+16x2e+16x3o",
        edge_hidden_irreps="16x0e+16x1o+16x2e+16x3o",
        num_bessel=4, max_ell=3, num_interactions=2, correlation=1,
        keep_last_layer_irreps=True, n_matrix_components=2, symmetric_matrix=False,
        edge_block_readout=VARIANTS[variant],
        edge_block_readout_kwargs=None if variant == "baseline" else {"rank": rank})


def common_hashes(model):
    return {"backbone": digest_tensors(model.model.mace.state_dict().items()),
            "onsite": digest_tensors(model.model.matrix_readouts.self_interactions.state_dict().items())}


def model_loss(model, batch):
    out = model(batch.clone())
    return ((out["node_labels"]-batch.point_labels).square().sum()
            + (out["edge_labels"]-batch.edge_labels).square().sum()) / (
                batch.point_labels.numel()+batch.edge_labels.numel())


def measure(models, loss_for, rounds, warmup):
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    times = {name: {phase: [] for phase in ("forward", "backward", "step")} for name in models}
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
        elapsed = (time.perf_counter()-start)*1000
        assert torch.isfinite(loss.detach())
        if phase != "forward":
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)
        return elapsed
    names = list(models)
    for phase in ("forward", "backward", "step"):
        for _ in range(warmup):
            for name in names:
                run(name, phase)
        for index in range(rounds):
            order = names[index % 3:] + names[:index % 3]
            if (index // 3) % 2:
                order.reverse()
            for name in order:
                times[name][phase].append(run(name, phase))
    return {name: {phase+"_ms": {"raw_ms": raw, **summary(raw)} for phase, raw in phases.items()}
            for name, phases in times.items()}


def _memory_worker(variant, points, rank, connection):
    try:
        if sys.platform != "linux":
            raise RuntimeError("process peak RSS units are certified only on Linux")
        import resource
        torch.set_default_dtype(torch.float32)
        with redirect_stdout(sys.stderr):
            dataset = dataset_for(points)
            batch = Batch.from_data_list([dataset[0]])
            model = model_for(dataset.data_processor.basis_table, variant, rank)
            shared = common_hashes(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = model_loss(model, batch)
                loss.backward()
                optimizer.step()
                assert torch.isfinite(loss.detach())
            record = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      "scope": "fresh spawned whole process: imports, construction, batch, four Adam steps; not readout-only",
                      "shared_initial_hashes": shared}
        connection.send({"result": record})
    except Exception:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def host_memory(points, rank, shared):
    ctx = multiprocessing.get_context("spawn")
    result = {}
    for variant in VARIANTS:
        receiver, sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_memory_worker, args=(variant, points, rank, sender))
        process.start()
        sender.close()
        try:
            record = receiver.recv()
        finally:
            receiver.close()
            process.join()
        if process.exitcode != 0 or "error" in record:
            raise RuntimeError(f"RSS worker {variant}: exit {process.exitcode}: {record}")
        assert record["result"]["shared_initial_hashes"] == shared
        result[variant] = record["result"]
    return result


def probe(points, rank, rounds, warmup):
    dataset = dataset_for(points)
    batch = Batch.from_data_list([dataset[0]])
    batch_hash = digest_tensors(batch.items())
    models = {name: model_for(dataset.data_processor.basis_table, name, rank) for name in VARIANTS}
    baseline = models["baseline"]
    for name, model in models.items():
        if name != "baseline":
            model.model.mace.load_state_dict(baseline.model.mace.state_dict(), strict=True)
            model.model.matrix_readouts.self_interactions.load_state_dict(
                baseline.model.matrix_readouts.self_interactions.state_dict(), strict=True)
        out = model(batch.clone())
        assert out["node_labels"].shape == batch.point_labels.shape
        assert out["edge_labels"].shape == batch.edge_labels.shape
    shared = common_hashes(baseline)
    assert all(common_hashes(model) == shared for model in models.values())
    initial = {name: digest_tensors(model.state_dict().items()) for name, model in models.items()}
    count = {name: {"total": parameters(model),
        "edge_operations": sum(parameters(block.operation) for block in model.model.matrix_readouts.interactions.values()),
        "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel()*b.element_size() for b in model.buffers())}
        for name, model in models.items()}
    full_times = measure(models, lambda model: model_loss(model, batch), rounds, warmup)
    for branch in ("sk_net", "residual.radial_gates"):
        grads = [p.grad for name, p in models["sk_residual"].named_parameters() if branch in name and p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    # Same orbital pair, Q, components and feature tensors for all isolated wrappers.
    point = PointBasis("pair", R=2.2, basis="1x0e+1x1o+1x2e+1x3o")
    source = o3.Irreps("16x0e+16x1o+16x2e+16x3o")
    ops = {}
    for name, cls in VARIANTS.items():
        torch.manual_seed(SEED)
        ops[name] = E3nnIrrepsMatrixBlock(point, point, "ij", cls,
            irreps={"node_feats_irreps": None, "edge_messages_irreps": source,
                    "edge_feats_irreps": o3.Irreps("4x0e")},
            n_matrix_components=2, **({} if name == "baseline" else {"rank": rank}))
        torch.testing.assert_close(ops[name].change_of_basis, ops["baseline"].change_of_basis, atol=0, rtol=0)
    edges = batch.edge_index.shape[1]
    torch.manual_seed(SEED)
    a, b = (torch.randn(edges, source.dim) for _ in range(2))
    radial, vectors = torch.randn(edges, 4), torch.randn(edges, 3)
    inputs = [("a", a), ("b", b), ("radial", radial), ("vectors", vectors)]
    op_hash = digest_tensors(inputs)
    def op_loss(block):
        kwargs = {"edge_messages": (a, b)}
        if not isinstance(block.operation, E3nnLinearEdgeBlock):
            kwargs["edge_feats"] = (radial, radial)
        if isinstance(block.operation, E3nnSlaterKosterEdgeBlock):
            kwargs["edge_vectors"] = (vectors, -vectors)
        out = block(**kwargs)
        return out.square().mean()
    assert all(block.block_shape == (16, 16, 2) for block in ops.values())
    op_times = measure(ops, op_loss, rounds, warmup)
    assert digest_tensors(batch.items()) == batch_hash and digest_tensors(inputs) == op_hash
    return {"points": points, "directed_edges": edges, "batch_tensor_sha256": batch_hash,
        "operation_input_sha256": op_hash, "initial_model_sha256": initial,
        "shared_initial_hashes": shared, "shared_weights_equal": True,
        "input_hashes_unchanged": True, "branches_effective": True,
        "parameters": count, "operation_parameters": {name: parameters(op) for name, op in ops.items()},
        "measure": {"full_model": full_times, "operation": op_times},
        "host_memory": host_memory(points, rank, shared)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=72)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.points < 2 or args.points % 2 or min(args.rank, args.rounds, args.warmup) < 1:
        parser.error("even points >=2 and positive rank/rounds/warmup required")
    if torch.cuda.is_available():
        parser.error("CPU only: disable CUDA")
    if sys.platform != "linux":
        parser.error("fresh-process RSS measurement currently requires Linux")
    torch.set_default_dtype(torch.float32)
    with redirect_stdout(sys.stderr):
        measured = probe(args.points, args.rank, args.rounds, args.warmup)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root/"src/graph2mat/bindings/e3nn/modules/slater_koster.py",
             root/"src/graph2mat/bindings/e3nn/modules/matrixblock.py", root/"src/graph2mat/models/mace.py",
             root/"src/graph2mat/bindings/e3nn/modules/edge_operations.py"]
    result = {"python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "e3nn", "mace-torch", "pytorch-lightning")},
        "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(), "seed": SEED,
        "rank": args.rank, "rounds": args.rounds, "warmup": args.warmup, "components": 2,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "timing_scope": "training forward includes input clone and MSE; backward excludes forward; step includes forward/backward/Adam, excludes zero_grad; checks outside timing",
        "update_policy": "forward/backward phases keep weights fixed; only warmup and timed step phase update Adam weights; same number/order per variant",
        "memory_scope": "parameter/buffer bytes are tensor payload only; host RSS is whole fresh process, imports/construction included; no accelerator memory measurement",
        "limits": "synthetic CPU only; host load not isolated; architectures differ; no VRAM or scientific accuracy conclusion",
        **measured}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
