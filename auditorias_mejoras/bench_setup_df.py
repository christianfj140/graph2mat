"""Opt-in CPU H+S setup comparison and guarded synthetic d/f CUDA profiling.

No queue submissions, training campaigns, scientific labels or defaults changed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from bench_pipeline_gpu import (CAMPAIGN, HERE, ROOT, compiled_extensions, describe,
    gpu_state, guarded_command, memory_fraction, process_tree_memory, variant_order)

BASE = "270d17d1f204293a059baa09e8f1b3756981e2bf"
SOURCE = "src/graph2mat/core/data/sparse.py"


def original_component_helper():
    import numpy as np
    source = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT, text=True)
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "_csr_to_block_dict_components")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{BASE}:{SOURCE}", "exec"), namespace)
    return namespace[node.name], hashlib.sha256(source.encode()).hexdigest()


def setup_summary(records):
    result = {}
    for system in sorted({row["system"] for row in records}):
        rows = [row for row in records if row["system"] == system]
        baseline = {row["round"]: row for row in rows if row["variant"] == "baseline"}
        candidate = {row["round"]: row for row in rows if row["variant"] == "candidate"}
        ratios = []
        for index in sorted(baseline.keys() & candidate.keys()):
            assert baseline[index]["samples_sha256"] == candidate[index]["samples_sha256"], "different setup outputs"
            ratios.append(candidate[index]["setup_ms"] / baseline[index]["setup_ms"])
        result[system] = {name: describe([row["setup_ms"] for row in group.values()])
                          for name, group in (("baseline_ms", baseline), ("candidate_ms", candidate)) if group}
        result[system]["paired_ratio"] = describe(ratios) if ratios else None
    return result


def profile_rows(events):
    scopes = {event.key: dict(cuda_ms=event.device_time_total / 1000,
        cpu_ms=event.cpu_time_total / 1000, calls=event.count)
        for event in events if event.key.startswith("g2m/") and event.device_type.name == "CPU"}
    forward = scopes.get("g2m/forward", {}).get("cuda_ms", 0)
    for name, row in scopes.items():
        # Autograd worker kernels are not children of the calling thread's range.
        if name == "g2m/backward":
            row["cuda_ms"] = None
        row["fraction_forward_cuda"] = row["cuda_ms"] / forward if forward and row["cuda_ms"] is not None else None
    return scopes


def setup_child(args):
    extensions = compiled_extensions()
    import torch
    import yaml
    from graph2mat.core.data import sparse
    from graph2mat.tools.lightning import MatrixDataModule
    from bench_readout_rango_bajo import digest_tensors

    assert not torch.cuda.is_available(), "CPU setup probe must not use CUDA"
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    original, baseline_hash = original_component_helper()
    if args.variant == "baseline":
        sparse._csr_to_block_dict_components = original
    config = CAMPAIGN / "runs" / args.system / "baseline-seed0/config.yaml"
    cfg = yaml.safe_load(config.read_text())
    data = copy.deepcopy(cfg["data"])
    data.update(loader_threads=0, store_in_memory=True)
    before = process_tree_memory(os.getpid())
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start = time.perf_counter()
    module = MatrixDataModule(**data)
    module.setup("fit")
    elapsed = (time.perf_counter() - start) * 1000
    memory = process_tree_memory(os.getpid())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    digest = hashlib.sha256()
    sizes = {}
    for split in ("train", "val"):
        dataset = getattr(module, split + "_dataset")
        sizes[split] = len(dataset)
        digest.update(split.encode())
        for index in range(len(dataset)):
            digest.update(digest_tensors(dataset[index].items()).encode())
    result = dict(system=args.system, variant=args.variant, round=args.round,
        setup_ms=elapsed, samples_sha256=digest.hexdigest(), structures=sizes,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), baseline_source_sha256=baseline_hash,
        host_before=before, host_after=memory, process_peak_rss_mib=peak,
        setup_peak_rss_increment_mib=max(0, peak-peak_before), extensions=extensions,
        scope="fresh process; constructor+setup H+S cached; imports, hashing and teardown excluded; OS cache not cold")
    module.teardown(None)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


def df_child(args):
    extensions = compiled_extensions()
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from torch_geometric.data import Batch
    from bench_cache_tensores_cpu import dataset_for
    from bench_readout_rango_bajo import digest_tensors
    from graph2mat.bindings.e3nn import E3nnLinearEdgeBlock
    from graph2mat.bindings.e3nn.modules.node_operations import E3nnLinearTSQNodeBlock
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261014)
    torch.cuda.set_per_process_memory_fraction(args.fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = dataset_for(args.points)
    module = MatrixDataModule(basis_table=dataset.data_processor.basis_table,
        train_runs=[dataset.input_data[0]], val_runs=[], n_matrix_components=2,
        symmetric_matrix=True, sub_point_matrix=False, loader_threads=0)
    module.setup("fit")
    from graph2mat.tools.lightning.data import TypeLayoutCollate
    cpu = TypeLayoutCollate(Batch.from_data_list, module.basis_table)([module.train_dataset[0]] * 4)
    model = LitMACEMatrixModel(basis_table=module.basis_table,
        hidden_irreps="16x0e+16x1o+16x2e+16x3o", edge_hidden_irreps="16x0e+16x1o+16x2e+16x3o",
        num_bessel=4, max_ell=3, num_interactions=2, correlation=1, keep_last_layer_irreps=True,
        n_matrix_components=2, symmetric_matrix=True, node_block_readout=E3nnLinearTSQNodeBlock,
        edge_block_readout=E3nnLinearEdgeBlock)
    initial_sha = digest_tensors(model.state_dict().items())
    inputs_sha = digest_tensors(cpu.items())
    model.cuda()
    batch = model.transfer_batch_to_device(cpu, "cuda", 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = ((out["node_labels"] - batch.point_labels).square().sum()
            + (out["edge_labels"] - batch.edge_labels).square().sum()
            ) / (batch.point_labels.numel()+batch.edge_labels.numel())
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(args.warmup):
        assert torch.isfinite(step().detach())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, cuda_times = [], []
    for _ in range(args.steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        loss = step()
        end.record()
        torch.cuda.synchronize()
        times.append((time.perf_counter()-wall_start)*1000)
        cuda_times.append(start.elapsed_time(end))
        assert torch.isfinite(loss.detach())
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    def annotate(owner, attribute, label):
        original = getattr(owner, attribute)
        def traced(*values, **kwargs):
            with record_function(label):
                return original(*values, **kwargs)
        setattr(owner, attribute, traced)
    annotate(model.model, "_checkpointed_readout", "g2m/readout")
    readout = model.model.matrix_readouts
    for kind, blocks in (("node", readout.self_interactions), ("edge", readout.interactions.values())):
        for block in blocks:
            if block is not None:
                annotate(block, "forward", "g2m/"+kind+"_block")
                annotate(block, "coefficients_to_block", "g2m/"+kind+"_change_of_basis")
                annotate(block.operation, "forward", "g2m/"+kind+"_operation")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with record_function("g2m/forward"):
                out = model(batch)
            with record_function("g2m/loss"):
                loss = ((out["node_labels"] - batch.point_labels).square().sum()
                    + (out["edge_labels"] - batch.edge_labels).square().sum()
                    ) / (batch.point_labels.numel()+batch.edge_labels.numel())
            with record_function("g2m/backward"):
                loss.backward()
            with record_function("g2m/adam"):
                optimizer.step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output)+".trace.json")
    scopes = profile_rows(prof.key_averages())
    result = dict(points=args.points, graphs=4, round=args.round, basis=["1x0e+1x1o+1x2e", "1x0e+1x1o+1x3o"],
        model_initial_sha256=initial_sha, inputs_sha256=inputs_sha,
        warmup=args.warmup, measured_steps=args.steps, wall_ms=times, cuda_elapsed_ms=cuda_times,
        peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved, scopes=scopes,
        profiler_steps=2, extensions=extensions, torch=torch.__version__, cuda=torch.version.cuda,
        scope="synthetic periodic symmetric C2 d/f,16 channels,corr1; unprofiled times/peaks; profiler inclusive GPU scopes overlap; not material accuracy or inference-only")
    module.teardown(None)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src/graph2mat").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    manifest = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=source_hash.hexdigest(), baseline_commit=BASE, rounds=args.rounds,
        warmup=args.warmup, steps=args.steps, cpu_only=args.cpu_only, gpu_only=args.gpu_only,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(__file__, args.out / "runner-measured.py")
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", ROOT / "src", HERE)))}
    records = []
    if not args.gpu_only:
        for system in ("w90", "6x6"):
            for index in range(args.rounds):
                for variant in variant_order(("baseline", "candidate"), index):
                    label = f"setup-{system}-{index:02d}-{variant}"
                    output = args.out / (label + ".json")
                    command = [sys.executable, __file__, "setup", "--system", system, "--variant", variant,
                               "--round", str(index), "--output", str(output)]
                    print("START", label, flush=True)
                    with (args.out / (label+".log")).open("w") as log:
                        subprocess.run(command, cwd=ROOT, env={**env, "CUDA_VISIBLE_DEVICES": ""},
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    records.append(json.loads(output.read_text()))
        (args.out / "setup-summary.json").write_text(json.dumps(setup_summary(records), indent=2)+"\n")
    if not args.cpu_only:
        for index in range(args.rounds):
            for points in variant_order((16, 72), index):
                label = f"df-{points}-{index:02d}"
                output = args.out / (label+".json")
                state = gpu_state()
                fraction = memory_fraction(state["memory.total"], state["memory.free"])
                command = [sys.executable, __file__, "df", "--points", str(points), "--round", str(index),
                    "--steps", str(args.steps), "--warmup", str(args.warmup), "--fraction", str(fraction),
                    "--output", str(output)]
                print("START", label, flush=True)
                status = guarded_command(command, args.out / (label+".log"), args.out / (label+".gpu.json"),
                                         {**env, "CUDA_VISIBLE_DEVICES": "0"})
                if status:
                    raise RuntimeError(label+" failed; see log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "setup", "df"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--system", choices=("w90", "6x6"))
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--points", type=int, choices=(16, 72), default=16)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu-only", action="store_true")
    group.add_argument("--gpu-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.steps) < 1:
        parser.error("positive rounds/warmup/steps required")
    {"run": run, "setup": setup_child, "df": df_child}[args.action](args)


if __name__ == "__main__":
    main()
