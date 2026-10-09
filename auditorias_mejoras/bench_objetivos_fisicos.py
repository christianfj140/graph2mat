"""Paired physical-loss CPU costs, including reference I/O and Lightning fit setup."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from statistics import median
import subprocess
import tempfile
from time import perf_counter

from smoke_objetivos_fisicos import (
    CASES, fit_steps, load_sample, loader_for, model_for, processor_for, self_check,
    source_hashes, validate_paths,
)


def summarize_seconds(values):
    center = median(values)
    return {"median_s": center, "mad_s": median(abs(x-center) for x in values), "samples_s": values}


def paired_times(baseline, candidate, warmup=3, rounds=12):
    if warmup < 0 or rounds < 1:
        raise ValueError("warmup >= 0 and rounds >= 1 are required")
    for _ in range(warmup):
        baseline()
        candidate()
    a, b = [], []
    for turn in range(rounds):
        pair = ((baseline, a), (candidate, b))
        for function, samples in pair if turn % 2 == 0 else reversed(pair):
            start = perf_counter()
            function()
            samples.append(perf_counter()-start)
    result = {"baseline": summarize_seconds(a), "physical": summarize_seconds(b)}
    result["physical_over_baseline"] = result["physical"]["median_s"] / result["baseline"]["median_s"]
    return result


def auxiliary_bytes(data):
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    def size(value):
        if hasattr(value, "numel"):
            return value.numel()*value.element_size()
        if hasattr(value, "nbytes"):
            return value.nbytes
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return 0
    return (sum(size(v) for k, v in data.items() if k.startswith("physical_"))
        + sum(size(v) for k, v in (getattr(data, LAYOUT_ATTR, None) or {}).items() if k.startswith("physical_"))
        + size(data.metadata.get("physical_reference", {})))


def measure_case(fdf, target, term, root, warmup, rounds):
    import torch
    from graph2mat.bindings.torch import InMemoryData
    from graph2mat.core.data.metrics import elementwise_mse

    processor = processor_for(fdf, target)
    start = perf_counter()
    baseline = load_sample(fdf, processor)
    first_baseline = perf_counter()-start
    start = perf_counter()
    try:
        candidate = load_sample(fdf, processor, term)
    except ValueError as error:
        if target == "hamiltonian" and "unknown/unobserved density partner where predicted H can vary" in str(error):
            return {"target": target, "term": term, "status": "blocked", "reason": str(error)}
        raise
    first_candidate = perf_counter()-start
    for key in ("point_labels", "edge_labels"):
        torch.testing.assert_close(baseline[key], candidate[key], rtol=0, atol=0, equal_nan=True)
    cache_a, cache_b = InMemoryData([baseline]), InMemoryData([candidate])
    collate = loader_for(candidate, processor.basis_table).collate_fn
    batch_a, batch_b = collate([baseline]), collate([candidate])
    control = model_for(processor.basis_table, {})
    physical_model = model_for(processor.basis_table, {term: .3})
    initial = deepcopy(control.state_dict())
    physical_model.load_state_dict(initial)
    with torch.no_grad():
        predictions = control(batch_a)
    n = predictions["node_labels"].clone().requires_grad_()
    e = predictions["edge_labels"].clone().requires_grad_()
    base_loss, physical_loss = elementwise_mse(), elementwise_mse(**{term: .3})

    def loss_backward(metric, batch):
        n.grad = e.grad = None
        loss, _ = metric(nodes_pred=n, nodes_ref=batch.point_labels, edges_pred=e,
            edges_ref=batch.edge_labels, batch=batch, basis_table=processor.basis_table)
        loss.backward()

    def full_step(model, data):
        model.load_state_dict(initial)
        torch.manual_seed(13)
        fit_steps(model, data, processor.basis_table, root, steps=1)

    phases = {}
    for name, a, b in (
        ("load_prepare_validate", lambda: load_sample(fdf, processor), lambda: load_sample(fdf, processor, term)),
        ("cached_access", lambda: cache_a[0], lambda: cache_b[0]),
        ("collate", lambda: collate([baseline]), lambda: collate([candidate])),
        ("loss_backward", lambda: loss_backward(base_loss, batch_a), lambda: loss_backward(physical_loss, batch_b)),
        ("lightning_one_step_fit", lambda: full_step(control, baseline), lambda: full_step(physical_model, candidate)),
    ):
        phases[name] = paired_times(a, b, warmup, rounds)
    return {"target": target, "term": term, "weight": .3, "status": "measured_cpu",
        "first_invocation_s": {"baseline": first_baseline, "physical": first_candidate},
        "atoms": int(candidate.num_nodes), "node_rows": len(candidate.point_labels),
        "edge_rows": len(candidate.edge_labels), "spin": candidate.physical_spin.tolist(),
        "auxiliary_bytes_sample": auxiliary_bytes(candidate), "auxiliary_bytes_batch1": auxiliary_bytes(batch_b),
        "phases": phases}


def main():
    import torch
    from graph2mat.tools.lightning.model import _training_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdf", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fdf, output = validate_paths(args.fdf, args.output)
    if args.warmup < 0 or args.rounds < 1 or torch.cuda.is_available():
        parser.error("warmup >= 0, rounds >= 1, CUDA disabled required")
    torch.set_default_dtype(torch.float64)
    self_check()
    assert summarize_seconds([1., 2., 3.])["mad_s"] == 1.
    counts = [0, 0]
    def tick(index):
        counts[index] += 1
    paired_times(lambda: tick(0), lambda: tick(1), 1, 2)
    assert counts == [3, 3]
    before = source_hashes(fdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"])
    provenance = _training_provenance()
    scripts = (Path(__file__).resolve(), Path(__file__).with_name("smoke_objetivos_fisicos.py"))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_before = os.getloadavg()
    try:
        with tempfile.TemporaryDirectory(prefix="g2m-physical-bench-", dir=output.parent) as root:
            cases = [measure_case(fdf, target, term, root, args.warmup, args.rounds) for target, term in CASES]
    finally:
        assert before == source_hashes(fdf), "scientific sources changed during benchmark"
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")), platform.processor())
    result = {"fdf": str(fdf), "source_hashes": before, "source_hashes_unchanged": True,
        "provenance": provenance, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "script_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        "cpu": cpu, "host_logical_cpus": os.cpu_count(), "threads": torch.get_num_threads(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "host_load_before": load_before, "host_load_after": os.getloadavg(),
        "dtype": "float64", "seed": 13, "warmup": args.warmup, "rounds": args.rounds,
        "rss_peak_before_KiB": rss_before, "rss_peak_after_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "whole benchmark process including imports/models/all cases; not paired allocation and not VRAM",
        "first_invocation_scope": "first invocation in this process; OS filesystem cache not evicted",
        "load_control_scope": "matched physical channels, original sparse support, no auxiliary certification; S read for geometry",
        "lightning_scope": "one-step Trainer.fit includes setup, logging, optimizer construction and prepared-data preflight",
        "auxiliary_scope": "physical tensors + physical layout arrays + k metadata array payload; excludes Python container overhead",
        "accuracy_or_gpu_claim": False, "cases": cases}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Paired CPU benchmark written: {output}")


if __name__ == "__main__":
    main()
