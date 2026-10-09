"""Paired CPU cost-batching proxy/collate probe; not a VRAM or accuracy benchmark.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python auditorias_mejoras/bench_batching_coste.py
Uses the frozen synthetic directed C2 s/p/d/f fixtures from the cache benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import torch
from torch.utils.data import BatchSampler, SequentialSampler

from bench_cache_tensores_cpu import dataset_for, rss_kib
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.data import _CostBatchSampler


SEQUENCE = (4, 4, 72, 4, 16, 128, 4, 16, 72, 4, 16, 4)
BUDGET = 100656


def inputs(mode):
    datasets = {points: dataset_for(points) for points in set(SEQUENCE)}
    table = datasets[4].data_processor.basis_table
    configs = [datasets[points].input_data[0] for points in SEQUENCE]
    module = MatrixDataModule(basis_table=table, train_runs=configs, val_runs=[],
        n_matrix_components=2, symmetric_matrix=False, sub_point_matrix=False,
        batch_size=5, loader_threads=0, max_batch_cost=BUDGET if mode == "cost" else None)
    module.setup("fit")
    return module


def memory_probe(mode):
    module = inputs(mode)
    before, peak_before = rss_kib(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    loader = module.train_dataloader()
    initial_ms = (time.perf_counter() - start) * 1000
    costs = loader.batch_sampler.costs if mode == "cost" else ()
    result = dict(initial_loader_ms=initial_ms, rss_delta_kib=rss_kib() - before,
        peak_rss_delta_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak_before,
        integer_cost_cache_bytes=(sys.getsizeof(costs) + sum(sys.getsizeof(cost) for cost in costs)
                                 if mode == "cost" else 0))
    module.teardown(None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--memory", choices=("baseline", "cost"))
    args = parser.parse_args()
    if args.rounds < 1 or torch.cuda.is_available():
        parser.error("positive rounds required; disable CUDA for this CPU-only probe")
    torch.set_default_dtype(torch.float32)
    if args.memory:
        print(json.dumps(memory_probe(args.memory)))
        return

    module = inputs("cost")
    loader = module.train_dataloader()
    costs = loader.batch_sampler.costs
    assert costs == tuple(points * 1398 for points in SEQUENCE)
    dataset = module.train_dataset
    frozen = [dataset[index] for index in range(len(dataset))]
    digest = hashlib.sha256()
    for sample in frozen:
        for key, value in sorted(sample.items()):
            if torch.is_tensor(value):
                digest.update(key.encode())
                digest.update(value.numpy().tobytes())

    def epoch(mode, read=True, describe=True):
        sampler = SequentialSampler(dataset)
        batches = (_CostBatchSampler(sampler, 5, False, costs, BUDGET) if mode == "cost"
                   else BatchSampler(sampler, 5, False))
        result = []
        for indices in batches:
            batch = loader.collate_fn([(dataset if read else frozen)[index] for index in indices])
            if describe:
                result.append(dict(indices=indices, proxy=sum(costs[index] for index in indices),
                    tensor_bytes=sum(value.numel() * value.element_size()
                                     for value in batch.values() if torch.is_tensor(value))))
        if describe:
            assert [index for batch in result for index in batch["indices"]] == list(range(12))
        return result

    descriptions = {mode: epoch(mode) for mode in ("baseline", "cost")}
    for batches in descriptions.values():
        assert sum(batch["proxy"] for batch in batches) == sum(costs)
    for batch in descriptions["cost"]:
        assert batch["proxy"] <= BUDGET or len(batch["indices"]) == 1
    times = {f"{mode}_{phase}": [] for mode in descriptions for phase in ("collate", "lazy_collate")}
    for name in times:
        mode, phase = name.split("_", 1)
        for _ in range(3):
            epoch(mode, read=phase == "lazy_collate", describe=False)
    for round_index in range(args.rounds):
        order = list(times) if round_index % 2 == 0 else list(reversed(times))
        for name in order:
            mode, phase = name.split("_", 1)
            start = time.perf_counter()
            epoch(mode, read=phase == "lazy_collate", describe=False)
            times[name].append((time.perf_counter() - start) * 1000)
    memory = {}
    for mode in descriptions:
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--memory", mode],
            env=dict(os.environ), text=True)
        memory[mode] = json.loads(output)
    root = Path(__file__).resolve().parent.parent
    result = dict(python=sys.version, torch=torch.__version__, dtype="float32", components=2,
        symmetric=False, sequence=SEQUENCE, sample_tensors_sha256=digest.hexdigest(), budget=BUDGET,
        source_sha256={name: hashlib.sha256((root / "src/graph2mat/tools/lightning" / name).read_bytes()).hexdigest()
                       for name in ("data.py", "model.py")}, rounds=args.rounds, warmup=3,
        batches=descriptions, fresh_process_initialization=memory,
        epoch_ms={name: dict(median=statistics.median(values), min=min(values), max=max(values))
                  for name, values in times.items()}, exact_coverage=True,
        limits="synthetic CPU proxy/tensor bytes, not GPU/VRAM/model throughput/physical accuracy; "
               "prepass includes config conversion, not disk parsing; host load not isolated, RSS includes allocator")
    print(json.dumps(result, indent=2))
    module.teardown(None)


if __name__ == "__main__":
    main()
