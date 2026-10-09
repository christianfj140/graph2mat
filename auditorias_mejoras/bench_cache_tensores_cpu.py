"""Paired synthetic CPU preprocessing/cache timings; no GPU/accuracy claim.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONPATH pointing to this repo's src.
Legacy implementations below are limited to BasisConfiguration inputs, without
FDF/spectral sidecars, and reproduce the pre-change lazy/cache behavior there.
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

import numpy as np
import torch
from torch_geometric.data import Batch

from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import InMemoryData, TorchBasisMatrixDataset
from graph2mat.core.data.matrices.basis_matrix import BasisMatrix
from graph2mat.tools.lightning.data import TypeLayoutCollate


class LegacyDataset(TorchBasisMatrixDataset):
    def __getitem__(self, index):
        item = self.input_data[index]
        data = self.data_cls.new(item, data_processor=self.data_processor, labels=self.load_labels)
        str(item).endswith(".fdf")  # The old suffix check formats the whole configuration.
        return data


class LegacyCache:
    def __init__(self, dataset):
        self.data_objects = [dataset[i] for i in range(len(dataset))]
    def __getitem__(self, index):
        return self.data_objects[index]


def dataset_for(points, cls=TorchBasisMatrixDataset):
    table = BasisTableWithEdges([
        PointBasis("A", R=2.2, basis="1x0e+1x1o+1x2e"),
        PointBasis("B", R=2.2, basis="1x0e+1x1o+1x3o"),
    ])
    counts = np.tile([9, 11], points // 2)
    config = BasisConfiguration(point_types=["A", "B"] * (points // 2),
        positions=np.column_stack((np.arange(points) * 1.2, np.zeros((points, 2)))),
        basis=table, cell=np.diag([points * 1.2, 20., 20.]), pbc=(True, False, False),
        matrix=BasisMatrix({(i, i, 0): np.ones((count, count, 2)) for i, count in enumerate(counts)},
                           np.array([3, 1, 1]), counts))
    processor = MatrixDataProcessor(basis_table=table, sub_point_matrix=False, n_matrix_components=2)
    return cls([config] * 8, processor)


def rss_kib():
    with open("/proc/self/status") as stream:
        return next(int(line.split()[1]) for line in stream if line.startswith("VmRSS:"))


def memory_probe(points, mode):
    dataset = dataset_for(points, LegacyDataset if mode == "baseline" else TorchBasisMatrixDataset)
    before, peak_before = rss_kib(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    cache = (LegacyCache if mode == "baseline" else InMemoryData)(dataset)
    initial_ms = (time.perf_counter() - start) * 1000
    after, peak_after = rss_kib(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sample = cache[0]
    return {"initial_ms_8_samples": initial_ms, "rss_delta_kib": after - before,
        "peak_rss_delta_kib": peak_after - peak_before,
        "tensor_bytes_per_sample": sum(value.numel() * value.element_size()
                                       for value in sample.values() if torch.is_tensor(value))}


def assert_same(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected.keys():
        if torch.is_tensor(expected[key]):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
            assert actual[key].dtype == expected[key].dtype


def measure(points, rounds):
    baseline, candidate = dataset_for(points, LegacyDataset), dataset_for(points)
    old_cache, new_cache = LegacyCache(baseline), InMemoryData(candidate)
    assert_same(candidate[0], baseline[0])
    assert_same(new_cache[0], old_cache[0])
    collate = TypeLayoutCollate(Batch.from_data_list, candidate.data_processor.basis_table,
                                symmetric=candidate.data_processor.symmetric_matrix)
    assert_same(collate([candidate[0], candidate[1]]), collate([baseline[0], baseline[1]]))
    functions = {}
    for name, dataset in (("lazy_baseline", baseline), ("lazy_candidate", candidate),
                          ("cache_baseline", old_cache), ("cache_candidate", new_cache)):
        functions[name] = lambda dataset=dataset: dataset[0]
        functions[name + "_collate2"] = lambda dataset=dataset: collate([dataset[0], dataset[1]])
    for call in functions.values():
        for _ in range(3):
            call()
    times = {name: [] for name in functions}
    for round_index in range(rounds):
        order = list(functions) if round_index % 2 == 0 else list(reversed(functions))
        for name in order:
            start = time.perf_counter()
            for _ in range(8):
                functions[name]()
            times[name].append((time.perf_counter() - start) * 1000 / 8)
    memory = {}
    for mode in ("baseline", "candidate"):
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            "--points", str(points), "--memory", mode], env=dict(os.environ), text=True)
        memory[mode] = json.loads(output)
    return {"points": points, "edges": candidate[0].edge_index.shape[1],
        "access_ms": {name: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                      for name, values in times.items()}, "fresh_process_memory": memory,
        "exact_tensor_equivalence": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=72)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--memory", choices=("baseline", "candidate"))
    args = parser.parse_args()
    if args.points < 2 or args.points % 2 or args.rounds < 1:
        parser.error("points must be positive even >=2 and rounds >=1")
    if torch.cuda.is_available():
        parser.error("disable CUDA for this CPU-only probe")
    torch.set_default_dtype(torch.float32)
    if args.memory:
        print(json.dumps(memory_probe(args.points, args.memory)))
        return
    root = Path(__file__).resolve().parent.parent
    source = root / "src/graph2mat/bindings/torch/data/dataset.py"
    print(json.dumps({"python": sys.version, "torch": torch.__version__, "dtype": "float32",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "components": 2,
        "symmetric": False,
        "rounds": args.rounds, "warmup": 3, "measure": measure(args.points, args.rounds),
        "limits": "synthetic CPU access/collate, host load not isolated; RSS includes runtime/allocator; not GPU/VRAM/accuracy"}, indent=2))


if __name__ == "__main__":
    main()
