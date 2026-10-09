"""Paired CPU sparse helpers against their exact pre-change Git definitions.

Only the two helper function ASTs are executed in an isolated namespace: no
module body, converters, checkout or source mutation. Synthetic fixtures reuse
the cache benchmark's periodic s/p/d/f graph, not user datasets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
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
from scipy.sparse import coo_array

from bench_cache_tensores_cpu import dataset_for, rss_kib
from graph2mat.core.data import _sparse, sparse

BASE = "cc334f0c5b3936a42c3a2baedb0aad0a0d5e0427"
ROOT = Path(__file__).resolve().parent.parent
SOURCE = "src/graph2mat/core/data/sparse.py"
HELPERS = {"coords": "_blockmatrix_coo_coords", "csr_components": "_csr_to_block_dict_components"}


def baseline_helpers():
    source = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT, text=True)
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS.values()]
    assert len(selected) == len(HELPERS)
    namespace = {"np": np, "itertools": itertools}
    exec(compile(ast.Module(body=selected, type_ignores=[]), f"{BASE}:{SOURCE}", "exec"), namespace)
    return {name: namespace[function] for name, function in HELPERS.items()}


def fixture(points):
    dataset = dataset_for(points)
    sample = dataset[0]
    orbitals = dataset.data_processor.basis_table.basis_size[sample.point_types.numpy()].astype(np.int32)
    edges, images = sample.edge_index[:, ::2].numpy(), sample.neigh_isc[::2].numpy()
    coords = dict(orbitals=orbitals, edge_index=edges, edge_neigh_isc=images,
                  n_supercells=3, symmetrize_edges=True)
    # Independent preparation: neither measured helper is warmed by CSR construction.
    first = np.r_[0, np.cumsum(orbitals)]
    rows, columns, values = [], [], []
    for atom, count in enumerate(orbitals):
        for i in range(count):
            for j in range(count):
                rows.append(first[atom] + i)
                columns.append(first[atom] + j)
                values.append(.25)
    for (source, target), image in zip(edges.T, images):
        opposite = 0 if image == 0 else 3 - image
        for i in range(orbitals[source]):
            for j in range(orbitals[target]):
                rows.extend((first[source] + i, first[target] + j))
                columns.extend((first[target] + j + first[-1] * image, first[source] + i + first[-1] * opposite))
                values.extend((.75, .75))
    csr = coo_array((values, (rows, columns)), (first[-1], first[-1] * 3)).tocsr()
    components = dict(data=np.column_stack((csr.data, csr.data + .5)),
        ptr=csr.indptr.astype(np.int32), cols=csr.indices.astype(np.int32),
        atom_first_orb=np.r_[0, np.cumsum(orbitals)].astype(np.int32), orbitals=orbitals,
        n_atoms=points, fill_value=0.0)
    return {"coords": coords, "csr_components": components}, csr.nnz, csr.shape


def memory_probe(points, mode, name):
    baseline = baseline_helpers()
    args, _, _ = fixture(points)
    call = baseline[name] if mode == "baseline" else getattr(sparse, HELPERS[name])
    before, peak_before = rss_kib(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    output = call(**args[name])
    initial_ms = (time.perf_counter() - start) * 1000
    after, peak_after = rss_kib(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    arrays = output[:2] if name == "coords" else output.values()
    return {"initial_call_ms": initial_ms, "rss_delta_kib": after - before,
        "peak_rss_delta_kib": peak_after - peak_before,
        "output_bytes": sum(array.nbytes for array in arrays)}


def measure(points, rounds):
    start = time.perf_counter()
    baseline = baseline_helpers()
    extraction_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    args, nnz, shape = fixture(points)
    fixture_ms = (time.perf_counter() - start) * 1000
    functions = {}
    for name in HELPERS:
        old, new = baseline[name], getattr(sparse, HELPERS[name])
        expected, actual = old(**args[name]), new(**args[name])
        if name == "coords":
            assert actual[2] == expected[2]
            actual_arrays, expected_arrays = actual[:2], expected[:2]
        else:
            assert list(actual) == list(expected)
            actual_arrays, expected_arrays = actual.values(), expected.values()
        for a, e in zip(actual_arrays, expected_arrays):
            np.testing.assert_array_equal(a, e)
            assert a.dtype == e.dtype
        functions[name + "_baseline"] = lambda old=old, name=name: old(**args[name])
        functions[name + "_candidate"] = lambda new=new, name=name: new(**args[name])
    for call in functions.values():
        for _ in range(3):
            call()
    times = {name: [] for name in functions}
    for round_index in range(rounds):
        order = list(functions) if round_index % 2 == 0 else list(reversed(functions))
        for name in order:
            start = time.perf_counter()
            for _ in range(4):
                functions[name]()
            times[name].append((time.perf_counter() - start) * 1000 / 4)
    memory = {}
    for name in HELPERS:
        memory[name] = {}
        for mode in ("baseline", "candidate"):
            output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                "--points", str(points), "--memory", mode, "--helper", name], env=dict(os.environ), text=True)
            memory[name][mode] = json.loads(output)
    return {"points": points, "unique_edges": args["coords"]["edge_index"].shape[1],
        "nnz": nnz, "shape": shape, "exact_array_and_order_equivalence": True,
        "baseline_extraction_ms": extraction_ms, "fixture_ms": fixture_ms,
        "ms": {name: {"median": statistics.median(values), "min": min(values), "max": max(values)}
               for name, values in times.items()}, "fresh_process_memory": memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=72)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--memory", choices=("baseline", "candidate"))
    parser.add_argument("--helper", choices=tuple(HELPERS), default="coords")
    args = parser.parse_args()
    if args.points < 2 or args.points % 2 or args.rounds < 1:
        parser.error("points must be positive even >=2 and rounds >=1")
    if torch.cuda.is_available():
        parser.error("disable CUDA for this CPU-only probe")
    if args.memory:
        print(json.dumps(memory_probe(args.points, args.memory, args.helper)))
        return
    print(json.dumps({"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "data_dtype": "float64", "components": 2, "baseline_commit": BASE,
        "source_sha256": hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(),
        "scalar_kernel_file_not_modified": _sparse.__file__, "rounds": args.rounds, "warmup": 3,
        "measure": measure(args.points, args.rounds),
        "limits": "synthetic CPU, host load not isolated; RSS includes allocator/runtime, peak delta0 is not zero memory; not GPU/VRAM/accuracy or compiled scalar kernel"}, indent=2))


if __name__ == "__main__":
    main()
