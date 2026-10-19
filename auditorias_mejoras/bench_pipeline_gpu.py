"""Opt-in 5.3/6.1/6.2/6.4 pipeline measurements; never submits queue jobs.

Run with the shared CUDA Python and PYTHONPATH=auditorias_mejoras/probe_site:src.
Each variant/round runs in a fresh child. Outputs belong to --out exclusively.
"""
import argparse
import copy
import cProfile
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMPAIGN = HERE / "experimentos_readout_w90_6x6_20261010_v1"


def memory_fraction(total, free):
    """Cap our allocator, leaving 30% free at launch (10 points of reserve)."""
    fraction = min(.55, free / total - .30)
    if fraction * total < 512:
        raise RuntimeError("insufficient VRAM headroom")
    return fraction


def gpu_state():
    keys = ("memory.total", "memory.free", "utilization.gpu", "temperature.gpu",
            "clocks.sm", "clocks.mem", "power.draw")
    output = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(keys),
                                      "--format=csv,noheader,nounits"], text=True)
    result = {key: float(value) for key, value in zip(keys, output.strip().split(","))}
    result["time"] = time.time()
    return result


def describe(values):
    median = statistics.median(values)
    return dict(median=median, min=min(values), max=max(values),
                mad=statistics.median(abs(v - median) for v in values))


def loader_options(task, variant):
    if task == "loader":
        options = {"w0_p0": (0, False), "w0_p1": (0, True),
                   "w4_p0": (4, False), "w4_p1": (4, True)}
        if variant not in options:
            raise ValueError("invalid loader variant: " + variant)
        return options[variant]
    workers = 4 if task == "cache" and not variant.endswith("0") else 0
    return workers, bool(workers)


def process_tree_memory(pid, proc_root=Path("/proc")):
    """One simultaneous-ish tree sample; RSS double counts shared pages, PSS does not."""
    pending, seen, rows = [pid], set(), {}
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        root = proc_root / str(current)
        try:
            status = root.joinpath("status").read_text()
            rss = float(status.split("VmRSS:")[1].split()[0]) / 1024
        except (OSError, IndexError):
            continue
        try:
            pss = float(root.joinpath("smaps_rollup").read_text().split("Pss:")[1].split()[0]) / 1024
        except (OSError, IndexError):
            pss = None
        rows[current] = (rss, pss)
        try:
            pending.extend(map(int, root.joinpath("task", str(current), "children").read_text().split()))
        except OSError:
            pass  # A worker may finish between the status and children reads.
    return dict(time=time.time(), pids=sorted(rows), workers=len(rows) - int(pid in rows),
        main_rss_mib=rows.get(pid, (0, None))[0],
        children_rss_mib=sum(value[0] for key, value in rows.items() if key != pid),
        rss_sum_mib=sum(value[0] for value in rows.values()),
        pss_sum_mib=sum(value[1] for value in rows.values())
            if rows and all(value[1] is not None for value in rows.values()) else None)


def variant_order(variants, index):
    shift = (index // 2) % len(variants)
    order = list(variants[shift:] + variants[:shift])
    return list(reversed(order)) if index % 2 else order


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["task"] + "/" + record["system"], {}).setdefault(
            record["variant"], []).append(record)
    result = {}
    for key, variants in groups.items():
        result[key] = {}
        for name, rows in variants.items():
            reference = {"coeff_opt": "gauge_dense" if name.startswith("gauge") else "mse_legacycoeff", "loader": "w0_p0", "cache": "legacy0" if name.endswith("0") else "legacy", "sparse": "legacy", "batching": "fixed",
                         "coeff": "gauge_dense" if name.startswith("gauge") else "mse_dense"}[key.split("/")[0]]
            refs = {r["round"]: r for r in variants.get(reference, [])}
            ratios = []
            for row in rows:
                if row["round"] not in refs:
                    continue
                ref = refs[row["round"]]
                assert row["initial_sha256"] == ref["initial_sha256"], "unpaired initial weights"
                assert row["inputs_sha256"] == ref["inputs_sha256"], "unpaired inputs"
                if key.startswith("sparse"):
                    assert row["exact_sparse_equivalence"] and ref["exact_sparse_equivalence"], "sparse oracle failed"
                ratios.append(statistics.median(row["wall_ms"]) / statistics.median(ref["wall_ms"]))
            result[key][name] = dict(rounds=len(rows), unit=rows[0]["unit"],
                wall_ms=describe([statistics.median(r["wall_ms"]) for r in rows]),
                peak_allocated_mib=describe([r["peak_allocated_mib"] for r in rows]),
                peak_reserved_mib=describe([r["peak_reserved_mib"] for r in rows]),
                paired_wall_ratio=ratios,
                paired_ratio_summary=describe(ratios) if ratios else None)
    return result


def compiled_extensions():
    """Reuse installed scalar kernels while importing current Python source."""
    import sysconfig
    installed = Path(sysconfig.get_paths()["purelib"]) / "graph2mat"
    paths = {}
    for suffix in ("core.data._sparse", "core.modules._labels_resort"):
        relative = suffix.replace(".", "/")
        found = list(installed.glob(relative + "*.so"))
        if len(found) != 1:
            raise RuntimeError("expected one installed extension: " + relative)
        name = "graph2mat." + suffix
        spec = importlib.util.spec_from_file_location(name, found[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        paths[name] = dict(path=str(found[0]), sha256=hashlib.sha256(found[0].read_bytes()).hexdigest())
    return paths


def resolve(value):
    if isinstance(value, str) and value.startswith("graph2mat."):
        value = value.replace("graph2mat.metrics.", "graph2mat.core.data.metrics.")
        module, name = value.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)
    return value


def child(args):
    extensions = compiled_extensions()
    import graph2mat
    import numpy as np
    import torch
    from torch.utils.data import BatchSampler, SequentialSampler
    from torch_geometric.loader import DataLoader
    from graph2mat.bindings.torch import InMemoryData
    from graph2mat.bindings.torch.data.coefficient_targets import CoefficientMatrixMSEProjector
    from graph2mat.core.data import sparse
    from graph2mat.tools.lightning.data import MatrixDataModule, _CostBatchSampler
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from readout_comparison_callback import tensor_hash

    baseline_projector_sha = None
    if args.variant == "mse_legacycoeff":
        original = subprocess.check_output(["git", "show",
            "270d17d1f204293a059baa09e8f1b3756981e2bf:src/graph2mat/bindings/torch/data/coefficient_targets.py"], cwd=ROOT)
        namespace = {"__name__": "projector_reference"}
        exec(compile(original, "projector_reference.py", "exec"), namespace)
        CoefficientMatrixMSEProjector.__call__ = namespace["CoefficientMatrixMSEProjector"].__call__
        baseline_projector_sha = hashlib.sha256(original).hexdigest()

    assert Path(graph2mat.__file__).resolve() == ROOT / "src/graph2mat/__init__.py"
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261013)
    torch.cuda.set_per_process_memory_fraction(args.fraction)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    sparse_oracles = None
    if args.task == "sparse":
        from bench_sparse_vectorizado import baseline_helpers, HELPERS
        old_helpers = baseline_helpers()
        sparse_oracles = (old_helpers["coords"], sparse._blockmatrix_coo_coords)
        if args.variant == "legacy":
            for name, function in old_helpers.items():
                setattr(sparse, HELPERS[name], function)
    if args.task == "cache" and args.variant.startswith("legacy"):
        # Historical read-only access, confined to this child; original snapshots checked below.
        InMemoryData.__getitem__ = lambda self, index: self.data_objects[index]

    config_path = CAMPAIGN / "runs" / args.system / "baseline-seed0/config.yaml"
    synthetic = args.system == "heterogeneous"
    t0 = time.perf_counter()
    setup_profile = cProfile.Profile() if args.profile else None
    if setup_profile:
        setup_profile.enable()
    if synthetic:
        from bench_batching_coste import inputs, BUDGET
        module = inputs("cost")
        model_kwargs = dict(hidden_irreps="16x0e+16x1o+16x2e+16x3o",
            edge_hidden_irreps="16x0e+16x1o+16x2e+16x3o", num_bessel=4,
            max_ell=3, num_interactions=2, keep_last_layer_irreps=True, correlation=1,
            symmetric_matrix=False, n_matrix_components=2)
        batch_size, workers = 5, 0
        budget = BUDGET
    else:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        data = copy.deepcopy(cfg["data"])
        workers, pin_memory = loader_options(args.task, args.variant)
        data["loader_threads"] = workers
        data["store_in_memory"] = args.variant != "lazy"
        if args.task.startswith("coeff") and args.variant.startswith("mse"):
            data.update(matrix_component_policy="h_only", n_matrix_components=1)
        module = MatrixDataModule(**data)
        module.setup("fit")
        model_kwargs = {key: resolve(value) for key, value in cfg["model"].items()}
        model_kwargs.update(symmetric_matrix=module.data_processor.symmetric_matrix,
                            n_matrix_components=module.n_matrix_components)
        if args.task.startswith("coeff") and args.variant.startswith("mse"):
            from graph2mat.core.data.metrics import elementwise_mse
            model_kwargs["loss"] = elementwise_mse
        batch_size = data["batch_size"]
    setup_ms = (time.perf_counter() - t0) * 1000
    if setup_profile:
        setup_profile.disable()
        setup_profile.dump_stats(str(args.output) + ".setup.prof")
    model = LitMACEMatrixModel(basis_table=module.basis_table, **model_kwargs)
    model.log = lambda *unused, **kwargs: None  # Same metric computation, no Trainer/logging backend.
    initial_sha = tensor_hash(model.state_dict().items())
    model.coefficient_training = args.task.startswith("coeff") and args.variant.endswith("coeff")
    if model.coefficient_training and args.variant.startswith("mse"):
        module.set_coefficient_mse_projector(CoefficientMatrixMSEProjector(model.model.matrix_readouts))
    dataset = module.train_dataset
    def samples_hash(samples):
        hashes = (tensor_hash((key, value) for key, value in sample.items() if torch.is_tensor(value))
                  for sample in samples)
        return hashlib.sha256("".join(hashes).encode()).hexdigest()
    input_sha = samples_hash(dataset[i] for i in range(len(dataset)))
    if args.task == "batching":
        sizes = module.basis_table.basis_size.astype(np.int64)
        costs = []
        for sample in (dataset[i] for i in range(len(dataset))):
            orbitals = sizes[sample.point_types.numpy()]
            edges = sample.edge_index.numpy()
            costs.append(int(module.n_matrix_components * (np.square(orbitals).sum()
                + (orbitals[edges[0]] * orbitals[edges[1]]).sum())))
        if not synthetic:
            budget = max(costs) * batch_size
        sampler = SequentialSampler(dataset)
        batches = (_CostBatchSampler(sampler, batch_size, False, costs, budget)
                   if args.variant == "cost" else BatchSampler(sampler, batch_size, False))
        indices = list(batches)
        assert [i for batch in indices for i in batch] == list(range(len(dataset)))
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=0)
    else:
        indices = [list(range(start, min(start + batch_size, len(dataset))))
                   for start in range(0, len(dataset), batch_size)]
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
            persistent_workers=bool(workers), pin_memory=pin_memory)
    loader = module._with_type_layout(loader, coefficient_projector=module._coefficient_mse_projector)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    # Compare the certified CUDA loss and every parameter gradient before timing.
    if args.profile:
        profile = cProfile.Profile()
        cpu_check = profile.runcall(next_batch)
        profile.dump_stats(str(args.output) + ".collate.prof")
    else:
        cpu_check = next_batch()
    check = model.transfer_batch_to_device(cpu_check, "cuda", 0)
    numeric = dict(coefficient_route=model._can_train_coefficients(check))
    if sparse_oracles is not None:
        with torch.no_grad():
            output = model(check)
        predictions = {key: output[key].detach().cpu() for key in ("node_labels", "edge_labels")}
        active = sparse._blockmatrix_coo_coords
        reconstructions = []
        for function in sparse_oracles:
            sparse._blockmatrix_coo_coords = function
            reconstructions.append(module.data_processor.matrix_from_data(cpu_check,
                predictions=predictions, out_format=graph2mat.Formats.SCIPY_CSR, threshold=0.))
        sparse._blockmatrix_coo_coords = active
        for old_matrix, new_matrix in zip(*reconstructions):
            assert old_matrix.shape == new_matrix.shape
            for key in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(old_matrix, key), getattr(new_matrix, key))
        numeric["exact_sparse_equivalence"] = True
        del output, predictions, reconstructions, old_matrix, new_matrix
    if args.task.startswith("coeff") and args.variant.endswith("coeff") and args.variant.startswith("mse"):
        assert numeric["coefficient_route"]
        out = model(check)
        reference, _ = model.loss_fn(nodes_pred=out["node_labels"], edges_pred=out["edge_labels"],
            nodes_ref=check.point_labels, edges_ref=check.edge_labels, batch=check,
            basis_table=model.basis_table, out=out, model=model.model)
        parameters = list(model.parameters())
        ref_grads = torch.autograd.grad(reference, parameters, allow_unused=True)
        loss = model.training_step(check, 0)["loss"]
        torch.testing.assert_close(loss, reference, rtol=3e-5, atol=3e-6)
        actual_grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        differences = []
        for actual, ref in zip(actual_grads, ref_grads):
            assert (actual is None) == (ref is None)
            if actual is not None:
                torch.testing.assert_close(actual, ref, rtol=3e-4, atol=3e-5)
                differences.append(float((actual-ref).abs().max()))
        numeric.update(loss=float(loss.detach()), dense_loss=float(reference.detach()), max_gradient_error=max(differences))
        del actual_grads, ref_grads, out, reference, loss, actual, ref
    if args.task.startswith("coeff") and args.variant == "gauge_coeff":
        assert not numeric["coefficient_route"]
    del check, cpu_check
    # Release only inactive preflight allocations; never empty the cache between timed steps.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    iterator = iter(loader)
    wall, data_times, transfer_times, forward_times, backward_times, optimizer_times, writer_times = ([] for _ in range(7))
    matrix_sha = None
    unit = "epoch" if args.task == "batching" else "step"
    updates_per_unit = len(indices) if unit == "epoch" else 1
    first_unit_ms = None
    loss_history = []
    for iteration in range(args.warmup + args.steps):
        if iteration == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        phases = [0.] * 6
        events = []
        unit_losses = []
        for _ in range(updates_per_unit):
            t = time.perf_counter()
            cpu = next_batch()
            phases[0] += (time.perf_counter() - t) * 1000
            markers = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            markers[0].record()
            batch = model.transfer_batch_to_device(cpu, "cuda", 0)
            markers[1].record()
            if args.task == "sparse":
                with torch.no_grad():
                    prediction = model(batch)
                markers[2].record()
                prediction = {key: prediction[key].detach().cpu() for key in ("node_labels", "edge_labels")}
                t = time.perf_counter()
                matrices = module.data_processor.matrix_from_data(cpu, predictions=prediction,
                    out_format=graph2mat.Formats.SCIPY_CSR, threshold=0.)
                phases[5] += (time.perf_counter() - t) * 1000
                if matrix_sha is None:
                    digest = hashlib.sha256()
                    for matrix in matrices:
                        for array in (matrix.data, matrix.indices, matrix.indptr):
                            digest.update(array.tobytes())
                    matrix_sha = digest.hexdigest()
                markers[3].record()
                markers[4].record()
                del prediction, matrices
            else:
                optimizer.zero_grad(set_to_none=True)
                result = model.training_step(batch, iteration)
                unit_losses.append(result["loss"].detach())
                markers[2].record()
                result["loss"].backward()
                markers[3].record()
                optimizer.step()
                markers[4].record()
                if iteration == 0:
                    assert torch.isfinite(result["loss"]).all()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    assert grads and all(torch.isfinite(grad).all() for grad in grads)
                    del grads
                del result
            events.append(markers)
            del batch, cpu
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        values = [float(loss) for loss in unit_losses]
        assert all(math.isfinite(value) for value in values), "nonfinite training loss"
        if first_unit_ms is None:
            first_unit_ms = elapsed
        if iteration >= args.warmup:
            loss_history.append(values)
            wall.append(elapsed)
            data_times.append(phases[0])
            transfer_times.append(sum(e[0].elapsed_time(e[1]) for e in events))
            forward_times.append(sum(e[1].elapsed_time(e[2]) for e in events))
            backward_times.append(sum(e[2].elapsed_time(e[3]) for e in events) if args.task != "sparse" else 0.)
            optimizer_times.append(sum(e[3].elapsed_time(e[4]) for e in events))
            writer_times.append(phases[5])
    peak_allocated = torch.cuda.max_memory_allocated()/2**20
    peak_reserved = torch.cuda.max_memory_reserved()/2**20
    assert all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite final parameters"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "nonfinite final gradients"
    after_sha = samples_hash(dataset[i] for i in range(len(dataset)))
    assert input_sha == after_sha, "benchmark mutated cached samples"
    result = dict(task=args.task, system=args.system, variant=args.variant, round=args.round,
        unit=unit, warmup=args.warmup, measured_units=args.steps, wall_ms=wall, loss_history=loss_history,
        data_ms=data_times, h2d_cuda_ms=transfer_times, forward_loss_cuda_ms=forward_times,
        backward_cuda_ms=backward_times, optimizer_cuda_ms=optimizer_times, writer_cpu_ms=writer_times,
        first_unit_ms=first_unit_ms, setup_ms=setup_ms, numeric=numeric,
        baseline_projector_sha256=baseline_projector_sha,
        initial_sha256=initial_sha, inputs_sha256=input_sha, matrix_sha256=matrix_sha,
        exact_sparse_equivalence=numeric.get("exact_sparse_equivalence"),
        peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved,
        workers=workers, pin_memory=loader.pin_memory, persistent_workers=bool(workers),
        profiled=args.profile, batch_indices=indices, updates_per_unit=updates_per_unit,
        structures_per_epoch=len(dataset), extensions=extensions,
        torch=torch.__version__, cuda=torch.version.cuda, dtype="float32",
        gpu=torch.cuda.get_device_name(), allocator_fraction=args.fraction,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if not synthetic else None,
        host_rss_mib=int(Path("/proc/self/status").read_text().split("VmRSS:")[1].split()[0])/1024,
        scope="actual training_step + Adam; log backend disabled, epoch metrics retained; "
              "end-to-end wall includes loading/H2D; CUDA events diagnostic; no accuracy experiment")
    if args.task == "batching":
        result.update(cost_budget=budget, batch_costs=[sum(costs[i] for i in b) for b in indices])
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("RESULT", args.task, args.system, args.variant, statistics.median(wall), flush=True)
    if workers:
        iterator._shutdown_workers()
    module.teardown(None)


def guarded_command(command, log_path, telemetry_path, env):
    """Supervise one owned process group; usable for measurements and CUDA pytest."""
    observations = []
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                state = gpu_state()
                state["host"] = process_tree_memory(proc.pid)
                observations.append(state)
                if state["memory.free"] < .25 * state["memory.total"] or state["temperature.gpu"] >= 80:
                    raise RuntimeError("VRAM/temperature guard: " + str(state))
                time.sleep(.5)
        finally:
            try:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                # The leader can exit before a worker that ignores TERM.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            finally:
                telemetry_path.write_text(json.dumps(observations, indent=2)+"\n")
    return proc.returncode


def validate_cuda(args):
    args.out.mkdir(parents=True, exist_ok=True)
    before = gpu_state()
    fraction = memory_fraction(before["memory.total"], before["memory.free"])
    code = ("import bench_pipeline_gpu as p; p.compiled_extensions(); import torch; "
            "torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(" + str(fraction) + "); "
            "torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; "
            "import pytest; raise SystemExit(pytest.main(["
            "'src/graph2mat/tools/lightning/tests/test_coefficient_cuda_training.py', '-q', '-s', '--tb=short']))")
    env = {**os.environ, "G2M_GPU_GUARDED": "1", "CUDA_VISIBLE_DEVICES": "0",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", ROOT / "src", HERE)))}
    status = guarded_command([sys.executable, "-c", code], args.out / "cuda-tests.log",
                             args.out / "cuda-tests.gpu.json", env)
    if status:
        raise RuntimeError("CUDA validation failed; see " + str(args.out / "cuda-tests.log"))


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "src/graph2mat").rglob("*.py")):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    specs = []
    for task in args.tasks.split(","):
        variants = {"loader": ("w0_p0", "w0_p1", "w4_p0", "w4_p1"),
                    "coeff_opt": ("mse_legacycoeff", "mse_coeff", "mse_dense", "gauge_dense", "gauge_coeff"),
                    "cache": ("legacy", "safe", "lazy", "legacy0", "safe0"),
                    "sparse": ("legacy", "vectorized"), "batching": ("fixed", "cost"),
                    "coeff": ("mse_dense", "mse_coeff", "gauge_dense", "gauge_coeff")}[task]
        for system in ("w90", "6x6", "heterogeneous") if task == "batching" else ("w90", "6x6"):
            specs.append((task, system, variants))
    manifest = dict(head=head, source_sha256=source_hash.hexdigest(), tasks=args.tasks,
                    rounds=args.rounds, steps=args.steps, warmup=args.warmup,
                    profiled=args.profile,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reserve_free_fraction=.20, stop_free_fraction=.25,
                    documentation=["https://docs.pytorch.org/docs/stable/notes/cuda.html",
                        "https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html"],
                    limits="shared GPU contexts permitted by user; concurrent activity can confound timing; "
                           "20% reserve guarded with 25% abort threshold, not a lock on foreign allocations")
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "probe manifest changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    records = []
    for task, system, variants in specs:
        for index in range(args.rounds):
            order = variant_order(variants, index)
            for variant in order:
                label = f"{task}-{system}-{index:02d}-{variant}"
                output = args.out / (label + ".json")
                if not output.exists():
                    partial = output.with_suffix(".partial")
                    before = gpu_state()
                    fraction = memory_fraction(before["memory.total"], before["memory.free"])
                    command = [sys.executable, str(Path(__file__).resolve()), "child", "--task", task,
                        "--system", system, "--variant", variant, "--round", str(index),
                        "--steps", str(args.steps), "--warmup", str(args.warmup),
                        "--fraction", str(fraction), "--output", str(partial)]
                    if args.profile:
                        command.append("--profile")
                    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", ROOT / "src", HERE)))}
                    print("START", label, flush=True)
                    status = guarded_command(command, args.out / (label + ".log"),
                                             args.out / (label + ".gpu.json"), env)
                    if status:
                        raise RuntimeError(f"{label} exit {status}; see {label}.log")
                    partial.replace(output)
                records.append(json.loads(output.read_text()))
                summary = summarize(records)
                (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                print("DONE", label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "child", "validate"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tasks", default="cache,sparse,batching,coeff")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--task")
    parser.add_argument("--system")
    parser.add_argument("--variant")
    parser.add_argument("--round", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true", help="separate cProfile diagnostic; not primary timings")
    args = parser.parse_args()
    if min(args.rounds, args.steps, args.warmup) < 1:
        parser.error("positive rounds, steps and warmup required")
    if args.action == "validate":
        validate_cuda(args)
    else:
        run(args) if args.action == "run" else child(args)


if __name__ == "__main__":
    main()
