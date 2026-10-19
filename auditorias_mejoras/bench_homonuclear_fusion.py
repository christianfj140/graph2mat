"""Guarded, paired opt-in measurements; no queue submissions or scientific fits."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import bench_pipeline_gpu as pipeline

HERE, ROOT = pipeline.HERE, pipeline.ROOT


def summarize(records):
    if not records:
        raise ValueError("empty campaign")
    result = {}
    for system in sorted({row["system"] for row in records}):
        pairs = {"off": {}, "on": {}}
        for row in (r for r in records if r["system"] == system):
            variant, index = row["fusion_variant"], row["round"]
            if variant not in pairs or index in pairs[variant]:
                raise ValueError("unknown variant or duplicate round")
            if row.get("aborted") or not row.get("preflight", {}).get("passed"):
                raise ValueError("aborted or uncertified record")
            if not row["wall_ms"] or any(not math.isfinite(t) or t <= 0 for t in row["wall_ms"]):
                raise ValueError("invalid timing")
            pairs[variant][index] = row
        if not pairs["off"] or pairs["off"].keys() != pairs["on"].keys():
            raise ValueError("missing paired round")
        if set(pairs["off"]) != set(range(max(pairs["off"]) + 1)):
            raise ValueError("missing campaign round")
        ratios, allocated_ratios, reserved_ratios = [], [], []
        for index in sorted(pairs["off"]):
            off, on = pairs["off"][index], pairs["on"][index]
            for key in ("initial_sha256", "inputs_sha256", "recipe_sha256", "warmup", "measured_steps"):
                if off[key] != on[key]:
                    raise ValueError("unpaired " + key)
            ratios.append(statistics.median(on["wall_ms"]) / statistics.median(off["wall_ms"]))
            for key, values in (("peak_allocated_mib", allocated_ratios), ("peak_reserved_mib", reserved_ratios)):
                if any(not math.isfinite(r[key]) or r[key] <= 0 for r in (off, on)):
                    raise ValueError("invalid VRAM peak")
                values.append(on[key] / off[key])
            if (not all(math.isfinite(r["preflight"]["reference_loss"]) for r in (off, on))
                    or not math.isclose(off["preflight"]["reference_loss"], on["preflight"]["reference_loss"],
                                        rel_tol=5e-5, abs_tol=5e-6)):
                raise ValueError("different initial loss")
        result[system] = dict(paired_wall_ratio=ratios, ratio=pipeline.describe(ratios),
            paired_allocated_ratio=allocated_ratios, paired_reserved_ratio=reserved_ratios,
            stable_improvement=len(ratios) >= 4 and all(r < 1 for r in ratios),
            variants={variant: {key: pipeline.describe([statistics.median(row[key]) if isinstance(row[key], list)
                                                        else row[key] for row in group.values()])
                                for key in ("wall_ms", "peak_allocated_mib", "peak_reserved_mib")}
                      for variant, group in pairs.items()})
    return result


def environment():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "G2M_GPU_GUARDED": "1",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "PYTHONPATH": os.pathsep.join(map(str, (HERE / "probe_site", ROOT / "src", HERE)))}


def available_fraction():
    foreign = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    if foreign:
        raise RuntimeError("GPU occupied; no job launched or queued: " + foreign)
    state = pipeline.gpu_state()
    if state["temperature.gpu"] >= 80:
        raise RuntimeError("GPU temperature limit")
    return pipeline.memory_fraction(state["memory.total"], state["memory.free"])


def validate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    fraction = available_fraction()
    code = ("import bench_pipeline_gpu as p; p.compiled_extensions(); import graph2mat; import torch; "
            "torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(" + str(fraction) + "); "
            "torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; "
            "import pytest; raise SystemExit(pytest.main(["
            "'src/graph2mat/tools/lightning/tests/test_homonuclear_fusion_model.py', "
            "'-q', '-s', '-k', 'cuda or optional_compile', '--tb=short']))")
    status = pipeline.guarded_command([sys.executable, "-c", code], args.out / "cuda-tests.log",
                                       args.out / "cuda-tests.gpu.json", environment())
    if status:
        raise RuntimeError("CUDA certification failed; see " + str(args.out / "cuda-tests.log"))


def child(args):
    pipeline.compiled_extensions()
    import graph2mat
    import torch
    import graph2mat.tools.lightning.models.mace as mace_module

    assert Path(graph2mat.__file__).resolve() == ROOT / "src/graph2mat/__init__.py"
    original = mace_module.LitMACEMatrixModel
    preflight = {}

    def factory(*values, **kwargs):
        options = dict(kwargs.get("edge_block_readout_kwargs") or {})
        options["fuse_homonuclear"] = args.variant == "on"
        kwargs["edge_block_readout_kwargs"] = options
        started = time.perf_counter()
        model = original(*values, **kwargs)
        print("CONSTRUCTED", args.system, args.variant, flush=True)
        preflight["construction_ms"] = (time.perf_counter() - started) * 1000
        preflight["edge_options"] = options
        blocks = [m for m in model.modules() if getattr(m, "symm_transpose", False)
                  and type(getattr(m, "operation", None)).__name__ == "E3nnLinearEdgeBlock"]
        if not blocks or (args.variant == "on" and not all(m._fused_homonuclear for m in blocks)):
            raise RuntimeError("unexpected eligibility/fallback")
        preflight["eligible_blocks"] = len(blocks)
        preflight["metadata_bytes"] = sum(b.numel() * b.element_size() for name, b in model.named_buffers()
                                           if "_homonuclear_" in name)

        def check(owner, inputs):
            handle.remove()
            saved = [m._fused_homonuclear for m in blocks]
            try:
                for m in blocks:
                    m._fused_homonuclear = False
                # Match the measured grad context: no_grad selects another costly JIT specialization.
                expected = owner(*inputs)
                for m, active in zip(blocks, saved):
                    m._fused_homonuclear = active
                calls = []

                def record(module, values, keywords):
                    if keywords.get("_homonuclear"):
                        pairs = [v for v in keywords.values() if isinstance(v, tuple)]
                        calls.append(pairs[0][0].shape[0])

                hooks = [m.operation.register_forward_pre_hook(record, with_kwargs=True) for m in blocks]
                try:
                    actual = owner(*inputs)
                finally:
                    for h in hooks:
                        h.remove()
                errors = {}
                for key in ("node_labels", "edge_labels"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=5e-5, atol=5e-6)
                    errors[key] = float((actual[key] - expected[key]).detach().abs().max())
                batch = inputs[0]

                def initial_loss(out):
                    value, _ = model.loss_fn(nodes_pred=out["node_labels"], edges_pred=out["edge_labels"],
                        nodes_ref=batch.point_labels, edges_ref=batch.edge_labels, batch=batch,
                        basis_table=model.basis_table, out=out, model=model.model)
                    return value

                expected_loss, actual_loss = initial_loss(expected), initial_loss(actual)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=5e-5, atol=5e-6)
                if args.variant == "on" and not calls:
                    raise RuntimeError("fused operation was not evaluated")
                preflight.update(passed=True, max_abs_errors=errors, fused_rows=calls,
                                 reference_loss=float(expected_loss.detach()), actual_loss=float(actual_loss.detach()))
                # Do not let the two validation graphs inflate the measured allocator reserve.
                del expected, actual, expected_loss, actual_loss
                torch.cuda.empty_cache()
                preflight["cache_cleared_before_warmup"] = True
                print("PREFLIGHT", args.system, args.variant, preflight["reference_loss"], flush=True)
            finally:
                for m, active in zip(blocks, saved):
                    m._fused_homonuclear = active

        # Lightning training_step calls MatrixMACE directly, bypassing Lightning.forward.
        handle = model.model.register_forward_pre_hook(check)
        return model

    mace_module.LitMACEMatrixModel = factory
    if args.system.startswith("df"):
        from bench_setup_df import df_child
        df_child(argparse.Namespace(points=int(args.system[2:]), round=args.round, warmup=args.warmup,
                                    steps=args.steps, fraction=args.fraction, output=args.output))
    else:
        pipeline.child(argparse.Namespace(task="coeff", system=args.system, variant="gauge_dense",
            round=args.round, warmup=args.warmup, steps=args.steps, fraction=args.fraction,
            output=args.output, profile=False))
    row = json.loads(args.output.read_text())
    row.update(system=args.system, fusion_variant=args.variant, preflight=preflight,
               measured_steps=args.steps)
    if "model_initial_sha256" in row:
        row["initial_sha256"] = row["model_initial_sha256"]
    row["recipe_sha256"] = row.get("config_sha256") or hashlib.sha256(
        (HERE / "bench_setup_df.py").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(row, indent=2) + "\n")


def source_hashes():
    paths = [*sorted((ROOT / "src/graph2mat").rglob("*.py")), Path(__file__),
             *[HERE / name for name in ("bench_pipeline_gpu.py", "bench_setup_df.py",
                                       "bench_cache_tensores_cpu.py", "bench_readout_rango_bajo.py",
                                       "readout_comparison_callback.py")]]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(args):
    if args.rounds < 1 or args.warmup < 1 or args.steps < 1:
        raise ValueError("positive rounds, warmup and steps required")
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    manifest = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    hashes=hashes, rounds=args.rounds, warmup=args.warmup, steps=args.steps,
                    primary="df72 wall_ms", reserve_initial=.30, stop_free=.25, allocator_cap=.55)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    for name in (Path(__file__).name, "bench_pipeline_gpu.py", "bench_setup_df.py"):
        shutil.copyfile(HERE / name, args.out / name)
    for system in ("w90", "6x6"):
        config = pipeline.CAMPAIGN / "runs" / system / "baseline-seed0/config.yaml"
        shutil.copyfile(config, args.out / (system + "-scientific-config.yaml"))
    records = []
    for index in range(args.rounds):
        for system in ("df16", "df72", "w90", "6x6"):
            for variant in pipeline.variant_order(("off", "on"), index):
                if source_hashes() != hashes:
                    raise RuntimeError("source changed during measurements")
                fraction = available_fraction()
                label = f"{system}-{index:02d}-{variant}"
                output = args.out / (label + ".json")
                partial = output.with_suffix(".partial")
                print("START", label, flush=True)
                command = [sys.executable, __file__, "child", "--system", system, "--variant", variant,
                    "--round", str(index), "--warmup", str(args.warmup), "--steps", str(args.steps),
                    "--fraction", str(fraction), "--output", str(partial)]
                status = pipeline.guarded_command(command, args.out / (label + ".log"),
                                                   args.out / (label + ".gpu.json"), environment())
                if status or not partial.exists() or source_hashes() != hashes:
                    raise RuntimeError("probe failed or source changed: " + label)
                row = json.loads(partial.read_text())
                if not row["preflight"].get("passed"):
                    raise RuntimeError("preflight missing: " + label)
                partial.replace(output)
                records.append(row)
    (args.out / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("validate", "run", "child"):
        p = sub.add_parser(action)
        if action != "child":
            p.add_argument("--out", type=Path, required=True)
        if action in ("run", "child"):
            p.add_argument("--warmup", type=int, default=10)
            p.add_argument("--steps", type=int, default=30)
        if action == "run":
            p.add_argument("--rounds", type=int, default=4)
        if action == "child":
            p.add_argument("--system", choices=("df16", "df72", "w90", "6x6"), required=True)
            p.add_argument("--variant", choices=("off", "on"), required=True)
            p.add_argument("--round", type=int, required=True)
            p.add_argument("--fraction", type=float, required=True)
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"validate": validate, "run": run, "child": child}[args.action](args)


if __name__ == "__main__":
    main()
