"""Exercise actual campaign run() bodies with IO/training dependencies replaced.

The AST extraction only avoids module-level imports of another repository. The
functions themselves run unchanged, against synthetic samples; no fit/GPU/SCF.
"""
import ast
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main():
    repo = Path(__file__).resolve().parents[1] / "source"
    results = {}
    with tempfile.TemporaryDirectory(prefix="g2m-updates-") as temporary:
        tmp = Path(temporary)
        for filename in ("run_soc_campaign.py", "run_magnetic_campaign.py"):
            source = repo / "auditorias_mejoras" / filename
            module = ast.parse(source.read_text(), filename=str(source))
            run_ast = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run")
            records, configs = {}, []
            def build(*args, **kwargs):
                return {"trainer": {"max_epochs": kwargs["max_epochs"]}}
            def patch_config(config, patch):
                for key, value in patch.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        patch_config(config[key], value)
                    else:
                        config[key] = value
            def train(config, directory):
                configs.append(config)
                directory.mkdir(parents=True, exist_ok=True)
                return directory / "fake.ckpt", 0., 0.
            c = SimpleNamespace(TRAIN_KW={}, build_graph2mat_config=build,
                apply_config_patch=patch_config, _cosine=lambda epochs: {},
                run_training=train, write_json=lambda path, value: records.update({str(path): value}))
            s4 = SimpleNamespace(torch_backend_preflight=lambda: {"effective_backend": "cpu"},
                build_graph2mat_config=build)
            samples = [SimpleNamespace(sample_id=str(i)) for i in range(10)]
            bench = SimpleNamespace(samples=lambda *args, **kwargs: samples,
                BASIS=tmp / "basis", WORK=tmp / "work")
            ns = dict(Path=Path, c=c, s4=s4, bench=bench, shutil=shutil,
                MATERIAL="synthetic", RUNS=tmp / filename, BASIS_GLOB="unused", MODEL={},
                HIDDEN="2x0e", EDGE_HIDDEN="2x0e", UPDATES=8000, BATCH=2,
                ARMS={"fe": ["delta"]},
                runs_dir=lambda material: tmp / filename,
                delta_sample=lambda sample, delta: (sample, 0.),
                arm_sample=lambda material, sample, arm: sample,
                samples=lambda *args, **kwargs: samples,
                evaluate=lambda *args, **kwargs: {})
            # The non-smoke magnetic path checks only existence of this audit file.
            (bench.WORK / "fe").mkdir(parents=True, exist_ok=True)
            (bench.WORK / "fe" / "moment_audit.json").write_text("{}")
            exec(compile(ast.Module(body=[run_ast], type_ignores=[]), str(source), "exec"), ns)
            if filename.startswith("run_soc"):
                ns["run"]("delta", seed=0, updates=7)
            else:
                ns["run"]("fe", "delta", seed=0, updates=7)
            result = next(v for k, v in records.items() if k.endswith("/result.json"))
            epochs = configs[0]["trainer"]["max_epochs"]
            actual_updates = epochs * 5  # ceil(10 structures / BATCH=2)
            results[filename] = dict(requested_updates=7, recorded_updates=result["updates"],
                training_epochs=epochs, actual_updates=actual_updates)
            assert result["updates"] == 7 and actual_updates == 5
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
