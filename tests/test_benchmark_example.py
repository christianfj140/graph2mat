"""docs/examples/benchmarks: the Delta-learning workflow runs end to end, the recipe is valid."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "benchmarks"


def test_recipe_class_paths_exist():
    recipe = yaml.safe_load((EXAMPLE / "delta_recipe.yaml").read_text())
    paths = [recipe["model"]["loss"], recipe["model"]["node_block_readout"], recipe["model"]["edge_block_readout"],
             recipe["lr_scheduler"]["class_path"]]
    for path in paths:
        # The CLI reads "graph2mat.metrics.X" as graph2mat.core.data.metrics.X.
        module, name = path.replace("graph2mat.metrics.", "graph2mat.core.data.metrics.").rsplit(".", 1)
        assert hasattr(importlib.import_module(module), name), path
    assert recipe["lr_scheduler"]["init_args"]["T_max"] == recipe["trainer"]["max_epochs"]


def test_delta_workflow_end_to_end(tmp_path):
    """prepare (fake SIESTA), a perfect "prediction" of H - H0, evaluate: zero error."""
    sisl = pytest.importorskip("sisl")
    from graph2mat.tools.conftest import _FAKE_SIESTA, make_hamiltonian, write_fdf
    from graph2mat.tools.siesta.delta import write_delta_sample

    for split in ("train", "validation", "test"):
        for i in range(2):
            run = tmp_path / "runs" / split / f"s{i}"
            H = make_hamiltonian(onsite=0.1 * i)
            write_fdf(run, H.geometry, label="g")
            H.write(str(run / "g.TSHS"))
    make_hamiltonian(onsite=-0.5, hopping=-2.0).write(str(tmp_path / "h0src.TSHS"))
    (tmp_path / "fake_siesta.py").write_text(_FAKE_SIESTA)
    env = {"FAKE_SIESTA_TSHS": str(tmp_path / "h0src.TSHS")}
    script = str(EXAMPLE / "delta_workflow.py")
    subprocess.run([sys.executable, script, "prepare", "runs", "--siesta", f"{sys.executable} {tmp_path / 'fake_siesta.py'}",
                    "--spectral-window"], cwd=tmp_path, check=True, env={**os.environ, **env})
    assert sorted(p.name for p in (tmp_path / "delta" / "train" / "s0").iterdir()) == ["RUN.fdf", "g.TSHS",
                                                                                       "spectral_window.npz"]
    for run in (tmp_path / "runs" / "test").iterdir():  # the model's output: H - H0, exactly
        label = write_delta_sample(run, run / "g.TSHS", tmp_path / "h0" / "test" / run.name / "g.TSHS",
                                   tmp_path / "exact" / run.name)
        sisl.get_sile(str(label)).read_hamiltonian().write(str(run / "ML_prediction.HSX"))
    subprocess.run([sys.executable, script, "evaluate", "runs", "--out", "m.json"], cwd=tmp_path, check=True)
    mean = json.loads((tmp_path / "m.json").read_text())["mean"]
    assert mean["H_MAE_meV"] < 1e-9 and mean["H_MAE_gauge_meV"] < 1e-9
