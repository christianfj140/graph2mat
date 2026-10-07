"""Real two-rank Gloo training, isolated from the pytest process."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def _fit(output_dir, layout):
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.strategies import DDPStrategy

    from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    class GradientCoverage(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            missing = [name for name, parameter in pl_module.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            assert not missing, f"Rank {trainer.global_rank}: trainable parameters without gradients: {missing}"

    pl.seed_everything(0, workers=True)
    table = BasisTableWithEdges([PointBasis(kind, R=2, basis=[1]) for kind in ("A", "B")])
    configurations = []
    for index in range(4):
        kinds = ["A", "A", "B", "B"] if layout != "missing_types" else ["A" if index % 2 else "B"] * 2
        configurations.append(BasisConfiguration(
            point_types=kinds, positions=np.array([[i * 0.7, 0., 0.] for i in range(len(kinds))]),
            basis=table, cell=np.eye(3) * 10, pbc=(False, False, False),
            matrix=np.full((len(kinds), len(kinds)), 0.5),
        ))
    module = MatrixDataModule(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=configurations, val_runs=[], loader_threads=0, batch_size=1,
    )
    model = LitMACEMatrixModel(
        basis_table=table, symmetric_matrix=True, hidden_irreps="2x0e+2x1o",
        edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1,
    )
    before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    devices = 1 if layout == "reference" else 2
    strategy = (DDPStrategy(process_group_backend="gloo", start_method="popen", find_unused_parameters=False)
                if devices == 2 else "auto")
    trainer = pl.Trainer(
        accelerator="cpu", devices=devices, strategy=strategy,
        max_epochs=2, limit_train_batches=1, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=output_dir,
        callbacks=[GradientCoverage()] if devices == 2 else [],
    )
    trainer.fit(model, datamodule=module)
    if trainer.is_global_zero:
        changed = any(not torch.equal(parameter.detach(), before[key])
                      for key, parameter in model.named_parameters())
        result = {"world_size": trainer.world_size, "global_step": trainer.global_step,
                  "loss": float(trainer.callback_metrics["train_loss_epoch"]), "weights_changed": changed}
        (Path(output_dir) / "result.json").write_text(json.dumps(result))
        torch.save(model.state_dict(), Path(output_dir) / "weights.pt")
    module.teardown(None)


def _run_training_script(output_dir, layout):
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), str(output_dir), layout],
        cwd=output_dir, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        # The isolated process group belongs to this test, including both ranks.
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Two-rank training timed out:\n" + stdout + stderr)
    (output_dir / "training.log").write_text(stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return json.loads((output_dir / "result.json").read_text())


@pytest.mark.parametrize("layout", ["all_types", "missing_types"])
def test_two_cpu_ranks_train_real_mace_for_multiple_steps(tmp_path, layout):
    import math
    import torch

    training = _run_training_script(tmp_path, layout)
    assert training["world_size"] == 2 and training["global_step"] == 2
    assert math.isfinite(training["loss"]) and training["weights_changed"]
    if layout == "all_types":
        # Identical samples on both ranks must yield the same updates as serial training.
        reference_dir = tmp_path / "reference"
        reference_dir.mkdir()
        reference = _run_training_script(reference_dir, "reference")
        assert reference["world_size"] == 1 and reference["global_step"] == 2
        actual = torch.load(tmp_path / "weights.pt", weights_only=True)
        expected = torch.load(reference_dir / "weights.pt", weights_only=True)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7, msg=name)


if __name__ == "__main__":
    _fit(sys.argv[1], sys.argv[2])
