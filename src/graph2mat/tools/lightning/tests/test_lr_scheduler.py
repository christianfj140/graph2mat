"""The CLI attaches ``lr_scheduler`` to the model's own optimizer (no ``optimizer:`` in the config).

Lightning only attaches a configured scheduler to an optimizer given in the config, and the
models build their own (MACE parameter groups), so ``optimizer: null`` + ``lr_scheduler:``
used to train at a constant learning rate without any warning.
"""
import json

import pytest
import pytorch_lightning as pl
import sisl

from graph2mat import AtomicTableWithEdges
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.cli import OrbitalMatrixCLI
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

ATOM = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
BASIS = AtomicTableWithEdges([ATOM])
COSINE = {"class_path": "torch.optim.lr_scheduler.CosineAnnealingLR", "init_args": {"T_max": 4, "eta_min": 0.0}}
QUIET = ["--trainer.accelerator=cpu", "--trainer.logger=false", "--trainer.enable_checkpointing=false",
         "--trainer.enable_progress_bar=false", "--trainer.enable_model_summary=false", "--trainer.callbacks=[]"]


class _Data(MatrixDataModule):
    def __init__(self, **kwargs):
        kwargs["basis_table"] = BASIS
        super().__init__(**kwargs)


class _Model(LitMACEMatrixModel):
    def __init__(self, **kwargs):
        kwargs["basis_table"] = BASIS
        super().__init__(**kwargs)


def _args(tmp_path, *extra):
    g = sisl.Geometry([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atoms=ATOM, lattice=[10.0] * 3)
    m = sisl.DensityMatrix(g)
    m[0, 0], m[1, 1], m[0, 1], m[1, 0] = 1.0, 2.0, 0.25, 0.25
    m.write(tmp_path / "sample.nc")
    return ["fit", f"--data.root_dir={tmp_path}", "--data.train_runs=sample.nc", "--data.val_runs=sample.nc",
            "--data.out_matrix=density_matrix", "--data.symmetric_matrix=true", "--data.sub_point_matrix=false",
            "--data.loader_threads=0", "--model.hidden_irreps=2x0e+2x1o", "--model.edge_hidden_irreps=2x0e+2x1o",
            "--model.num_bessel=2", "--model.max_ell=1", "--trainer.max_epochs=4", *QUIET, *extra]


def _cli(args):
    return OrbitalMatrixCLI(model_class=_Model, datamodule_class=_Data, save_config_callback=None, args=args)


def test_scheduler_without_optimizer_is_attached(tmp_path):
    cli = _cli(_args(tmp_path, "--lr_scheduler", json.dumps(COSINE)))
    assert len(cli.trainer.lr_scheduler_configs) == 1
    groups = cli.trainer.optimizers[0].param_groups
    # Cosine with T_max = 4 epochs, stepped once per epoch: eta_min after 4 epochs.
    assert max(g["lr"] for g in groups) == pytest.approx(0.0, abs=1e-12)
    # The model's own optimizer (MACE parameter groups) is the one being scheduled.
    assert {g["name"] for g in groups} >= {"embedding", "readouts"}


def test_without_scheduler_the_learning_rate_is_constant(tmp_path):
    cli = _cli(_args(tmp_path))
    assert cli.trainer.lr_scheduler_configs == []
    assert {g["lr"] for g in cli.trainer.optimizers[0].param_groups} == {1e-3}


def test_scheduler_with_stage_learning_rates_is_rejected(tmp_path):
    stages = json.dumps([{"name": "a", "max_epochs": 2, "optim_lr": 1e-3}, {"name": "b", "max_epochs": 2, "optim_lr": 1e-4}])
    with pytest.raises(ValueError, match="optim_lr"):
        _cli(_args(tmp_path, "--lr_scheduler", json.dumps(COSINE), f"--model.training_stages={stages}"))


def test_resuming_a_checkpoint_trained_without_scheduler_is_rejected(tmp_path):
    args = _args(tmp_path)
    module = _Data(out_matrix="density_matrix", root_dir=str(tmp_path), symmetric_matrix=True,
                   sub_point_matrix=False, loader_threads=0, train_runs="sample.nc", val_runs="sample.nc")
    model = _Model(symmetric_matrix=True, hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
                   num_bessel=2, max_ell=1)
    trainer = pl.Trainer(accelerator="cpu", fast_dev_run=True, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=module)
    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    with pytest.raises(ValueError, match="without a learning-rate scheduler"):
        _cli([*args, "--ckpt_path", str(ckpt), "--lr_scheduler", json.dumps(COSINE)])
