"""Exercise campaign entry points without importing an external benchmark repo."""
import __future__
import ast
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


def _function(filename, name, namespace):
    # Only module-level setup imports the external harness. Execute the actual
    # function body, with training/process IO replaced at its boundary.
    path = Path(__file__).parents[1] / "auditorias_mejoras" / filename
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def _patch_config(config, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            _patch_config(config[key], value)
        else:
            config[key] = value


@pytest.mark.parametrize("filename", ["run_soc_campaign.py", "run_magnetic_campaign.py"])
@pytest.mark.parametrize("updates", [0, -1, 2, 7, 10])
def test_requested_update_budget_is_enforced(filename, updates, tmp_path):
    samples = [SimpleNamespace(sample_id=str(i)) for i in range(10)]
    configs, records = [], {}

    def train(config, directory):
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader

        class ToyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)

            def training_step(self, batch, batch_idx):
                return self.layer(batch).square().mean()

            def validation_step(self, batch, batch_idx):
                self.log("val_loss", self.layer(batch).square().mean())

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        configs.append(config)
        directory.mkdir(parents=True)
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=directory, filename="best-{step}", monitor="val_loss")
        contract = {key: config["trainer"][key] for key in
                    ("max_steps", "max_epochs", "check_val_every_n_epoch", "val_check_interval")
                    if key in config["trainer"]}
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False,
            enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
            callbacks=[checkpoint], **contract)
        loader = DataLoader(torch.ones(10, 1), batch_size=2)
        trainer.fit(ToyModel(), train_dataloaders=loader, val_dataloaders=loader)
        assert trainer.global_step == updates
        assert checkpoint.best_model_path and Path(checkpoint.best_model_path).is_file()
        assert len(list(directory.glob("best-*.ckpt"))) == 1
        return Path(checkpoint.best_model_path), 0., 0.

    build = lambda *args, **kw: {"trainer": {"max_epochs": kw["max_epochs"]}}
    c = SimpleNamespace(TRAIN_KW={}, apply_config_patch=_patch_config,
                        _cosine=lambda epochs: {}, run_training=train,
                        write_json=lambda path, value: records.update({path.name: value}))
    s4 = SimpleNamespace(build_graph2mat_config=build,
                         torch_backend_preflight=lambda: {"effective_backend": "cpu"})
    bench = SimpleNamespace(samples=lambda *args: samples, BASIS=tmp_path, WORK=tmp_path)
    (tmp_path / "fe").mkdir()
    (tmp_path / "fe/moment_audit.json").write_text("{}")
    ns = dict(Path=Path, c=c, s4=s4, bench=bench, shutil=shutil, MATERIAL="synthetic",
              RUNS=tmp_path / "runs", BASIS_GLOB="unused", MODEL={}, HIDDEN="2x0e",
              EDGE_HIDDEN="2x0e", UPDATES=8000, BATCH=2, ARMS={"fe": ["delta"]},
              runs_dir=lambda material: tmp_path / "runs",
              delta_sample=lambda sample, delta: (sample, 0.),
              arm_sample=lambda material, sample, arm: sample,
              samples=lambda *args, **kwargs: samples, evaluate=lambda *args, **kwargs: {})
    run = _function(filename, "run", ns)
    args = ("delta", 0, updates) if filename == "run_soc_campaign.py" else ("fe", "delta", 0, updates)
    if updates <= 0:
        with pytest.raises(ValueError, match="positive"):
            run(*args)
        assert not configs
        return
    run(*args)
    trainer = configs[0]["trainer"]
    assert trainer["max_steps"] == updates
    assert trainer["max_epochs"] * 5 >= updates
    assert records["result.json"]["updates"] == updates


@pytest.mark.parametrize("room_allows", [False, True])
def test_first_energy_training_respects_room(tmp_path, room_allows):
    class StopBeforeSleep(Exception):
        pass

    calls, started = [], []

    def room(n_running, parallel):
        calls.append((n_running, parallel))
        return room_allows

    def start(*args):
        if not room_allows:
            pytest.fail("A training started despite the temperature/VRAM gate rejecting it")
        started.append(args[1])
        raise StopBeforeSleep

    def sleep(seconds):
        raise StopBeforeSleep

    ns = dict(Path=Path, json=json, _parallel=lambda root: 3, _room=room,
              start=start, time=SimpleNamespace(sleep=sleep))
    train_all = _function("run_energy_loss_experiment.py", "train_all", ns)
    with pytest.raises(StopBeforeSleep):
        train_all(tmp_path, [{"arm": "control", "seed": 0}])
    assert calls == [(0, 3)]
    assert started == ([{"arm": "control", "seed": 0}] if room_allows else [])
