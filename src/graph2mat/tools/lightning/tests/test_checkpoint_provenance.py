"""Additive provenance and real checkpoint round trips on CPU."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis, __version__
from graph2mat.bindings.torch.load import load_from_lit_ckpt
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning import model as model_module
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel


def _basis():
    return BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])


def _checkpoint():
    model = LitBasisMatrixModel(model_cls=torch.nn.Identity, basis_table=_basis())
    checkpoint = {"state_dict": {"example": torch.tensor([2.])},
                  "hyper_parameters": {}, "datamodule_hyper_parameters": {}}
    model.on_save_checkpoint(checkpoint)
    return model, checkpoint


def test_checkpoint_records_versions_and_source_git_from_any_cwd(tmp_path, monkeypatch):
    # A wheel has no checkout. Create a real tracked source fixture instead of
    # relying on whichever directory happens to contain the installed package.
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "model.py"
    source_file.write_text("# tracked source fixture\n")
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "model.py"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "--quiet",
                    "-m", "fixture"], check=True)
    expected_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                             text=True).strip()
    monkeypatch.setattr(model_module, "__file__", str(source_file))
    monkeypatch.chdir(tmp_path)
    _, checkpoint = _checkpoint()
    provenance = checkpoint["provenance"]
    assert provenance["git_commit"] == expected_commit
    assert provenance["git_dirty"] is False
    assert provenance["python"] == platform.python_version()
    for name in ("torch", "torch-geometric", "pytorch-lightning", "e3nn", "mace-torch", "sisl"):
        assert provenance["versions"][name] == importlib.metadata.version(name)
    torch.testing.assert_close(checkpoint["state_dict"]["example"], torch.tensor([2.]))


def test_checkpoint_provenance_without_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "__file__", str(tmp_path / "installed_model.py"))
    _, checkpoint = _checkpoint()
    provenance = checkpoint["provenance"]
    assert provenance["git_commit"] is None and provenance["git_dirty"] is None
    assert provenance["versions"]["torch"] == importlib.metadata.version("torch")


def test_checkpoint_can_be_saved_without_git_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    _, checkpoint = _checkpoint()
    assert checkpoint["provenance"]["git_commit"] is None
    assert checkpoint["provenance"]["git_dirty"] is None
    assert checkpoint["version"] == __version__


def test_installed_package_does_not_report_enclosing_repositories_commit(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture",
                    "-c", "user.email=fixture@example.invalid", "commit", "--quiet",
                    "--allow-empty", "-m", "fixture"], check=True)
    installed = tmp_path / "venv" / "site-packages" / "graph2mat"
    installed.mkdir(parents=True)
    monkeypatch.setattr(model_module, "__file__", str(installed / "model.py"))
    _, checkpoint = _checkpoint()
    assert checkpoint["provenance"]["git_commit"] is None


def test_checkpoint_provenance_marks_missing_optional_dependency(monkeypatch):
    version = importlib.metadata.version

    def without_mace(name):
        if name == "mace-torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return version(name)

    monkeypatch.setattr(importlib.metadata, "version", without_mace)
    _, checkpoint = _checkpoint()
    assert checkpoint["provenance"]["versions"]["mace-torch"] is None
    assert checkpoint["provenance"]["versions"]["torch"] == version("torch")


def test_legacy_checkpoint_without_provenance_still_loads():
    model, checkpoint = _checkpoint()
    checkpoint.pop("provenance", None)
    model.on_load_checkpoint(checkpoint)
    assert model.basis_table is checkpoint["basis_table"]
    assert checkpoint["version"] == __version__
    torch.testing.assert_close(checkpoint["state_dict"]["example"], torch.tensor([2.]))


@pytest.mark.parametrize("legacy", [False, True])
def test_real_mace_checkpoint_roundtrip_preserves_predictions_on_cpu(tmp_path, legacy):
    pl.seed_everything(0, workers=True)
    basis = _basis()
    config = BasisConfiguration(
        point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
        basis=basis, cell=np.eye(3)*10, pbc=(False, False, False), matrix=np.ones((2, 2)),
    )
    module = MatrixDataModule(
        basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
        train_runs=[config], val_runs=[config], loader_threads=0, split_seed=7,
    )
    model = LitMACEMatrixModel(
        basis_table=basis, symmetric_matrix=True, hidden_irreps="2x0e+2x1o",
        edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, fast_dev_run=True, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, datamodule=module)
    checkpoint_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if legacy:
        checkpoint.pop("provenance", None)
        checkpoint["datamodule_hyper_parameters"].pop("split_seed", None)
        checkpoint["datamodule_hyper_parameters"].pop("val_fraction", None)
        torch.save(checkpoint, checkpoint_path)
    else:
        assert checkpoint["provenance"]["versions"]["torch"] == importlib.metadata.version("torch")
        assert checkpoint["datamodule_hyper_parameters"]["split_seed"] == 7
    restored, processor = load_from_lit_ckpt(checkpoint_path)
    batch = next(iter(module.val_dataloader()))
    model.eval()
    restored.eval()
    with torch.no_grad():
        before, after = model(batch), restored(batch)
    for field in ("node_labels", "edge_labels"):
        torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)
    assert next(restored.parameters()).device.type == "cpu"
    assert processor.symmetric_matrix and not processor.sub_point_matrix
    module.teardown(None)
