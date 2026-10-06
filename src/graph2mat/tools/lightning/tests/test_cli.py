import pytest
import torch

from graph2mat.tools.lightning.cli import OrbitalMatrixCLI


class _CheckpointObject:
    pass


def _checkpoint(extra=None):
    return {
        "basis_table": "basis",
        "datamodule_hyper_parameters": {"out_matrix": "hamiltonian"},
        "hyper_parameters": {"hidden_irreps": "1x0e"},
        **(extra or {}),
    }


@pytest.mark.parametrize(
    "disabled", ["E3MAT_FROMCKPT_MODEL", "E3MAT_FROMCKPT_DATAPROC"]
)
def test_checkpoint_basis_table_when_defaults_disabled(
    tmp_path, monkeypatch, disabled
):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(_checkpoint(), checkpoint)
    monkeypatch.setenv(disabled, "off")

    config = OrbitalMatrixCLI.__new__(OrbitalMatrixCLI)._config_from_ckpt(checkpoint)

    assert config["model"]["basis_table"] == "basis"
    assert config["data"]["basis_table"] == "basis"


def test_checkpoint_config_loads_lightning_objects(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(_checkpoint({"lightning_object": _CheckpointObject()}), checkpoint)

    config = OrbitalMatrixCLI.__new__(OrbitalMatrixCLI)._config_from_ckpt(checkpoint)

    assert config["model"]["basis_table"] == "basis"
