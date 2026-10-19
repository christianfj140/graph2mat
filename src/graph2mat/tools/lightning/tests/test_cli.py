import pytest
import torch
import json
import numpy as np
import pytorch_lightning as pl
import sisl

from graph2mat.tools.lightning.cli import OrbitalMatrixCLI
from graph2mat import AtomicTableWithEdges
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from pytorch_lightning.cli import LightningArgumentParser


class _CheckpointObject:
    pass


@pytest.mark.parametrize("spin", [1, 2])
def test_cli_multitarget_fit_resolves_layout(tmp_path, spin):
    from jsonargparse import Namespace
    from graph2mat.core.data.tests.physical_fixtures import analytic_matrices
    from graph2mat.core.data.tests.test_multitarget import TARGETS, write_fdf_bundle
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    matrices = analytic_matrices(spin=spin)
    path = write_fdf_bundle(tmp_path, matrices)
    config = Namespace(data=Namespace(matrix_targets=list(TARGETS), matrix_target_spin=None,
        n_matrix_components=1, root_dir=str(tmp_path), train_runs=[str(path)],
        basis_table=AtomicTableWithEdges(matrices["overlap"].atoms.atom),
        out_matrix="hamiltonian", sub_point_matrix=False),
        model=Namespace(matrix_targets=None, matrix_target_spin=None, n_matrix_components=1))
    _autoconfigure_n_matrix_components(config, stage="fit")
    assert config.data.matrix_target_spin == spin
    assert config.model.matrix_target_spin == spin
    assert config.model.matrix_targets == list(TARGETS)
    assert config.data.n_matrix_components == config.model.n_matrix_components == (4 if spin == 1 else 7)


def test_cli_rejects_mismatched_descriptor_and_unlabeled_inference():
    from jsonargparse import Namespace
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = Namespace(data=Namespace(matrix_targets=["hamiltonian", "overlap", "density_matrix"],
        matrix_target_spin=1, n_matrix_components=1, sub_point_matrix=False),
        model=Namespace(matrix_targets=["hamiltonian", "overlap", "energy_density_matrix"],
                        matrix_target_spin=1, n_matrix_components=1))
    with pytest.raises(ValueError, match="semantic|descriptor|targets"):
        _autoconfigure_n_matrix_components(config, stage="predict")
    config.model.matrix_targets = None
    config.data.matrix_target_spin = None
    config.model.matrix_target_spin = None
    with pytest.raises(ValueError, match="spin"):
        _autoconfigure_n_matrix_components(config, stage="predict")
    config.data.matrix_target_spin = 2
    _autoconfigure_n_matrix_components(config, stage="predict")
    assert config.model.n_matrix_components == 5


@pytest.mark.parametrize("callback", ["MatrixWriter", "SamplewiseMetricsLogger"])
@pytest.mark.parametrize("splits", ["test", ["test"]])
def test_cli_parses_callback_split_string_or_sequence(tmp_path, callback, splits):
    parser = LightningArgumentParser()
    parser.add_subclass_arguments(pl.Callback, "callback")
    config = parser.parse_object({"callback": {
        "class_path": "graph2mat.tools.lightning.callbacks." + callback,
        "init_args": {"splits": splits, "output_file": str(tmp_path / "metrics.csv")},
    }})
    instance = parser.instantiate_classes(config).callback
    assert instance.splits == ["test"]


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


@pytest.mark.parametrize("command", ["validate", "test", "predict"])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_cli_checkpoint_runs_preserve_weights_predictions_and_overrides(tmp_path, command, legacy):
    pl.seed_everything(0, workers=True)
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    basis = AtomicTableWithEdges([atom])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[10.]*3)
    matrix = sisl.DensityMatrix(geometry)
    matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0] = 1., 2., .25, .25
    matrix.write(tmp_path / "sample.nc")
    geometry.write(tmp_path / "geometry.xyz")
    module = MatrixDataModule(
        out_matrix="density_matrix", root_dir=str(tmp_path), basis_table=basis,
        symmetric_matrix=True, sub_point_matrix=False, loader_threads=0, split_seed=7,
        train_runs="sample.nc", val_runs="sample.nc", test_runs="sample.nc",
        predict_structs="geometry.xyz",
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
    if legacy:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint.pop("provenance", None)
        checkpoint["datamodule_hyper_parameters"].pop("split_seed", None)
        checkpoint["datamodule_hyper_parameters"].pop("val_fraction", None)
        torch.save(checkpoint, checkpoint_path)
    module.setup("test")
    batch = next(iter(module.test_dataloader()))
    model.eval()
    with torch.no_grad():
        expected = model(batch)
    callbacks = []
    prediction_file = tmp_path / "prediction.DM"
    if command == "predict":
        callbacks = [{"class_path": "graph2mat.tools.lightning.callbacks.MatrixWriter",
                      "init_args": {"output_file": str(prediction_file), "splits": ["predict"]}}]
    safe_globals = torch.serialization.get_safe_globals().copy()
    runs_json = tmp_path / "runs.json"
    runs_json.write_text(json.dumps({"train": ["missing-inactive.nc"]}))
    cli = OrbitalMatrixCLI(
        model_class=LitMACEMatrixModel, datamodule_class=MatrixDataModule,
        save_config_callback=None, args=[
            command, "--ckpt_path", str(checkpoint_path), "--seed_everything=0",
            "--trainer.accelerator=cpu", "--trainer.devices=1", "--trainer.logger=false",
            "--trainer.enable_checkpointing=false", "--trainer.enable_progress_bar=false",
            "--trainer.enable_model_summary=false", "--trainer.callbacks", json.dumps(callbacks),
            "--data.batch_size=1", "--model.optim_lr=0.002",
            "--data.train_runs=null", "--data.runs_json", str(runs_json),
        ],
    )
    assert cli.model.hparams.hidden_irreps == "2x0e+2x1o"
    assert cli.model.hparams.optim_lr == .002
    assert cli.datamodule.batch_size == 1
    assert cli.datamodule.hparams.split_seed == (32 if legacy else 7)
    assert next(cli.model.parameters()).device.type == "cpu"
    assert torch.serialization.get_safe_globals() == safe_globals
    for key, value in model.state_dict().items():
        torch.testing.assert_close(cli.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        actual = cli.model(batch)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    if command != "predict":
        loader = cli.datamodule.val_dataloader() if command == "validate" else cli.datamodule.test_dataloader()
        cli_batch = next(iter(loader))
        assert cli.datamodule.basis_table.maxR() == module.basis_table.maxR()
        torch.testing.assert_close(cli_batch.point_labels, batch.point_labels, rtol=0, atol=0)
        torch.testing.assert_close(cli_batch.edge_labels, batch.edge_labels, rtol=0, atol=0)
        errors = torch.cat([
            (expected["node_labels"] - batch.point_labels).reshape(-1),
            (expected["edge_labels"] - batch.edge_labels).reshape(-1),
        ])
        # Epoch sums accumulate in float64; do not compare against an extra
        # float32 reduction that rounds the independent reference again.
        split = "val" if command == "validate" else "test"
        assert float(cli.trainer.callback_metrics[f"{split}_loss"]) == pytest.approx(float(errors.double().square().mean()), abs=1e-12)
    else:
        expected_matrix = module.data_processor.matrix_from_data(batch, expected)[0]
        written = sisl.get_sile(prediction_file).read_density_matrix(geometry=geometry)
        np.testing.assert_allclose(written.tocsr().toarray(), expected_matrix.tocsr().toarray(), rtol=0, atol=1e-12)
    cli.datamodule.teardown(None)
    module.teardown(None)
