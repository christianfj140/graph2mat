"""Opt-in spin-orbit MACE model: equivariance, CPU training, checkpoints and fail-closed options."""

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.e3nn import (
    E3nnLinearTSQNodeBlock,
    E3nnLowRankEdgeBlock,
    E3nnSimpleEdgeBlock,
    E3nnSlaterKosterEdgeBlock,
    E3nnSpinOrbitMatrixBlock,
)
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data import metrics
from graph2mat.core.data.spin_orbit import siesta_to_soc_channels
from graph2mat.core.data.tests.test_spin_orbit import make_soc_system, soc_processor
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

TINY = dict(hidden_irreps="2x0e+2x1o+2x2e", edge_hidden_irreps="2x0e+2x1o+2x2e",
            num_bessel=3, max_ell=2, num_interactions=2, correlation=2)


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _shell(l):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, R=2.1) for m in range(-l, l + 1)]


# ---------------------------------------------------------------------------
# Equivariance of the full model: h0 rotates as a matrix, g also as an axial vector
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["sp", "spd"])
def soc_model(request, float64):
    sp = _shell(0) + _shell(1)
    atoms = [sisl.Atom("H", orbitals=sp),
             sisl.Atom("O", orbitals=sp + (_shell(2) if request.param == "spd" else []))]
    table = AtomicTableWithEdges(atoms)
    torch.manual_seed(0)
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True, spin_orbit=True, **TINY).eval()
    return model, soc_processor(atoms, symmetric=True), atoms


@pytest.fixture(params=["bimolecule", "chain"])
def soc_geometry(request, soc_model):
    _, _, (h, o) = soc_model
    if request.param == "bimolecule":
        return sisl.Geometry([[0., 0., 0.], [1., 0.3, 0.2]], atoms=[h, o], lattice=[20, 20, 20])
    geom = sisl.Geometry([[0., 0., 0.], [1.3, 0.2, 0.]], atoms=[h, o], lattice=[4, 20, 20])
    geom.set_nsc([3, 1, 1])
    return geom


def _channels(model, processor, geometry):
    """Predicted [h0, gx, gy, gz] supercell arrays, read back from the reconstructed sisl H."""
    data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geometry), data_processor=processor)
    with torch.no_grad():
        predictions = model(data)
    H = processor.matrix_from_data(data, predictions, out_format="sisl_H")
    return siesta_to_soc_channels(np.stack([H.tocsr(i).toarray() for i in range(8)], -1))


def test_mace_spin_orbit_equivariance(soc_model, soc_geometry):
    model, processor, _ = soc_model
    geometry = soc_geometry
    original = _channels(model, processor, geometry)
    assert np.isfinite(original).all()
    assert np.max(np.abs(original[..., 0])) > 1e-6 and np.max(np.abs(original[..., 1:])) > 1e-6
    cob = processor.basis_table.change_of_basis
    for rotation in Rotation.random(16, random_state=np.random.default_rng(0)).as_matrix():
        rotated = geometry.copy()
        rotated.xyz[:] = geometry.xyz @ rotation.T
        rotated.cell[:] = geometry.cell @ rotation.T
        actual = _channels(model, processor, rotated)
        basis_rotation = torch.as_tensor(cob @ rotation @ cob.T)
        D = block_diag(*[
            basis.e3nn_irreps.D_from_matrix(basis_rotation).numpy()
            for basis in (processor.basis_table.basis[int(species)] for species in geometry.atoms.species)
        ])
        expected = np.concatenate([
            np.einsum("ij,jkc,lk->ilc", D, cell, D)
            for cell in np.split(original, original.shape[1] // geometry.no, axis=1)
        ], axis=1)
        # g is a Cartesian (axial) vector in its channel index: g'_a = sum_b R_ab g_b.
        expected[..., 1:] = expected[..., 1:] @ rotation.T
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# CPU training, checkpoint round trip
# ---------------------------------------------------------------------------

def _soc_inputs(spin_orbit=True, **kwargs):
    samples, atoms = [], None
    for seed in (0, 1):
        H, atoms, _, _ = make_soc_system(seed=seed)
        samples.append(TorchBasisMatrixData.new(H, data_processor=soc_processor(atoms, symmetric=True)))
    torch.manual_seed(0)
    model = LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), symmetric_matrix=True,
                               spin_orbit=spin_orbit, **{**TINY, **kwargs})
    return model, samples


def _data_module(model, samples, **kwargs):
    options = dict(out_matrix="hamiltonian", matrix_component_policy="spin_orbit", n_matrix_components=4)
    options.update(kwargs)
    module = MatrixDataModule(basis_table=model.basis_table, symmetric_matrix=True, sub_point_matrix=False,
                              loader_threads=0, train_runs=[], val_runs=[], batch_size=len(samples), **options)
    module.setup("fit")
    module.train_dataset, module.val_dataset = samples, []
    return module


def _trainer(**kwargs):
    return pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, **kwargs)


def test_spin_orbit_fit_step_and_checkpoint(tmp_path):
    model, samples = _soc_inputs()
    assert model.hparams.n_matrix_components == 4
    blocks = [*model.model.matrix_readouts.self_interactions, *model.model.matrix_readouts.interactions.values()]
    assert blocks and all(isinstance(b, E3nnSpinOrbitMatrixBlock) for b in blocks)
    batch = Batch.from_data_list(samples)
    assert batch.point_labels.shape[1] == 4

    loss = model.training_step(batch.clone(), 0)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    model.zero_grad()

    initial = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = _trainer(max_steps=3, max_epochs=3)
    trainer.fit(model, datamodule=_data_module(model, samples))
    assert trainer.global_step == 3
    assert torch.isfinite(trainer.callback_metrics["train_loss_epoch"])
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())

    path = tmp_path / "soc.ckpt"
    trainer.save_checkpoint(path)
    restored = LitMACEMatrixModel.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    assert restored.hparams.spin_orbit and restored.hparams.n_matrix_components == 4
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected, actual = model(batch.clone()), restored(batch.clone())
    for key in ("node_labels", "edge_labels"):
        assert torch.isfinite(expected[key]).all()
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Fail-closed options
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"loss": metrics.gauge_invariant_mse},
    {"loss": metrics.hamiltonian_composite_loss},
    {"loss": metrics.coefficient_space_mse},
    {"loss_kwargs": {"electron_weight": 1.0}},
    {"training_stages": [{"max_epochs": 1, "loss": "gauge_invariant_mse"}]},
    {"output_scaleshift": True},
    {"coefficient_training": True},
    {"return_coefficients": True},
    {"long_range": "charges"},
    {"hamiltonian_local_context": True},
    {"basis_grouping": "max"},
    {"basis_grouping": "basis_shape"},
    {"edge_block_readout": E3nnSlaterKosterEdgeBlock},
    {"edge_block_readout": E3nnLowRankEdgeBlock},
    {"node_block_readout": E3nnLinearTSQNodeBlock},
    {"matrix_targets": ["hamiltonian", "overlap"]},
    {"n_output_components": 4},
    {"n_matrix_components": 2},
    {"symmetric_matrix": False},
], ids=lambda bad: "-".join(f"{k}={getattr(v, '__name__', v)}" for k, v in bad.items()))
def test_spin_orbit_rejects_uncertified_options(bad):
    atoms = [sisl.Atom("H", orbitals=_shell(0))]
    # symmetric_matrix=True in the base: otherwise every case fails on that rule alone.
    kwargs = {"symmetric_matrix": True, **TINY, **bad}
    with pytest.raises(ValueError, match="spin_orbit"):
        LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), spin_orbit=True, **kwargs)


def test_spin_orbit_requires_symmetric_matrix():
    # False (also the default) predicts both edge directions apart and on-site blocks as "ij":
    # H would not be Hermitian.
    atoms = [sisl.Atom("H", orbitals=_shell(0))]
    for kwargs in ({"symmetric_matrix": False}, {}):
        with pytest.raises(ValueError, match=r"spin_orbit.*symmetric_matrix.*Hermiticity"):
            LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), spin_orbit=True, **TINY, **kwargs)


@pytest.mark.parametrize("ok", [
    {"loss": metrics.block_type_mae},
    {"training_stages": [{"max_epochs": 1, "loss": "elementwise_mae"}]},
    {"edge_block_readout": E3nnSimpleEdgeBlock},
    {"n_matrix_components": 4},
], ids=["block_type_mae", "stage_mae", "simple_edge", "n4"])
def test_spin_orbit_accepts_certified_options(ok):
    atoms = [sisl.Atom("H", orbitals=_shell(0))]
    model = LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), symmetric_matrix=True,
                               spin_orbit=True, **{**TINY, **ok})
    assert model.hparams.n_matrix_components == 4


@pytest.mark.parametrize("model_flag,data", [
    (False, {}),
    (True, {"matrix_component_policy": "h_only", "n_matrix_components": 4}),
], ids=["soc_data_plain_model", "plain_data_soc_model"])
@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_policy_and_model_flag_must_agree(model_flag, data, stage):
    model, samples = _soc_inputs(spin_orbit=model_flag, n_matrix_components=4)
    module = _data_module(model, samples, **data)
    with pytest.raises(ValueError, match="disagrees with data matrix_component_policy"):
        getattr(_trainer(), stage)(model, datamodule=module)


@pytest.mark.parametrize("stage", ["fit", "validate"])
def test_model_rechecks_data_attributes_set_after_construction(stage):
    model, samples = _soc_inputs()
    module = _data_module(model, samples)
    module.spectral_window = True
    with pytest.raises(ValueError, match=r"spin_orbit data do not support: spectral_window"):
        getattr(_trainer(), stage)(model, datamodule=module)


@pytest.mark.parametrize("data,word", [
    ({"spectral_window": True}, "spectral_window"),
    ({"physical_labels": ["charge"]}, "physical_labels"),
    ({"matrix_targets": ["hamiltonian", "overlap"]}, "matrix_targets"),
])
def test_data_module_rejects_uncertified_options(data, word):
    atoms = [sisl.Atom("H", orbitals=_shell(0))]
    with pytest.raises(ValueError, match=rf"(spin_orbit|spin-orbit).*{word}"):
        MatrixDataModule(basis_table=AtomicTableWithEdges(atoms), out_matrix="hamiltonian",
                         matrix_component_policy="spin_orbit", n_matrix_components=4, **data)


@pytest.mark.parametrize("out_format", ["scipy_csr", "scipy_coo", "numpy"])
def test_non_sisl_formats_rejected_for_spin_orbit(out_format):
    H, atoms, _, _ = make_soc_system()
    processor = soc_processor(atoms, symmetric=True)
    data = TorchBasisMatrixData.new(H, data_processor=processor)
    with pytest.raises(ValueError, match=r"(spin_orbit|spin-orbit).*sign.*sisl"):
        processor.matrix_from_data(data, out_format=out_format)
    with pytest.raises(ValueError, match=r"(spin_orbit|spin-orbit).*sign.*sisl"):
        processor.labels_to(out_format, data)


def test_writer_writes_spin_orbit_predictions_without_overlap(tmp_path):
    """Roadmap 8.1: spin-orbit predictions are written as an orthogonal spin-orbit H (labels
    carry no S), with a note on how to attach S; they used to be refused."""
    import json
    from types import SimpleNamespace

    import sisl

    from graph2mat.tools.lightning.callbacks import MatrixWriter

    H, atoms, _, _ = make_soc_system()
    processor = soc_processor(atoms, symmetric=True)
    sample = TorchBasisMatrixData.new(H, data_processor=processor)
    batch = Batch.from_data_list([sample])
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=processor, out_matrix="hamiltonian"))
    writer = MatrixWriter(str(tmp_path / "out.TSHS"))
    writer._on_batch_end("predict", trainer, None,
                         {"node_labels": sample.point_labels, "edge_labels": sample.edge_labels}, batch, 0, 0)
    written = sisl.get_sile(tmp_path / "out.TSHS").read_hamiltonian()
    assert written.spin.is_spinorbit and written.orthogonal
    metadata = json.loads((tmp_path / "out.TSHS.metadata.json").read_text())
    assert "attach S" in metadata["overlap_source"]


def test_plot_matrix_error_runs_with_spin_orbit_data(monkeypatch):
    from types import SimpleNamespace
    from graph2mat.tools.lightning import callbacks

    H, atoms, _, _ = make_soc_system()
    processor = soc_processor(atoms, symmetric=True)
    sample = TorchBasisMatrixData.new(H, data_processor=processor)
    trainer = SimpleNamespace(datamodule=SimpleNamespace(
        data_processor=processor, basis_table=processor.basis_table,
        symmetric_matrix=True, out_matrix="hamiltonian"))
    titles = []
    monkeypatch.setattr(callbacks, "plot_basis_matrix", lambda matrix, **kw: SimpleNamespace(
        update_layout=lambda **layout: titles.append(layout["title"])))
    callback = callbacks.PlotMatrixError(show=False, store_in_logger=False)
    outputs = {"node_labels": sample.point_labels + 0.5, "edge_labels": sample.edge_labels + 0.5}
    callback.on_test_epoch_start(trainer, None)
    callback.on_test_batch_end(trainer, None, outputs, Batch.from_data_list([sample]), 0)
    callback.on_test_epoch_end(trainer, None)
    assert sorted(titles) == sorted(f"{m}_component{c}" for m in ("MAE", "RMSE") for c in range(4))


def test_server_rejects_spin_orbit_processor():
    from graph2mat.tools.server.server_app import create_server_app

    _, atoms, _, _ = make_soc_system()
    with pytest.raises(ValueError, match=r"soc: .*(spin_orbit|spin-orbit).*not supported"):
        create_server_app({"soc": {"data_processor": soc_processor(atoms, symmetric=True)}})


def test_cli_sets_spin_orbit_from_policy():
    from jsonargparse import Namespace
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = Namespace(data=Namespace(matrix_component_policy="spin_orbit", n_matrix_components=1),
                       model=Namespace(spin_orbit=False, n_matrix_components=1))
    _autoconfigure_n_matrix_components(config, stage="predict")
    assert config.model.spin_orbit is True
    assert config.data.n_matrix_components == config.model.n_matrix_components == 4

    config.data.matrix_component_policy = "h_only"
    config.model.spin_orbit = False  # unset: derived from the policy
    config.data.n_matrix_components = config.model.n_matrix_components = 1
    _autoconfigure_n_matrix_components(config, stage="predict")
    assert config.model.spin_orbit is False and config.model.n_matrix_components == 1

    config.data.matrix_component_policy = "spin_orbit"
    config.data.n_matrix_components = 2
    with pytest.raises(ValueError, match="spin_orbit"):
        _autoconfigure_n_matrix_components(config, stage="predict")


def test_cli_rejects_explicit_spin_orbit_without_policy():
    from jsonargparse import Namespace
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = Namespace(data=Namespace(matrix_component_policy="h_only", n_matrix_components=1),
                       model=Namespace(spin_orbit=True, n_matrix_components=1))
    with pytest.raises(ValueError, match=r"model\.spin_orbit=True.*matrix_component_policy"):
        _autoconfigure_n_matrix_components(config, stage="fit")
    assert config.model.spin_orbit is True  # never overwritten behind the user's back


def test_spin_orbit_disables_e3nn_torchscript_codegen():
    # TorchScript fx codegen of the SOC node TensorSquare stalled Lightning's sanity check.
    from e3nn import get_optimization_defaults, set_optimization_defaults

    previous = get_optimization_defaults()["jit_script_fx"]
    try:
        set_optimization_defaults(jit_script_fx=True)
        atoms = [sisl.Atom("H", orbitals=_shell(0))]
        LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), symmetric_matrix=True, spin_orbit=True, **TINY)
        assert get_optimization_defaults()["jit_script_fx"] is False
    finally:
        set_optimization_defaults(jit_script_fx=previous)
