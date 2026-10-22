"""Lightning layer of the magnetic models (roadmap 8.1, magnetism).

``LitMACEMatrixModel(magnetic_moments=...)`` has three certified variants besides the plain one:
``spin_orbit=True`` (4 channels, no moments), ``spin_orbit=True, magnetic_moments="vector"`` (policy
``spin_orbit_full``: 8 channels) and ``magnetic_moments="collinear"`` (policy ``spin_h_only``: up, down).
The math lives in ``models/tests/test_magnetic_mace.py``; here the model, the data module and the CLI have to
agree, uncertified options fail before training, and checkpoints reproduce the model.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from e3nn import o3
from scipy.sparse import csr_array
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.e3nn import (
    E3nnLinearTSQNodeBlock,
    E3nnSimpleEdgeBlock,
    E3nnSlaterKosterEdgeBlock,
    E3nnSpinOrbitFullMatrixBlock,
    E3nnSpinOrbitMatrixBlock,
)
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.load import load_from_lit_ckpt
from graph2mat.core.data import metrics
from graph2mat.core.data.spin_orbit import SOC_FULL_TIME_REVERSAL_PARITY
from graph2mat.models import mace as mace_module
from graph2mat.core.data.tests.test_spin_orbit import make_soc_system, soc_processor
from graph2mat.core.data.tests.test_spin_orbit_full import full_processor, make_magnetic_system
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.tests.test_spin_orbit_model import TINY, _trainer

EVEN, ODD = slice(0, 4), slice(4, 8)

def _shell(l, zeta=1):
    return [sisl.AtomicOrbital(n=l + 1, l=l, m=m, zeta=zeta, R=2.1) for m in range(-l, l + 1)]


# Variant -> LitMACEMatrixModel options, data policy, data moments and number of channels.
VARIANTS = {
    "plain": (dict(), "h_only", None, 1),
    "soc": (dict(spin_orbit=True), "spin_orbit", None, 4),
    "vector": (dict(spin_orbit=True, magnetic_moments="vector"), "spin_orbit_full", "vector", 8),
    "collinear": (dict(magnetic_moments="collinear"), "spin_h_only", "collinear", 2),
}


@pytest.fixture(scope="module", autouse=True)
def float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture
def float32():
    """A test that runs in the precision of real training (the module default is float64)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------

def _atoms():
    return make_soc_system(0, orthogonal=True)[0].geometry.atoms.atom  # H (s+p) and O (s)


def _model(kind, atoms=None, **kwargs):
    options = VARIANTS[kind][0]
    atoms = _atoms() if atoms is None else atoms
    torch.manual_seed(0)
    return LitMACEMatrixModel(basis_table=AtomicTableWithEdges(atoms), symmetric_matrix=True,
                              **{**TINY, **options, **kwargs})


@pytest.fixture(scope="module")
def models():
    return {kind: _model(kind) for kind in VARIANTS}


def _polarized_system(seed):
    """Spin-polarized (up, down) Hamiltonian on the geometry of ``make_soc_system``, symmetric blocks."""
    geometry = make_soc_system(0, orthogonal=True)[0].geometry
    rng = np.random.default_rng(seed)
    components = []
    for _ in range(2):
        onsite = rng.normal(size=(geometry.no,) * 2)
        image = rng.normal(size=(geometry.no,) * 2)
        blocks = {0: onsite + onsite.T, 1: image, -1: image.T}
        components.append(csr_array(np.concatenate([blocks[int(off[0])] for off in geometry.sc_off], axis=1)))
    return sisl.Hamiltonian.fromsp(geometry, components, spin=sisl.Spin("polarized"))


def _samples(kind, n=2):
    """``n`` labelled structures (same geometry, different matrices and moments) of a magnetic variant."""
    samples = []
    for seed in range(n):
        rng = np.random.default_rng(10 + seed)
        if kind == "vector":
            H, _, atoms = make_magnetic_system(seed)
            processor = full_processor(atoms, True).copy(magnetic_moments="vector")
            moments = rng.normal(0, 2.0, (H.geometry.na, 3))
        else:
            H = _polarized_system(seed)
            processor = MatrixDataProcessor(
                basis_table=AtomicTableWithEdges(H.geometry.atoms.atom), out_matrix="hamiltonian",
                symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=2,
                matrix_component_policy="spin_h_only", magnetic_moments="collinear")
            moments = rng.normal(0, 2.0, H.geometry.na)
        samples.append(TorchBasisMatrixData.new(H, data_processor=processor, magnetic_moments=moments))
    return samples


def _module(kind, samples=(), n_matrix_components=None, **kwargs):
    _, policy, moments, n = VARIANTS[kind]
    options = dict(out_matrix="hamiltonian", matrix_component_policy=policy, magnetic_moments=moments,
                   n_matrix_components=n if n_matrix_components is None else n_matrix_components)
    options.update(kwargs)
    module = MatrixDataModule(basis_table=AtomicTableWithEdges(_atoms()), symmetric_matrix=True,
                              sub_point_matrix=False, loader_threads=0, train_runs=[], val_runs=[],
                              batch_size=max(len(samples), 1), **options)
    module.setup("fit")
    module.train_dataset, module.val_dataset = list(samples), []
    return module


# ---------------------------------------------------------------------------
# The variants and what they build
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,channels,block,layer", [
    ("plain", 1, None, False),
    ("soc", 4, E3nnSpinOrbitMatrixBlock, False),
    ("vector", 8, E3nnSpinOrbitFullMatrixBlock, True),
    ("collinear", 2, None, True),
])
def test_variants_build_their_channels(models, kind, channels, block, layer):
    model = models[kind]
    assert model.hparams.n_matrix_components == channels
    assert (model.model.magnetic_layer is not None) is layer
    assert model.model.magnetic_moments == VARIANTS[kind][0].get("magnetic_moments")
    readout = model.model.matrix_readouts
    blocks = [b for b in [*readout.self_interactions, *readout.interactions.values()] if b is not None]
    if block is not None:
        assert blocks and all(type(b) is block for b in blocks)
    else:
        assert not any(isinstance(b, E3nnSpinOrbitMatrixBlock) for b in blocks)


@pytest.mark.parametrize("kind,n", [("vector", 1), ("vector", 8), ("collinear", 1), ("collinear", 2)])
def test_n_matrix_components_is_normalised_to_the_variant(kind, n):
    # The processor and the CLI may leave the default 1; the model always has the variant's channels.
    model = _model(kind, n_matrix_components=n)
    assert model.hparams.n_matrix_components == VARIANTS[kind][3]


@pytest.mark.parametrize("kind,n", [("vector", 4), ("vector", 2), ("collinear", 4), ("collinear", 8)])
def test_other_channel_counts_are_rejected(kind, n):
    with pytest.raises(ValueError, match="n_matrix_components"):
        _model(kind, n_matrix_components=n)


@pytest.mark.parametrize("options", [
    {"magnetic_moments": "vector"},                       # needs spin_orbit
    {"spin_orbit": True, "magnetic_moments": "collinear"},  # collinear is spin_h_only
    {"magnetic_moments": "spin"},
    {"magnetic_moments": "Vector"},
], ids=["vector-without-spin-orbit", "collinear-with-spin-orbit", "unknown", "case"])
def test_other_combinations_are_rejected(options):
    with pytest.raises(ValueError, match="magnetic_moments"):
        LitMACEMatrixModel(basis_table=AtomicTableWithEdges(_atoms()), symmetric_matrix=True, **TINY, **options)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
@pytest.mark.parametrize("bad", [
    {"loss": metrics.gauge_invariant_mse},
    {"loss_kwargs": {"electron_weight": 1.0}},
    {"training_stages": [{"max_epochs": 1, "loss": "gauge_invariant_mse"}]},
    {"output_scaleshift": True},
    {"coefficient_training": True},
    {"return_coefficients": True},
    {"long_range": "charges"},
    {"hamiltonian_local_context": True},
    {"readout_per_interaction": True},
    {"torch_compile": True},
    {"cueq": True},
    {"gradient_checkpointing": True},
    {"basis_grouping": "max"},
    {"edge_block_readout": E3nnSlaterKosterEdgeBlock},
    {"node_block_readout": E3nnLinearTSQNodeBlock},
    {"matrix_targets": ["hamiltonian", "overlap"]},
    {"n_output_components": 2},
], ids=lambda bad: "-".join(f"{k}={getattr(v, '__name__', v)}" for k, v in bad.items()))
def test_magnetic_models_reject_uncertified_options(kind, bad):
    with pytest.raises(ValueError, match="magnetic_moments does not support|magnetic_moments only supports"):
        _model(kind, **bad)


def test_magnetic_models_accept_certified_options():
    for kind in ("vector", "collinear"):
        model = _model(kind, loss=metrics.block_type_mae, edge_block_readout=E3nnSimpleEdgeBlock,
                       training_stages=[{"max_epochs": 1, "loss": "elementwise_mae"}], readout_fp32=True)
        assert model.hparams.n_matrix_components == VARIANTS[kind][3]
        assert model.model.readout_fp32


def test_per_interaction_readout_stays_allowed_without_moments():
    # Only the models with moments gain this rejection: 4-channel spin_orbit keeps its options.
    assert _model("soc", readout_per_interaction=True).hparams.readout_per_interaction


def test_vector_model_disables_e3nn_torchscript_codegen():
    from e3nn import get_optimization_defaults, set_optimization_defaults

    previous = get_optimization_defaults()["jit_script_fx"]
    try:
        set_optimization_defaults(jit_script_fx=True)
        _model("vector")
        assert get_optimization_defaults()["jit_script_fx"] is False
    finally:
        set_optimization_defaults(jit_script_fx=previous)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
def test_lit_model_applies_exact_time_reversal(models, kind):
    # The model built by the Lightning layer must run the m / -m double pass of MatrixMACE.
    model = models[kind].model.eval()
    sample = _samples(kind, 1)[0]
    flipped = sample.clone()
    flipped["magnetic_moments"] = -sample["magnetic_moments"]
    with torch.no_grad():
        plus, minus = model(sample), model(flipped)
    for key in ("node_labels", "edge_labels"):
        assert plus[key].abs().max() > 1e-6
        if kind == "vector":
            torch.testing.assert_close(minus[key], plus[key] * plus[key].new_tensor(SOC_FULL_TIME_REVERSAL_PARITY),
                                       rtol=0, atol=1e-10)
        else:
            torch.testing.assert_close(minus[key], plus[key].flip(-1), rtol=0, atol=1e-12)


def test_readout_fp32_keeps_exact_time_reversal_under_bf16_autocast(float32, monkeypatch):
    # The m / -m double pass runs inside the readout that readout_fp32 takes out of the autocast region.
    # e3nn's Linear cannot consume the bf16 features that autocast makes on CPU, so the backbone is
    # emulated: it runs in fp32 and hands over bf16 node features, as MACE does under bf16-mixed.
    real, handed_over = mace_module.mace_node_feats, []

    def bf16_backbone(*args, **kwargs):
        with torch.autocast("cpu", enabled=False):
            handed_over.append(real(*args, **kwargs).to(torch.bfloat16))
        return handed_over[-1]

    monkeypatch.setattr(mace_module, "mace_node_feats", bf16_backbone)
    model = _model("vector", readout_fp32=True).model.eval()
    sample = _samples("vector", 1)[0]
    flipped = sample.clone()
    flipped["magnetic_moments"] = -sample["magnetic_moments"]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        plus, minus = model(sample), model(flipped)
    assert len(handed_over) == 2 and all(f.dtype == torch.bfloat16 for f in handed_over)
    for key in ("node_labels", "edge_labels"):
        assert plus[key].dtype == minus[key].dtype == torch.float32
        scale = plus[key].abs().max()
        assert scale > 1e-3 and plus[key][:, ODD].abs().max() > 1e-4 * scale
        torch.testing.assert_close(minus[key][:, EVEN], plus[key][:, EVEN], rtol=0, atol=1e-6 * scale)
        torch.testing.assert_close(minus[key][:, ODD], -plus[key][:, ODD], rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
def test_model_needs_the_moments_in_the_data(models, kind):
    sample = _samples(kind, 1)[0]
    del sample["magnetic_moments"]
    with pytest.raises(ValueError, match="magnetic_moments"):
        models[kind].model(sample)


# ---------------------------------------------------------------------------
# Data and model must agree
# ---------------------------------------------------------------------------

# (model, data variant, text of the error). Moments are validated in every stage, also the data side
# of spin_orbit_full without moments (a module whose attribute changed after construction).
DISAGREEMENTS = [
    ("vector", "soc", "disagrees with data matrix_component_policy"),
    ("vector", "plain", "disagrees with data matrix_component_policy"),
    ("vector", "collinear", "disagrees with data matrix_component_policy"),
    ("soc", "vector", "disagrees with data matrix_component_policy"),
    ("plain", "vector", "disagrees with data matrix_component_policy"),
    ("collinear", "plain", "disagrees with data matrix_component_policy"),
    ("collinear", "vector", "disagrees with data matrix_component_policy"),
    ("plain", "collinear", "disagrees with data magnetic_moments"),
]


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("model_kind,data_kind,match", DISAGREEMENTS,
                         ids=[f"{m}-model-{d}-data" for m, d, _ in DISAGREEMENTS])
def test_model_and_data_must_agree(models, model_kind, data_kind, match, stage):
    with pytest.raises(ValueError, match=match):
        getattr(_trainer(), stage)(models[model_kind], datamodule=_module(data_kind))


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("model_kind,moments", [
    ("vector", None), ("vector", "collinear"), ("collinear", None), ("collinear", "vector"),
    ("plain", "vector"), ("soc", "vector"),
])
def test_moments_of_model_and_data_must_agree(models, model_kind, moments, stage):
    # Same policy, other moments: the data module was built with the right ones and changed afterwards.
    module = _module(model_kind)
    module.magnetic_moments = moments
    with pytest.raises(ValueError, match="magnetic_moments"):
        getattr(_trainer(), stage)(models[model_kind], datamodule=module)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
def test_magnetic_models_need_a_datamodule_to_fit(models, kind):
    # Unverifiable agreement: the fit hook fails instead of silently training on another policy.
    with pytest.raises(ValueError, match="disagrees with data"):
        _trainer(max_steps=1).fit(models[kind], train_dataloaders=torch.utils.data.DataLoader([0]))


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("moments", [None, "collinear"], ids=["no-moments", "moments"])
def test_collinear_model_rejects_h_and_overlap_data(models, stage, moments):
    # h_and_overlap also has 2 components: it would be flipped as up/down if only the shape were checked.
    module = _module("plain", matrix_component_policy="h_and_overlap", n_matrix_components=2)
    module.magnetic_moments = moments
    with pytest.raises(ValueError, match="disagrees with data matrix_component_policy='h_and_overlap'"):
        getattr(_trainer(), stage)(models["collinear"], datamodule=module)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
@pytest.mark.parametrize("stage", ["fit", "validate"])
@pytest.mark.parametrize("option,value", [("spectral_window", True), ("physical_labels", ["charge"]),
                                          ("matrix_targets", ["hamiltonian", "overlap"])])
def test_model_rechecks_data_attributes_set_after_construction(models, kind, stage, option, value):
    module = _module(kind)
    setattr(module, option, value)
    with pytest.raises(ValueError, match=rf"data do not support: {option}"):
        getattr(_trainer(), stage)(models[kind], datamodule=module)


@pytest.mark.parametrize("kind", ["vector", "collinear"])
@pytest.mark.parametrize("option,value", [("spectral_window", True), ("physical_labels", ["charge"]),
                                          ("matrix_targets", ["hamiltonian", "overlap"])])
def test_data_module_rejects_uncertified_options_with_moments(kind, option, value):
    with pytest.raises(ValueError, match=rf"data do not support.*{option}"):
        _module(kind, **{option: value})


# ---------------------------------------------------------------------------
# CLI: model options come from the data
# ---------------------------------------------------------------------------

def _config(policy="h_only", moments=None, n=1, **model):
    from jsonargparse import Namespace

    return Namespace(data=Namespace(matrix_component_policy=policy, magnetic_moments=moments, n_matrix_components=n),
                     model=Namespace(spin_orbit=False, magnetic_moments=None, n_matrix_components=n, **model))


@pytest.mark.parametrize("policy,moments,spin_orbit,derived,channels", [
    ("spin_orbit_full", "vector", True, "vector", 8),
    ("spin_h_only", "collinear", False, "collinear", 2),
    ("spin_orbit", None, True, None, 4),
    ("h_only", None, False, None, 1),
])
@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_cli_derives_model_options_from_the_data(policy, moments, spin_orbit, derived, channels, stage):
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = _config(policy, moments)
    _autoconfigure_n_matrix_components(config, stage=stage)
    assert config.model.spin_orbit is spin_orbit
    assert config.model.magnetic_moments == derived
    assert config.data.n_matrix_components == config.model.n_matrix_components == channels


@pytest.mark.parametrize("policy,moments,model,match", [
    ("spin_orbit_full", None, {}, r"spin_orbit_full.*data\.magnetic_moments='vector'"),
    ("spin_orbit_full", "collinear", {}, r"spin_orbit_full.*data\.magnetic_moments='vector'"),
    ("spin_orbit", "vector", {}, r"spin_orbit.*magnetic_moments"),
    ("spin_h_only", "vector", {}, r"magnetic_moments='vector'.*spin_orbit_full"),
    ("h_only", "vector", {}, r"magnetic_moments='vector'.*spin_orbit_full"),
    ("h_only", "collinear", {}, r"magnetic_moments='collinear'.*spin_h_only"),
    ("spin_h_only", None, {"magnetic_moments": "collinear"}, r"model\.magnetic_moments='collinear'.*data\.magnetic_moments"),
    ("h_only", None, {"magnetic_moments": "vector"}, r"model\.magnetic_moments='vector'.*data\.magnetic_moments"),
    ("spin_orbit_full", "vector", {"magnetic_moments": "collinear"}, r"model\.magnetic_moments='collinear'"),
    ("spin_h_only", "collinear", {"spin_orbit": True}, r"model\.spin_orbit=True.*matrix_component_policy"),
    ("h_only", None, {"spin_orbit": True}, r"model\.spin_orbit=True.*matrix_component_policy"),
], ids=["full-no-moments", "full-collinear", "soc-vector", "spin-vector", "plain-vector", "plain-collinear",
        "explicit-collinear-no-data", "explicit-vector-no-data", "explicit-collinear-full",
        "explicit-spin-orbit-collinear", "explicit-spin-orbit-plain"])
def test_cli_rejects_conflicts(policy, moments, model, match):
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = _config(policy, moments)
    for name, value in model.items():
        setattr(config.model, name, value)
    with pytest.raises(ValueError, match=match):
        _autoconfigure_n_matrix_components(config, stage="fit")
    # The user's explicit options are never overwritten behind their back.
    for name, value in model.items():
        assert getattr(config.model, name) == value


@pytest.mark.parametrize("policy,moments,n", [("spin_orbit_full", "vector", 4), ("spin_orbit_full", "vector", 2),
                                              ("spin_h_only", "collinear", 4)])
def test_cli_rejects_other_channel_counts(policy, moments, n):
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    with pytest.raises(ValueError, match="n_matrix_components"):
        _autoconfigure_n_matrix_components(_config(policy, moments, n), stage="fit")


def test_cli_accepts_the_explicit_options_the_data_imply():
    from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components

    config = _config("spin_orbit_full", "vector")
    config.model.spin_orbit, config.model.magnetic_moments = True, "vector"
    _autoconfigure_n_matrix_components(config, stage="fit")
    assert config.model.magnetic_moments == "vector"


def test_cli_parser_takes_the_options_and_derives_them():
    import pytorch_lightning as pl
    from pytorch_lightning.cli import LightningArgumentParser

    from graph2mat.tools.lightning.cli import OrbitalMatrixCLI, _autoconfigure_n_matrix_components

    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, "trainer")
    parser.add_lightning_class_args(LitMACEMatrixModel, "model")
    parser.add_lightning_class_args(MatrixDataModule, "data")
    OrbitalMatrixCLI.add_arguments_to_parser(None, parser)
    config = parser.parse_args(["--data.matrix_component_policy=spin_orbit_full", "--data.magnetic_moments=vector"])
    assert config.data.magnetic_moments == "vector" and config.model.magnetic_moments is None
    _autoconfigure_n_matrix_components(config, stage="fit")
    assert (config.model.spin_orbit, config.model.magnetic_moments) == (True, "vector")
    assert config.data.n_matrix_components == config.model.n_matrix_components == 8

    config = parser.parse_args(["--data.matrix_component_policy=spin_orbit_full", "--data.magnetic_moments=vector",
                                "--model.magnetic_moments=collinear"])
    with pytest.raises(ValueError, match=r"model\.magnetic_moments='collinear'"):
        _autoconfigure_n_matrix_components(config, stage="fit")


# ---------------------------------------------------------------------------
# Writers and servers
# ---------------------------------------------------------------------------

def test_writer_writes_spin_orbit_full_predictions(tmp_path):
    # spin_orbit_full predictions are written as an orthogonal spin-orbit H (the labels carry no S).
    from graph2mat.tools.lightning.callbacks import MatrixWriter

    H, _, atoms = make_magnetic_system()
    processor = full_processor(atoms, True).copy(magnetic_moments="vector")
    sample = TorchBasisMatrixData.new(H, data_processor=processor, magnetic_moments=np.zeros((H.geometry.na, 3)))
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=processor, out_matrix="hamiltonian"))
    MatrixWriter(str(tmp_path / "out.TSHS"))._on_batch_end(
        "predict", trainer, None, {"node_labels": sample.point_labels, "edge_labels": sample.edge_labels},
        Batch.from_data_list([sample]), 0, 0)
    written = sisl.get_sile(tmp_path / "out.TSHS").read_hamiltonian()
    assert written.spin.is_spinorbit and written.orthogonal
    for i in range(8):
        np.testing.assert_allclose(written.tocsr(i).toarray(), H.tocsr(i).toarray(), atol=1e-10)


def test_writer_still_writes_collinear_predictions(models, tmp_path):
    # spin_h_only predictions are ordinary polarized Hamiltonians: only spin_orbit_full is refused.
    from graph2mat.tools.lightning.callbacks import MatrixWriter

    samples = _samples("collinear", 1)
    batch, processor = Batch.from_data_list(samples), samples[0].metadata["data_processor"]
    with torch.no_grad():
        prediction = models["collinear"].model.eval()(batch.clone())
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=processor, out_matrix="hamiltonian"))
    MatrixWriter(str(tmp_path / "out.HSX"))._on_batch_end("predict", trainer, None, prediction, batch, 0, 0)
    assert (tmp_path / "out.HSX").exists()


@pytest.mark.parametrize("kind", ["collinear", "vector"])
def test_server_rejects_processors_with_moments(kind):
    # Its extrapolation path rebuilds configurations without moments. The vector processor is relabelled
    # h_only so that only the moments are at fault.
    from graph2mat.tools.server.server_app import create_server_app

    processor = _samples(kind, 1)[0].metadata["data_processor"]
    if kind == "vector":
        processor = processor.copy(matrix_component_policy="h_only", n_matrix_components=1)
    with pytest.raises(ValueError, match=r"mag: .*magnetic_moments.*not supported"):
        create_server_app({"mag": {"data_processor": processor}})


def test_server_rejects_spin_orbit_full_policy_even_without_moments():
    from graph2mat.tools.server.server_app import create_server_app

    _, _, atoms = make_magnetic_system()
    with pytest.raises(ValueError, match=r"full: .*spin_orbit_full.*not supported"):
        create_server_app({"full": {"data_processor": full_processor(atoms, True)}})


def test_plot_matrix_error_runs_with_spin_orbit_full_data(monkeypatch):
    from graph2mat.tools.lightning import callbacks

    sample = _samples("vector", 1)[0]
    processor = sample.metadata["data_processor"]
    trainer = SimpleNamespace(datamodule=SimpleNamespace(
        data_processor=processor, basis_table=processor.basis_table, symmetric_matrix=True, out_matrix="hamiltonian"))
    titles = []
    monkeypatch.setattr(callbacks, "plot_basis_matrix", lambda matrix, **kw: SimpleNamespace(
        update_layout=lambda **layout: titles.append(layout["title"])))
    callback = callbacks.PlotMatrixError(show=False, store_in_logger=False)
    outputs = {"node_labels": sample.point_labels + 0.5, "edge_labels": sample.edge_labels + 0.5}
    callback.on_test_epoch_start(trainer, None)
    callback.on_test_batch_end(trainer, None, outputs, Batch.from_data_list([sample]), 0)
    callback.on_test_epoch_end(trainer, None)
    assert sorted(titles) == sorted(f"{m}_component{c}" for m in ("MAE", "RMSE") for c in range(8))


# ---------------------------------------------------------------------------
# Training on CPU, normalisation and checkpoints
# ---------------------------------------------------------------------------

class _StepLosses(pl.Callback):
    def __init__(self):
        self.values = []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.values.append(float(outputs["loss"]))


def _mean_neighbours(samples):
    return sum(s["edge_index"].shape[1] for s in samples) / sum(len(s["positions"]) for s in samples)


@pytest.fixture(scope="module", params=["vector", "collinear"])
def trained(request, tmp_path_factory):
    """A magnetic model after 3 training steps, its losses, the initial weights and its checkpoint."""
    kind = request.param
    samples = _samples(kind)
    model = _model(kind, optim_lr=1e-2)
    assert model.hparams.avg_num_neighbors == 1.0
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    losses = _StepLosses()
    trainer = _trainer(max_steps=3, max_epochs=3, callbacks=[losses])
    trainer.fit(model, datamodule=_module(kind, samples))
    assert trainer.global_step == 3
    path = tmp_path_factory.mktemp(kind) / f"{kind}.ckpt"
    trainer.save_checkpoint(path)
    return SimpleNamespace(kind=kind, samples=samples, model=model, losses=losses.values, initial=initial,
                           path=path, trainer=trainer)


def test_three_step_training_gives_finite_decreasing_loss(trained):
    assert len(trained.losses) == 3 and np.isfinite(trained.losses).all()
    assert all(a > b for a, b in zip(trained.losses, trained.losses[1:])), trained.losses


def test_training_updates_the_magnetic_layer_and_the_readout(trained):
    state = trained.model.state_dict()
    for prefix in ("model.magnetic_layer.", "model.matrix_readouts.", "model.mace."):
        keys = [k for k in state if k.startswith(prefix)]
        assert keys and any(not torch.equal(trained.initial[k], state[k]) for k in keys), prefix


def test_fit_normalises_the_neighbour_term_with_the_training_graphs(trained):
    expected = _mean_neighbours(trained.samples)
    assert expected > 1.0
    model = trained.model
    assert model.hparams.avg_num_neighbors == pytest.approx(expected)
    assert model.model.magnetic_layer.avg_num_neighbors == pytest.approx(expected)
    assert all(i.avg_num_neighbors == pytest.approx(expected) for i in model.model.mace.interactions)


def test_checkpoint_reproduces_the_trained_model(trained):
    restored = LitMACEMatrixModel.load_from_checkpoint(trained.path, map_location="cpu", weights_only=False)
    assert restored.hparams.magnetic_moments == VARIANTS[trained.kind][2]
    assert restored.hparams.avg_num_neighbors == pytest.approx(_mean_neighbours(trained.samples))
    assert restored.model.magnetic_layer.avg_num_neighbors == trained.model.model.magnetic_layer.avg_num_neighbors
    batch = Batch.from_data_list(trained.samples)
    trained.model.eval(), restored.eval()
    with torch.no_grad():
        expected, actual = trained.model(batch.clone()), restored(batch.clone())
    for key in ("node_labels", "edge_labels"):
        assert torch.isfinite(expected[key]).all()
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_ckpt_path_restores_the_derived_normalisation_into_a_default_model(trained):
    # trainer.predict(fresh_model, ckpt_path=...) with a model built with the default (a config saved before fit):
    # the derived value is not in the state dict, so loading the checkpoint has to restore it.
    fresh = _model(trained.kind)
    assert fresh.hparams.avg_num_neighbors == 1.0
    module = _module(trained.kind, trained.samples)
    module.predict_dataset = trained.samples
    _trainer().predict(fresh, datamodule=module, ckpt_path=trained.path)
    expected = _mean_neighbours(trained.samples)
    assert fresh.hparams.avg_num_neighbors == pytest.approx(expected)
    assert fresh.model.magnetic_layer.avg_num_neighbors == pytest.approx(expected)
    assert all(i.avg_num_neighbors == pytest.approx(expected) for i in fresh.model.mace.interactions)
    batch = Batch.from_data_list(trained.samples)
    trained.model.eval(), fresh.eval()
    with torch.no_grad():
        actual, reference = fresh(batch.clone()), trained.model(batch.clone())
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("stored", [None, 1.0])
def test_resumed_fit_does_not_derive_over_a_restored_value(trained, tmp_path, stored):
    # The checkpoint's value wins over the training data's mean, also when it is the default 1.0.
    path = trained.path
    if stored is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["hyper_parameters"]["avg_num_neighbors"] = stored
        path = tmp_path / "stored.ckpt"
        torch.save(checkpoint, path)
    fresh = _model(trained.kind)
    _trainer(max_steps=4, max_epochs=4).fit(
        fresh, datamodule=_module(trained.kind, trained.samples), ckpt_path=path)
    expected = _mean_neighbours(trained.samples) if stored is None else stored
    assert fresh.hparams.avg_num_neighbors == pytest.approx(expected)
    assert fresh.model.magnetic_layer.avg_num_neighbors == pytest.approx(expected)


def test_load_from_lit_ckpt_restores_model_and_processor(trained):
    model, processor = load_from_lit_ckpt(trained.path, as_torch=True)
    assert processor.magnetic_moments == VARIANTS[trained.kind][2]
    assert processor.matrix_component_policy == VARIANTS[trained.kind][1]
    assert model.magnetic_moments == VARIANTS[trained.kind][2]
    sample = trained.samples[0]
    trained.model.eval(), model.eval()
    with torch.no_grad():
        expected, actual = trained.model.model(sample.clone()), model(sample.clone())
    torch.testing.assert_close(actual["edge_labels"], expected["edge_labels"], rtol=0, atol=0)


def test_explicit_avg_num_neighbors_is_kept():
    # Also what a restored model does when fitted again: its stored value is not 1.0.
    model = _model("vector", avg_num_neighbors=7.0)
    trainer = _trainer(max_steps=1, max_epochs=1)
    trainer.fit(model, datamodule=_module("vector", _samples("vector")))
    assert model.hparams.avg_num_neighbors == 7.0
    assert model.model.magnetic_layer.avg_num_neighbors == 7.0
    assert all(i.avg_num_neighbors == 7.0 for i in model.model.mace.interactions)


def test_models_without_moments_keep_avg_num_neighbors():
    model = _model("soc")
    _trainer(max_steps=1, max_epochs=1).fit(model, datamodule=_module("soc", _soc_samples()))
    assert model.hparams.avg_num_neighbors == 1.0
    assert all(i.avg_num_neighbors == 1.0 for i in model.model.mace.interactions)


def test_training_stage_groups_include_the_magnetic_layer(models):
    model = models["vector"]
    layer = {id(p) for p in model.model.magnetic_layer.parameters()}
    assert layer
    for group in ("readout", "readouts", "matrix_readouts"):
        assert layer <= {id(p) for p in model._parameter_group_modules()[group].parameters()}
    assert not layer & {id(p) for p in model._parameter_group_modules()["mace"].parameters()}


def test_optimizer_trains_the_magnetic_layer(models):
    optimizer = models["collinear"].configure_optimizers()
    trained = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert {id(p) for p in models["collinear"].model.magnetic_layer.parameters()} <= trained


# ---------------------------------------------------------------------------
# Checkpoints written before the magnetic options existed
# ---------------------------------------------------------------------------

def _plain_samples():
    geometry = make_soc_system(0, orthogonal=True)[0].geometry
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="hamiltonian",
                                    symmetric_matrix=True, sub_point_matrix=False)
    samples = []
    for seed in range(2):
        rng = np.random.default_rng(seed)
        onsite, image = rng.normal(size=(geometry.no,) * 2), rng.normal(size=(geometry.no,) * 2)
        blocks = {0: onsite + onsite.T, 1: image, -1: image.T}
        H = sisl.Hamiltonian.fromsp(
            geometry, [csr_array(np.concatenate([blocks[int(off[0])] for off in geometry.sc_off], axis=1))])
        samples.append(TorchBasisMatrixData.new(H, data_processor=processor))
    return samples


def _soc_samples():
    samples = []
    for seed in range(2):
        H, atoms, _, _ = make_soc_system(seed)
        samples.append(TorchBasisMatrixData.new(H, data_processor=soc_processor(atoms, symmetric=True)))
    return samples


@pytest.mark.parametrize("kind,samples", [("plain", _plain_samples), ("soc", _soc_samples)])
def test_checkpoints_without_the_magnetic_keys_load_and_predict_identically(tmp_path, kind, samples):
    samples = samples()
    model = _model(kind)
    trainer = _trainer(max_steps=1, max_epochs=1)
    trainer.fit(model, datamodule=_module(kind, samples))
    path = tmp_path / f"{kind}.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["hyper_parameters"].pop("magnetic_moments") is None  # what an older version never wrote
    assert checkpoint["datamodule_hyper_parameters"].pop("magnetic_moments") is None
    assert not [k for k in checkpoint["state_dict"] if "magnetic" in k]
    torch.save(checkpoint, path)

    restored = LitMACEMatrixModel.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    assert restored.hparams.get("magnetic_moments") is None and restored.model.magnetic_layer is None
    batch = Batch.from_data_list(samples)
    model.eval(), restored.eval()
    with torch.no_grad():
        expected, actual = model(batch.clone()), restored(batch.clone())
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    loaded, processor = load_from_lit_ckpt(path, as_torch=True)
    assert processor.magnetic_moments is None
    with torch.no_grad():
        torch.testing.assert_close(loaded.eval()(batch.clone())["edge_labels"], expected["edge_labels"], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Reachability with the real Fe basis
# ---------------------------------------------------------------------------

# Campaign recipe of roadmap 8.1 (auditorias_mejoras/run_soc_campaign.py).
CAMPAIGN = dict(hidden_irreps="16x0e+16x0o+16x1o+16x1e+16x2e+16x2o",
                edge_hidden_irreps="8x0e+8x0o+8x1o+8x1e+8x2e+8x2o+8x3o+8x3e", max_ell=3, correlation=3,
                num_interactions=2, num_bessel=8)


def test_fe_basis_reaches_every_coefficient_of_the_eight_channels(float32):
    """Structural check, so it runs in float32 (seconds; float64 takes minutes).
    Fe SIESTA basis: 3 s, 2 p and 2 d shells (``3x0e+2x1o+2x2e``). With the campaign hidden irreps
    no irrep instance of any of the eight channels (on-site ``a`` has a ``4o`` part) is structurally zero:
    the coefficients of both passes, combined as time reversal combines them, are non-zero everywhere."""
    fe = sisl.Atom("Fe", orbitals=[o for l, zetas in ((0, 3), (1, 2), (2, 2)) for z in range(1, zetas + 1)
                                   for o in _shell(l, z)])
    start = time.perf_counter()
    model = _model("vector", atoms=[fe], **CAMPAIGN).eval()
    build_seconds = time.perf_counter() - start
    readout = model.model.matrix_readouts
    onsite = readout.self_interactions[0]
    assert str(onsite.i_irreps) == "3x0e+2x1o+2x2e"
    assert o3.Irrep("4o") in [ir for _, ir in onsite._a_irreps_out]
    blocks = [b for b in [*readout.self_interactions, *readout.interactions.values()] if b is not None]
    captured = {id(b): [] for b in blocks}
    hooks = [b.operation.register_forward_hook(lambda mod, args, out, key=id(b): captured[key].append(out))
             for b in blocks]
    xyz = np.array([[0.0, 0.0, 0.0], [2.5, 0.2, 0.1], [1.2, 2.4, -0.2], [1.1, 0.9, 2.3], [-1.3, 1.0, 1.9]])
    geometry = sisl.Geometry(xyz, atoms=fe, lattice=[30.0, 30.0, 30.0])
    processor = MatrixDataProcessor(
        basis_table=model.basis_table, out_matrix="hamiltonian", symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=8, magnetic_moments="vector", matrix_component_policy="spin_orbit_full")
    moments = np.random.default_rng(1).normal(0, 2.0, (geometry.na, 3))
    config = OrbitalConfiguration.from_geometry(geometry, magnetic_moments=moments)
    try:
        with torch.no_grad():
            model.model(TorchBasisMatrixData.from_config(config, data_processor=processor))
    finally:
        for hook in hooks:
            hook.remove()
    missing = []
    for block in blocks:
        plus, minus = captured[id(block)]
        groups = [(block._single_irreps_out, (plus + minus) / 2), (block._g_irreps_out, (plus + minus) / 2),
                  (block._b0_irreps_out, (plus - minus) / 2), (block._a_irreps_out, (plus - minus) / 2)]
        start = 0
        for name, (irreps, coefficients) in zip(["a0", "b", "b0", "a"], groups):
            for mul, ir in irreps:
                for _ in range(mul):
                    # fp32 rounding is ~1e-7 of the largest coefficient; the smallest real ones are ~2e-4 of it.
                    if coefficients[:, start:start + ir.dim].norm() < 1e-5 * coefficients.abs().max():
                        missing.append(f"{block.i_irreps} x {block.j_irreps} {name} {ir}")
                    start += ir.dim
        assert start == block._irreps_out.dim
    assert not missing, missing
    print(f"Fe full-block model build: {build_seconds:.1f} s")
