"""Roadmap 8.2: the model's readout radial range follows the data's edge_cutoff_padding."""

import numpy as np
import pytest
import sisl
import torch
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.tools.lightning.tests.test_spin_orbit_model import _trainer

R = 2.1
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


def _atoms():
    sp = [sisl.AtomicOrbital(n=1, l=0, m=0, R=R)] + [
        sisl.AtomicOrbital(n=2, l=1, m=m, R=R) for m in (-1, 0, 1)]
    return [sisl.Atom("H", orbitals=sp), sisl.Atom("O", orbitals=sp)]


def _r_max(module):
    return float(module.cutoff_fn.r_max)


@pytest.mark.parametrize("padding", [0.0, 0.5])
@pytest.mark.parametrize("mp_cutoff", [None, 3.0])
def test_model_radial_range_follows_padding(padding, mp_cutoff):
    model = LitMACEMatrixModel(basis_table=AtomicTableWithEdges(_atoms()), pair_envelope=True,
                               mp_cutoff=mp_cutoff, edge_cutoff_padding=padding, **TINY)
    readout = model.model.readout_radial_embedding if mp_cutoff else model.model.mace.radial_embedding
    assert _r_max(readout) == pytest.approx(2 * (R + padding))
    if mp_cutoff:
        assert _r_max(model.model.mace.radial_embedding) == pytest.approx(mp_cutoff)
    np.testing.assert_allclose(model.model.pair_envelope_radii.numpy(), [R + padding] * 2)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_model_rejects_invalid_padding(bad):
    with pytest.raises(ValueError, match="edge_cutoff_padding"):
        LitMACEMatrixModel(basis_table=AtomicTableWithEdges(_atoms()), edge_cutoff_padding=bad, **TINY)


def test_long_edges_equivariant_with_mp_cutoff():
    padding = 0.5
    atoms = _atoms()
    table = AtomicTableWithEdges(atoms)
    processor = MatrixDataProcessor(basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
                                    edge_cutoff_padding=padding)
    torch.manual_seed(0)
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True, mp_cutoff=2 * R,
                               edge_cutoff_padding=padding, **TINY).eval()
    h, o = atoms
    # 0-2 at 4.64 Å, in (2R, 2R + 2p): a matrix edge but no message-passing edge.
    geometry = sisl.Geometry([[0., 0., 0.], [1., .3, .2], [4.6, .5, .3]], atoms=[h, o, h],
                             lattice=[30, 30, 30])
    long_d = np.linalg.norm(geometry.xyz[2] - geometry.xyz[0])
    assert 2 * R < long_d < 2 * (R + padding)

    def matrix(geom):
        data = TorchBasisMatrixData.from_config(OrbitalConfiguration.from_geometry(geom), data_processor=processor)
        with torch.no_grad():
            return processor.matrix_from_data(data, model(data), out_format="numpy")

    original = matrix(geometry)
    rows, cols = slice(geometry.firsto[0], geometry.firsto[1]), slice(geometry.firsto[2], geometry.firsto[3])
    assert np.isfinite(original).all() and np.max(np.abs(original[rows, cols])) > 1e-6
    cob = processor.basis_table.change_of_basis
    for rotation in Rotation.random(8, random_state=np.random.default_rng(0)).as_matrix():
        rotated = geometry.copy()
        rotated.xyz[:] = geometry.xyz @ rotation.T
        rotated.cell[:] = geometry.cell @ rotation.T
        basis_rotation = torch.as_tensor(cob @ rotation @ cob.T)
        D = block_diag(*[table.basis[int(s)].e3nn_irreps.D_from_matrix(basis_rotation).numpy()
                         for s in geometry.atoms.species])
        np.testing.assert_allclose(matrix(rotated), D @ original @ D.T, rtol=1e-8, atol=1e-10)


def _module(table, padding):
    module = MatrixDataModule(basis_table=table, out_matrix="hamiltonian", symmetric_matrix=True,
                              sub_point_matrix=False, loader_threads=0, train_runs=[], val_runs=[],
                              edge_cutoff_padding=padding)
    module.setup("fit")
    module.train_dataset, module.val_dataset = [], []
    return module


@pytest.mark.parametrize("model_p,data_p", [(0.0, 0.5), (0.5, 0.0)])
@pytest.mark.parametrize("stage", ["fit", "validate"])
def test_padding_disagreement_fails(model_p, data_p, stage):
    table = AtomicTableWithEdges(_atoms())
    model = LitMACEMatrixModel(basis_table=table, symmetric_matrix=True, edge_cutoff_padding=model_p, **TINY)
    with pytest.raises(ValueError, match="edge_cutoff_padding"):
        getattr(_trainer(max_steps=1), stage)(model, datamodule=_module(table, data_p))


def test_cli_links_padding():
    from pytorch_lightning.cli import LightningArgumentParser

    from graph2mat.tools.lightning.cli import OrbitalMatrixCLI

    import pytorch_lightning as pl

    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, "trainer")
    parser.add_lightning_class_args(LitMACEMatrixModel, "model")
    parser.add_lightning_class_args(MatrixDataModule, "data")
    OrbitalMatrixCLI.add_arguments_to_parser(None, parser)
    config = parser.parse_args(["--data.edge_cutoff_padding=0.5"])
    assert config.model.edge_cutoff_padding == 0.5
