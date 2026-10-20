from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch

from graph2mat import (
    AtomicTableWithEdges,
    BasisConfiguration,
    BasisMatrixData,
    BasisTableWithEdges,
    MatrixDataProcessor,
    PointBasis,
)


def _atoms():
    return [
        sisl.Atom(
            1,
            [
                sisl.AtomicOrbital("1s", R=1.0),
                sisl.AtomicOrbital("2px", R=1.0),
                sisl.AtomicOrbital("2py", R=1.0),
                sisl.AtomicOrbital("2pz", R=1.0),
            ],
        )
    ]


def _processor(atoms, **kwargs):
    return MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(atoms),
        out_matrix="hamiltonian",
        sub_point_matrix=False,
        **kwargs,
    )


def _hamiltonian(positions, lattice, nsc=(1, 1, 1), offdiag=(), seed=0):
    atoms = _atoms()
    geom = sisl.Geometry(positions, atoms=atoms[0], lattice=lattice)
    geom.set_nsc(list(nsc))
    H = sisl.Hamiltonian(geom)
    rng = np.random.default_rng(seed)
    n = geom.no
    for io in range(n):
        for jo in range(n):
            if io // 4 == jo // 4:
                H[io, jo] = 0.5 * (1 + abs(io - jo)) if io <= jo else H[jo, io]
    for io, jo, val in offdiag:
        H[io, jo] = val
        H[jo, io] = val
    return H, atoms


def test_zero_padding_is_identical():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, size=(3, 3))
    H, atoms = _hamiltonian(pos, sisl.Lattice(4.0, nsc=[3, 3, 3]), nsc=(3, 3, 3))
    for i in range(H.no):
        H[i, i] = 1.0 + i
    default = BasisMatrixData.new(H, _processor(atoms, symmetric_matrix=True))
    zero = BasisMatrixData.new(
        H, _processor(atoms, symmetric_matrix=True, edge_cutoff_padding=0.0)
    )
    assert default.edge_index.shape[1] > 0
    for name in ("edge_index", "neigh_isc", "point_labels", "edge_labels"):
        np.testing.assert_array_equal(
            np.asarray(getattr(default, name)), np.asarray(getattr(zero, name))
        )


def test_padding_adds_edges_and_their_labels():
    # R_i + R_j = 2 < d = 2.5 < 2 * (1 + 0.3) = 2.6
    H, atoms = _hamiltonian(
        [[0, 0, 0], [2.5, 0, 0]],
        sisl.Lattice(30.0),
        offdiag=[(0, 4, 0.37), (1, 5, -0.21)],
    )
    plain = BasisMatrixData.new(H, _processor(atoms, symmetric_matrix=True))
    assert plain.edge_index.shape[1] == 0

    proc = _processor(atoms, symmetric_matrix=True, edge_cutoff_padding=0.3)
    padded = BasisMatrixData.new(H, proc)
    assert padded.edge_index.shape[1] > 0
    assert np.abs(np.asarray(padded.edge_labels)).max() == pytest.approx(0.37)
    H2 = proc.matrix_from_data(padded, out_format="sisl_H", threshold=None)
    np.testing.assert_allclose(
        H2.tocsr().toarray(), H.tocsr().toarray(), atol=1e-12
    )

    # Below the padded cutoff the edge is gone again.
    short = _processor(atoms, symmetric_matrix=True, edge_cutoff_padding=0.2)
    assert BasisMatrixData.new(H, short).edge_index.shape[1] == 0


def test_padding_cutoff_per_species_and_scalar():
    atoms = _atoms()
    proc = _processor(atoms, edge_cutoff_padding=0.3)
    types = np.array([0, 0])
    np.testing.assert_allclose(proc.get_cutoff(types), np.array([1.3, 1.3]))
    assert proc.copy(edge_cutoff_padding=0.0).get_cutoff(types)[0] == 1.0

    # Tables with a single scalar radius use a global cutoff 2 * (R + p).
    scalar = MatrixDataProcessor(
        basis_table=SimpleNamespace(R=1.5), edge_cutoff_padding=0.25
    )
    assert scalar.get_cutoff(types) == pytest.approx(2 * (1.5 + 0.25))


def test_insufficient_nsc_for_padded_edges_raises():
    # Periodic cell 2.5: padded cutoff 2.6 needs the first image, nsc=1 cannot hold it.
    H, atoms = _hamiltonian([[0, 0, 0]], sisl.Lattice([2.5, 30, 30], nsc=[1, 1, 1]))
    proc = _processor(atoms, symmetric_matrix=True, edge_cutoff_padding=0.3)
    with pytest.raises(ValueError, match="nsc"):
        BasisMatrixData.new(H, proc)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_padding_validation(bad):
    with pytest.raises(ValueError, match="edge_cutoff_padding"):
        _processor(_atoms(), edge_cutoff_padding=bad)


def _checkpoint(tmp_path, pop_padding):
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    basis = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    config = BasisConfiguration(
        point_types=["A", "A"],
        positions=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        basis=basis,
        cell=np.eye(3) * 10,
        pbc=(False, False, False),
        matrix=np.ones((2, 2)),
    )
    module = MatrixDataModule(
        basis_table=basis,
        symmetric_matrix=True,
        sub_point_matrix=False,
        train_runs=[config],
        val_runs=[config],
        loader_threads=0,
        edge_cutoff_padding=0.5,
    )
    model = LitMACEMatrixModel(
        basis_table=basis,
        symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o",
        edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2,
        max_ell=1,
        edge_cutoff_padding=0.5,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=module)
    path = tmp_path / "m.ckpt"
    trainer.save_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["datamodule_hyper_parameters"]["edge_cutoff_padding"] == 0.5
    if pop_padding:
        ckpt["datamodule_hyper_parameters"].pop("edge_cutoff_padding")
        torch.save(ckpt, path)
    module.teardown(None)
    return path


@pytest.mark.parametrize("pop_padding, expected", [(False, 0.5), (True, 0.0)])
def test_checkpoint_loader_keeps_padding(tmp_path, pop_padding, expected):
    from graph2mat.bindings.torch.load import load_from_lit_ckpt

    _, processor = load_from_lit_ckpt(_checkpoint(tmp_path, pop_padding))
    assert processor.edge_cutoff_padding == expected
