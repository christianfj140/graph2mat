import numpy as np
import pytest
import sisl
import torch
from jsonargparse import Namespace
from scipy.sparse import csr_array
from sisl.io.siesta import _siesta

from graph2mat import BasisTableWithEdges, OrbitalConfiguration, PointBasis
from graph2mat.core.data.sparse import csr_to_sisl_sparse_orbital
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.cli import _autoconfigure_n_matrix_components
from graph2mat.tools.lightning.data import infer_n_matrix_components_from_data_inputs
from graph2mat.tools.lightning.model import LitBasisMatrixModel


def _polarized_single_orbital_config():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atoms=[atom, atom.copy()],
        lattice=[10, 10, 10],
    )

    h = sisl.Hamiltonian(geometry, spin=sisl.Spin("polarized"))
    h._csr = h._csr.fromsp(
        [
            csr_array([[1.0, 0.1], [0.1, 3.0]]),
            csr_array([[2.0, 0.2], [0.2, 4.0]]),
        ]
    )

    return OrbitalConfiguration.from_matrix(h, labels=True)


def test_datamodule_raises_on_matrix_component_mismatch():
    config = _polarized_single_orbital_config()
    basis_table = BasisTableWithEdges([PointBasis(1, R=np.array([2.0]), basis=[1])])

    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        basis_table=basis_table,
        symmetric_matrix=True,
        sub_point_matrix=False,
        n_matrix_components=1,
        train_runs=[config],
    )

    with pytest.raises(ValueError, match="n_matrix_components mismatch"):
        datamodule.setup("fit")


def test_shape_validation_raises_clear_error_message():
    out = {
        "node_labels": torch.zeros(4),
        "edge_labels": torch.zeros(2),
    }
    batch = {
        "point_labels": torch.zeros(4, 2),
        "edge_labels": torch.zeros(2, 2),
    }

    with pytest.raises(ValueError, match="n_matrix_components"):
        LitBasisMatrixModel._validate_pred_ref_shapes(out, batch)


def test_cli_autoconfigures_matrix_components(monkeypatch):
    config = Namespace(
        data=Namespace(
            root_dir=".",
            basis_files=None,
            no_basis=None,
            basis_table=None,
            out_matrix="hamiltonian",
            symmetric_matrix=True,
            sub_point_matrix=False,
            initial_node_feats="OneHotZ",
            train_runs=["dummy"],
            val_runs=None,
            test_runs=None,
            runs_json=None,
            n_matrix_components=1,
        ),
        model=Namespace(n_matrix_components=1),
    )

    monkeypatch.setattr(
        "graph2mat.tools.lightning.cli.infer_n_matrix_components_from_data_inputs",
        lambda **kwargs: 3,
    )

    _autoconfigure_n_matrix_components(config)

    assert config.data.n_matrix_components == 3
    assert config.model.n_matrix_components == 3


def test_infer_matrix_components_accepts_iterables(monkeypatch):
    monkeypatch.setattr(
        "graph2mat.tools.lightning.data.TorchBasisMatrixData.new",
        lambda *args, **kwargs: Namespace(
            point_labels=torch.zeros(4, 3), edge_labels=None
        ),
    )

    inferred = infer_n_matrix_components_from_data_inputs(
        basis_table=object(),
        train_runs=(run for run in ["dummy"]),
    )

    assert inferred == 3


def test_hamiltonian_multicomponent_conversion_preserves_spin_and_overlap(tmp_path):
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atoms=[atom, atom.copy()],
        lattice=[10, 10, 10],
    )

    matrix = csr_to_sisl_sparse_orbital(
        [
            csr_array([[1.0, 0.1], [0.1, 3.0]]),
            csr_array([[2.0, 0.2], [0.2, 4.0]]),
            csr_array([[1.0, 0.0], [0.0, 1.0]]),
        ],
        geometry=geometry,
        sp_class=sisl.Hamiltonian,
    )

    assert isinstance(matrix, sisl.Hamiltonian)
    assert matrix.spin.is_polarized
    assert not matrix.orthogonal
    assert matrix.S_idx == 2
    assert matrix._csr._D.shape == (4, 3)

    out_file = tmp_path / "polarized_non_orthogonal.HSX"
    sisl.get_sile(str(out_file), "w").write_hamiltonian(matrix)

    assert _siesta.read_hsx_sizes(str(out_file))[0] == 2

    reloaded = sisl.get_sile(str(out_file)).read_hamiltonian(geometry=geometry)
    assert reloaded.spin.is_polarized
    assert not reloaded.orthogonal
    assert reloaded._csr._D.shape == (4, 3)
