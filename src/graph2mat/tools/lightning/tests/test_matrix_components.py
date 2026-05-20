import json
from types import SimpleNamespace

import numpy as np
import pytest
import sisl
import torch
from jsonargparse import Namespace
from scipy.sparse import csr_array
from sisl.io.siesta import _siesta

from graph2mat import (
    BasisTableWithEdges,
    MatrixDataProcessor,
    OrbitalConfiguration,
    PointBasis,
)
from graph2mat.bindings.torch.data import TorchBasisMatrixData
from graph2mat.core.data.sparse import csr_to_sisl_sparse_orbital
from graph2mat.tools.lightning import MatrixDataModule, MatrixWriter
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


def _nonspin_nonorthogonal_hamiltonian():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atoms=[atom, atom.copy()],
        lattice=[10, 10, 10],
    )

    return sisl.Hamiltonian.fromsp(
        geometry,
        csr_array([[1.0, 0.1], [0.2, 2.0]]),
        S=csr_array([[10.0, 0.3], [0.4, 20.0]]),
    )


def _symmetric_nonspin_nonorthogonal_hamiltonian():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atoms=[atom, atom.copy()],
        lattice=[10, 10, 10],
    )

    return sisl.Hamiltonian.fromsp(
        geometry,
        csr_array([[1.0, 0.125], [0.125, 2.0]]),
        S=csr_array([[10.0, 0.75], [0.75, 20.0]]),
    )


def _single_orbital_basis_table():
    return BasisTableWithEdges([PointBasis(1, R=np.array([2.0]), basis=[1])])


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


def test_datamodule_validates_h_only_post_policy_components():
    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        basis_table=_single_orbital_basis_table(),
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
        train_runs=[_nonspin_nonorthogonal_hamiltonian()],
    )

    datamodule.setup("fit")
    sample = datamodule.train_dataset[0]

    assert sample.point_labels.ndim == 1
    assert sample.edge_labels.ndim == 1


def test_datamodule_h_only_rejects_raw_component_count():
    datamodule = MatrixDataModule(
        out_matrix="hamiltonian",
        basis_table=_single_orbital_basis_table(),
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="h_only",
        train_runs=[_nonspin_nonorthogonal_hamiltonian()],
    )

    with pytest.raises(ValueError, match="matrix_component_policy='h_only'"):
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


def test_shape_validation_accepts_h_only_one_component_labels():
    out = {
        "node_labels": torch.zeros(4),
        "edge_labels": torch.zeros(2),
    }
    batch = {
        "point_labels": torch.zeros(4),
        "edge_labels": torch.zeros(2),
    }

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
            matrix_component_policy="raw_components",
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


def test_infer_matrix_components_uses_hamiltonian_policy():
    hamiltonian = _nonspin_nonorthogonal_hamiltonian()
    basis_table = _single_orbital_basis_table()

    inferred_h_only = infer_n_matrix_components_from_data_inputs(
        basis_table=basis_table,
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        matrix_component_policy="h_only",
        train_runs=[hamiltonian],
    )
    inferred_raw = infer_n_matrix_components_from_data_inputs(
        basis_table=basis_table,
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        matrix_component_policy="raw_components",
        train_runs=[hamiltonian],
    )

    assert inferred_h_only == 1
    assert inferred_raw == 2


def test_infer_matrix_components_h_only_symmetric_hamiltonian_is_one():
    inferred = infer_n_matrix_components_from_data_inputs(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=True,
        sub_point_matrix=False,
        matrix_component_policy="h_only",
        train_runs=[_symmetric_nonspin_nonorthogonal_hamiltonian()],
    )

    assert inferred == 1


def test_torch_data_h_only_exposes_h_not_overlap():
    hamiltonian = _nonspin_nonorthogonal_hamiltonian()
    data_processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
    )

    sample = TorchBasisMatrixData.new(
        hamiltonian, data_processor=data_processor, labels=True
    )

    np.testing.assert_allclose(sample.point_labels.numpy(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        np.sort(sample.edge_labels.numpy()), np.array([0.1, 0.2])
    )


def test_raw_components_do_not_serialize_as_spin():
    hamiltonian = _nonspin_nonorthogonal_hamiltonian()
    data_processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="raw_components",
    )
    sample = TorchBasisMatrixData.new(
        hamiltonian, data_processor=data_processor, labels=True
    )

    with pytest.raises(ValueError, match="raw_components"):
        sample.convert_to("sisl_H")


def test_h_and_overlap_writes_unpolarized_nonorthogonal_hamiltonian():
    hamiltonian = _nonspin_nonorthogonal_hamiltonian()
    data_processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=2,
        matrix_component_policy="h_and_overlap",
    )
    sample = TorchBasisMatrixData.new(
        hamiltonian, data_processor=data_processor, labels=True
    )

    matrix = sample.convert_to("sisl_H")

    assert not matrix.spin.is_polarized
    assert not matrix.orthogonal
    assert matrix.S_idx == 1


def test_matrix_writer_h_only_outputs_unpolarized_hamiltonian_metadata(tmp_path):
    hamiltonian = _nonspin_nonorthogonal_hamiltonian()
    data_processor = MatrixDataProcessor(
        basis_table=_single_orbital_basis_table(),
        out_matrix="hamiltonian",
        symmetric_matrix=False,
        sub_point_matrix=False,
        n_matrix_components=1,
        matrix_component_policy="h_only",
    )
    sample = TorchBasisMatrixData.new(
        hamiltonian, data_processor=data_processor, labels=True
    )

    class WriterDataProcessor:
        default_out_format = data_processor.default_out_format
        matrix_component_policy = data_processor.matrix_component_policy
        n_matrix_components = data_processor.n_matrix_components

        def yield_from_batch(self, batch, predictions=None):
            yield sample

    out_file = tmp_path / "prediction.HSX"
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            data_processor=WriterDataProcessor(),
            out_matrix="hamiltonian",
        )
    )

    MatrixWriter(str(out_file), splits=["test"])._on_batch_end(
        "test",
        trainer=trainer,
        pl_module=None,
        prediction={},
        batch=None,
        batch_idx=0,
        dataloader_idx=0,
    )

    written = sisl.get_sile(str(out_file)).read_hamiltonian(
        geometry=hamiltonian.geometry
    )
    assert not written.spin.is_polarized
    assert written._csr.data.shape[1] == 1

    metadata = json.loads(out_file.with_suffix(".HSX.metadata.json").read_text())
    assert metadata["matrix_component_policy"] == "h_only"
    assert metadata["serialized_spin_polarized"] is False
    assert "without an overlap matrix" in metadata["note"]


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
        matrix_component_policy="h_and_overlap",
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


def test_hamiltonian_multicomponent_conversion_requires_policy():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry([[0.0, 0.0, 0.0]], atoms=[atom], lattice=[10, 10, 10])

    with pytest.raises(ValueError, match="matrix_component_policy"):
        csr_to_sisl_sparse_orbital(
            [csr_array([[1.0]]), csr_array([[2.0]])],
            geometry=geometry,
            sp_class=sisl.Hamiltonian,
        )


def test_hamiltonian_spin_conversion_is_explicit():
    atom = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])
    geometry = sisl.Geometry([[0.0, 0.0, 0.0]], atoms=[atom], lattice=[10, 10, 10])

    matrix = csr_to_sisl_sparse_orbital(
        [csr_array([[1.0]]), csr_array([[2.0]])],
        geometry=geometry,
        sp_class=sisl.Hamiltonian,
        matrix_component_policy="spin_h_only",
    )

    assert matrix.spin.is_polarized
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), [[1.0]])
    np.testing.assert_allclose(matrix.tocsr(1).toarray(), [[2.0]])
