import json
from types import SimpleNamespace

import numpy as np
import pytest
import sisl
import torch
from jsonargparse import Namespace
from scipy.sparse import csr_array
from scipy.linalg import eigh
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


def test_shape_validation_accepts_h_only_prediction_of_h_and_overlap_labels():
    """gauge_invariant_mse with n_output_components=1: H predicted, (H, S) labels."""
    out = {"node_labels": torch.zeros(4), "edge_labels": torch.zeros(2)}
    batch = {"point_labels": torch.zeros(4, 2), "edge_labels": torch.zeros(2, 2)}

    LitBasisMatrixModel._validate_pred_ref_shapes(out, batch, overlap_optional=True)
    with pytest.raises(ValueError):  # two components fewer is still an error
        LitBasisMatrixModel._validate_pred_ref_shapes(
            out, {"point_labels": torch.zeros(4, 3), "edge_labels": torch.zeros(2, 3)},
            overlap_optional=True,
        )


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


@pytest.mark.parametrize("runs", [{}, {"train_runs": []}, {"val_runs": []}, {"test_runs": []}])
def test_infer_matrix_components_absent_splits_are_empty(runs):
    assert infer_n_matrix_components_from_data_inputs(
        basis_table=_single_orbital_basis_table(), sub_point_matrix=False, **runs,
    ) is None


@pytest.mark.parametrize("split", ["val", "test"])
def test_infer_matrix_components_can_use_a_single_evaluation_split(split):
    assert infer_n_matrix_components_from_data_inputs(
        basis_table=_single_orbital_basis_table(), out_matrix="hamiltonian",
        sub_point_matrix=False, matrix_component_policy="raw_components",
        **{f"{split}_runs": [_nonspin_nonorthogonal_hamiltonian()]},
    ) == 2


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_cli_component_inference_does_not_read_inactive_splits(stage):
    def inactive():
        raise AssertionError("inactive dataset was read")
        yield

    runs = {f"{split}_runs": inactive() for split in ("train", "val", "test")}
    if stage != "predict":
        runs[f"{'val' if stage == 'validate' else stage}_runs"] = [_nonspin_nonorthogonal_hamiltonian()]
    config = Namespace(data=Namespace(basis_table=_single_orbital_basis_table(), out_matrix="hamiltonian",
                                     sub_point_matrix=False, matrix_component_policy="raw_components",
                                     n_matrix_components=1, **runs), model=Namespace(n_matrix_components=1))
    _autoconfigure_n_matrix_components(config, stage=stage)
    assert config.data.n_matrix_components == config.model.n_matrix_components == (1 if stage == "predict" else 2)


def test_component_inference_validate_uses_automatic_train_split():
    config = Namespace(data=Namespace(basis_table=_single_orbital_basis_table(), out_matrix="hamiltonian",
                                     sub_point_matrix=False, matrix_component_policy="raw_components",
                                     n_matrix_components=1, train_runs=[_nonspin_nonorthogonal_hamiltonian()]),
                       model=Namespace(n_matrix_components=1))
    _autoconfigure_n_matrix_components(config, stage="validate")
    assert config.data.n_matrix_components == config.model.n_matrix_components == 2


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


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_h_and_overlap_batch_predictions_keep_distinct_structures(spin, symmetric):
    from torch_geometric.data import Batch
    from graph2mat import AtomicTableWithEdges

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix="hamiltonian",
        symmetric_matrix=symmetric, sub_point_matrix=False,
        n_matrix_components=spin + 1, matrix_component_policy="h_and_overlap",
    )
    samples, predicted_samples, expected, reference = [], [], [], []
    for n in (2, 3):
        geometry = sisl.Geometry([[float(i), 0., 0.] for i in range(n)], atoms=atom, lattice=[20.] * 3)
        h = [np.diag(np.arange(n) + n + c) + .125 * np.ones((n, n)) for c in range(spin)]
        s = np.eye(n) + .03125 * n * np.ones((n, n))
        hs = [channel + n * .25 * np.eye(n) + .0625 * (n + c + 1) * np.ones((n, n))
              for c, channel in enumerate(h)]
        ss = s + .0625 * n * (np.eye(n) + np.ones((n, n)))
        for matrices, target in (((h, s), samples), ((hs, ss), predicted_samples)):
            channels, overlap = matrices
            matrix = sisl.Hamiltonian.fromsp(
                geometry, [csr_array(channel) for channel in channels], S=csr_array(overlap),
                spin="polarized" if spin == 2 else "unpolarized",
            )
            sample = TorchBasisMatrixData.new(matrix, processor)
            sample.point_labels = sample.point_labels.double()
            sample.edge_labels = sample.edge_labels.double()
            target.append(sample)
        expected.append((geometry, hs, ss))
        reference.append((samples[-1].point_labels.clone(), samples[-1].edge_labels.clone()))
    batch = Batch.from_data_list(samples)
    before = batch.point_labels.clone(), batch.edge_labels.clone(), batch.edge_index.clone()
    predictions = {
        "node_labels": torch.cat([sample.point_labels for sample in predicted_samples]),
        "edge_labels": torch.cat([sample.edge_labels for sample in predicted_samples]),
    }
    converted = processor.matrix_from_data(batch, predictions=predictions, threshold=None)
    explicit = tuple(processor.yield_from_batch(batch, predictions=predictions, threshold=None, as_matrix=True))
    assert len(converted) == len(explicit) == 2
    for result in (converted, explicit):
        for matrix, (geometry, h, s) in zip(result, expected):
            np.testing.assert_array_equal(matrix.geometry.xyz, geometry.xyz)
            np.testing.assert_array_equal(matrix.geometry.cell, geometry.cell)
            np.testing.assert_array_equal(matrix.geometry.orbitals, geometry.orbitals)
            assert matrix.spin.is_polarized == (spin == 2)
            np.testing.assert_array_equal(matrix.tocsr(matrix.S_idx).toarray(), s)
            for channel in range(spin):
                np.testing.assert_array_equal(matrix.tocsr(channel).toarray(), h[channel])
                np.testing.assert_allclose(matrix.eigenvalue(spin=channel).eig,
                    eigh(h[channel], s, eigvals_only=True), rtol=1e-10, atol=1e-10)
    for actual, saved in zip((batch.point_labels, batch.edge_labels, batch.edge_index), before):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    for sample, (nodes, edges) in zip(samples, reference):
        torch.testing.assert_close(sample.point_labels, nodes, rtol=0, atol=0)
        torch.testing.assert_close(sample.edge_labels, edges, rtol=0, atol=0)


@pytest.mark.parametrize("spin", [1, 2])
def test_matrix_writer_h_and_overlap_hsx_preserves_generalized_levels(tmp_path, spin):
    from torch_geometric.data import Batch
    from graph2mat import AtomicTableWithEdges

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[20.] * 3)
    h = [np.array([[1.375, -.125], [-.125, 1.375]]) + .5 * c * np.array([[1., .25], [.25, 1.]])
         for c in range(spin)]
    s = np.array([[1., .25], [.25, 1.]])
    original = sisl.Hamiltonian.fromsp(geometry, [csr_array(channel) for channel in h],
        S=csr_array(s), spin="polarized" if spin == 2 else "unpolarized")
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix="hamiltonian",
        symmetric_matrix=True, sub_point_matrix=False,
        n_matrix_components=spin + 1, matrix_component_policy="h_and_overlap",
    )
    sample = TorchBasisMatrixData.new(original, processor)
    batch = Batch.from_data_list([sample])
    nodes, edges = batch.point_labels.clone(), batch.edge_labels.clone()
    trainer = SimpleNamespace(datamodule=SimpleNamespace(data_processor=processor, out_matrix="hamiltonian"))
    path = tmp_path / "prediction.HSX"
    MatrixWriter(str(path))._on_batch_end("predict", trainer, None,
        {"node_labels": nodes, "edge_labels": edges}, batch, 0, 0)
    matrix = sisl.get_sile(path).read_hamiltonian(geometry=geometry)
    assert not matrix.orthogonal
    assert matrix.spin.is_polarized == (spin == 2)
    np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), s, rtol=1e-10, atol=1e-10)
    for channel in range(spin):
        np.testing.assert_allclose(matrix.tocsr(channel).toarray(), h[channel], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(matrix.eigenvalue(spin=channel).eig,
            [1. + .5 * channel, 2. + .5 * channel], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(batch.point_labels, nodes, rtol=0, atol=0)
    torch.testing.assert_close(batch.edge_labels, edges, rtol=0, atol=0)
    np.testing.assert_array_equal(original.tocsr(original.S_idx).toarray(), s)
    for channel in range(spin):
        np.testing.assert_array_equal(original.tocsr(channel).toarray(), h[channel])


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
    # One Hamiltonian channel; HSX always stores S, which adds a column when read back.
    assert written._csr.data.shape[1] - (not written.orthogonal) == 1

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
