"""Nominal per-atom magnetic moments as node data (roadmap 8.1, magnetism).

The moments come from ``<stem>.magnetic_moments.json`` next to the run or structure file (for SIESTA
``.fdf`` runs also ``magnetic_moments.json``), or from the configuration in the Python API. The processor
validates them and they travel as the node array ``magnetic_moments`` of the data objects and their batches.
"""

import json
import re
import zipfile

import numpy as np
import pytest
import pytorch_lightning as pl
import sisl
import torch
from torch_geometric.loader import DataLoader

from graph2mat import (
    AtomicTableWithEdges,
    BasisConfiguration,
    BasisMatrixData,
    BasisTableWithEdges,
    MatrixDataProcessor,
    OrbitalConfiguration,
    PointBasis,
)
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data import configuration as configuration_module

FILE = "magnetic_moments.json"
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [0.3, 1.2, 0.9]])
VECTOR = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 0.5], [2.5, 0.0, -0.25]])
COLLINEAR = np.array([2.2, -2.2, 0.0])


def _atoms():
    return [sisl.Atom(1, [sisl.AtomicOrbital("1s", R=2.0)])]


def _geometry(positions=POSITIONS):
    return sisl.Geometry(positions, atoms=_atoms()[0], lattice=[10.0] * 3)


def _processor(magnetic_moments=None, policy="h_only", **kwargs):
    kwargs.setdefault("out_matrix", "hamiltonian")
    return MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(_atoms()),
        sub_point_matrix=False,
        matrix_component_policy=policy,
        magnetic_moments=magnetic_moments,
        **kwargs,
    )


def _data(moments, processor, cls=BasisMatrixData, positions=POSITIONS):
    return cls.new(_geometry(positions), processor, labels=False, magnetic_moments=moments)


def _write_run(folder, moments=None, extra=None, labels=False, file=FILE):
    """A SIESTA run folder: ``sample.fdf`` (+ ``sample.DM``) and optionally the moments file ``file``."""
    folder.mkdir(parents=True, exist_ok=True)
    geometry = _geometry()
    run = folder / "sample.fdf"
    geometry.write(run)
    run.write_text("SystemLabel sample\n" + run.read_text())
    if labels:
        matrix = sisl.DensityMatrix(geometry)
        for i in range(geometry.na):
            matrix[i, i] = 1.0 + i
        matrix.write(folder / "sample.DM")
    if moments is not None:
        (folder / file).write_text(json.dumps({"nominal": moments, **(extra or {})}))
    return run


def _write_structure(folder, name, moments=None):
    """A bare structure file (any format sisl reads) and optionally its ``<stem>.magnetic_moments.json``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    _geometry().write(path)
    if moments is not None:
        (folder / f"{path.stem}.{FILE}").write_text(json.dumps({"nominal": moments}))
    return path


def _from_run(run, labels=False):
    return OrbitalConfiguration.from_run(
        run, basis=_geometry().atoms, out_matrix="density_matrix" if labels else None
    )


# ---------------------------------------------------------------------------
# Reading magnetic_moments.json next to the run file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("nominal", [VECTOR, COLLINEAR])
def test_from_run_reads_nominal_moments_next_to_fdf(tmp_path, nominal, labels):
    run = _write_run(
        tmp_path, nominal.tolist(), extra={"converged": (nominal + 1).tolist()}, labels=labels
    )
    config = _from_run(run, labels)
    np.testing.assert_array_equal(config.magnetic_moments, nominal)
    assert (config.matrix is not None) == labels


@pytest.mark.parametrize("labels", [False, True])
def test_from_run_without_json_has_no_moments(tmp_path, labels):
    assert _from_run(_write_run(tmp_path, labels=labels), labels).magnetic_moments is None


@pytest.mark.parametrize("file", [FILE, f"sample.{FILE}"])
@pytest.mark.parametrize("legacy_python", [False, True])
@pytest.mark.parametrize("labels", [False, True])
def test_from_run_reads_moments_inside_zip(tmp_path, monkeypatch, legacy_python, labels, file):
    folder = tmp_path / "source"
    _write_run(folder, VECTOR.tolist(), labels=labels, file=file)
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in folder.iterdir():
            archive.write(path, "nested/" + path.name)
    if legacy_python:
        # Python 3.9 has no native ZIP support in sisl: the run is extracted first.
        native_get_sile = sisl.get_sile

        def without_native_zip(path, *args, **options):
            if isinstance(path, zipfile.Path):
                raise RuntimeError("sisl ZIP requires Python 3.10")
            return native_get_sile(path, *args, **options)

        monkeypatch.setattr(sisl, "get_sile", without_native_zip)
        monkeypatch.setattr(configuration_module, "version_info", (3, 9), raising=False)
    with zipfile.ZipFile(archive_path) as archive:
        config = _from_run(zipfile.Path(archive, "nested/sample.fdf"), labels)
    np.testing.assert_array_equal(config.magnetic_moments, VECTOR)


def test_zip_without_json_has_no_moments(tmp_path):
    run = _write_run(tmp_path / "source")
    with zipfile.ZipFile(tmp_path / "dataset.zip", "w") as archive:
        archive.write(run, "nested/sample.fdf")
    with zipfile.ZipFile(tmp_path / "dataset.zip") as archive:
        config = _from_run(zipfile.Path(archive, "nested/sample.fdf"))
    assert config.magnetic_moments is None


def test_processor_with_moments_needs_the_json_file(tmp_path):
    run = _write_run(tmp_path)
    with pytest.raises(ValueError, match=re.escape(f"{run} has none") + rf".*Provide sample\.{FILE}"):
        BasisMatrixData.new(run, _processor("vector"), labels=False)
    # The same run is fine for a processor that does not use moments (data unchanged).
    assert BasisMatrixData.new(run, _processor(None), labels=False)["magnetic_moments"] is None


@pytest.mark.parametrize("content", ['{"converged": [[0, 0, 1]]}', "[1, 2, 3]", "not json"])
def test_json_without_nominal_or_malformed_names_the_file(tmp_path, content):
    run = _write_run(tmp_path)
    (tmp_path / FILE).write_text(content)
    with pytest.raises(ValueError, match=FILE):
        _from_run(run)


def test_fdf_prefers_its_own_file_over_the_directory_one(tmp_path):
    run = _write_run(tmp_path, VECTOR.tolist())
    (tmp_path / f"sample.{FILE}").write_text(json.dumps({"nominal": (-VECTOR).tolist()}))
    np.testing.assert_array_equal(_from_run(run).magnetic_moments, -VECTOR)


@pytest.mark.parametrize("suffix", ["xyz", "fdf"])
def test_structures_in_one_directory_get_their_own_moments(tmp_path, suffix):
    # afm and fm share the directory (and, for fdf, a directory-level file that must not win).
    afm = _write_structure(tmp_path, f"afm.{suffix}", [[0, 0, 2.2], [0, 0, -2.2], [0, 0, 2.2]])
    fm = _write_structure(tmp_path, f"fm.{suffix}", [[0, 0, 2.2]] * 3)
    (tmp_path / FILE).write_text(json.dumps({"nominal": [[1.0, 1.0, 1.0]] * 3}))
    afm_moments, fm_moments = (_from_run(path).magnetic_moments for path in (afm, fm))
    np.testing.assert_array_equal(afm_moments, [[0, 0, 2.2], [0, 0, -2.2], [0, 0, 2.2]])
    np.testing.assert_array_equal(fm_moments, [[0, 0, 2.2]] * 3)
    processor = _processor("vector")
    first, second = (BasisMatrixData.new(path, processor, labels=False) for path in (afm, fm))
    assert not np.array_equal(first["magnetic_moments"], second["magnetic_moments"])


def test_directory_level_file_is_only_for_siesta_runs(tmp_path):
    # One structure, only the directory-level file: other structure files are not "one run per directory".
    structure = _write_structure(tmp_path, "afm.xyz")
    (tmp_path / FILE).write_text(json.dumps({"nominal": VECTOR.tolist()}))
    assert _from_run(structure).magnetic_moments is None
    with pytest.raises(ValueError, match=rf"afm\.{FILE}"):
        BasisMatrixData.new(structure, _processor("vector"), labels=False)
    # A processor without moments still loads it, and the same file works for an fdf run.
    assert BasisMatrixData.new(structure, _processor(None), labels=False)["magnetic_moments"] is None
    fdf = _write_run(tmp_path)
    np.testing.assert_array_equal(_from_run(fdf).magnetic_moments, VECTOR)


def test_dataset_item_path_loads_moments(tmp_path):
    from graph2mat.bindings.torch import TorchBasisMatrixDataset

    run = _write_run(tmp_path, VECTOR.tolist())
    dataset = TorchBasisMatrixDataset(
        [run], _processor("vector"), data_cls=TorchBasisMatrixData, load_labels=False
    )
    cob = dataset.data_processor.basis_table.change_of_basis
    expected = (VECTOR @ cob.T).astype(np.float32)
    np.testing.assert_allclose(dataset[0].magnetic_moments.numpy(), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Shapes and finiteness
# ---------------------------------------------------------------------------


def test_configuration_stores_moments_as_float_array():
    config = OrbitalConfiguration.from_geometry(_geometry(), magnetic_moments=VECTOR.tolist())
    assert isinstance(config.magnetic_moments, np.ndarray)
    assert config.magnetic_moments.dtype == float
    base = BasisConfiguration(
        point_types=["A"], positions=np.zeros((1, 3)),
        basis=BasisTableWithEdges([PointBasis("A", R=2, basis=[1])]),
        magnetic_moments=[[0, 0, 1]],
    )
    np.testing.assert_array_equal(base.magnetic_moments, [[0.0, 0.0, 1.0]])


def test_collinear_and_vector_shapes_are_stored():
    collinear = _data(COLLINEAR, _processor("collinear", "spin_h_only", n_matrix_components=2))
    assert collinear["magnetic_moments"].shape == (3,)
    vector = _data(VECTOR, _processor("vector"))
    assert vector["magnetic_moments"].shape == (3, 3)


def _bad_cases():
    nan = VECTOR.copy()
    nan[1, 2] = np.nan
    inf = COLLINEAR.copy()
    inf[0] = np.inf
    return [
        ("vector", VECTOR[:, :2], "shape"),
        ("vector", COLLINEAR, "shape"),
        ("vector", VECTOR[:2], "shape"),
        ("vector", np.zeros((3, 3, 1)), "shape"),
        ("vector", nan, "finite"),
        ("collinear", VECTOR, "shape"),
        ("collinear", COLLINEAR[:2], "shape"),
        ("collinear", inf, "finite"),
    ]


@pytest.mark.parametrize("kind, moments, match", _bad_cases())
def test_wrong_moments_are_rejected(kind, moments, match):
    policy = "spin_h_only" if kind == "collinear" else "h_only"
    with pytest.raises(ValueError, match=match):
        _data(moments, _processor(kind, policy, n_matrix_components=2 if kind == "collinear" else 1))


@pytest.mark.parametrize("moments, match", [
    ([[0.0, 0.0, 1.0]] * 2, "shape"),
    ([[0.0, 0.0, float("nan")]] * 3, "finite"),
])
def test_wrong_moments_of_a_run_name_the_sample(tmp_path, moments, match):
    run = _write_run(tmp_path, moments)
    with pytest.raises(ValueError, match=match) as error:
        BasisMatrixData.new(run, _processor("vector"), labels=False)
    assert str(run) in str(error.value)


def test_missing_moments_in_the_python_api_name_the_file():
    with pytest.raises(ValueError, match=FILE):
        _data(None, _processor("vector"))


def test_torch_predict_takes_the_moments_of_the_geometry():
    def model(data):
        seen.append(data.magnetic_moments.clone())
        return {"node_labels": torch.zeros(3), "edge_labels": torch.zeros(data.edge_index.shape[1] // 2)}

    seen = []
    processor = _processor("vector", symmetric_matrix=True)
    processor.torch_predict(model, _geometry(), magnetic_moments=VECTOR)
    expected = processor.cartesian_to_basis(VECTOR)
    np.testing.assert_allclose(seen[0].numpy(), expected, rtol=1e-6)
    # A geometry has no moments file: asking for them is an error, and so is giving them for nothing.
    with pytest.raises(ValueError, match="pass the nominal moment"):
        processor.torch_predict(model, _geometry())
    with pytest.raises(ValueError, match="must not be given"):
        _processor(None, symmetric_matrix=True).torch_predict(model, _geometry(), magnetic_moments=VECTOR)
    assert len(seen) == 1


@pytest.mark.parametrize("bad", ["scalar", "Vector", True, 1])
def test_unknown_kind_is_rejected(bad):
    with pytest.raises(ValueError, match="magnetic_moments"):
        _processor(bad)


@pytest.mark.parametrize("policy", ["h_only", "spin_orbit", "spin_orbit_full", "raw_components"])
def test_collinear_needs_spin_h_only(policy):
    with pytest.raises(ValueError, match="spin_h_only"):
        _processor("collinear", policy)
    _processor("collinear", "spin_h_only")


# ---------------------------------------------------------------------------
# Data content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [BasisMatrixData, TorchBasisMatrixData])
def test_vector_moments_are_converted_like_positions(cls):
    processor = _processor("vector")
    data = _data(VECTOR, processor, cls)
    cob = processor.basis_table.change_of_basis
    assert not np.allclose(cob, np.eye(3))  # the conversion really changes the numbers
    stored = data.numpy_arrays().magnetic_moments
    np.testing.assert_allclose(stored, VECTOR @ cob.T, rtol=1e-6)
    np.testing.assert_allclose(stored, processor.cartesian_to_basis(VECTOR), rtol=1e-6)
    # Same rule as the positions: if the moments were the positions, they would coincide.
    same = _data(POSITIONS, processor, cls)
    np.testing.assert_array_equal(
        same.numpy_arrays().magnetic_moments, same.numpy_arrays().positions
    )


@pytest.mark.parametrize("cls", [BasisMatrixData, TorchBasisMatrixData])
def test_collinear_moments_are_not_converted(cls):
    data = _data(COLLINEAR, _processor("collinear", "spin_h_only", n_matrix_components=2), cls)
    np.testing.assert_allclose(data.numpy_arrays().magnetic_moments, COLLINEAR, rtol=1e-6)


def test_torch_data_holds_a_tensor():
    data = _data(VECTOR, _processor("vector"), TorchBasisMatrixData)
    assert isinstance(data.magnetic_moments, torch.Tensor)
    assert data.magnetic_moments.dtype == torch.get_default_dtype()
    assert "magnetic_moments" in data.keys()


def test_same_geometry_different_moments_give_different_data():
    processor = _processor("vector")
    first = _data(VECTOR, processor, TorchBasisMatrixData)
    second = _data(-VECTOR, processor, TorchBasisMatrixData)
    flipped_one = VECTOR.copy()
    flipped_one[1] *= -1
    third = _data(flipped_one, processor, TorchBasisMatrixData)
    for other in (second, third):
        torch.testing.assert_close(first.positions, other.positions, rtol=0, atol=0)
        assert not torch.equal(first.magnetic_moments, other.magnetic_moments)
    torch.testing.assert_close(first.magnetic_moments, -second.magnetic_moments, rtol=0, atol=0)


def test_data_does_not_alias_the_configuration_moments():
    processor = _processor("collinear", "spin_h_only", n_matrix_components=2)
    moments = COLLINEAR.copy()
    config = OrbitalConfiguration.from_geometry(_geometry(), magnetic_moments=moments)
    data = BasisMatrixData.from_config(config, processor)
    config.magnetic_moments[0] = 99.0
    moments[1] = 99.0
    np.testing.assert_array_equal(data["magnetic_moments"], COLLINEAR)


def test_copy_and_subgraph_keep_moments():
    processor = _processor("vector")
    torch_data = _data(VECTOR, processor, TorchBasisMatrixData)
    back = torch_data.copy(cls=BasisMatrixData)
    np.testing.assert_allclose(
        back["magnetic_moments"], torch_data.magnetic_moments.numpy(), rtol=0, atol=0
    )
    again = back.copy(cls=TorchBasisMatrixData)
    assert torch.equal(again.magnetic_moments, torch_data.magnetic_moments)
    # Nodes are filtered together with the moments.
    two_types = BasisMatrixData.new(_geometry(), processor, labels=False, magnetic_moments=VECTOR)
    two_types["point_types"][:] = np.array([0, 1, 0])
    sub = two_types.node_types_subgraph(np.array([0]))
    np.testing.assert_array_equal(sub["magnetic_moments"], two_types["magnetic_moments"][[0, 2]])


def test_batch_concatenates_moments_in_node_order():
    processor = _processor("vector")
    other = np.array([[0.0, 0.0, 1.0], [4.0, 5.0, 6.0]])
    a = _data(VECTOR, processor, TorchBasisMatrixData)
    b = _data(other, processor, TorchBasisMatrixData, positions=POSITIONS[:2])
    batch = next(iter(DataLoader([a, b], batch_size=2)))
    assert batch.magnetic_moments.shape == (5, 3)
    assert torch.equal(batch.magnetic_moments[:3], a.magnetic_moments)
    assert torch.equal(batch.magnetic_moments[3:], b.magnetic_moments)
    assert batch.magnetic_moments.shape[0] == batch.positions.shape[0]

    scalar_processor = _processor("collinear", "spin_h_only", n_matrix_components=2)
    c = _data(COLLINEAR, scalar_processor, TorchBasisMatrixData)
    d = _data(COLLINEAR[:2] * -1, scalar_processor, TorchBasisMatrixData, positions=POSITIONS[:2])
    scalars = next(iter(DataLoader([c, d], batch_size=2)))
    assert torch.equal(scalars.magnetic_moments, torch.cat([c.magnetic_moments, d.magnetic_moments]))


# ---------------------------------------------------------------------------
# A processor without moments changes nothing (ruling R1)
# ---------------------------------------------------------------------------


def test_processor_without_moments_leaves_data_and_batches_untouched():
    config_with = OrbitalConfiguration.from_geometry(_geometry(), magnetic_moments=VECTOR)
    config_without = OrbitalConfiguration.from_geometry(_geometry())
    processor = _processor(None)
    assert processor.magnetic_moments is None
    with_moments = TorchBasisMatrixData.from_config(config_with, processor)
    without = TorchBasisMatrixData.from_config(config_without, processor)

    assert "magnetic_moments" not in with_moments.keys()
    assert sorted(with_moments.keys()) == sorted(without.keys())
    for key in without.keys():
        if torch.is_tensor(without[key]):
            assert with_moments[key].dtype == without[key].dtype
            assert torch.equal(with_moments[key], without[key]), key

    batch_a = next(iter(DataLoader([with_moments, with_moments], batch_size=2)))
    batch_b = next(iter(DataLoader([without, without], batch_size=2)))
    assert sorted(batch_a.keys()) == sorted(batch_b.keys())
    for key in batch_b.keys():
        if torch.is_tensor(batch_b[key]):
            assert torch.equal(batch_a[key], batch_b[key]), key
    # The numpy data object keeps the entry as None.
    assert BasisMatrixData.from_config(config_with, processor)["magnetic_moments"] is None


def test_default_processor_has_no_moments():
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges(_atoms()))
    assert processor.magnetic_moments is None
    assert processor.copy(magnetic_moments="vector").magnetic_moments == "vector"


# ---------------------------------------------------------------------------
# Datamodule and checkpoints
# ---------------------------------------------------------------------------


def _module(**kwargs):
    from graph2mat.tools.lightning import MatrixDataModule

    basis = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
    config = BasisConfiguration(
        point_types=["A", "A"],
        positions=np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        basis=basis,
        cell=np.eye(3) * 10,
        pbc=(False, False, False),
        matrix=np.ones((2, 2)),
        magnetic_moments=np.array([[0.0, 0, 1], [0, 0, -1]]),
    )
    module = MatrixDataModule(
        basis_table=basis,
        symmetric_matrix=True,
        sub_point_matrix=False,
        train_runs=[config],
        val_runs=[config],
        predict_structs=[config],
        loader_threads=0,
        **kwargs,
    )
    return basis, module


def test_datamodule_passes_moments_to_the_processor_and_hparams():
    _, module = _module(magnetic_moments="vector")
    assert module.hparams["magnetic_moments"] == "vector"
    module.setup("fit")
    assert module.data_processor.magnetic_moments == "vector"
    sample = module.train_dataset[0]
    assert sample.magnetic_moments.shape == (2, 3)
    module.teardown(None)


def test_datamodule_defaults_to_no_moments():
    _, module = _module()
    assert module.hparams["magnetic_moments"] is None
    module.setup("fit")
    assert module.data_processor.magnetic_moments is None
    assert "magnetic_moments" not in module.train_dataset[0].keys()
    module.teardown(None)


@pytest.mark.parametrize("moments", [None, "collinear"])
def test_datamodule_spin_orbit_full_requires_vector(moments):
    with pytest.raises(ValueError, match="vector"):
        _module(
            out_matrix="hamiltonian", matrix_component_policy="spin_orbit_full",
            n_matrix_components=8, magnetic_moments=moments,
        )
    _module(
        out_matrix="hamiltonian", matrix_component_policy="spin_orbit_full",
        n_matrix_components=8, magnetic_moments="vector",
    )


@pytest.mark.parametrize("policy", ["h_only", "spin_orbit"])
def test_datamodule_collinear_requires_spin_h_only(policy):
    with pytest.raises(ValueError, match="spin_h_only"):
        _module(out_matrix="hamiltonian", matrix_component_policy=policy, magnetic_moments="collinear")
    _module(out_matrix="hamiltonian", matrix_component_policy="spin_h_only",
            n_matrix_components=2, magnetic_moments="collinear")


def test_datamodule_rejects_unknown_kind():
    with pytest.raises(ValueError, match="magnetic_moments"):
        _module(magnetic_moments="scalar")


def _checkpoint(tmp_path, pop_moments):
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    # A plain model does not take data with moments (the model checks it): its checkpoint is edited to
    # carry the option. A real magnetic model's checkpoint is covered by lightning/tests/test_magnetic_model.py.
    basis, module = _module()
    model = LitMACEMatrixModel(
        basis_table=basis,
        symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o",
        edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2,
        max_ell=1,
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
    assert ckpt["datamodule_hyper_parameters"]["magnetic_moments"] is None
    if pop_moments:
        ckpt["datamodule_hyper_parameters"].pop("magnetic_moments")
    else:
        ckpt["datamodule_hyper_parameters"]["magnetic_moments"] = "vector"
    torch.save(ckpt, path)
    module.teardown(None)
    return path


@pytest.mark.parametrize("pop_moments, expected", [(False, "vector"), (True, None)])
def test_checkpoint_loader_restores_moments(tmp_path, pop_moments, expected):
    from graph2mat.bindings.torch.load import load_from_lit_ckpt

    _, processor = load_from_lit_ckpt(_checkpoint(tmp_path, pop_moments))
    assert processor.magnetic_moments == expected
