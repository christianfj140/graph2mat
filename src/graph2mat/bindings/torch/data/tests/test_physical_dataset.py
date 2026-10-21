"""Opt-in physical preparation, observation support and batch ownership."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import sisl
import torch
from scipy.sparse import csr_array
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.bindings.torch import InMemoryData, TorchBasisMatrixData, TorchBasisMatrixDataset
from graph2mat.core.data import physical
from graph2mat.core.data.basis import NoBasisAtom
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices, prepared_sample
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.bindings.torch.modules.graph2mat import layout_to
from graph2mat.tools.lightning.data import TypeLayoutCollate


def processor_for(matrices, target="density_matrix", symmetric=True):
    return MatrixDataProcessor(basis_table=AtomicTableWithEdges(matrices["density"].geometry.atoms.atom),
        out_matrix=target, symmetric_matrix=symmetric, sub_point_matrix=False)


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("target", ["density_matrix", "hamiltonian", "energy_density_matrix"])
def test_preparation_excludes_embedded_overlap(spin, target):
    data, _ = prepared_sample(spin=spin, target=target)
    assert data.physical_spin.tolist() == [spin]
    assert data.physical_reference_nodes.shape == (2, spin)
    assert data.point_labels.ndim == (1 if spin == 1 else 2)
    np.testing.assert_allclose(data.physical_population_reference, [1., 1.], atol=1e-14)


def test_density_baseline_is_restored_without_changing_reference():
    data, _ = prepared_sample(baseline=True)
    np.testing.assert_allclose(data.point_labels, [-1/6, -1/6], atol=1e-14)
    np.testing.assert_allclose(data.physical_reference_nodes[:, 0], [5/6, 5/6], atol=1e-14)
    np.testing.assert_array_equal(data.physical_baseline_nodes, [[1.], [1.]])


def test_collinear_baseline_is_not_broadcast_implicitly():
    with pytest.raises(ValueError, match="collinear|baseline"):
        prepared_sample(spin=2, baseline=True)


@pytest.mark.parametrize("unknown", ["absent", "nan"])
def test_unknown_partner_is_not_an_observed_zero(unknown):
    m = analytic_matrices()
    d = m["density"].tocsr(0).tolil()
    d[0, 1] = d[1, 0] = 0. if unknown == "absent" else np.nan
    m["density"] = sisl.DensityMatrix.fromsp(m["density"].geometry, csr_array(d))
    with pytest.raises(ValueError, match="unknown|observed"):
        physical.prepare_physical_data(processor_for(m, "hamiltonian"), **m, requested=("energy",))


def test_stored_zero_remains_observed():
    m = analytic_matrices()
    d = csr_array(([5/6, 0., 0., 5/6], [0, 1, 0, 1], [0, 2, 4]), shape=(2, 2))
    m["density"] = sisl.DensityMatrix.fromsp(m["density"].geometry, d)
    assert m["density"].nnz == 4
    data = physical.prepare_physical_data(processor_for(m), **m, requested=("charge",))
    np.testing.assert_allclose(data.physical_population_reference, [5/6, 5/6], atol=1e-14)
    assert data.physical_reference_known_edges.all()
    assert not hasattr(data, "physical_partner_edges")


def test_energy_h_rejects_unknown_partner_even_when_reference_h_is_zero():
    m = analytic_matrices()
    s = csr_array(np.eye(2))
    g = m["density"].geometry
    m.update(overlap=sisl.Overlap.fromsp(g, s),
        hamiltonian=sisl.Hamiltonian.fromsp(g, .5*s, S=s),
        density=sisl.DensityMatrix.fromsp(g, s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, .5*s))
    # Full HD=ES, but a predicted H edge can vary where D is unobserved.
    with pytest.raises(ValueError, match="unknown|observed"):
        physical.prepare_physical_data(processor_for(m, "hamiltonian"), **m, requested=("energy",))


def test_gauge_labels_keep_overlap_but_auxiliary_reference_does_not():
    data, _ = prepared_sample(target="hamiltonian", gauge=True)
    np.testing.assert_allclose(data.point_labels, [[.5, 1.], [.5, 1.]])
    np.testing.assert_allclose(data.edge_labels, [[-.3, .2]])
    assert data.physical_reference_nodes.shape == (2, 1)


def test_half_edge_auxiliaries_preserve_archived_opposite_partners():
    m = analytic_matrices()
    s = m["overlap"].tocsr(0).tolil()
    s[0, 1] += 1e-9
    s[1, 0] -= 1e-9
    s = csr_array(s)
    g = m["overlap"].geometry
    m["overlap"] = sisl.Overlap.fromsp(g, s)
    for key in ("density", "hamiltonian", "energy_density"):
        previous = m[key]
        m[key] = type(previous).fromsp(g, previous.tocsr(0), S=s)
    data = physical.prepare_physical_data(processor_for(m), **m, requested=("charge", "energy"))
    np.testing.assert_allclose(data.physical_overlap_edges, [[.2-1e-9]], rtol=0, atol=1e-16)
    np.testing.assert_allclose(data.physical_overlap_reverse_edges, [[.2+1e-9]], rtol=0, atol=1e-16)


def test_partial_empty_basis_retains_atoms_in_normalization():
    m = analytic_matrices()
    g = m["density"].geometry.add(sisl.Geometry([[8., 0., 0.]], atoms=NoBasisAtom("He", R=0.)))
    s = m["overlap"].tocsr(0)
    m.update(overlap=sisl.Overlap.fromsp(g, s),
        density=sisl.DensityMatrix.fromsp(g, m["density"].tocsr(0), S=s),
        hamiltonian=sisl.Hamiltonian.fromsp(g, m["hamiltonian"].tocsr(0), S=s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, m["energy_density"].tocsr(0), S=s))
    processor = processor_for(m)
    data = physical.prepare_physical_data(processor, **m, requested=("charge", "energy"))
    layout = physical.physical_layout(data, processor.basis_table)
    assert layout["physical_atoms_per_structure"].tolist() == [3]
    np.testing.assert_array_equal(data.physical_population_reference, [1., 1., 0.])


@pytest.mark.parametrize("block", [None, ["2 1 0 0", "0 2 0 0", "0 0 1 0"],
    ["2 0 0 .5", "0 2 0 0", "0 0 1 0"], ["0 0 0 0", "0 2 0 0", "0 0 1 0"]])
def test_fdf_requires_supported_explicit_grid(tmp_path, monkeypatch, block):
    path = tmp_path / "input.fdf"
    path.write_text("SystemLabel analytic\n")
    fake = SimpleNamespace(get=lambda name, default=None: block if "monkhorst" in name.lower() else default)
    monkeypatch.setattr(physical.sisl, "get_sile", lambda path: fake)
    with pytest.raises(ValueError, match="grid|mesh"):
        physical.load_physical_data(path, processor_for(analytic_matrices()), requested=("charge",))


@pytest.mark.parametrize("symmetric", [False, True])
def test_periodic_layout_preserves_both_self_image_endpoints(symmetric):
    m = periodic_matrices()
    processor = processor_for(m, symmetric=symmetric)
    data = physical.prepare_physical_data(processor, **m, requested=("charge",))
    layout = physical.physical_layout(data, processor.basis_table)
    np.testing.assert_array_equal(layout["physical_node_owner"], [0])
    np.testing.assert_array_equal(layout["physical_edge_source"], np.zeros(1 if symmetric else 2))
    np.testing.assert_array_equal(layout["physical_edge_target"], np.zeros(1 if symmetric else 2))
    assert layout["physical_symmetric"] is symmetric


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_batch_copies_and_cache_preserve_all_auxiliaries(spin, symmetric):
    data, table = prepared_sample(spin=spin, symmetric=symmetric, data_cls=TorchBasisMatrixData)
    samples = [data, deepcopy(data)]
    cache = InMemoryData(samples)
    changed = cache[0]
    changed.physical_population_reference.add_(9)
    torch.testing.assert_close(cache[0].physical_population_reference, data.physical_population_reference)
    batch = TypeLayoutCollate(Batch.from_data_list, table, symmetric=symmetric)(samples)
    for i, original in enumerate(samples):
        retrieved = batch.get_example(i)
        for key, value in original.items():
            if key.startswith("physical_"):
                torch.testing.assert_close(retrieved[key], value, rtol=0, atol=0)
    layout = getattr(batch, LAYOUT_ATTR)
    assert layout["physical_atoms_per_structure"].tolist() == [2, 2]
    assert layout["physical_node_owner"].tolist() == [0, 1, 2, 3]
    moved = layout_to(layout, torch.device("meta"))
    assert moved["physical_node_owner"].device.type == "meta"


def test_public_numpy_copy_preserves_physical_contract():
    data, _ = prepared_sample()
    copied = data.copy()
    for key, value in data._data.items():
        if key.startswith("physical_"):
            np.testing.assert_array_equal(copied[key], value)
    torch_data, _ = prepared_sample(data_cls=TorchBasisMatrixData)
    converted = torch_data.copy(cls=BasisMatrixData)
    np.testing.assert_array_equal(converted.physical_population_reference, [1., 1.])


@pytest.mark.parametrize("labels", [False, True])
def test_dataset_off_and_unlabelled_prediction_do_not_read_auxiliary(tmp_path, monkeypatch, labels):
    m = analytic_matrices()
    def forbidden(*args, **kwargs):
        raise AssertionError("auxiliary read while physical preparation is off")
    monkeypatch.setattr(physical, "load_physical_data", forbidden)
    ds = TorchBasisMatrixDataset([m["density"]], processor_for(m), load_labels=labels,
        physical_labels=None if labels else ("charge", "energy"))
    assert ds[0].num_nodes == 2


def test_dataset_rejects_physics_for_unprepared_non_fdf_input():
    m = analytic_matrices()
    ds = TorchBasisMatrixDataset([m["density"]], processor_for(m), physical_labels=("charge",))
    with pytest.raises(ValueError, match="FDF|prepared|physical"):
        ds[0]


@pytest.mark.parametrize("grid", [2, 20])
@pytest.mark.parametrize("requested", [("charge",), ("energy",)])
def test_fdf_loader_reads_declared_binary_references_without_optional_reads(tmp_path, monkeypatch, grid, requested):
    m = analytic_matrices()
    path = tmp_path / "RUN.fdf"
    path.write_text("SystemLabel analytic\n")
    m["hamiltonian"].write(tmp_path / "analytic.TSHS")
    with sisl.get_sile(tmp_path / "analytic.TSDE", "w") as writer:
        writer.write_density_matrices(m["density"], m["energy_density"])
    block = [f"{grid} 0 0 0", f"0 {grid if grid == 20 else 1} 0 0", "0 0 1 0"]
    def get(name, default=None):
        return block if "monkhorst" in name.lower() else "analytic" if name == "SystemLabel" else default
    def forbidden(*args, **kwargs):
        raise AssertionError("charge mode read H or EDM")
    fake = SimpleNamespace(get=get, dir_file=lambda name: tmp_path / name,
        read_overlap=lambda **kwargs: m["overlap"].copy(),
        read_hamiltonian=(lambda **kwargs: m["hamiltonian"].copy()) if "energy" in requested else forbidden)
    real_get_sile = sisl.get_sile
    def reader(p, *args, **kwargs):
        if str(p).endswith(".fdf"):
            return fake
        result = real_get_sile(p, *args, **kwargs)
        if "charge" in requested:
            result.read_energy_density_matrix = forbidden
        return result
    monkeypatch.setattr(physical.sisl, "get_sile", reader)
    data = physical.load_physical_data(path, processor_for(m), requested=requested)
    reference = data.metadata["physical_reference"]
    points = np.asarray(reference["kpoints"])
    weights = np.asarray(reference["kweights"])
    if grid == 2:
        np.testing.assert_allclose(points, [[0., 0., 0.], [.5, 0., 0.]], atol=1e-14)
        np.testing.assert_array_equal(weights, [.5, .5])
    else:
        axis = np.r_[np.arange(-9, 0)/20, np.arange(0, 11)/20]
        expected = np.array([[x, y, 0.] for x in axis for y in axis])
        np.testing.assert_allclose(points, expected, atol=1e-14)
        np.testing.assert_allclose(weights, np.full(400, 1/400), atol=1e-14)
    assert reference["density"].endswith("analytic.TSDE")
    assert data.physical_capabilities.tolist() == [1 if "charge" in requested else 2]


def test_opt_in_fdf_dataset_and_workers_preserve_references(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    path = tmp_path / "RUN.fdf"
    path.write_text("NetCharge 0.25\n")
    def load(*args, **kwargs):
        return deepcopy(data)
    monkeypatch.setattr(physical, "load_physical_data", load)
    ds = TorchBasisMatrixDataset([path, path], data.metadata["data_processor"], physical_labels=("charge", "energy"))
    for workers in (0, 2):
        loader = DataLoader(ds, batch_size=2, num_workers=workers,
            collate_fn=TypeLayoutCollate(Batch.from_data_list, table))
        batch = next(iter(loader))
        torch.testing.assert_close(batch.physical_population_reference, torch.ones(4, dtype=torch.float64))
        assert batch.total_charge.tolist() == [.25, .25]


def test_physical_zip_is_explicitly_unsupported(tmp_path):
    import zipfile
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("RUN.fdf", "SystemLabel absent\n")
    with zipfile.ZipFile(archive_path) as archive:
        ds = TorchBasisMatrixDataset([zipfile.Path(archive, "RUN.fdf")],
            processor_for(analytic_matrices()), physical_labels=("charge",))
        with pytest.raises(ValueError, match="ZIP|zip"):
            ds[0]


def test_direct_prepared_dataset_does_not_reload_or_share_mutable_references(monkeypatch):
    data, _ = prepared_sample(data_cls=TorchBasisMatrixData)
    def forbidden(*args, **kwargs):
        raise AssertionError("a prepared sample was sent to the FDF loader")
    monkeypatch.setattr(physical, "load_physical_data", forbidden)
    ds = TorchBasisMatrixDataset([data], data.metadata["data_processor"], physical_labels=("charge",))
    result = ds[0]
    torch.testing.assert_close(result.physical_population_reference, data.physical_population_reference)
    result.physical_population_reference.add_(8)
    torch.testing.assert_close(data.physical_population_reference, torch.ones(2, dtype=torch.float64))


def test_heterogeneous_batch_counts_atoms_instead_of_matrix_elements():
    first, table = prepared_sample(data_cls=TorchBasisMatrixData)
    m = periodic_matrices()
    for key in ("overlap", "density", "hamiltonian", "energy_density"):
        m[key].geometry.atoms[0] = analytic_matrices()["density"].geometry.atoms[0]
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False)
    second = physical.prepare_physical_data(processor, **m, requested=("charge", "energy"), data_cls=TorchBasisMatrixData)
    batch = TypeLayoutCollate(Batch.from_data_list, table)([first, second])
    layout = getattr(batch, LAYOUT_ATTR)
    assert layout["physical_atoms_per_structure"].tolist() == [2, 1]
    assert layout["physical_atom_structure"].tolist() == [0, 0, 1]
    assert layout["physical_edge_source"].tolist() == [0, 2]
    assert layout["physical_edge_target"].tolist() == [1, 2]
    np.testing.assert_allclose(batch.get_example(1).physical_population_reference, [.84], atol=1e-7)


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
def test_datamodule_opt_in_stage_propagation_and_unlabelled_predict(stage, tmp_path):
    from graph2mat.tools.lightning.data import MatrixDataModule
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    path = tmp_path / "predict.XV"
    analytic_matrices()["density"].geometry.write(path)
    module = MatrixDataModule(basis_table=table, out_matrix="density_matrix", sub_point_matrix=False,
        train_runs=[data, data], val_runs=[data], test_runs=[data], predict_structs=[path],
        physical_labels=("charge", "energy"), loader_threads=0)
    module.setup(stage)
    split = {"fit": "train", "validate": "val", "test": "test", "predict": "predict"}[stage]
    result = getattr(module, split + "_dataset")[0]
    if stage == "predict":
        assert getattr(result, "point_labels", None) is None
        assert getattr(result, "physical_version", None) is None
    else:
        assert result.physical_capabilities.tolist() == [3]
    module.teardown(stage)


def test_preparation_uses_known_scalar_overlap_type_without_fallback_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not a known matrix type.*")
        prepared_sample()


def test_net_charge_follows_fdf_label_rules_and_includes(tmp_path):
    """Audit M16: 'Net.Charge' or a NetCharge in an %include file gave a total charge of 0."""
    import zipfile

    from graph2mat.bindings.torch.data.dataset import _net_charge

    (tmp_path / "a.fdf").write_text("SystemLabel x\nNet.Charge -1.0\n")
    (tmp_path / "b.fdf").write_text("SystemLabel x\n%include b_inc.fdf\n")
    (tmp_path / "b_inc.fdf").write_text("netcharge 2.0\n")
    (tmp_path / "c.fdf").write_text("SystemLabel x\n")
    assert [_net_charge(tmp_path / f"{n}.fdf") for n in "abc"] == [-1.0, 2.0, 0.0]
    with zipfile.ZipFile(tmp_path / "s.zip", "w") as archive:
        archive.writestr("s.fdf", "Net_Charge 0.5\n")
    with zipfile.ZipFile(tmp_path / "s.zip") as reader:
        assert _net_charge(zipfile.Path(reader, "s.fdf")) == 0.5
