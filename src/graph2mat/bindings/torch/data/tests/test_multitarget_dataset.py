"""Joint references survive datasets, caches and heterogeneous PyG batches."""

from copy import deepcopy

import numpy as np
import pytest
import sisl
import torch
from scipy.sparse import csr_array
from torch_geometric.data import Batch

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
from graph2mat.bindings.torch import InMemoryData, TorchBasisMatrixData, TorchBasisMatrixDataset
from graph2mat.core.data import multitarget
from graph2mat.core.data.node_feats import OneHotZ
from graph2mat.core.data.tests.test_multitarget import TARGETS
from graph2mat.tools.lightning.data import MatrixDataModule, TypeLayoutCollate


def bundle(natoms=2, spin=1):
    atoms = [sisl.Atom(name, orbitals=[sisl.AtomicOrbital("1s", R=2.8)]) for name in ("H", "He")]
    geometry = sisl.Geometry([[i, 0., 0.] for i in range(natoms)],
                             atoms=[atoms[i % 2] for i in range(natoms)], lattice=[20.] * 3)
    s = csr_array(np.eye(natoms) + .1 * (np.ones((natoms, natoms)) - np.eye(natoms)))
    h, d, e = 2 * s, .5 * s, .25 * s
    def channels(a):
        return a if spin == 1 else [a, a * .7]
    return dict(overlap=sisl.Overlap.fromsp(geometry, s),
                hamiltonian=sisl.Hamiltonian.fromsp(geometry, channels(h), S=s),
                density=sisl.DensityMatrix.fromsp(geometry, channels(d), S=s),
                energy_density=sisl.EnergyDensityMatrix.fromsp(geometry, channels(e), S=s))


def processor(spin=1, targets=TARGETS, symmetric=True):
    table = AtomicTableWithEdges(bundle()["overlap"].atoms.atom)
    return MatrixDataProcessor(basis_table=table, symmetric_matrix=symmetric,
        out_matrix="hamiltonian", sub_point_matrix=False, matrix_targets=list(targets),
        matrix_target_spin=spin, node_attr_getters=[OneHotZ])


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_multitarget_batch_heterogeneous_ownership(spin, symmetric):
    p = processor(spin, symmetric=symmetric)
    dataset = TorchBasisMatrixDataset([bundle(2, spin), bundle(3, spin)], p)
    samples = [dataset[i] for i in range(2)]
    batch = TypeLayoutCollate(Batch.from_data_list, p.basis_table, symmetric=symmetric)(samples)
    assert batch.matrix_target_signature.shape == (2, 6)
    for index, sample in enumerate(samples):
        retrieved = batch.get_example(index)
        for name in ("point_labels", "edge_labels", "matrix_known_nodes", "matrix_known_edges",
                     "matrix_target_signature", "positions", "neigh_isc", "point_types"):
            torch.testing.assert_close(retrieved[name], sample[name], rtol=0, atol=0)
    assert batch.ptr.tolist() == [0, 2, 5]


def test_collate_rejects_semantic_mismatch():
    a = processor(targets=TARGETS[:3])
    b = processor(targets=("hamiltonian", "overlap", "energy_density_matrix"))
    samples = [TorchBasisMatrixDataset([bundle()], p)[0] for p in (a, b)]
    with pytest.raises(ValueError, match="layout|descriptor|semantic"):
        TypeLayoutCollate(Batch.from_data_list, a.basis_table)(samples)


def test_cache_does_not_alias_labels():
    p = processor()
    dataset = TorchBasisMatrixDataset([bundle()], p)
    original = dataset[0]
    cache = InMemoryData(dataset)
    changed = cache[0]
    changed.point_labels.add_(50)
    changed.matrix_known_nodes.fill_(False)
    torch.testing.assert_close(cache[0].point_labels, original.point_labels)
    assert cache[0].matrix_known_nodes.all()
    copied = original.copy(cls=BasisMatrixData)
    np.testing.assert_array_equal(copied.matrix_target_signature, [[1, 1, 1, 1, 1, 1]])


def test_geometry_only_never_loads_references(monkeypatch):
    p = processor()
    def forbidden(*args, **kwargs):
        raise AssertionError("geometry inference loaded references")
    monkeypatch.setattr(multitarget, "load_multitarget_data", forbidden)
    data = TorchBasisMatrixDataset([bundle()["overlap"].geometry], p, load_labels=False)[0]
    assert getattr(data, "point_labels", None) is None
    assert data.matrix_target_signature.tolist() == [[1, 1, 1, 1, 1, 1]]
    assert data["node_attrs"].shape == (2, 2)


def test_label_changes_leave_features_exactly_equal():
    p = processor()
    first = bundle()
    second = deepcopy(first)
    for key, factor in (("density", .4), ("hamiltonian", 1.2)):
        original = second[key]
        second[key] = type(original).fromsp(original.geometry, original.tocsr(0) * factor,
                                             S=second["overlap"].tocsr(0))
    a, b = [TorchBasisMatrixDataset([item], p)[0] for item in (first, second)]
    for key in ("positions", "edge_index", "shifts", "cell", "node_attrs", "point_types"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert not torch.equal(a.point_labels, b.point_labels)


def test_datamodule_prepared_samples_and_direct_configuration():
    p = processor(2)
    samples = [multitarget.prepare_multitarget_data(p, **bundle(n, 2), data_cls=TorchBasisMatrixData)
               for n in (2, 3)]
    dm = MatrixDataModule(basis_table=p.basis_table, out_matrix="hamiltonian",
        sub_point_matrix=False, matrix_targets=list(TARGETS), matrix_target_spin=2,
        train_runs=samples, val_fraction=0, loader_threads=0)
    dm.setup("fit")
    assert dm.n_matrix_components == 7
    batch = next(iter(dm.train_dataloader()))
    assert batch.point_labels.shape[1] == 7


def test_physical_selection_by_target_survives_dataset_and_cache():
    from graph2mat.core.data.tests.physical_fixtures import analytic_matrices
    from graph2mat.core.data.tests.test_multitarget import processor_for

    matrices = analytic_matrices()
    p = processor_for(matrices)
    requested = {"density_matrix": ["charge", "energy"]}
    dataset = TorchBasisMatrixDataset([matrices], p, physical_labels=requested)
    cached = InMemoryData(dataset)[0]
    view = multitarget.target_data_view(cached, p.matrix_target_layout, "density_matrix")
    torch.testing.assert_close(view.physical_population_reference, torch.ones(2))
    unprepared = multitarget.prepare_multitarget_data(p, **matrices, data_cls=TorchBasisMatrixData)
    with pytest.raises(ValueError, match="physical|capabilit|reference"):
        TorchBasisMatrixDataset([unprepared], p, physical_labels=requested)[0]
