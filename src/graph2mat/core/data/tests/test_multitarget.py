"""Semantic layouts and compatible references for joint matrix prediction."""

import importlib

import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices


TARGETS = ("hamiltonian", "overlap", "density_matrix", "energy_density_matrix")


def processor_for(matrices, spin=1, symmetric=True, targets=TARGETS):
    return MatrixDataProcessor(basis_table=AtomicTableWithEdges(matrices["overlap"].atoms.atom),
        out_matrix="hamiltonian", symmetric_matrix=symmetric, sub_point_matrix=False,
        matrix_targets=list(targets), matrix_target_spin=spin)


@pytest.mark.parametrize("spin,stops", [(1, [(0, 1), (1, 2), (2, 3), (3, 4)]),
                                      (2, [(0, 2), (2, 3), (3, 5), (5, 7)])])
def test_layout_canonical_spin_slices(spin, stops):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    targets = ("hamiltonian", "overlap", "density_matrix", "energy_density_matrix")
    layout = module.MatrixTargetLayout(tuple(reversed(targets)), spin)
    assert layout.targets == targets
    assert [(s.start, s.stop) for s in layout.slices.values()] == stops
    assert layout.n_components == (4 if spin == 1 else 7)
    assert module.MatrixTargetLayout.from_dict(layout.to_dict()) == layout
    np.testing.assert_array_equal(layout.signature, [[1, spin, 1, 1, 1, 1]])


@pytest.mark.parametrize("targets,spin", [
    (("overlap",), 1), (("density_matrix",), 1),
    (("hamiltonian", "overlap", "overlap"), 1),
    (("hamiltonian", "overlap", "unknown"), 1),
    (("hamiltonian", "overlap"), 4), (("hamiltonian", "overlap"), True),
    (("hamiltonian", "overlap"), 1.0),
])
def test_layout_rejects_invalid_semantics(targets, spin):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    with pytest.raises(ValueError):
        module.MatrixTargetLayout(targets, spin)


@pytest.mark.parametrize("extra", [(), ("density_matrix",), ("energy_density_matrix",)])
def test_selected_layout_roundtrip_and_tampering(extra):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    layout = module.MatrixTargetLayout(("hamiltonian", "overlap") + extra, 2)
    assert module.MatrixTargetLayout.from_dict(layout.to_dict()) == layout
    for key, bad in [("version", 2), ("slices", {}), ("units", {})]:
        descriptor = layout.to_dict()
        descriptor[key] = bad
        with pytest.raises(ValueError, match="descriptor|layout|version"):
            module.MatrixTargetLayout.from_dict(descriptor)


def test_numpy_integer_spin_descriptor_is_json_serializable():
    import json

    module = importlib.import_module("graph2mat.core.data.multitarget")
    layout = module.MatrixTargetLayout(TARGETS, np.int64(2))
    assert module.MatrixTargetLayout.from_dict(json.loads(json.dumps(layout.to_dict()))) == layout


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_preparation_matches_independent_channels(spin, symmetric):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices(spin=spin)
    processor = processor_for(matrices, spin, symmetric)
    data = module.prepare_multitarget_data(processor, **matrices)
    points = [.5, 1., 5/6, 5/36] if spin == 1 else [.5, .5, 1., 5/12, 5/12, 5/72, 5/72]
    edge = [-.3, .2, 5/6, 5/36] if spin == 1 else [-.3, -.3, .2, 5/12, 5/12, 5/72, 5/72]
    np.testing.assert_allclose(data.point_labels, [points, points], rtol=0, atol=1e-14)
    np.testing.assert_allclose(data.edge_labels, [edge] * (1 if symmetric else 2), rtol=0, atol=1e-14)
    assert data.matrix_known_nodes.all() and data.matrix_known_edges.all()
    assert processor.n_matrix_components == (4 if spin == 1 else 7)
    view = module.target_data_view(data, processor.matrix_target_layout, "density_matrix")
    np.testing.assert_allclose(np.asarray(view.point_labels).reshape(2, spin),
                               [[5/6] if spin == 1 else [5/12, 5/12]] * 2)
    assert data.point_labels.shape[1] == processor.n_matrix_components


def test_periodic_preparation_keeps_opposite_images():
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = periodic_matrices()
    data = module.prepare_multitarget_data(processor_for(matrices), **matrices)
    np.testing.assert_allclose(data.point_labels, [[.5, 1., .8, .4]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(data.edge_labels, [[.1, .2, .1, .05]], rtol=0, atol=1e-14)


def test_component_support_is_not_row_support():
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices(spin=2)
    masks = {"density_matrix": [csr_array(np.eye(2, dtype=bool)), csr_array(np.ones((2, 2), dtype=bool))]}
    data = module.prepare_multitarget_data(processor_for(matrices, 2), **matrices,
                                         observation_masks=masks)
    assert np.isnan(data.edge_labels[0, 3])
    assert data.edge_labels[0, 4] == pytest.approx(5/12)
    np.testing.assert_array_equal(data.matrix_known_edges[0], [True, True, True, False, True, True, True])
    assert np.isfinite(matrices["density"]._csr._D).all()
    bad = dict(masks, density_matrix=[csr_array([[True, True], [False, True]]), masks["density_matrix"][1]])
    with pytest.raises(ValueError, match="mask|Hermitian"):
        module.prepare_multitarget_data(processor_for(matrices, 2), **matrices, observation_masks=bad)


@pytest.mark.parametrize("target", ["density", "energy_density"])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_archived_zero_support_must_be_hermitian(target, symmetric, periodic):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = periodic_matrices() if periodic else analytic_matrices()
    geometry = matrices["overlap"].geometry
    cls = type(matrices[target])
    if periodic:
        for image in (-1, 1):
            other = geometry.sc_index([image, 0, 0])
            values = csr_array(([.8, 0.], ([0, 0], [0, other])), shape=(1, 3))
            matrices[target] = cls.fromsp(geometry, values)
            with pytest.raises(ValueError, match="support.*Hermitian"):
                module.prepare_multitarget_data(processor_for(matrices, symmetric=symmetric), **matrices)
    else:
        values = csr_array(([.8, 0., .8], [0, 1, 1], [0, 2, 3]), shape=(2, 2))
        for values in (values, values.T.tocsr()):
            matrices[target] = cls.fromsp(geometry, values)
            with pytest.raises(ValueError, match="support.*Hermitian"):
                module.prepare_multitarget_data(processor_for(matrices, symmetric=symmetric), **matrices)


@pytest.mark.parametrize("bad", ["missing", "nan", "inf", "geometry", "basis", "spin", "overlap"])
def test_incompatible_bundle_rejected(bad):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    processor = processor_for(matrices)
    matrices["density"] = matrices["density"].copy()
    if bad == "missing":
        matrices["density"] = None
    elif bad in ("nan", "inf"):
        # sisl's public scalar setter normalizes NaN to zero; emulate corrupt archived bytes.
        matrices["density"]._csr._D[0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "geometry":
        matrices["density"].geometry.xyz[0, 0] += .01
    elif bad == "basis":
        matrices["density"].geometry.atoms[0] = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.81)])
    elif bad == "spin":
        matrices["density"] = analytic_matrices(spin=2)["density"]
    else:
        matrices["density"][0, 0, matrices["density"].S_idx] = 1.1
    with pytest.raises(ValueError, match="density|reference|basis|geometry|spin|overlap"):
        module.prepare_multitarget_data(processor, **matrices)


def test_hs_does_not_require_density_or_mesh():
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    data = module.prepare_multitarget_data(processor_for(matrices, targets=TARGETS[:2]),
        hamiltonian=matrices["hamiltonian"], overlap=matrices["overlap"])
    assert data.point_labels.shape == (2, 2)


def test_multitarget_rejects_baseline_and_inconsistent_component_count():
    matrices = analytic_matrices()
    kwargs = dict(basis_table=AtomicTableWithEdges(matrices["overlap"].atoms.atom),
                  matrix_targets=list(TARGETS), matrix_target_spin=1)
    with pytest.raises(ValueError, match="baseline|sub_point_matrix"):
        MatrixDataProcessor(**kwargs)
    with pytest.raises(ValueError, match="components"):
        MatrixDataProcessor(**kwargs, sub_point_matrix=False, n_matrix_components=3)


def write_fdf_bundle(directory, matrices, *, tsde=True, ef=0.):
    path = directory / "RUN.fdf"
    matrices["overlap"].geometry.write(path)
    path.write_text(path.read_text() + "\nSystemLabel analytic\n")
    matrices["hamiltonian"].copy().write(directory / "analytic.TSHS")
    if tsde:
        with sisl.get_sile(directory / "analytic.TSDE", "w") as writer:
            writer.write_density_matrices(matrices["density"].copy(),
                                          matrices["energy_density"].copy(), Ef=ef)
    return path


@pytest.mark.parametrize("spin", [1, 2])
def test_real_fdf_tshs_tsde_roundtrip(tmp_path, spin):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices(spin=spin)
    path = write_fdf_bundle(tmp_path, matrices)
    data = module.load_multitarget_data(path, processor_for(matrices, spin))
    points = [.5, 1., 5/6, 5/36] if spin == 1 else [.5, .5, 1., 5/12, 5/12, 5/72, 5/72]
    edge = [-.3, .2, 5/6, 5/36] if spin == 1 else [-.3, -.3, .2, 5/12, 5/12, 5/72, 5/72]
    np.testing.assert_allclose(data.point_labels, [points, points], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(data.edge_labels, [edge], rtol=1e-10, atol=1e-10)
    assert data.metadata["physical_reference"]["density"].endswith("analytic.TSDE")


def test_fdf_hs_needs_no_tsde_or_grid(tmp_path):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    path = write_fdf_bundle(tmp_path, matrices, tsde=False)
    processor = processor_for(matrices, targets=TARGETS[:2])
    data = module.load_multitarget_data(path, processor)
    np.testing.assert_allclose(data.point_labels, [[.5, 1.], [.5, 1.]], rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match="density|TSDE|absent"):
        module.load_multitarget_data(path, processor_for(matrices))


def test_fdf_shifted_edm_partner_rejected(tmp_path):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    path = write_fdf_bundle(tmp_path, matrices, ef=.5)
    with pytest.raises(ValueError, match="gauge|Fermi"):
        module.load_multitarget_data(path, processor_for(matrices))


def test_stored_zero_and_absent_density_are_distinct():
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    geometry = matrices["overlap"].geometry
    explicit = csr_array(([.8, 0., 0., .8], [0, 1, 0, 1], [0, 2, 4]), shape=(2, 2))
    matrices["density"] = sisl.DensityMatrix.fromsp(geometry, explicit)
    observed = module.prepare_multitarget_data(processor_for(matrices), **matrices)
    assert observed.matrix_known_edges[0, 2] and observed.edge_labels[0, 2] == 0.
    matrices["density"] = sisl.DensityMatrix.fromsp(geometry, csr_array(np.diag([.8, .8])))
    unknown = module.prepare_multitarget_data(processor_for(matrices), **matrices)
    assert not unknown.matrix_known_edges[0, 2] and np.isnan(unknown.edge_labels[0, 2])
    with pytest.raises(ValueError, match="absent|mask"):
        module.prepare_multitarget_data(processor_for(matrices), **matrices,
            observation_masks={"density_matrix": [csr_array(np.ones((2, 2), dtype=bool))]})


@pytest.mark.parametrize("masks", [[csr_array(np.eye(3, dtype=bool))],
                                  [csr_array(np.ones((2, 2)))], []])
def test_invalid_component_masks_rejected(masks):
    module = importlib.import_module("graph2mat.core.data.multitarget")
    matrices = analytic_matrices()
    with pytest.raises(ValueError, match="mask"):
        module.prepare_multitarget_data(processor_for(matrices), **matrices,
                                        observation_masks={"density_matrix": masks})


def test_zip_input_rejected_before_reading(tmp_path):
    import zipfile

    module = importlib.import_module("graph2mat.core.data.multitarget")
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as writer:
        writer.writestr("RUN.fdf", "SystemLabel absent\n")
    with zipfile.ZipFile(archive_path) as reader:
        with pytest.raises(ValueError, match="local FDF"):
            module.load_multitarget_data(zipfile.Path(reader, "RUN.fdf"),
                                         processor_for(analytic_matrices()))
