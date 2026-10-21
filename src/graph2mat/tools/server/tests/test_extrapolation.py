"""Extrapolation must preserve raw inputs, channel policies and missing labels."""

import numpy as np
import pytest
import sisl
import subprocess
import sys

from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor, OrbitalConfiguration
from graph2mat.tools.server.extrapolation import (
    DescriptorManager,
    HistoryCopy, HistorySOAPs, MatrixTimeSeriesState, extrapolate_from_series,
    get_history_block_dicts,
)


def _geometry(two_orbitals=True):
    orbitals = [sisl.AtomicOrbital("1s", R=1.)]
    if two_orbitals:
        orbitals.append(sisl.AtomicOrbital("2s", R=1.))
    atom = sisl.Atom("H", orbitals=orbitals)
    return sisl.Geometry([[0., 0., 0.], [.75, 0., 0.]], atoms=atom,
                         lattice=sisl.Lattice([10., 10., 10.], nsc=[1, 1, 1]))


@pytest.mark.parametrize("kind,cls", [
    ("density_matrix", sisl.DensityMatrix),
    ("energy_density_matrix", sisl.EnergyDensityMatrix),
    ("hamiltonian", sisl.Hamiltonian),
])
@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("all_dims", [False, True])
def test_series_channel_conversion_preserves_input_and_missing_entry_policy(kind, cls, polarized, all_dims):
    geometry = _geometry()
    kwargs = {"spin": "polarized" if polarized else "unpolarized"}
    if kind == "hamiltonian":
        kwargs["orthogonal"] = False
    matrix = cls(geometry, **kwargs)
    raw = [9., 11.] if polarized else [9.]
    if kind == "hamiltonian":
        raw.append(.25)
    matrix[0, 0] = raw
    matrix.finalize()
    before = matrix._csr.data.copy()
    columns = matrix._csr.col.copy()
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix=kind,
        symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=2 if polarized else 1,
        matrix_component_policy="spin_h_only" if polarized else "h_only",
    )
    series = MatrixTimeSeriesState(2, processor)
    series.add_next_geometry(geometry)
    series.add_last_matrix(matrix, all_dims=all_dims)
    stored = series.configs[-1].matrix
    wanted = raw if all_dims else raw[:2 if polarized else 1]
    fill = 0. if kind == "hamiltonian" else np.nan
    block = np.full((2, 2, len(wanted)), fill)
    block[0, 0] = wanted
    if len(wanted) == 1:
        block = block[:, :, 0]
    np.testing.assert_array_equal(stored.block_dict[0, 0, 0], block)
    np.testing.assert_array_equal(stored._get_block(1, 1, 0), np.full_like(block, fill))
    np.testing.assert_array_equal(matrix._csr.data, before)
    np.testing.assert_array_equal(matrix._csr.col, columns)
    assert stored._n_components == len(wanted)


def test_series_delta_preserves_both_caller_matrices():
    geometry = _geometry()
    matrix = sisl.DensityMatrix(geometry, spin="polarized")
    reference = sisl.DensityMatrix(geometry, spin="polarized")
    matrix[0, 0], reference[0, 0] = [9., 11.], [4., 5.]
    matrix.finalize()
    reference.finalize()
    before, before_reference = matrix._csr.data.copy(), reference._csr.data.copy()
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=2,
    )
    series = MatrixTimeSeriesState(2, processor)
    series.add_next_geometry(geometry)
    series.add_last_matrix_ref(reference)
    series.add_last_matrix(matrix, all_dims=True)
    np.testing.assert_array_equal(series.configs[-1].matrix.block_dict[0, 0, 0][0, 0], [5., 6.])
    np.testing.assert_array_equal(matrix._csr.data, before)
    np.testing.assert_array_equal(reference._csr.data, before_reference)


@pytest.mark.parametrize("cls", [sisl.DensityMatrix, sisl.EnergyDensityMatrix])
@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("all_dims", [False, True])
def test_density_history_separates_auxiliary_overlap_from_physical_spin_channels(cls, polarized, all_dims):
    geometry = _geometry()
    matrix = cls(geometry, spin="polarized" if polarized else "unpolarized", orthogonal=False)
    raw = [2., 5., .25] if polarized else [2., .25]
    matrix[0, 0] = raw
    matrix.finalize()
    before = matrix._csr.data.copy()
    series = MatrixTimeSeriesState(1)
    series.add_next_geometry(geometry)
    series.add_last_matrix(matrix, all_dims=all_dims)
    stored = series.configs[-1].matrix
    wanted = raw if all_dims else raw[:-1]
    assert stored._n_components == len(wanted)
    np.testing.assert_array_equal(np.atleast_1d(stored.block_dict[0, 0, 0][0, 0]), wanted)
    np.testing.assert_array_equal(matrix._csr.data, before)
    assert not matrix.orthogonal


def test_history_copy_accepts_extrapolator_kwargs_without_changing_complete_objective():
    history = {(0, 0, 0): np.array([[[1.]], [[2.]], [[4.]]]),
               (0, 1, 0): np.array([[[10.]], [[20.]], [[40.]]])}
    descriptors, irreps = HistoryCopy.get_descriptors(
        None, None, history, {}, None, r_cut=6., n_max=1, l_max=1, rbf="gto",
    )
    assert irreps is None
    for value in descriptors.values():
        np.testing.assert_array_equal(value, [[1., 10.], [2., 20.], [4., 40.]])
    result = HistoryCopy.predict_from_descriptors(descriptors, history,
                                                 {"rcond": {key: 1e-8 for key in history}})
    np.testing.assert_allclose(result[0, 0, 0], [[8.]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[0, 1, 0], [[80.]], rtol=0, atol=1e-12)


def test_history_copy_runs_through_real_series_and_processor():
    geometry = _geometry(two_orbitals=False)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False,
    )
    series = MatrixTimeSeriesState(3, processor)
    for value in (1., 2., 4.):
        series.add_next_geometry(geometry.copy())
        matrix = sisl.DensityMatrix(geometry)
        for i in range(2):
            for j in range(2):
                matrix[i, j] = value
        matrix.finalize()
        series.add_last_matrix(matrix)
    series.add_next_geometry(geometry.copy())
    result = extrapolate_from_series(series, descriptor_manager=HistoryCopy, descriptor_order=1)
    assert set(result) == {(0, 0, 0), (1, 1, 0), (0, 1, 0)}
    for block in result.values():
        np.testing.assert_allclose(block, [[8.]], rtol=0, atol=1e-12)


def _radial(r):
    return np.exp(-r)


@pytest.mark.parametrize("periodic,custom_tag", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("variable_cell", [False, True])
@pytest.mark.parametrize("vector_branch", ["node", "edge"])
def test_soap_uses_configuration_pbc_and_atomic_species(periodic, custom_tag, variable_cell, vector_branch):
    SOAP = pytest.importorskip("dscribe.descriptors").SOAP
    orbitals = sisl.SphericalOrbital(0, _radial, R=1.).toAtomicOrbital(n=1)
    atom = sisl.Atom("H", orbitals=orbitals, tag="H_custom" if custom_tag else "H")
    geometry = sisl.Geometry([[0., 0., 0.], [2.6, 0., 0.]], atoms=atom,
                             lattice=sisl.Lattice([3., 10., 10.], nsc=[3, 1, 1]))
    matrix = sisl.DensityMatrix(geometry)
    for i in range(2):
        for j in range(6):
            matrix[i, j] = 1.
    matrix.finalize()
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False,
    )
    kwargs = {"metadata": {"geometry": geometry}, "pbc": (periodic, False, False)}
    configs = [OrbitalConfiguration.from_matrix(matrix, **kwargs) for _ in range(2)]
    target_geometry = geometry.copy()
    if variable_cell:
        target_geometry.set_lattice(sisl.Lattice([3.5, 10., 10.], nsc=[3, 1, 1]))
    configs.append(OrbitalConfiguration.from_geometry(
        target_geometry, pbc=(periodic, False, False), metadata={"geometry": target_geometry},
    ))
    data = [BasisMatrixData.from_config(config, processor,
                                       nsc=config.metadata["geometry"].nsc) for config in configs]
    history, missing = get_history_block_dicts(configs, data[-1])
    descriptors, _ = HistorySOAPs.get_descriptors(data, configs, history, missing, processor,
                                                r_cut=1.2, n_max=1, l_max=1)
    soap = SOAP(species=[1], periodic=periodic, r_cut=1.2, n_max=1, l_max=1)
    expected = []
    for config in configs:
        atoms = config.metadata["geometry"].to.ase()
        atoms.set_pbc([periodic, False, False])
        expected.append(soap.create(atoms))
    for index in range(2):
        np.testing.assert_allclose(descriptors[index, index, 0][:, :2],
                                   np.array(expected)[:, index], rtol=0, atol=1e-12)
        if periodic and vector_branch == "node":
            # The closest image is at -.4 in historical cells and -.9 in the new cell.
            np.testing.assert_allclose(descriptors[index, index, 0][:, 2],
                                       [.4, .4, .9 if variable_cell else .4], rtol=0, atol=1e-12)
    if periodic and vector_branch == "edge":
        for key, value in descriptors.items():
            if key[0] != key[1] or key[2] != 0:
                np.testing.assert_allclose(value[:, 2], [.4, .4, .9 if variable_cell else .4],
                                           rtol=0, atol=1e-12)


def test_missing_soap_dependency_names_the_optional_extra():
    code = r'''
import importlib.abc
import sys
class MissingSOAP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == "dscribe":
            raise ModuleNotFoundError("missing optional SOAP", name="dscribe")
sys.meta_path.insert(0, MissingSOAP())
from graph2mat.tools.server.extrapolation import HistorySOAPs
try:
    HistorySOAPs.get_descriptors(None, None, None, None, None)
except ModuleNotFoundError as exc:
    assert exc.name == "dscribe"
    assert "graph2mat[extrapolation]" in str(exc)
else:
    raise AssertionError("missing SOAP must fail explicitly")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("missing_images", [False, True])
def test_history_blocks_preserve_all_directions_and_mark_absent_images(symmetric, missing_images):
    geometry = _geometry(two_orbitals=False)
    geometry.xyz[1, 0] = 2.6
    geometry.set_lattice(sisl.Lattice([3., 10., 10.], nsc=[3, 1, 1]))
    past_geometry = geometry.copy()
    if missing_images:
        past_geometry.set_nsc([1, 1, 1])
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="density_matrix",
        symmetric_matrix=symmetric, sub_point_matrix=False,
    )
    configs = []
    for geom, value in ((past_geometry, 1.), (geometry, 9.)):
        matrix = sisl.DensityMatrix(geom)
        for i in range(2):
            for j in range(2 * geom.n_s):
                matrix[i, j] = value
        matrix.finalize()
        configs.append(OrbitalConfiguration.from_matrix(
            matrix, pbc=(True, False, False), metadata={"geometry": geom},
        ))
    configs.append(OrbitalConfiguration.from_geometry(
        geometry, pbc=(True, False, False), metadata={"geometry": geometry},
    ))
    target = BasisMatrixData.from_config(configs[-1], processor, nsc=geometry.nsc)
    history, missing = get_history_block_dicts(configs, target)
    # The canonical representative points from atom 1 to atom 0 in the +x cell.
    wanted_edges = {(1, 0, 2)} if symmetric else {(0, 1, 1), (1, 0, 2)}
    assert set(history) == {(0, 0, 0), (1, 1, 0)} | wanted_edges
    for key in wanted_edges:
        np.testing.assert_array_equal(history[key], [[[9.]]] if missing_images else [[[1.]], [[9.]]])
        assert missing.get(key, []) == ([0] if missing_images else [])
    np.testing.assert_array_equal(history[0, 0, 0], [[[1.]], [[9.]]])


def test_history_copy_aligns_global_descriptors_to_available_block_steps():
    history = {(0, 0, 0): np.array([[[1.]], [[2.]], [[4.]]]),
               (0, 1, 0): np.array([[[10.]], [[40.]]])}
    before = {key: value.copy() for key, value in history.items()}
    descriptors, irreps = HistoryCopy.get_descriptors(
        None, None, history, {(0, 1, 0): [1]}, None,
    )
    assert irreps is None
    # Label at step p pairs with the global matrix of step p - 1 (audit M6): block (0, 1, 0)
    # has labels at steps 0 and 2, so its descriptors are steps 1 and 2 (the target's), where
    # only the (0, 0, 0) column is complete.
    np.testing.assert_array_equal(descriptors[0, 0, 0], [[1.], [2.], [4.]])
    np.testing.assert_array_equal(descriptors[0, 1, 0], [[2.], [4.]])
    result = HistoryCopy.predict_from_descriptors(
        descriptors, history, {"rcond": {key: 1e-8 for key in history}},
    )
    # Both blocks double at every step: the next values are 8 and 80.
    np.testing.assert_allclose(result[0, 0, 0], [[8.]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result[0, 1, 0], [[80.]], rtol=0, atol=1e-12)
    for key in history:
        np.testing.assert_array_equal(history[key], before[key])


@pytest.mark.parametrize("history", [{}, {(0, 0, 0): np.empty((0, 1, 1))}])
def test_history_copy_empty_blocks_do_not_create_fake_predictions(history):
    descriptors, irreps = HistoryCopy.get_descriptors(None, None, history, {}, None)
    assert descriptors == {} and irreps is None
    assert HistoryCopy.predict_from_descriptors(descriptors, history, {}) == {}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("missing", ["labels", "all_labels", "historical_descriptor", "target_descriptor"])
def test_descriptor_fit_uses_finite_observations_per_label_component(dtype, missing):
    descriptors = np.array([[1., 1.], [2., 2.], [4., 4.]], dtype=dtype)
    labels = np.array([[10., 20.], [20., 40.]], dtype=dtype)
    expected = np.array([40., 80.])
    if missing == "labels":
        labels[0, 0], labels[1, 1] = np.nan, np.nan
    elif missing == "all_labels":
        labels[:] = np.nan
        expected[:] = 0
    elif missing == "historical_descriptor":
        descriptors[0, 1] = np.nan
    else:
        descriptors[-1, 1] = np.nan
    before_labels, before_descriptors = labels.copy(), descriptors.copy()
    actual = DescriptorManager.predict_block(descriptors, labels)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5 if dtype == np.float32 else 1e-12)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(labels, before_labels)
    np.testing.assert_array_equal(descriptors, before_descriptors)


def test_descriptor_fit_rejects_unaligned_observations_and_empty_target():
    with pytest.raises(ValueError, match="one more descriptor"):
        DescriptorManager.predict_block(np.ones((3, 2)), np.ones((3, 1, 1)))
    with pytest.raises(ValueError, match="at least one descriptor"):
        DescriptorManager.predict_block(np.empty((0, 2)), np.empty((0, 1, 1)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_descriptor_fit_preserves_its_objective_and_dtype(dtype):
    descriptors = np.array([[1., 1.], [2., 2.], [4., 4.]], dtype=dtype)
    labels = np.array([[10., 20.], [20., 40.]], dtype=dtype)
    coefficients = DescriptorManager.block_coefficients(descriptors)
    prediction = DescriptorManager.predict_block(descriptors, labels)
    tolerance = 2e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(coefficients, [.8, 1.6], rtol=0, atol=tolerance)
    np.testing.assert_allclose(prediction, [40., 80.], rtol=0, atol=tolerance)
    assert coefficients.dtype == prediction.dtype == dtype


@pytest.mark.parametrize("missing", ["history", "target", "both"])
def test_descriptor_coefficients_keep_zero_weights_for_unusable_history_rows(missing):
    descriptors = np.array([[1., 1.], [2., 2.], [4., 4.]])
    if missing in ("history", "both"):
        descriptors[0, 1] = np.nan
    if missing in ("target", "both"):
        descriptors[-1, 1] = np.nan
    expected = [0., 2.] if missing == "history" else [.8, 1.6]
    actual = DescriptorManager.block_coefficients(descriptors)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("manager", [HistoryCopy, HistorySOAPs])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_real_extrapolation_with_missing_onsite_and_edge_blocks(manager, components, symmetric):
    orbital = sisl.SphericalOrbital(0, _radial, R=1.).toAtomicOrbital(n=1)
    geometry = sisl.Geometry([[0., 0., 0.], [.75, 0., 0.]],
                             atoms=sisl.Atom(1, orbitals=orbital), lattice=[10.]*3)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="density_matrix",
        symmetric_matrix=symmetric, sub_point_matrix=False, n_matrix_components=components,
    )
    series = MatrixTimeSeriesState(3, processor)
    for step, value in enumerate((1., 2., 4.)):
        series.add_next_geometry(geometry.copy())
        matrix = sisl.DensityMatrix(geometry, spin="polarized" if components == 2 else "unpolarized")
        for i in range(2):
            for j in range(2):
                if step == 1 and (i != 0 or j != 0):
                    continue
                factor = 2 if symmetric and i != j else 1 + 2*i + j
                raw = value * factor * np.array([1., 10.][:components])
                matrix[i, j] = raw
        matrix.finalize()
        series.add_last_matrix(matrix)
    before = [{key: value.copy() for key, value in config.matrix.block_dict.items()}
              for config in series.configs]
    series.add_next_geometry(geometry.copy())
    actual = extrapolate_from_series(series, descriptor_manager=manager, descriptor_order=1)
    # The blocks other than (0, 0, 0) miss step 1: their one observed pair is the label of
    # step 2 with the global matrix of step 1 (audit M6), where only the constant
    # descriptor and (0, 0, 0) (2 * [1, 10]) are complete; the target uses step 2 (4 * [1, 10]).
    # The least-squares prediction is label * (D2 . D1) / (D1 . D1).
    squared_features = 101 if components == 2 else 1
    weight = (1 + 8*squared_features) / (1 + 4*squared_features)
    expected = {(0, 0, 0): 8., (1, 1, 0): 16*weight, (0, 1, 0): 8*weight}
    if manager is HistorySOAPs:
        expected = {(0, 0, 0): 7/3, (1, 1, 0): 10., (0, 1, 0): 5.}
    if not symmetric:
        expected[1, 0, 0] = 12*weight if manager is HistoryCopy else 7.5
    assert set(actual) == set(expected)
    for key, value in expected.items():
        block = value * np.array([1., 10.][:components])
        block = block.reshape(1, 1, components) if components == 2 else block.reshape(1, 1)
        np.testing.assert_allclose(actual[key], block, rtol=0, atol=1e-10)
    for config, original in zip(list(series.configs)[:-1], before):
        assert set(config.matrix.block_dict) == set(original)
        for key, value in original.items():
            np.testing.assert_array_equal(config.matrix.block_dict[key], value)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_extrapolation_without_any_historical_matrix_is_explicit(depth):
    geometry = _geometry(two_orbitals=False)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False,
    )
    series = MatrixTimeSeriesState(3, processor)
    for _ in range(depth):
        series.add_next_geometry(geometry.copy())
    with pytest.raises(ValueError, match="historical matrix"):
        extrapolate_from_series(series, descriptor_manager=HistoryCopy)


@pytest.mark.parametrize("kind,cls,known", [
    ("hamiltonian", sisl.Hamiltonian, True),
    ("density_matrix", sisl.DensityMatrix, False),
])
def test_history_empty_sparse_blocks_keep_the_converters_missing_policy(kind, cls, known):
    geometry = _geometry(two_orbitals=False)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix=kind,
        symmetric_matrix=True, sub_point_matrix=False,
    )
    configs = [OrbitalConfiguration.from_matrix(
        cls(geometry), metadata={"geometry": geometry},
    ) for _ in range(2)]
    configs.append(OrbitalConfiguration.from_geometry(geometry, metadata={"geometry": geometry}))
    target = BasisMatrixData.from_config(configs[-1], processor, nsc=geometry.nsc)
    history, missing = get_history_block_dicts(configs, target)
    keys = {(0, 0, 0), (1, 1, 0), (0, 1, 0)}
    assert set(history) == (keys if known else set())
    for key in keys:
        if known:
            np.testing.assert_array_equal(history[key], np.zeros((2, 1, 1)))
            assert not missing.get(key)
        else:
            assert missing[key] == [0, 1]
    assert all(config.matrix.block_dict == {} for config in configs[:-1])


@pytest.mark.parametrize("kind,cls,known", [
    ("hamiltonian", sisl.Hamiltonian, True),
    ("density_matrix", sisl.DensityMatrix, False),
])
def test_single_history_empty_sparse_matrix_is_zero_h_or_unknown_dm(kind, cls, known):
    geometry = _geometry(two_orbitals=False)
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix=kind,
        symmetric_matrix=True, sub_point_matrix=False,
    )
    series = MatrixTimeSeriesState(1, processor)
    series.add_next_geometry(geometry.copy())
    series.add_last_matrix(cls(geometry))
    series.add_next_geometry(geometry.copy())
    if known:
        result = extrapolate_from_series(series, descriptor_manager=HistoryCopy)
        assert set(result) == {(0, 0, 0), (1, 1, 0), (0, 1, 0)}
        for block in result.values():
            np.testing.assert_array_equal(block, [[0.]])
    else:
        with pytest.raises(ValueError, match="observed historical matrix block"):
            extrapolate_from_series(series, descriptor_manager=HistoryCopy)
    assert series.configs[0].matrix.block_dict == {}
