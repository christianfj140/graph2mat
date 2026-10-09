"""Full reference observables, not a comparison between two new code paths."""

import importlib
import warnings

import numpy as np
import pytest
import sisl
from scipy.sparse import csr_array

from graph2mat.core.data.matrices import OrbitalMatrix, get_matrix_cls
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices, periodic_matrices


@pytest.fixture
def physical():
    return importlib.import_module("graph2mat.core.data.physical")


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("fractional", [False, True])
def test_reference_population_and_energy_are_independent(physical, spin, fractional):
    matrices = analytic_matrices(spin, fractional)
    actual = physical.physical_reference_targets(**matrices, requested=("charge", "energy"))
    np.testing.assert_allclose(actual["population"], [.6, .6] if fractional else [1., 1.], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual["energy_residual"], [0., 0.], rtol=0, atol=1e-14)
    assert actual["spin_components"].tolist() == [spin]
    assert actual["capabilities"].tolist() == [3]
    if fractional and spin == 1:
        density = matrices["density"].tocsr(0).toarray()
        overlap = matrices["overlap"].tocsr(0).toarray()
        assert not np.allclose(density @ overlap @ density, 2*density)


def test_periodic_trace_matches_independent_bloch_sum_not_gamma(physical):
    matrices = periodic_matrices()
    actual = physical.physical_reference_targets(**matrices, requested=("charge", "energy"))
    np.testing.assert_allclose(actual["population"], [.84], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual["energy_residual"], [0.], rtol=0, atol=1e-14)
    kx = matrices["kpoints"][:, 0]
    np.testing.assert_allclose(np.mean((.8+.2*np.cos(2*np.pi*kx))*(1.+.4*np.cos(2*np.pi*kx))), .84, atol=1e-14)
    assert (1.4*1.) == pytest.approx(1.4)
    np.testing.assert_allclose(matrices["density"].mulliken("atom"), [.84], atol=1e-14)


@pytest.mark.parametrize("small", [0., -1e-3, 1e-13, 1e-12])
def test_bad_overlap_is_rejected_without_regularization(physical, small):
    matrices = analytic_matrices()
    matrices["overlap"] = sisl.Overlap.fromsp(matrices["overlap"].geometry, csr_array(np.diag([1., small])))
    with pytest.raises(ValueError, match="singular|ill-conditioned|positive"):
        physical.physical_reference_targets(**matrices, requested=("charge", "energy"))


def test_energy_reference_does_not_shift_itself_to_pass(physical):
    matrices = analytic_matrices()
    matrices["energy_density"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="energy|gauge|residual"):
        physical.physical_reference_targets(**matrices, requested=("energy",))


@pytest.mark.parametrize("missing", ["overlap", "density", "hamiltonian", "energy_density"])
def test_missing_matrix_is_not_replaced_by_zero(physical, missing):
    matrices = analytic_matrices()
    matrices[missing] = None
    with pytest.raises(ValueError, match=missing):
        physical.physical_reference_targets(**matrices, requested=("energy",))


def test_charge_needs_no_hamiltonian_or_edm(physical):
    matrices = analytic_matrices()
    del matrices["hamiltonian"], matrices["energy_density"]
    result = physical.physical_reference_targets(**matrices, requested=("charge",))
    np.testing.assert_allclose(result["population"], [1., 1.], atol=1e-14)
    assert "energy_residual" not in result
    assert result["capabilities"].tolist() == [1]


@pytest.mark.parametrize("bad", ["position", "cell", "nsc", "basis"])
def test_incompatible_references_are_not_relabelled(physical, bad):
    matrices = analytic_matrices()
    matrix = matrices["density"] = matrices["density"].copy()
    assert all(a.equal(b) for a, b in zip(matrices["overlap"].geometry.atoms, matrix.geometry.atoms))
    if bad == "position":
        matrix.geometry.xyz[0, 0] += .01
    elif bad == "cell":
        matrix.geometry.lattice.cell[0, 0] += .01
    elif bad == "nsc":
        matrix.set_nsc([3, 1, 1])
    else:
        matrix.geometry.atoms[0] = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.81, q0=1.)])
    with pytest.raises(ValueError, match="geometry|cell|basis|nsc|image"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


@pytest.mark.parametrize("spin,size", [("non-colinear", 4), ("spin-orbit", 8)])
def test_unsupported_spin_is_rejected(physical, spin, size):
    matrices = analytic_matrices()
    matrices["density"] = sisl.DensityMatrix.fromsp(matrices["density"].geometry,
        [csr_array(np.eye(2)) for _ in range(size)], spin=spin)
    with pytest.raises(ValueError, match="spin|collinear"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


def test_spin_components_must_agree(physical):
    matrices = analytic_matrices()
    matrices["hamiltonian"] = analytic_matrices(spin=2)["hamiltonian"]
    with pytest.raises(ValueError, match="spin|components"):
        physical.physical_reference_targets(**matrices, requested=("energy",))


def test_complex_coefficients_are_rejected(physical):
    matrices = analytic_matrices()
    matrices["density"] = matrices["density"].astype(np.complex128)
    with pytest.raises(ValueError, match="real|complex"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


@pytest.mark.parametrize("kpoints,kweights", [
    ([], []), ([[0., 0.]], [1.]), ([[0., 0., np.nan]], [1.]),
    ([[0., 0., 0.]], [-1.]), ([[0., 0., 0.]], [0.]),
    ([[0., 0., 0.]], [.9]), ([[0., 0., 0.]], [np.inf]),
    ([[0., 0., 0.]], [[1.]]),
])
def test_invalid_mesh_is_not_normalized_or_replaced(physical, kpoints, kweights):
    matrices = analytic_matrices()
    matrices.update(kpoints=np.array(kpoints), kweights=np.array(kweights))
    with pytest.raises(ValueError, match="mesh|kpoint|weight"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


def test_nonhermitian_real_space_reference_is_rejected(physical):
    matrices = periodic_matrices()
    matrices["density"][0, matrices["density"].geometry.no, 0] += .03
    with pytest.raises(ValueError, match="Hermitian|hermitian"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


@pytest.mark.parametrize("unknown", ["absent", "nan"])
def test_unknown_density_with_nonzero_overlap_is_rejected(physical, unknown):
    matrices = analytic_matrices()
    density = csr_array(np.diag([5/6, 5/6]))
    if unknown == "nan":
        density = csr_array([[5/6, np.nan], [np.nan, 5/6]])
    matrices["density"] = sisl.DensityMatrix.fromsp(matrices["density"].geometry, density)
    with pytest.raises(ValueError, match="unknown|observed"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


def test_explicit_stored_zero_is_observed(physical):
    matrices = analytic_matrices()
    density = csr_array((np.array([5/6, 0., 0., 5/6]), [0, 1, 0, 1], [0, 2, 4]), shape=(2, 2))
    matrices["density"] = sisl.DensityMatrix.fromsp(matrices["density"].geometry, density)
    assert matrices["density"].tocsr(0).nnz == 4
    result = physical.physical_reference_targets(**matrices, requested=("charge",))
    np.testing.assert_allclose(result["population"], [5/6, 5/6], atol=1e-14)


def test_unknown_times_exact_zero_is_excluded_before_multiplication(physical):
    matrices = analytic_matrices()
    geometry = matrices["overlap"].geometry
    matrices["overlap"] = sisl.Overlap.fromsp(geometry, csr_array(np.eye(2)))
    matrices["density"] = sisl.DensityMatrix.fromsp(geometry, csr_array([[.7, np.nan], [np.nan, .8]]))
    result = physical.physical_reference_targets(**matrices, requested=("charge",))
    np.testing.assert_allclose(result["population"], [.7, .8], atol=1e-14)


def test_bounded_archive_asymmetry_uses_true_opposite_partner_without_changing_input(physical):
    matrices = analytic_matrices()
    epsilon = 1e-9
    for name, matrix in matrices.items():
        if name in ("kpoints", "kweights"):
            continue
        component = 0 if name == "overlap" else matrix.S_idx
        matrix[0, 1, component] += epsilon
        matrix[1, 0, component] -= epsilon
    before = {name: matrix.tocsr(0).copy() for name, matrix in matrices.items()
              if name not in ("kpoints", "kweights")}
    result = physical.physical_reference_targets(**matrices, requested=("charge", "energy"))
    # D_01 * S_10 belongs to atom 0, not D_01 * S_01.
    np.testing.assert_allclose(result["population"], [1-5*epsilon/6, 1+5*epsilon/6], rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["energy_residual"], [5*epsilon/36, -5*epsilon/36], rtol=0, atol=1e-15)
    for name, original in before.items():
        np.testing.assert_array_equal(matrices[name].tocsr(0).toarray(), original.toarray())


def test_large_archive_asymmetry_is_rejected(physical):
    matrices = analytic_matrices()
    matrices["overlap"][0, 1, 0] += 1e-4
    with pytest.raises(ValueError, match="Hermitian|hermitian"):
        physical.physical_reference_targets(**matrices, requested=("charge",))


@pytest.mark.parametrize("name", ["electron_weight", "mulliken_weight", "energy_weight"])
@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_physical_weight_is_rejected(physical, name, value):
    with pytest.raises(ValueError, match="finite|nonnegative"):
        physical.validate_physical_weights(**{name: value})


@pytest.mark.parametrize("labels,expected", [(None, ()), ([], ()), (["energy", "charge", "charge"], ("charge", "energy"))])
def test_physical_label_selection_is_explicit(physical, labels, expected):
    assert physical.normalize_physical_labels(labels) == expected


@pytest.mark.parametrize("labels", ["charge", ["missing"], [None]])
def test_invalid_label_selection_is_rejected(physical, labels):
    with pytest.raises(ValueError, match="physical_labels"):
        physical.normalize_physical_labels(labels)


@pytest.mark.parametrize("key", ["energy_density_matrix", sisl.EnergyDensityMatrix])
def test_edm_registration_does_not_fall_back_or_invent_baseline(key):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cls = get_matrix_cls(key)
    assert not captured
    assert cls is OrbitalMatrix
