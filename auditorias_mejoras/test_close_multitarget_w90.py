"""Numerical closure oracles; CPU only, no external data."""

import numpy as np
import pytest


def evaluator():
    from auditorias_mejoras import close_multitarget_w90
    return close_multitarget_w90


def test_packed_alignment_equal_node_edge_weight_and_auxiliary_units():
    refn = np.array([[1., 1., .5, .1]])
    refe = np.array([[2., 2., .25, .2]])
    predn, prede = refn.copy(), refe.copy()
    predn[0, 0], prede[0, 0] = 5., 7.
    result = evaluator().packed_errors(refn, refe, predn, prede)
    assert result["shift_eV"] == pytest.approx(2.8)
    assert result["hamiltonian"]["mse"] == pytest.approx(.9)
    for target in ("overlap", "density_matrix", "energy_density_matrix"):
        assert result[target]["rmse"] == 0


def test_exact_and_cS_spectra_full_space_and_gap():
    ev = evaluator()
    s = np.array([np.diag(np.linspace(.001, 1., 8))] * 2)
    h = np.array([np.diag(np.linspace(.001, 1., 8) * np.arange(8.))] * 2)
    ref = np.array([np.arange(8.)] * 2)
    for shift in (0., .375, -2.):
        result = ev.spectral_errors(h + shift * s, s, ref, shift)
        assert result["status"] == "valid"
        assert result["levels_rmse_eV"] < 1e-12
        assert result["direct_gap_mae_eV"] < 1e-12
        assert result["indirect_gap_error_eV"] < 1e-12
        assert result["min_s"] == pytest.approx(.001)


def test_non_spd_overlap_keeps_failure_and_no_conditional_zero_error():
    s = np.array([np.eye(8), np.diag([-1e-8] + [1.] * 7)])
    result = evaluator().spectral_errors(np.array([np.eye(8)] * 2), s, np.ones((2, 8)), 0.)
    assert result["status"] == "invalid"
    assert result["non_spd_k"] == [1]
    assert result["min_s"] == pytest.approx(-1e-8)
    assert result["levels_rmse_eV"] is None


def test_summary_counts_all_failed_spectra_in_denominator():
    rows = [dict(sample_id="a", spectra={"path": dict(common=dict(status="valid", levels_rmse_eV=.1),
            predicted_s=dict(status="invalid", levels_rmse_eV=None))}),
            dict(sample_id="b", spectra={"path": dict(common=dict(status="valid", levels_rmse_eV=.3),
            predicted_s=dict(status="valid", levels_rmse_eV=.4))})]
    result = evaluator().spectral_summary(rows, "path", "predicted_s")
    assert result["total"] == 2 and result["valid"] == 1 and result["invalid_ids"] == ["a"]
    assert result["mean_valid"]["levels_rmse_eV"] == .4


def test_no_clobber_closure(tmp_path):
    (tmp_path / "preserve").write_text("keep")
    with pytest.raises(FileExistsError):
        evaluator().evaluate(tmp_path)
    assert (tmp_path / "preserve").read_text() == "keep"


@pytest.mark.parametrize("arm", ["h", "joint"])
def test_real_geometry_only_prediction_and_writer(arm):
    from graph2mat.bindings.torch.data.tests.test_multitarget_dataset import bundle, processor
    ev = evaluator()
    table = processor().basis_table
    model = ev.pilot.make_arm(arm, table, small=True).eval()
    proc = ev.pilot.arm_processor(table, arm)
    data, prediction = ev.predict_one(model, proc, bundle(2))
    assert getattr(data, "point_labels", None) is None and getattr(data, "edge_labels", None) is None
    h = proc.matrix_from_data(data, predictions=prediction, matrix_target="hamiltonian", threshold=None)
    assert h.geometry.no == 2
    assert np.isfinite(h.tocsr(0).data).all()


def test_physical_contractions_reference_exact():
    import sisl
    from scipy.sparse import csr_array
    ev = evaluator()
    g = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)]), lattice=[10.] * 3)
    s = csr_array(np.eye(2))
    matrices = dict(hamiltonian=sisl.Hamiltonian.fromsp(g, 2*s, S=s),
        overlap=sisl.Overlap.fromsp(g, s), density=sisl.DensityMatrix.fromsp(g, .5*s, S=s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, s, S=s))
    result = ev.auxiliary_physics(matrices, matrices)
    assert result["N_reference"] == pytest.approx(1.)
    assert result["N_pred_fixed_Sref"] == pytest.approx(1.)
    assert result["N_pred_own_Spred"] == pytest.approx(1.)
    assert result["HD_minus_SE_max_abs_per_orbital_eV"] == pytest.approx(0.)


@pytest.mark.parametrize("pred_nsc,ref_nsc,nref,npred", [(1, 3, .54, .5), (3, 5, .78, .51)])
def test_fixed_overlap_contraction_matches_periodic_images_and_zero_writer_tails(pred_nsc, ref_nsc, nref, npred):
    import sisl
    from scipy.sparse import csr_array
    ev = evaluator()
    g = sisl.Geometry([[0, 0, 0]], atoms=sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)]), lattice=[10.] * 3)
    g.set_nsc([pred_nsc, 1, 1])
    unit = csr_array([[1. if not any(offset) else .1 for offset in g.sc_off]])
    predicted = dict(hamiltonian=sisl.Hamiltonian.fromsp(g, 2*unit, S=unit),
        overlap=sisl.Overlap.fromsp(g, unit), density=sisl.DensityMatrix.fromsp(g, .5*unit, S=unit),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, unit, S=unit))
    other = g.copy()
    other.set_nsc([ref_nsc, 1, 1])
    s = csr_array([[1. if not any(offset) else .1 if abs(offset[0]) == 1 else .3 for offset in other.sc_off]])
    d = csr_array([[.5 if not any(offset) else .2 if abs(offset[0]) == 1 else .4 for offset in other.sc_off]])
    reference = dict(overlap=sisl.Overlap.fromsp(other, s), density=sisl.DensityMatrix.fromsp(other, d, S=s))
    result = ev.auxiliary_physics(predicted, reference)
    assert result["N_reference"] == pytest.approx(nref)
    assert result["N_pred_fixed_Sref"] == pytest.approx(npred)
    assert result["N_pred_own_Spred"] == pytest.approx(npred)
    assert result["HD_minus_SE_max_abs_per_orbital_eV"] == pytest.approx(0.)
