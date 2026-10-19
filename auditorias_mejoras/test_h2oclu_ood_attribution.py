"""Analytic controls for the saved-matrix diagnostic, never training or inference."""
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    path = Path(__file__).with_name("diagnose_h2oclu_ood.py")
    assert path.is_file(), "saved-matrix attribution diagnostic is missing"
    return importlib.import_module("diagnose_h2oclu_ood")


def fixture():
    H = np.diag([-.004, 1., 4.])
    S = np.diag([.002, 1., 1.])
    H0 = np.diag([-.002, .9, 4.])
    return H, S, H0


def test_exact_reference_and_shared_h0_cancel():
    a = api()
    H, S, H0 = fixture()
    frame = a.reference_frame(H, S, H0, 1)
    out = a.diagnose(H, frame, np.array([0, 1, 1]), np.array([1, 8]), ["H:s", "O:s", "O:p"])
    assert out["baseline"]["gap_eV"] == pytest.approx(3.)
    assert out["baseline"]["gap_error_eV"] == pytest.approx(0., abs=1e-12)
    assert out["whitened_frobenius_eV"] == pytest.approx(0., abs=1e-12)
    assert out["h0_cancellation_max_eV"] < 1e-12
    assert frame["h0_metrics"]["gap_eV"] == pytest.approx(1.9)


def test_small_overlap_error_and_full_basis_counterfactual():
    a = api()
    H, S, H0 = fixture()
    P = H + np.diag([.004, -.000008, 0.])  # S-orthogonal residual: gauge c=0.
    before = P.copy()
    frame = a.reference_frame(H, S, H0, 1)
    out = a.diagnose(P, frame, np.array([0, 1, 1]), np.array([1, 8]), ["H:s", "O:s", "O:p"])
    assert out["baseline"]["gap_eV"] == pytest.approx(.999992)
    assert out["baseline"]["gap_error_eV"] == pytest.approx(2.000008)
    assert out["overlap_sectors"]["low_low"]["whitened_squared_share"] > .999999
    fixed = out["overlap_sectors"]["low_touch"]["oracle"]
    assert fixed["gap_eV"] == pytest.approx(2.999992)
    assert out["species"]["H-H"]["whitened_signed_share"] > .999999
    assert sum(g["whitened_signed_share"] for g in out["families"].values()) == pytest.approx(1.)
    assert out["atomic_blocks"][0]["atoms"] == [0, 0]
    assert out["atomic_block_signed_share_sum"] == pytest.approx(1.)
    assert fixed["level_count"] == 3  # Repair residual; never discard a basis vector.
    np.testing.assert_array_equal(P, before)


def test_cS_shift_changes_neither_gap_nor_attribution():
    a = api()
    H, S, H0 = fixture()
    frame = a.reference_frame(H, S, H0, 1)
    args = (frame, np.array([0, 1, 1]), np.array([1, 8]), ["H:s", "O:s", "O:p"])
    P = H + np.diag([.004, -.000008, 0.])
    x, y = a.diagnose(P, *args), a.diagnose(P+3*S, *args)
    assert y["gauge_c_eV"]-x["gauge_c_eV"] == pytest.approx(3.)
    assert y["baseline"]["gap_eV"] == pytest.approx(x["baseline"]["gap_eV"])
    assert y["whitened_frobenius_eV"] == pytest.approx(x["whitened_frobenius_eV"])


def test_physical_gate_precedes_residual_projection_with_cancellation():
    a = api()
    H, S, H0 = fixture()
    P = H+np.diag([.004, -.000008, 0.])
    P[0, 1] = 1e-10  # Valid raw H, but normalized defect of tiny residual is larger.
    out = a.diagnose(P, a.reference_frame(H, S, H0, 1), np.array([0, 1, 1]),
                     np.array([1, 8]), ["H:s", "O:s", "O:p"])
    assert out["baseline"]["gap_eV"] == pytest.approx(.999992)


def test_subspace_weight_invariant_under_degenerate_rotations():
    a = api()
    vectors = np.eye(3)[:, :2]
    theta = .71
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert a.subspace_low_weight(vectors, np.array([True, False, False])) == pytest.approx(.5)
    assert a.subspace_low_weight(vectors @ rotation, np.array([True, False, False])) == pytest.approx(.5)


def test_state_selection_includes_whole_degenerate_cluster():
    a = api()
    levels = np.array([0., 1., 1.+1e-10, 3.])
    np.testing.assert_array_equal(a.complete_degenerate_mask(levels, np.array([False, True, False, False])),
                                  [False, True, True, False])


def test_spurious_state_changes_occupied_subspace():
    a = api()
    H, S = np.diag([-2., .002, 4.]), np.diag([1., .002, 1.])
    P = H+np.diag([.000016, -.008, 0.])
    out = a.diagnose(P, a.reference_frame(H, S, H, 1), np.array([0, 1, 1]),
                     np.array([1, 8]), ["H:s", "O:s", "O:p"])
    assert out["occupied_subspace_leakage_states"] == pytest.approx(1.)
    assert out["predicted_occupied_low_trace"] == pytest.approx(1.)
    assert out["reference_occupied_low_trace"] == pytest.approx(0.)


def test_cli_refuses_to_overwrite_existing_output(tmp_path):
    target = tmp_path/"keep.json"
    target.write_text("keep")
    script = Path(__file__).with_name("diagnose_h2oclu_ood.py")
    result = subprocess.run([sys.executable, str(script), "--output", str(target)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "new diagnostic output" in result.stderr
    assert target.read_text() == "keep"


@pytest.mark.parametrize("bad", ["nonfinite", "asymmetric", "indefinite"])
def test_bad_physical_inputs_rejected_before_projection(bad):
    a = api()
    H, S, H0 = fixture()
    if bad == "nonfinite":
        H[0, 0] = np.nan
    elif bad == "asymmetric":
        H[0, 1] = 1.
    else:
        S[0, 0] = -1.
    with pytest.raises(ValueError):
        a.reference_frame(H, S, H0, 1)
