import math

import numpy as np
import pytest
import sisl

from graph2mat.tools.analysis.metrics import (
    band_dos_metrics,
    band_window_metrics,
    gamma_metrics,
    layer_masks,
    layer_metrics,
    path_bands,
    species_pair_errors,
)
from graph2mat.tools.conftest import make_hamiltonian, write_fdf


def _write(H, path):
    H.write(str(path))
    return path


def _shifted(H, c):
    """H + c S: the same physics with another energy zero."""
    out = H.copy()
    out._csr._D[:, 0] += c * out._csr._D[:, out.S_idx]
    return out


def _hbn():
    geometry = sisl.geom.honeycomb(1.44, atoms=[sisl.Atom(5, R=1.6), sisl.Atom(7, R=1.6)])
    H = make_hamiltonian(geometry)
    H[0, 0] = (2.0, 1.0)
    H[1, 1] = (-2.0, 1.0)
    return H


def test_exact_prediction_has_zero_errors(tmp_path):
    ref = _write(make_hamiltonian(), tmp_path / "ref.TSHS")
    gamma = gamma_metrics(ref, ref, window_ev=10.0)
    assert gamma["H_MAE_meV"] == gamma["H_MAE_gauge_meV"] == gamma["rel_Frob"] == 0
    assert gamma["spectral_err_meV"] == pytest.approx(0, abs=1e-9)
    assert band_dos_metrics(ref, ref, grid=(4, 4, 1), n_per_segment=10) == pytest.approx(
        {"band_rmse_meV": 0, "dos_rel_L1": 0}, abs=1e-9
    )


def test_gauge_shift_is_removed(tmp_path):
    H = make_hamiltonian()
    ref = _write(H, tmp_path / "ref.TSHS")
    pred = _write(_shifted(H, 0.2), tmp_path / "pred.TSHS")
    gamma = gamma_metrics(pred, ref)
    assert gamma["H_MAE_meV"] > 10 and gamma["H_MAE_gauge_meV"] == pytest.approx(0, abs=1e-9)
    layer = layer_metrics(pred, ref, n_per_segment=10)
    assert layer["gauge_c_meV"] == pytest.approx(200)
    assert layer["band_rmse_meV"] == pytest.approx(200, rel=1e-6)
    assert layer["band_rmse_aligned_meV"] == pytest.approx(0, abs=1e-6)


def test_identity_overlap_prediction_uses_the_reference_overlap(tmp_path):
    H = make_hamiltonian()
    ref = _write(H, tmp_path / "ref.TSHS")
    h_only = make_hamiltonian(overlap=0.0)  # same H, S = 1 as h_only predictions are written
    h_only._csr._D[:, 0] = H._csr._D[:, 0]
    pred = _write(h_only, tmp_path / "pred.TSHS")
    eig_pred, eig_ref = path_bands(pred, ref, n_per_segment=10)
    np.testing.assert_allclose(eig_pred, eig_ref, atol=1e-10)


def test_empty_energy_window_gives_nan(tmp_path):
    ref = _write(make_hamiltonian(), tmp_path / "ref.TSHS")
    assert math.isnan(gamma_metrics(ref, ref, window_ev=-1)["spectral_err_meV"])
    layer = layer_metrics(ref, ref, window_ev=-1, n_per_segment=5)
    assert math.isnan(layer["band_rmse_meV"]) and math.isnan(layer["band_rmse_aligned_meV"])


def test_layer_masks_of_a_bilayer(tmp_path):
    H = make_hamiltonian(sisl.geom.bilayer(bottom_atoms=sisl.Atom(6, R=1.6), top_atoms=sisl.Atom(6, R=1.6)))
    intra, inter = layer_masks(_write(H, tmp_path / "bl.TSHS"))
    z = H.geometry.xyz[:, 2]
    assert inter[0, 2] == (abs(z[0] - z[2]) > 1.5) and not inter[0, 1]
    assert (intra ^ inter).all()


def test_insulator_band_window_and_species_pairs(tmp_path):
    H = _hbn()
    ref = _write(H, tmp_path / "ref.TSHS")
    out = band_window_metrics(ref, ref)
    assert out["gap_ref_eV"] > 1 and out["gap_err_meV"] == pytest.approx(0, abs=1e-9)
    assert out["canon_band_edges_rmse_aligned_meV"] == pytest.approx(0, abs=1e-9)
    shifted = band_window_metrics(_write(_shifted(H, 0.1), tmp_path / "p.TSHS"), ref)
    assert shifted["gap_err_meV"] == pytest.approx(0, abs=1e-6)
    assert shifted["band_edges_rmse_aligned_meV"] == pytest.approx(0, abs=1e-6)

    fdf = write_fdf(tmp_path / "s", H.geometry)
    pairs = species_pair_errors(_write(_shifted(H, 0.1), tmp_path / "p2.TSHS"), ref, fdf)
    assert set(pairs) == {"B-B", "B-N", "N-N"}
    assert max(pairs.values()) == pytest.approx(0, abs=1e-9)


def test_band_window_needs_occupied_and_empty_bands(tmp_path):
    ref = _write(make_hamiltonian(onsite=-30.0), tmp_path / "ref.TSHS")
    with pytest.raises(ValueError, match="occupied/unoccupied"):
        band_window_metrics(ref, ref)
