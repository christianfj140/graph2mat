import numpy as np
import pytest
import sisl

from graph2mat.tools.analysis.attribution import CLASSES, attribute, counterfactuals
from graph2mat.tools.analysis.breakdown import structure_breakdown, summarize_breakdown
from graph2mat.tools.analysis.kb_tail import without_kb_tail
from graph2mat.tools.conftest import make_hamiltonian, write_fdf


def _carbon(R=1.6):
    r = np.linspace(0, R, 50)
    return sisl.Atom(6, R=R, orbitals=[sisl.AtomicOrbital("2pzZ", (r, np.exp(-r)), q0=1)])


def _noisy(H, rng, scale=0.01):
    out = H.copy()
    out._csr._D[:, 0] += scale * rng.normal(size=out._csr._D.shape[0])
    return out


def _bilayer():
    geometry = sisl.geom.bilayer(bottom_atoms=_carbon(), top_atoms=_carbon())
    H = sisl.Hamiltonian(geometry, orthogonal=False)
    H.construct([(0.1, 1.6, 3.5), ((0.0, 1.0), (-2.7, 0.1), (0.3, 0.01))])
    return H


def test_breakdown_categories(tmp_path, rng):
    H = make_hamiltonian(sisl.geom.graphene(atoms=_carbon()))
    H.write(str(tmp_path / "ref.TSHS"))
    _noisy(H, rng).write(str(tmp_path / "pred.TSHS"))
    fdf = write_fdf(tmp_path / "s", H.geometry)
    row = structure_breakdown(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", fdf, [_carbon()])
    assert row["total_n"] == H.no**2
    shells = [k for k in row if k.startswith("shell_") and k.endswith("_sum")]
    assert sum(row[k] for k in shells) == pytest.approx(row["total_sum"])
    assert row["l_pp_sum"] == pytest.approx(row["total_sum"])
    summary = summarize_breakdown([row, row])
    assert summary["n_structures"] == 2
    assert summary["categories"]["total"]["share_of_error"] == 1.0

    exact = structure_breakdown(tmp_path / "ref.TSHS", tmp_path / "ref.TSHS", fdf, [_carbon()])
    assert exact["total_sum"] == 0 and exact["raw_sum"] == 0


def test_attribution_shares_add_up(tmp_path, rng):
    H = _bilayer()
    H.write(str(tmp_path / "ref.TSHS"))
    _noisy(H, rng).write(str(tmp_path / "pred.TSHS"))
    out = attribute(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", n_per_segment=6)
    assert sum(out[f"share_{c}"] for c in CLASSES) == pytest.approx(1.0)
    assert out["band_aligned_first_order_meV"] == pytest.approx(out["band_aligned_exact_meV"], rel=0.2)

    cf = counterfactuals(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", tmp_path / "ref.TSHS", n_per_segment=6)
    assert cf["cf_H0_only_meV"] == pytest.approx(0, abs=1e-6)  # H0 = reference here
    assert set(cf) == {f"cf_{n}_meV" for n in ("prediction", "inter_from_H0", "H0_only", "inter_exact")}
    assert cf["cf_prediction_meV"] > 0


def test_without_kb_tail_cuts_only_far_pairs():
    H = _bilayer()
    radii = [1.0] * H.geometry.na  # 2 Ang reach: keeps on-site and 1.42 Ang bonds only
    truncated, report = without_kb_tail(H, radii)
    assert report["n_removed_elements"] > 0
    before, after = H.tocsr(0).toarray(), truncated.tocsr(0).toarray()
    kept = before == after
    assert np.all(np.diag(before)[: H.no] == np.diag(after)[: H.no])
    assert not kept.all()
    np.testing.assert_array_equal(truncated.tocsr(truncated.S_idx).toarray(), H.tocsr(H.S_idx).toarray())

    # With a baseline equal to H, the tail of H - H0 is zero: nothing changes.
    same, report = without_kb_tail(H, radii, baseline=H)
    assert report["n_removed_elements"] == 0
    np.testing.assert_allclose(same.tocsr(0).toarray(), before, atol=1e-12)


def _polarized_pair(tmp_path):
    H = make_hamiltonian(sisl.geom.graphene(atoms=_carbon()), spin="polarized")
    pred = H.copy()
    pred._csr._D[:, 1] += 0.3  # error in the spin-down channel only
    H.write(str(tmp_path / "ref.TSHS"))
    pred.write(str(tmp_path / "pred.TSHS"))
    return H


def test_breakdown_covers_every_spin_channel(tmp_path):
    H = _polarized_pair(tmp_path)
    fdf = write_fdf(tmp_path / "s", H.geometry)
    row = structure_breakdown(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", fdf, [_carbon()])
    assert row["raw_sum"] > 0 and row["total_n"] == 2 * H.no**2


def test_attribution_rejects_spin_polarized_matrices(tmp_path):
    _polarized_pair(tmp_path)
    with pytest.raises(NotImplementedError, match="unpolarized"):
        attribute(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", n_per_segment=3)


def test_without_kb_tail_cuts_every_spin_channel():
    H = make_hamiltonian(sisl.geom.graphene(atoms=_carbon()), spin="polarized")
    truncated, report = without_kb_tail(H, [0.1] * H.geometry.na)  # every off-site pair is "outside"
    for spin in range(2):
        off = truncated.tocsr(spin).toarray()
        off[np.arange(H.no), np.arange(H.no)] = 0
        assert np.abs(off).max() == 0
    np.testing.assert_array_equal(truncated.tocsr(truncated.S_idx).toarray(), H.tocsr(H.S_idx).toarray())
    restored, _ = without_kb_tail(H, [0.1] * H.geometry.na, baseline=H)
    assert restored.spin.is_polarized
    for spin in range(2):
        np.testing.assert_allclose(restored.tocsr(spin).toarray(), H.tocsr(spin).toarray(), atol=1e-12)


def test_attribution_keeps_prediction_couplings_beyond_the_reference_supercell(tmp_path, rng):
    """Couplings the prediction has in images the reference does not reach are errors too."""
    H = _bilayer()
    H.write(str(tmp_path / "ref.TSHS"))
    pred = _noisy(H, rng)
    pred.set_nsc([5, 5, 1])
    far = pred.geometry.sc_index([2, 0, 0])
    pred[0, far * pred.no] = (0.05, 0.0)  # a coupling only the prediction has
    pred.write(str(tmp_path / "pred.TSHS"))
    with_far = attribute(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", n_per_segment=6)
    pred[0, far * pred.no] = (0.0, 0.0)
    pred.write(str(tmp_path / "pred.TSHS"))
    without = attribute(tmp_path / "pred.TSHS", tmp_path / "ref.TSHS", n_per_segment=6)
    assert with_far["band_aligned_exact_meV"] != pytest.approx(without["band_aligned_exact_meV"])
