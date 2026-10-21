import numpy as np
import pytest
import sisl

from graph2mat.tools.conftest import make_hamiltonian, write_fdf
from graph2mat.tools.siesta import dm_pilot
from graph2mat.tools.siesta.label_noise import VARIANTS, gamma_mae_mev, noise_fdf, run_label_noise

FDF = """SystemLabel g
DM.Tolerance 1.d-4
%block AtomicCoordinatesAndAtomicSpecies
 0.0 0.0 0.0 1
 1.42 0.0 0.0 1
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def test_noise_fdf_shifts_coordinates_and_sets_the_tolerance():
    out = noise_fdf(FDF, True, "1.d-6", np.array([0.1, 0.2, 0.3])).splitlines()
    assert out[1] == "DM.Tolerance           1.d-6"
    assert [float(x) for x in out[3].split()[:3]] == [0.1, 0.2, 0.3]
    assert [float(x) for x in out[4].split()[:3]] == pytest.approx([1.52, 0.2, 0.3])
    assert out[0] == "SystemLabel g" and out[-1].startswith("%endblock")
    unshifted = noise_fdf(FDF.replace("DM.Tolerance 1.d-4\n", ""), False, "1.d-6")
    assert " 1.42 0.0 0.0 1" in unshifted and unshifted.endswith("DM.Tolerance           1.d-6\n")


def test_run_label_noise_with_fake_siesta(tmp_path, fake_siesta, monkeypatch):
    H = make_hamiltonian()
    ref = tmp_path / "ref"
    write_fdf(ref, H.geometry, label="g")
    H.write(str(ref / "g.TSHS"))
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(ref / "g.TSHS"))
    out = run_label_noise(ref, tmp_path / "noise", siesta=fake_siesta)
    assert set(out) == set(VARIANTS)
    assert "1.d-6" in (tmp_path / "noise" / "tight" / "RUN.fdf").read_text()
    assert gamma_mae_mev(out["shift"], ref / "g.TSHS") == 0


def _noise(H, rng):
    csr = H.tocsr(0).copy()
    csr.data = rng.normal(size=csr.nnz)
    return csr


def test_symmetric_and_keep_charges(rng):
    H = make_hamiltonian(sisl.geom.graphene(atoms=sisl.Atom(6, R=1.6)).tile(2, 0))
    g = H.geometry
    sym = dm_pilot.symmetric(g, _noise(H, rng))
    dm = sisl.DensityMatrix.fromsp(g, sym)
    np.testing.assert_allclose(dm.tocsr(0).toarray(), dm.transpose().tocsr(0).toarray(), atol=1e-14)

    S = H.tocsr(H.S_idx)
    kept = dm_pilot.keep_charges(sym, S, g)
    charge = np.add.reduceat(np.asarray(kept.multiply(S).sum(axis=1)).ravel(), g.firsto[:-1])
    np.testing.assert_allclose(charge, 0, atol=1e-12)


def test_pilot_structure_rows(tmp_path, monkeypatch):
    H = make_hamiltonian()
    ref = tmp_path / "ref"
    write_fdf(ref, H.geometry, label="g")
    H.write(str(ref / "g.TSHS"))
    dm = sisl.DensityMatrix.fromsp(H.geometry, H.tocsr(0) * 0.1 + 0.5 * sisl.DensityMatrix.fromsp(H.geometry, H.tocsr(H.S_idx)).tocsr(0), S=H.tocsr(H.S_idx))
    dm.write(str(ref / "g.DM"))
    make_hamiltonian(onsite=0.3).write(str(tmp_path / "h0.TSHS"))
    calls = []

    def fake_run(reference_dir, work_dir, dm=None, siesta=None):
        calls.append(work_dir.name)
        work_dir.mkdir(parents=True, exist_ok=True)
        H.write(str(work_dir / "g.TSHS"))
        return work_dir / "g.TSHS"

    monkeypatch.setattr(dm_pilot, "run_from_dm", fake_run)
    out = dm_pilot.pilot_structure(ref, tmp_path / "h0.TSHS", tmp_path / "work", sample_id="s0")
    assert out["sample"] == "s0" and len(out["rows"]) == 2 + 4 * 4 + 4 == len(calls)
    exact = out["rows"][0]
    assert exact["kind"] == "exact" and exact["dm_err"] == 0
    assert "vs_h0_meV" in out["rows"][1]
    assert [r["level"] for r in out["rows"] if r["kind"] == "t"] == [0.7, 0.9, 0.97, 0.99]


def test_pilot_runs_of_random_errors_are_named_by_seed(tmp_path, monkeypatch):
    """Audit M5: another seed reused the SIESTA runs (and thus H errors) of the first one."""
    H = make_hamiltonian()
    ref = tmp_path / "ref"
    write_fdf(ref, H.geometry, label="g")
    H.write(str(ref / "g.TSHS"))
    dm = sisl.DensityMatrix.fromsp(H.geometry, H.tocsr(0) * 0.1, S=H.tocsr(H.S_idx))
    dm.write(str(ref / "g.DM"))
    make_hamiltonian(onsite=0.3).write(str(tmp_path / "h0.TSHS"))
    names = []

    def fake_run(reference_dir, work_dir, dm=None, siesta=None):
        names.append(work_dir.name)
        work_dir.mkdir(parents=True, exist_ok=True)
        H.write(str(work_dir / "g.TSHS"))
        return work_dir / "g.TSHS"

    monkeypatch.setattr(dm_pilot, "run_from_dm", fake_run)
    for seed in (0, 1):
        dm_pilot.pilot_structure(ref, tmp_path / "h0.TSHS", tmp_path / "work", seed=seed)
    random_runs = [n for n in names if n.startswith(("rel_", "abs_", "relq_", "absq_"))]
    assert len(set(random_runs)) == len(random_runs) == 2 * 16
    assert names.count("exact_0.0") == 2  # deterministic runs keep their name (reused)
