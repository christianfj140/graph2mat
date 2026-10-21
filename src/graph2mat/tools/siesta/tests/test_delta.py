import numpy as np
import pytest
import sisl

from graph2mat.tools.conftest import make_hamiltonian, write_fdf
from graph2mat.tools.siesta.delta import (
    H0_KEYS,
    add_h0,
    combine_hamiltonians,
    compute_h0,
    h0_fdf,
    stage_h0_overlap,
    write_delta_sample,
)


def _read(path):
    return sisl.get_sile(str(path)).read_hamiltonian()


def _sample(tmp_path, spin="unpolarized"):
    H = make_hamiltonian(spin=spin)
    H0 = make_hamiltonian(onsite=-0.3, hopping=-2.1, spin=spin)
    ref_dir = tmp_path / "sample"
    write_fdf(ref_dir, H.geometry)
    H.write(str(ref_dir / "sample.TSHS"))
    H0.write(str(tmp_path / "h0.TSHS"))
    return H, H0, ref_dir


def test_h0_fdf_sets_every_key_once():
    out = h0_fdf("SystemLabel x\nMaxSCFIterations 200\nDM.Tolerance 1.d-4\n")
    assert out.startswith("SystemLabel x\n")
    for key, value in H0_KEYS.items():
        assert out.count(f"{key}  {value}\n") == 1
    assert "200" not in out and "1.d-4" not in out


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_delta_label_round_trip(tmp_path, spin):
    H, H0, ref_dir = _sample(tmp_path, spin)
    label = write_delta_sample(ref_dir, ref_dir / "sample.TSHS", tmp_path / "h0.TSHS", tmp_path / "delta")
    assert label == tmp_path / "delta" / "sample.TSHS"
    assert (tmp_path / "delta" / "RUN.fdf").read_text() == (ref_dir / "RUN.fdf").read_text()

    delta = _read(label)
    assert delta.spin.kind == H.spin.kind
    for i in range(H.spin.size(H.dtype)):
        np.testing.assert_allclose((delta.tocsr(i) - (H.tocsr(i) - H0.tocsr(i))).toarray(), 0, atol=1e-12)
    np.testing.assert_allclose((delta.tocsr(delta.S_idx) - H.tocsr(H.S_idx)).toarray(), 0, atol=1e-12)

    restored = _read(add_h0(label, tmp_path / "h0.TSHS", tmp_path / "restored.TSHS"))
    for i in range(H.spin.size(H.dtype)):
        np.testing.assert_allclose((restored.tocsr(i) - H.tocsr(i)).toarray(), 0, atol=1e-12)


def test_add_h0_in_place_keeps_prediction_overlap(tmp_path):
    make_hamiltonian(overlap=0.0).write(str(tmp_path / "ML_prediction.TSHS"))  # h_only: S = 1
    pred = _read(tmp_path / "ML_prediction.TSHS")
    make_hamiltonian(onsite=1.0).write(str(tmp_path / "h0.TSHS"))
    out = add_h0(tmp_path / "ML_prediction.TSHS", tmp_path / "h0.TSHS")
    assert out == tmp_path / "ML_prediction.TSHS"
    result = _read(out)
    np.testing.assert_allclose(result.Sk().toarray(), pred.Sk().toarray())
    np.testing.assert_allclose(
        result.Hk().toarray(), pred.Hk().toarray() + make_hamiltonian(onsite=1.0).Hk().toarray(), atol=1e-12
    )


def test_combine_uses_the_larger_supercell():
    a = make_hamiltonian()
    b = make_hamiltonian(onsite=1.0)
    b.set_nsc([5, 5, 1])
    out = combine_hamiltonians(a, b, -1.0)
    assert np.array_equal(out.nsc, [5, 5, 1])
    np.testing.assert_allclose(out.Hk().toarray(), a.Hk().toarray() - make_hamiltonian(onsite=1.0).Hk().toarray(), atol=1e-12)


def test_combine_refuses_different_spins():
    with pytest.raises(ValueError, match="spin"):
        combine_hamiltonians(make_hamiltonian(), make_hamiltonian(spin="polarized"), -1.0)


def test_compute_h0_runs_siesta_once_and_checks_geometry(tmp_path, fake_siesta, monkeypatch):
    H, H0, ref_dir = _sample(tmp_path)
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(tmp_path / "h0.TSHS"))
    out = compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta)
    assert out == tmp_path / "h0run" / "sample.TSHS"
    assert "MaxSCFIterations  1" in (tmp_path / "h0run" / "RUN.fdf").read_text()
    assert (tmp_path / "h0run" / "C.psf").exists()
    # Cached: SIESTA is not run again.
    monkeypatch.setenv("FAKE_SIESTA_FAIL", "1")
    assert compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta) == out


def test_compute_h0_rejects_moved_atoms(tmp_path, fake_siesta, monkeypatch):
    H, H0, ref_dir = _sample(tmp_path)
    moved = H0.copy()
    moved.geometry.xyz[0, 0] += 0.01
    moved.write(str(tmp_path / "moved.TSHS"))
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(tmp_path / "moved.TSHS"))
    with pytest.raises(RuntimeError, match="geometry differs"):
        compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta)
    assert not (tmp_path / "h0run" / "sample.TSHS").exists()


def test_compute_h0_without_output(tmp_path, fake_siesta):
    _, _, ref_dir = _sample(tmp_path)
    with pytest.raises(RuntimeError, match="SIESTA wrote no sample.TSHS"):
        compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta)


def test_stage_h0_overlap(tmp_path):
    _, _, ref_dir = _sample(tmp_path)
    structure = tmp_path / "predict" / "0"
    write_fdf(structure, make_hamiltonian().geometry, label="frame")
    dest = stage_h0_overlap(tmp_path / "h0.TSHS", structure / "RUN.fdf")
    assert dest == structure / "frame.TSHS" and dest.exists()
    with pytest.raises(FileExistsError):
        stage_h0_overlap(tmp_path / "h0.TSHS", structure / "RUN.fdf")


def test_delta_cli(tmp_path, fake_siesta, monkeypatch):
    from typer.testing import CliRunner

    from graph2mat.tools.cli.cli import app

    H, H0, ref_dir = _sample(tmp_path)
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(tmp_path / "h0.TSHS"))
    runner = CliRunner()
    result = runner.invoke(app, ["siesta", "delta", "h0", str(ref_dir), "--out-root", str(tmp_path / "h0s"), "--siesta", fake_siesta])
    assert result.exit_code == 0, result.output
    (h0,) = (tmp_path / "h0s").glob("sample-*/sample.TSHS")
    result = runner.invoke(app, ["siesta", "delta", "label", str(ref_dir), str(h0), str(tmp_path / "delta")])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["siesta", "delta", "add-h0", str(tmp_path / "delta" / "sample.TSHS"), str(h0),
                                 "--out", str(tmp_path / "back.TSHS")])
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose((_read(tmp_path / "back.TSHS").tocsr(0) - H.tocsr(0)).toarray(), 0, atol=1e-12)


def test_add_h0_keeps_h0_couplings_beyond_the_prediction_supercell(tmp_path):
    """SIESTA's H0 reaches more periodic images than a model's prediction: none is dropped."""
    atom = sisl.Atom(6, R=3.0)
    chain = lambda nsc: sisl.Geometry([[0.0, 0.0, 0.0]], atoms=atom, lattice=sisl.Lattice([1.5, 20.0, 20.0], nsc=nsc))
    h0 = sisl.Hamiltonian(chain([5, 1, 1]), orthogonal=False)
    h0.construct([(0.1, 1.6, 3.1), ((0.5, 1.0), (-2.0, 0.2), (-0.3, 0.05))])  # second neighbours in images +-2
    pred = sisl.Hamiltonian(chain([3, 1, 1]), orthogonal=False)
    pred.construct([(0.1, 1.6), ((0.1, 1.0), (-0.4, 0.2))])
    h0.write(str(tmp_path / "h0.TSHS"))
    pred.write(str(tmp_path / "pred.TSHS"))
    out = _read(add_h0(tmp_path / "pred.TSHS", tmp_path / "h0.TSHS", tmp_path / "out.TSHS"))
    k = [0.23, 0.0, 0.0]
    np.testing.assert_allclose(out.Hk(k=k).toarray(), pred.Hk(k=k).toarray() + h0.Hk(k=k).toarray(), atol=1e-12)


def test_delta_cli_keeps_runs_with_the_same_directory_name_apart(tmp_path, fake_siesta, monkeypatch):
    """Audit G5: A/run and B/run used to share one H0 directory, and "." emptied OUT_ROOT."""
    from typer.testing import CliRunner

    from graph2mat.tools.cli.cli import app

    runs = []
    for parent, onsite in (("A", 0.1), ("B", 0.4)):
        H = make_hamiltonian(onsite=onsite)
        run = write_fdf(tmp_path / parent / "run", H.geometry)
        H.write(str(run.parent / "sample.TSHS"))
        runs.append(run.parent)
    make_hamiltonian(onsite=-0.3).write(str(tmp_path / "h0.TSHS"))
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(tmp_path / "h0.TSHS"))
    out_root = tmp_path / "h0s"
    keep = out_root / "unrelated.txt"
    out_root.mkdir()
    keep.write_text("not an H0")
    result = CliRunner().invoke(app, ["siesta", "delta", "h0", *map(str, runs), "--out-root", str(out_root),
                                      "--siesta", fake_siesta])
    assert result.exit_code == 0, result.output
    assert len(list(out_root.glob("run-*/sample.TSHS"))) == 2
    monkeypatch.chdir(runs[0])
    result = CliRunner().invoke(app, ["siesta", "delta", "h0", ".", "--out-root", str(out_root),
                                      "--siesta", fake_siesta])
    assert result.exit_code == 0, result.output
    assert keep.read_text() == "not an H0"


def test_compute_h0_never_reuses_the_output_of_a_failed_siesta(tmp_path, fake_siesta, monkeypatch):
    """Audit G6: SIESTA writing the TSHS and then failing left a file that was reused."""
    import subprocess

    _, _, ref_dir = _sample(tmp_path)
    moved = make_hamiltonian(onsite=-0.3)
    moved.geometry.xyz[0, 0] += 0.5
    moved.write(str(tmp_path / "moved.TSHS"))
    monkeypatch.setenv("FAKE_SIESTA_TSHS", str(tmp_path / "moved.TSHS"))
    monkeypatch.setenv("FAKE_SIESTA_FAIL_AFTER", "1")
    with pytest.raises(subprocess.CalledProcessError):
        compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta)
    assert not (tmp_path / "h0run").exists()
    monkeypatch.delenv("FAKE_SIESTA_FAIL_AFTER")
    with pytest.raises(RuntimeError, match="geometry differs"):
        compute_h0(ref_dir, tmp_path / "h0run", siesta=fake_siesta)


def test_a_reused_h0_is_checked_against_the_label_geometry(tmp_path):
    _, _, ref_dir = _sample(tmp_path)
    stale = make_hamiltonian(onsite=-0.3)
    stale.geometry.xyz[0, 0] += 0.5
    (tmp_path / "h0run").mkdir()
    stale.write(str(tmp_path / "h0run" / "sample.TSHS"))
    with pytest.raises(RuntimeError, match="geometry differs"):
        compute_h0(ref_dir, tmp_path / "h0run")
    assert (tmp_path / "h0run" / "sample.TSHS").exists()  # not ours to delete
