import subprocess

import pytest

from graph2mat.tools.siesta.runner import (
    find_pseudos,
    replace_fdf_keys,
    run_siesta,
    siesta_command,
    system_label,
)


def test_replace_fdf_keys_ignores_case_dots_dashes_and_underscores():
    text = "SystemLabel x\nDM.Tolerance 1e-4\ndm-tolerance 1e-5\nMaxSCFIterations 100\n"
    out = replace_fdf_keys(text, {"DM.Tolerance": "100", "Max.SCF.Iterations": "1"})
    assert out == "SystemLabel x\nDM.Tolerance  100\nMax.SCF.Iterations  1\n"


def test_system_label(tmp_path):
    (tmp_path / "a.fdf").write_text("  systemlabel  graphene\n")
    (tmp_path / "b.fdf").write_text("NumberOfAtoms 2\n")
    assert system_label(tmp_path / "a.fdf") == "graphene"
    assert system_label(tmp_path / "b.fdf") == "siesta"


def test_find_pseudos_in_a_parent_directory(tmp_path):
    (tmp_path / "C.psf").write_text("")
    (tmp_path / "B.psml").write_text("")
    run = tmp_path / "runs" / "0"
    run.mkdir(parents=True)
    assert [p.name for p in find_pseudos(run)] == ["B.psml", "C.psf"]
    with pytest.raises(RuntimeError, match="no pseudopotential"):
        find_pseudos(tmp_path.parent / "nowhere_with_pseudos_xyz")


def test_siesta_command_lookup(monkeypatch):
    monkeypatch.setenv("SIESTA", "mpirun -np 2 siesta")
    assert siesta_command() == ["mpirun", "-np", "2", "siesta"]
    assert siesta_command("/opt/siesta") == ["/opt/siesta"]
    monkeypatch.delenv("SIESTA")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError, match="SIESTA executable not found"):
        siesta_command()


def test_run_siesta_writes_inputs_and_output(tmp_path, fake_siesta, monkeypatch):
    pseudo = tmp_path / "C.psf"
    pseudo.write_text("p")
    extra = tmp_path / "old.DM"
    extra.write_text("dm")
    work = tmp_path / "work"
    (work / "stale").mkdir(parents=True)  # leftovers of a previous run in the same directory
    (work / "RUN.fdf").write_text("SystemLabel old\n")
    run_siesta(work, "SystemLabel g\n", [pseudo], siesta=fake_siesta, extra_files={"g.DM": extra})
    assert sorted(p.name for p in work.iterdir()) == ["C.psf", "RUN.fdf", "RUN.out", "g.DM"]
    assert (work / "RUN.out").read_text() == "fake siesta\n"

    monkeypatch.setenv("FAKE_SIESTA_FAIL", "1")
    with pytest.raises(subprocess.CalledProcessError):
        run_siesta(work, "SystemLabel g\n", [], siesta=fake_siesta)


def test_run_siesta_refuses_to_empty_an_unrelated_directory(tmp_path, fake_siesta):
    (tmp_path / "data.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not a previous SIESTA run"):
        run_siesta(tmp_path, "SystemLabel g\n", [], siesta=fake_siesta)
    assert (tmp_path / "data.txt").read_text() == "keep"


def test_system_label_follows_fdf_label_rules(tmp_path):
    (tmp_path / "a.fdf").write_text("System.Label graphene\n")
    (tmp_path / "b.fdf").write_text("system_label  bn\n")
    assert system_label(tmp_path / "a.fdf") == "graphene" and system_label(tmp_path / "b.fdf") == "bn"
