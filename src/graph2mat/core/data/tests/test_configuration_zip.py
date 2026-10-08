"""Python 3.9 ZIP loading keeps related files, metadata and temporary ownership."""

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import sisl

from graph2mat import OrbitalConfiguration
from graph2mat.core.data import configuration as configuration_module


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("broken", [False, True])
def test_zip_run_without_native_sisl_zip_support(tmp_path, monkeypatch, labels, broken):
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]],
                             atoms=sisl.Atom(1, R=2), lattice=[10]*3)
    folder = tmp_path / "source"
    folder.mkdir()
    run = folder / "sample.fdf"
    geometry.write(run)
    run.write_text("SystemLabel sample\n" + run.read_text())
    if labels:
        matrix = sisl.DensityMatrix(geometry)
        matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0] = 2., 3., .25, .25
        matrix.write(folder / "sample.DM")
    kwargs = {"basis": geometry.atoms,
              "out_matrix": "density_matrix" if labels else None}
    expected = OrbitalConfiguration.from_run(run, **kwargs)
    filename = tmp_path / "dataset.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        for path in folder.iterdir():
            archive.write(path, "nested/" + path.name)

    native_get_sile = sisl.get_sile

    def without_native_zip(path, *args, **options):
        if isinstance(path, zipfile.Path):
            raise RuntimeError("sisl ZIP requires Python 3.10")
        if broken:
            raise ValueError("broken structure")
        return native_get_sile(path, *args, **options)

    temporaries = []

    def temporary_directory(*args, **options):
        temporary = tempfile.TemporaryDirectory(*args, dir=tmp_path, **options)
        temporaries.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(sisl, "get_sile", without_native_zip)
    monkeypatch.setattr(configuration_module, "version_info", (3, 9), raising=False)
    monkeypatch.setattr(configuration_module, "TemporaryDirectory", temporary_directory,
                        raising=False)
    with zipfile.ZipFile(filename) as archive:
        member = zipfile.Path(archive, "nested/sample.fdf")
        if broken:
            with pytest.raises(ValueError, match="broken structure"):
                OrbitalConfiguration.from_run(member, **kwargs)
        else:
            actual = OrbitalConfiguration.from_run(member, **kwargs)
            np.testing.assert_array_equal(actual.positions, expected.positions)
            np.testing.assert_array_equal(actual.cell, expected.cell)
            assert actual.metadata["path"] == str(member)
            if labels:
                for key, block in expected.matrix.block_dict.items():
                    np.testing.assert_array_equal(actual.matrix.block_dict[key], block)
        assert archive.read("nested/sample.fdf") == run.read_bytes()
    assert len(temporaries) == 1
    assert not temporaries[0].exists()


@pytest.mark.parametrize("name", ["../outside.XV", "/outside.XV"])
def test_legacy_zip_materialization_rejects_escaping_members(tmp_path, monkeypatch, name):
    filename = tmp_path / "dataset.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.writestr(name, "invalid member")
    monkeypatch.setattr(configuration_module, "version_info", (3, 9), raising=False)
    with zipfile.ZipFile(filename) as archive:
        member = zipfile.Path(archive, name)
        with pytest.raises(ValueError, match="Unsafe ZIP member"):
            OrbitalConfiguration.from_run(member)
        assert archive.read(name) == b"invalid member"
    assert list(tmp_path.iterdir()) == [filename]
