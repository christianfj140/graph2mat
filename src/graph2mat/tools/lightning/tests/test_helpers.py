"""ZIP patterns must have the same character-class semantics as pathlib."""

import zipfile

import pytest

from graph2mat.tools.lightning._helpers import glob


@pytest.mark.parametrize("pattern", ["frame_[ab]/*.fdf", "frame_[!c]/*.fdf", "frame_[a-b]/*.fdf", "frame_[a&~|]/*.fdf"])
def test_zip_glob_character_classes(tmp_path, pattern):
    names = ["frame_a/run.fdf", "frame_b/run.fdf", "frame_c/run.fdf", "frame_&/run.fdf", "frame_~/run.fdf", "frame_|/run.fdf"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir()
        path.write_text(name)
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        for name in names:
            writer.write(tmp_path / name, name)

    expected = sorted(str(path.relative_to(tmp_path)) for path in glob(tmp_path, pattern))
    with zipfile.ZipFile(archive) as reader:
        actual = sorted(path.at for path in glob(zipfile.Path(reader), pattern))
    assert actual == expected

