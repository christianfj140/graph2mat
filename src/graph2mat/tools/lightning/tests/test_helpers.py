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



def test_glob_order_is_sorted_and_the_same_for_directories_and_zips(tmp_path):
    """The automatic train/val split shuffles the glob result: its order must not depend on
    how the filesystem or the archive lists the files (audit G3)."""
    import random

    names = [f"run{i:02d}/RUN.fdf" for i in range(30)]
    shuffled = random.Random(0).sample(names, len(names))
    for name in shuffled:  # creation order unrelated to the names
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True)
        path.write_text(name)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        for name in shuffled:  # archive order unrelated to the names either
            writer.write(tmp_path / "data" / name, name)

    in_dir = [str(p.relative_to(tmp_path / "data")) for p in glob(tmp_path / "data", "*/RUN.fdf")]
    with zipfile.ZipFile(archive) as reader:
        in_zip = [p.at for p in glob(zipfile.Path(reader), "*/RUN.fdf")]
    assert in_dir == sorted(names) and in_zip == sorted(names)
