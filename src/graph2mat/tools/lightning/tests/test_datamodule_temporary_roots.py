"""Independent temporary roots that remain usable between Lightning stages."""

import multiprocessing
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import sisl

from graph2mat import BasisTableWithEdges, PointBasis
from graph2mat.tools.lightning import MatrixDataModule


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=[1, 1], lattice=[10]*3).write(
        source / "sample.XV"
    )
    return source


def _module(root, copy_root_to_tmp=True):
    return MatrixDataModule(
        root_dir=str(root), basis_table=BasisTableWithEdges([PointBasis(1, R=2, basis=[1])]),
        predict_structs="*.XV", copy_root_to_tmp=copy_root_to_tmp, sub_point_matrix=False,
        store_in_memory=False, loader_threads=0,
    )


def _finish(module):
    module.teardown(None)
    if isinstance(module.root_dir, zipfile.Path):
        module.root_dir.root.close()


def test_two_modules_with_same_source_own_independent_temporary_roots(tmp_path):
    source = _source(tmp_path)
    first, second = _module(source), _module(source)
    try:
        first.prepare_data()
        second.prepare_data()
        assert first.tmp_dir != second.tmp_dir
        second_path = second.tmp_dir
        first_path = first.tmp_dir
        first.teardown(None)
        assert not first_path.exists()
        assert second_path.exists()
        second.setup("predict")
        assert second.predict_dataset[0].num_nodes == 2
        assert (source / "sample.XV").exists()
    finally:
        _finish(first)
        _finish(second)


def test_temporary_root_survives_stage_teardown_and_reuses_data(tmp_path):
    module = _module(_source(tmp_path))
    try:
        module.prepare_data()
        temporary = module.tmp_dir
        module.prepare_data()
        assert module.tmp_dir == temporary
        module.setup("predict")
        dataset = module.predict_dataset
        assert dataset[0].num_nodes == 2
        module.teardown("predict")
        assert temporary.exists()
        module.setup("predict")
        assert module.predict_dataset is dataset
        assert dataset[0].num_nodes == 2
        module.teardown(None)
        assert not temporary.exists()
    finally:
        _finish(module)


@pytest.mark.parametrize("prefix", ["", "nested/"])
def test_copy_zip_root_preserves_members_and_subdirectory(tmp_path, prefix):
    source = _source(tmp_path)
    filename = tmp_path / "dataset.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.write(source / "sample.XV", prefix + "sample.XV")
    module = _module(str(filename) + ("/" + prefix if prefix else ""))
    try:
        module.prepare_data()
        module.setup("predict")
        assert module.predict_dataset[0].num_nodes == 2
        # The default spherical convention stores Cartesian (x,y,z) as (y,z,x).
        np.testing.assert_allclose(module.predict_dataset[0].positions.numpy()[1], [0., 0., 1.])
        temporary = module.tmp_dir
        member = module.predict_dataset.input_data[0]
        assert isinstance(member, zipfile.Path)
        assert member.root.filename != str(filename)
        module.teardown("predict")
        assert module.predict_dataset[0].num_nodes == 2
        module.teardown(None)
        assert not temporary.exists()
        assert filename.exists()
    finally:
        _finish(module)


def test_final_teardown_closes_source_zip_and_allows_reuse(tmp_path):
    source = _source(tmp_path)
    filename = tmp_path / "dataset.zip"
    with zipfile.ZipFile(filename, "w") as archive:
        archive.write(source / "sample.XV", "sample.XV")
    module = _module(filename, copy_root_to_tmp=False)
    try:
        module.setup("predict")
        assert module.predict_dataset[0].num_nodes == 2
        source_fd = module.root_dir.root.fp.fileno()
        module.teardown(None)
        with pytest.raises(OSError):
            os.fstat(source_fd)
        module.setup("predict")
        assert module.predict_dataset[0].num_nodes == 2
    finally:
        _finish(module)


def _predict_in_child(module, connection):
    try:
        module.setup("predict")
        temporary = module.tmp_dir
        result = {
            "temporary": str(temporary),
            "member": str(module.predict_dataset.input_data[0]),
            "nodes": module.predict_dataset[0].num_nodes,
        }
        module.teardown(None)
        result["removed"] = not temporary.exists()
        connection.send(result)
    except Exception as error:
        connection.send({"error": repr(error)})
    finally:
        connection.close()


def test_fork_uses_own_temporary_root_even_after_parent_setup(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This regression targets inherited temporary roots after fork")
    module = _module(_source(tmp_path))
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_predict_in_child, args=(module, sender))
    try:
        module.setup("predict")
        parent_temporary = module.tmp_dir
        assert module.predict_dataset[0].num_nodes == 2
        process.start()
        sender.close()
        assert receiver.poll(10), "Temporary-root child did not respond"
        result = receiver.recv()
        process.join(10)
        assert process.exitcode == 0
        assert "error" not in result, result
        assert result["nodes"] == 2 and result["removed"]
        assert result["temporary"] != str(parent_temporary)
        assert Path(result["member"]).parent == Path(result["temporary"])
        assert parent_temporary.exists()
        assert module.predict_dataset[0].num_nodes == 2
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()
        sender.close()
        _finish(module)


@pytest.mark.parametrize("archive", [False, True])
def test_process_exit_removes_owned_temporary_root(tmp_path, archive):
    source = _source(tmp_path)
    root = source
    if archive:
        root = tmp_path / "dataset.zip"
        with zipfile.ZipFile(root, "w") as output:
            output.write(source / "sample.XV", "sample.XV")
    command = (
        "import json, sys; from graph2mat.tools.lightning import MatrixDataModule; "
        "module = MatrixDataModule(root_dir=sys.argv[1], copy_root_to_tmp=True); "
        "module.prepare_data(); print(json.dumps(str(module.tmp_dir)))"
    )
    result = subprocess.run([sys.executable, "-c", command, str(root)],
                            capture_output=True, text=True, timeout=30, check=True)
    temporary = Path(json.loads(result.stdout))
    assert not temporary.exists()
    assert root.exists() and (source / "sample.XV").exists()
