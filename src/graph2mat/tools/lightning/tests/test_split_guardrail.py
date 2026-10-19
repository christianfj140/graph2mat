"""Split alarms use explicit evidence, never repartition or parse lazy data."""

import json
from io import BytesIO
import logging
import warnings
import zipfile

import pytest
from torch.utils.data import Subset

from graph2mat.bindings.torch import TorchBasisMatrixDataset
from graph2mat.tools.lightning.tests.test_datamodule_split import _module, _sample_ids


def guard():
    from graph2mat.tools.lightning.split_guardrail import SplitGuardrail
    return SplitGuardrail()


class ReadTrap(TorchBasisMatrixDataset):
    def __getitem__(self, index):
        raise AssertionError("guardrail parsed a lazy matrix")


def inputs(*items):
    return ReadTrap(items, data_processor=None)


def test_source_alias_is_reported_without_reading_files(tmp_path, caplog):
    file = tmp_path / "a.fdf"
    file.write_text("original")
    alias = tmp_path / "alias.fdf"
    alias.symlink_to(file)
    checker = guard()
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING):
        warnings.simplefilter("error")
        report = checker.inspect({"train": inputs(file), "val": inputs(alias)}, tmp_path)
    assert report["status"] == "overlap_detected"
    assert any(x["kind"] == "source" and x["splits"] == ["train", "val"] for x in report["overlaps"])
    assert "split guardrail" in caplog.text.lower()
    assert file.read_text() == "original"


def test_declared_sample_hash_and_group_are_separate_evidence(tmp_path):
    report = guard().inspect({"train": inputs(tmp_path / "a"), "val": inputs(tmp_path / "b")}, tmp_path,
        {"a": {"sample_id": "same", "group_id": "trajectory-a", "sha256": "A"*64},
         "b": {"sample_id": "same", "group_id": "trajectory-a", "sha256": "a"*64}})
    assert {x["kind"] for x in report["overlaps"]} == {"sample_id", "group_id", "sha256"}
    assert report["group_coverage"] == {"train": {"known": 1, "total": 1}, "val": {"known": 1, "total": 1}}


def test_missing_groups_are_not_reported_as_contamination(tmp_path):
    report = guard().inspect({"train": inputs(tmp_path / "a"), "val": inputs(tmp_path / "b")}, tmp_path)
    assert report["status"] == "not_evaluable"
    assert report["overlaps"] == []


def test_disjoint_declared_groups_are_not_a_certificate_of_independence(tmp_path):
    report = guard().inspect({"train": inputs(tmp_path / "a"), "val": inputs(tmp_path / "b")}, tmp_path,
        {"a": {"group_id": "one"}, "b": {"group_id": "two"}})
    assert report["status"] == "no_known_overlap"
    assert report["independence_certified"] is False


def test_zip_source_identity_includes_archive_and_member(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("a/RUN.fdf", "same text")
        stream.writestr("b/RUN.fdf", "same text")
    with zipfile.ZipFile(archive) as one, zipfile.ZipFile(archive) as two:
        report = guard().inspect({"train": inputs(zipfile.Path(one, "a/RUN.fdf")),
            "val": inputs(zipfile.Path(two, "a/RUN.fdf")),
            "test": inputs(zipfile.Path(two, "b/RUN.fdf"))}, tmp_path)
    sources = [x for x in report["overlaps"] if x["kind"] == "source"]
    assert len(sources) == 1 and sources[0]["splits"] == ["train", "val"]


@pytest.mark.parametrize("cached", [False, True])
def test_real_setup_preserves_partition_and_repeated_warning_is_deduplicated(cached, caplog):
    module = _module(store_in_memory=cached)
    for index, config in enumerate(module.train_runs):
        config.metadata = {"split_guardrail": {"sample_id": f"frame{index}", "group_id": "MD1"}}
    with caplog.at_level(logging.WARNING):
        module.setup("fit")
        report = module.split_guardrail_report
        alarms = [r for r in caplog.records if "split guardrail" in r.message.lower()]
        assert alarms and report["status"] == "overlap_detected"
        assert any(x["kind"] == "group_id" for x in report["overlaps"])
        assert _sample_ids(module.val_dataset) == [8, 15]
        assert sorted(_sample_ids(module.train_dataset)+[8, 15]) == list(range(20))
        caplog.clear()
        module.setup("fit")
        assert module.split_guardrail_report == report
        assert not [r for r in caplog.records if "split guardrail" in r.message.lower()]
    module.teardown(None)


def test_subset_indices_are_respected_without_dataset_getitem(tmp_path):
    parent = inputs(tmp_path / "a", tmp_path / "b", tmp_path / "c")
    report = guard().inspect({"train": Subset(parent, [0, 2]), "val": Subset(parent, [1])}, tmp_path,
        {"a": {"group_id": "a"}, "b": {"group_id": "b"}, "c": {"group_id": "c"}})
    assert report["status"] == "no_known_overlap"
    assert report["group_coverage"]["train"] == {"known": 2, "total": 2}


def test_fit_does_not_load_test_and_predict_does_not_run_guardrail(caplog):
    module = _module(test_runs=["nonexistent-guardrail.fdf"])
    module.setup("fit")
    assert module.test_dataset is None
    assert "test" not in module.split_guardrail_report["group_coverage"]
    report = module.split_guardrail_report
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        module.setup("predict")
        assert not [r for r in caplog.records if "split guardrail" in r.message.lower()]
    assert module.split_guardrail_report == report
    module.teardown(None)


@pytest.mark.parametrize("metadata", [[1], {"a": ["wrong"]}, {"a": {"group_id": ["not-an-id"], "sha256": "short"}}])
def test_malformed_alarm_metadata_does_not_raise(tmp_path, metadata):
    report = guard().inspect({"train": inputs(tmp_path / "a"), "val": inputs(tmp_path / "b")}, tmp_path, metadata)
    assert report["status"] == "not_evaluable"
    assert report["metadata_issues"] > 0


def test_cached_duplicate_identity_remains_known_when_test_is_added():
    module = _module(store_in_memory=True)
    module.test_runs = [module.train_runs[0]]
    module.setup("fit")
    module.setup("test")
    assert any(x["kind"] == "source" and x["splits"] == ["train", "test"]
               for x in module.split_guardrail_report["overlaps"])
    module.teardown(None)


def test_runs_json_optional_annotations_are_consumed_without_changing_inputs(tmp_path):
    module = _module(count=1, root_dir=str(tmp_path), runs_json="runs.json")
    other = _module(count=1).train_runs[0]
    module.train_runs[0].metadata = {"path": str(tmp_path / "a.fdf")}
    other.metadata = {"path": str(tmp_path / "b.fdf")}
    module.val_runs = [other]
    (tmp_path / "runs.json").write_text(json.dumps({"split_metadata": {
        "a.fdf": {"sample_id": "a", "group_id": "MD1"},
        "b.fdf": {"sample_id": "b", "group_id": "MD1"}}}))
    module.setup("fit")
    assert {x["kind"] for x in module.split_guardrail_report["overlaps"]} == {"group_id"}
    assert _sample_ids(module.train_dataset) == [0] and _sample_ids(module.val_dataset) == [0]
    module.teardown(None)


def test_rotating_pool_reports_all_parent_samples_not_only_active_pool():
    module = _module(rotating_pool_size=2)
    for index, config in enumerate(module.train_runs):
        config.metadata = {"split_guardrail": {"group_id": f"frame{index}"}}
    module.setup("fit")
    assert module.split_guardrail_report["group_coverage"]["train"] == {"known": 18, "total": 18}
    assert module.split_guardrail_report["status"] == "no_known_overlap"
    module.setup("fit")
    assert module.split_guardrail_report["group_coverage"]["train"]["total"] == 18
    module.teardown(None)


def test_metadata_edit_is_rechecked_even_with_same_cached_inputs(tmp_path):
    checker = guard()
    datasets = {"train": inputs(tmp_path / "a"), "val": inputs(tmp_path / "b")}
    annotations = {"a": {"group_id": "one"}, "b": {"group_id": "two"}}
    assert checker.inspect(datasets, tmp_path, annotations)["status"] == "no_known_overlap"
    checker.remember(datasets)
    annotations["b"]["group_id"] = "one"
    assert checker.inspect(datasets, tmp_path, annotations)["status"] == "overlap_detected"


def test_memory_backed_zip_does_not_raise_or_close_reader(tmp_path):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        writer.writestr("a.fdf", "not read")
        writer.writestr("b.fdf", "not read")
    with zipfile.ZipFile(buffer) as reader:
        report = guard().inspect({"train": inputs(zipfile.Path(reader, "a.fdf")),
            "val": inputs(zipfile.Path(reader, "a.fdf")),
            "test": inputs(zipfile.Path(reader, "b.fdf"))}, tmp_path)
        sources = [entry for entry in report["overlaps"] if entry["kind"] == "source"]
        assert len(sources) == 1 and sources[0]["splits"] == ["train", "val"]
        assert reader.fp is not None


def test_symlink_lexical_relative_annotations_are_not_lost(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name).touch()
        (tmp_path / (name + "-alias")).symlink_to(tmp_path / name)
    report = guard().inspect({"train": inputs(tmp_path / "one-alias"),
        "val": inputs(tmp_path / "two-alias")}, tmp_path,
        {"one-alias": {"group_id": "MD1"}, "two-alias": {"group_id": "MD1"}})
    assert {entry["kind"] for entry in report["overlaps"]} == {"group_id"}


@pytest.mark.parametrize("copy_root", [False, True])
def test_original_absolute_annotations_survive_root_copy(tmp_path, copy_root, monkeypatch):
    module = _module(count=0, root_dir=str(tmp_path), runs_json="runs.json",
                     copy_root_to_tmp=copy_root)
    (tmp_path / "runs.json").write_text(json.dumps({"train": ["a.fdf"], "val": ["b.fdf"],
        "split_metadata": {str(tmp_path / "a.fdf"): {"group_id": "MD1"},
                           str(tmp_path / "b.fdf"): {"group_id": "MD1"}}}))
    module.train_runs = None
    # Stop the normal matrix-component probe, not the guardrail or path loader.
    monkeypatch.setattr(module, "_validate_n_matrix_components", lambda splits: None)
    try:
        module.setup("fit")
        assert {entry["kind"] for entry in module.split_guardrail_report["overlaps"]} == {"group_id"}
    finally:
        module.teardown(None)


def test_single_split_malformed_annotations_still_log_diagnostic(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        report = guard().inspect({"train": inputs(tmp_path / "a"), "val": inputs()},
                                 tmp_path, {"a": ["invalid"]})
    assert report["metadata_issues"] == 1
    assert "metadata issues 1" in caplog.text
