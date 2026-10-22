"""Warning-only split evidence. Never read labels or certify independence."""

from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
import json
import logging
from pathlib import Path
import re
import zipfile

from torch.utils.data import Subset

from graph2mat.bindings.torch import InMemoryData, RotatingPoolData, TorchBasisMatrixDataset


def _items(dataset):
    if isinstance(dataset, RotatingPoolData):
        return _items(dataset.parent_data)  # Check all future training samples, not today's pool.
    if isinstance(dataset, Subset):
        parent = _items(dataset.dataset)
        return None if parent is None else [parent[i] for i in dataset.indices]
    if isinstance(dataset, TorchBasisMatrixDataset):
        return list(dataset.input_data)
    if isinstance(dataset, InMemoryData):
        return list(dataset.data_objects)
    return None  # Custom/streaming datasets cannot be inspected without consuming them.


def _record(item, root, declared, original_root=None):
    result = dict(source=("object", id(item)), metadata_issues=0)
    metadata = item.get("metadata") if isinstance(item, Mapping) else getattr(item, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    path = item if isinstance(item, (str, Path, zipfile.Path)) else metadata.get("path")
    keys = []
    if isinstance(path, (str, Path, zipfile.Path)):
        try:
            if isinstance(path, zipfile.Path) and path.root.filename is None:
                # A BytesIO archive has no filesystem name or safe str(Path). Python 3.9 wraps
                # the archive anew for every zipfile.Path: the shared buffer identifies it.
                result["source"] = ("zip-memory", id(path.root.fp), path.at)
            else:
                lexical = Path(str(path)).absolute()
                source = lexical.resolve()
                result["source"] = (("zip", str(Path(path.root.filename).resolve()), path.at)
                                    if isinstance(path, zipfile.Path) else ("file", str(source)))
                for candidate, base in ((lexical, Path(str(root)).absolute()),
                                        (source, Path(str(root)).resolve())):
                    try:
                        relative = candidate.relative_to(base)
                    except ValueError:
                        continue
                    keys.append(str(relative))
                    if original_root is not None:
                        keys.append(str(Path(str(original_root)).absolute() / relative))
                keys.extend((str(lexical), str(source), str(path)))
        except (OSError, RuntimeError, TypeError, ValueError):
            result["metadata_issues"] += 1
    if declared is None:
        declared = {}
    if not isinstance(declared, Mapping):
        result["metadata_issues"] += 1
        declared = {}
    fields = next((declared[key] for key in keys if key in declared), {})
    inline = metadata.get("split_guardrail", {})
    for values in (fields, inline):
        if not isinstance(values, Mapping):
            result["metadata_issues"] += 1
            continue
        for field in ("sample_id", "group_id", "sha256"):
            if field not in values:
                continue
            value = values[field]
            if (not isinstance(value, str) or not value.strip()
                    or (field == "sha256" and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None)):
                result["metadata_issues"] += 1
                result.pop(field, None)
            else:
                result[field] = value.lower() if field == "sha256" else value
    return result


def _summary(records):
    report = dict(status="not_comparable", independence_certified=False, overlaps=[], group_coverage={},
                  metadata_issues=0, unavailable_splits=[])
    present = {}
    for split, rows in records.items():
        if rows is None:
            report["unavailable_splits"].append(split)
            continue
        report["group_coverage"][split] = dict(known=sum("group_id" in row for row in rows), total=len(rows))
        report["metadata_issues"] += sum(row["metadata_issues"] for row in rows)
        if rows:
            present[split] = rows
    if len(present) < 2:
        if report["unavailable_splits"]:
            report["status"] = "not_evaluable"
        return report
    for field in ("source", "sample_id", "sha256", "group_id"):
        index = defaultdict(set)
        for split, rows in present.items():
            for row in rows:
                if field in row:
                    index[row[field]].add(split)
        pairs = defaultdict(list)
        for token, splits in index.items():
            for pair in combinations(present, 2):
                if set(pair).issubset(splits):
                    pairs[pair].append(token)
        for pair, shared in pairs.items():
            examples = ["<same in-memory input>" if field == "source" and token[0] in ("object", "zip-memory")
                        else repr(token)[:100] for token in shared[:3]]
            report["overlaps"].append(dict(kind=field, splits=list(pair), count=len(shared), examples=examples))
    complete_groups = all(count["known"] == count["total"] for count in report["group_coverage"].values())
    report["status"] = ("overlap_detected" if report["overlaps"] else "no_known_overlap"
        if complete_groups and not report["unavailable_splits"] and not report["metadata_issues"] else "not_evaluable")
    return report


class SplitGuardrail:
    """Keep raw identities through cache/pool wrapping and deduplicate log alarms."""

    def __init__(self):
        self._cache = {}
        self._current = {}
        self._logged = None

    def inspect(self, datasets, root, declared=None, original_root=None):
        self._current = {}
        records = {}
        for split, dataset in datasets.items():
            if dataset is None:
                continue
            cached = self._cache.get(split)
            items = cached[1] if cached is not None and cached[0] == id(dataset) else _items(dataset)
            self._current[split] = items
            records[split] = None if items is None else [
                _record(item, root, declared, original_root) for item in items]
        report = _summary(records)
        signature = json.dumps(report, sort_keys=True)
        if signature != self._logged:
            for overlap in report["overlaps"]:
                logging.warning("Split guardrail: shared %s between %s (%s shared identities; examples %s). "
                    "Alarm only; partitions unchanged.", overlap["kind"], "/".join(overlap["splits"]),
                    overlap["count"], overlap["examples"])
            missing = any(c["known"] < c["total"] for c in report["group_coverage"].values())
            if (report["metadata_issues"] or (report["status"] != "not_comparable"
                    and (missing or report["unavailable_splits"]))):
                logging.warning("Split guardrail: independence not evaluable from available group metadata "
                    "(coverage %s; metadata issues %s; unavailable splits %s). This is not proof of leakage; "
                    "no geometric similarity threshold is applied. Partitions unchanged.",
                    report["group_coverage"], report["metadata_issues"], report["unavailable_splits"])
            self._logged = signature
        return report

    def remember(self, datasets):
        self._cache = {split: (id(datasets[split]), items) for split, items in self._current.items()
                       if datasets.get(split) is not None}
