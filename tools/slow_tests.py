"""Regenerate ``src/graph2mat/_slow_tests.py`` from a JUnit report of the full suite.

    pytest --pyargs graph2mat tests --junitxml=junit.xml
    python tools/slow_tests.py junit.xml [threshold_seconds]

Tests above the threshold (default 1 s) get the ``slow`` marker, so that
``pytest -m "not slow"`` gives a quick check and the full suite stays in CI.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(junit, threshold=1.0):
    keys = set()
    for case in ET.parse(junit).iter("testcase"):
        if float(case.get("time", 0)) <= threshold:
            continue
        module = case.get("classname")
        if module.startswith("src.graph2mat."):
            module = module[len("src.graph2mat."):]
        keys.add(module.replace(".", "/") + ".py::" + case.get("name"))
    out = Path(__file__).resolve().parents[1] / "src" / "graph2mat" / "_slow_tests.py"
    body = "".join(f"    {key!r},\n" for key in sorted(keys))
    out.write_text(
        f'"""Tests above {threshold:g} s in the last timed full run (``tools/slow_tests.py``)."""\n'
        "from pathlib import Path\n\n"
        f"SLOW = frozenset((\n{body}))\n\n\n"
        "def mark_slow(items, base, prefix=\"\"):\n"
        "    \"\"\"Add the ``slow`` marker to the collected items listed in ``SLOW``.\"\"\"\n"
        "    import pytest\n\n"
        "    for item in items:\n"
        "        try:\n"
        "            relative = Path(item.path).resolve().relative_to(Path(base).resolve())\n"
        "        except ValueError:\n"
        "            continue\n"
        "        if f\"{prefix}{relative.as_posix()}::{item.name}\" in SLOW:\n"
        "            item.add_marker(pytest.mark.slow)\n"
    )
    print(f"{len(keys)} slow tests -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], *map(float, sys.argv[2:]))
