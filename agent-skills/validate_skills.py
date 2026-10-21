#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "siesta-matrix-physics",
    "graph2mat-validation",
    "hamiltonian-derivatives",
    "electron-phonon-validation",
    "materials-dataset-design",
    "scientific-experiment-audit",
}

errors = []
for name in sorted(EXPECTED):
    p = ROOT / name / "SKILL.md"
    if not p.exists():
        errors.append(f"missing: {p}")
        continue
    txt = p.read_text(encoding="utf-8")
    if not txt.startswith("---\n"):
        errors.append(f"{name}: missing YAML frontmatter")
    if f"name: {name}" not in txt:
        errors.append(f"{name}: frontmatter name mismatch")
    if "description:" not in txt.split("---", 2)[1]:
        errors.append(f"{name}: missing description")
    if len(txt.splitlines()) > 500:
        errors.append(f"{name}: exceeds 500 lines")

if errors:
    print("Skill validation FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("Skill validation OK")
