# Graph2Mat core without Torch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. The user approved direct execution and working in place on `g2m-v2` on 2026-10-07.

**Goal:** Make importing Graph2Mat and processing NumPy matrix labels work without installing PyTorch.

**Architecture:** Keep the existing core/bindings split and public interfaces. Make the two unconditional core Torch imports optional, and guard existing NumPy/Torch dispatch without rewriting losses or conversion code. Verify the base installation in a separate CPU CI job.

**Tech Stack:** Python, NumPy, SciPy, ASE, sisl, pytest; the existing Torch stack remains optional.

**Spec:** `auditorias_mejoras/roadmap.md`, item 3.5, import-only subtask; `auditorias_mejoras/claude_opus5.5ultracode.txt`, ENG_QUALITY-12; framework-independent core contract in `src/graph2mat/bindings/__init__.py`.

## Global Constraints

- Repository: `/home/christian/repositorios/graph2mat`; preserve unrelated untracked files.
- Preserve `requires-python = ">=3.9"` and all public names and signatures.
- No new dependency, dependency upgrade, training run, checkpoint-format change, or phonon work.
- Keep Torch gradients, loss definitions, and the existing full CPU suite unchanged.
- CLI dependencies, dependency version ranges, DataModule, and other roadmap items are separate deliverables. Do not mark all of 3.5 complete after this subtask.
- Do not alter the shared training environment or the running SiO2/autorun processes.

## Review Focus

- A real base-only installation must work, not just a simulated missing import: covered by the minimal CI job.
- Torch present but torch_geometric absent must preserve the existing core fallback: covered by the second subprocess mode.
- Missing internal dependencies of an installed Torch must propagate, not be mistaken for absent Torch: covered by the broken-Torch mode.
- NumPy relative errors with zero references and multicomponent labels with NaN edges must retain their semantics: covered by the subprocess smoke.
- Installed-Torch training and gradients must remain unchanged: covered by the full existing test suite before committing.

## Task 1: Optional core Torch imports and their regression gate

**Files:**

- Modify: `src/graph2mat/core/data/metrics.py` — optional import and tensor detection.
- Modify: `src/graph2mat/core/data/processing.py` — optional import; retain local imports in Torch-specific operations.
- Create: `tests/test_minimal_install.py` — subprocess tests outside the package's Torch-dependent `conftest.py`.
- Modify: `.github/workflows/tests.yaml` — separate base-only job; keep the existing full-stack job.
- Modify: `auditorias_mejoras/roadmap.md` — record the import subtask as complete, leaving the remaining parts of 3.5 open.

**Interfaces:**

- Consumes: `BasisTableWithEdges`, `PointBasis`, `BasisConfiguration`, `MatrixDataProcessor`, `BasisMatrixData.new(config, processor)` and existing NumPy metrics.
- Produces: the same public interfaces, now usable without Torch; no new public API.

- [x] **Step 1: Add this subprocess regression test.**

```python
import subprocess
import sys

import pytest


SMOKE = r'''
import importlib.abc
import sys

mode = sys.argv[1]
blocked = {"torch_geometric"}
if mode != "torch_only":
    blocked |= {"torch", "e3nn", "mace", "pytorch_lightning"}

class MissingML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked:
            name = "sympy" if mode == "broken_torch" and root == "torch" else root
            raise ModuleNotFoundError("simulated missing dependency: " + name, name=name)

sys.meta_path.insert(0, MissingML())

import numpy as np
import graph2mat as g
from graph2mat.core.data.metrics import elementwise_mse, block_type_mape

basis = g.BasisTableWithEdges([
    g.PointBasis("A", R=2, basis=[1], basis_convention="cartesian")
])
processor = g.MatrixDataProcessor(
    basis_table=basis, symmetric_matrix=True,
    sub_point_matrix=False, node_attr_getters=[],
)
config = g.BasisConfiguration(
    point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
    basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
    matrix=np.array([[1., .25], [.25, 1.]]),
)
data = g.BasisMatrixData.new(config, processor)
assert len(data.point_labels) == 2 and len(data.edge_labels) == 1
loss, _ = elementwise_mse(
    nodes_pred=data.point_labels+1, nodes_ref=data.point_labels,
    edges_pred=data.edge_labels+1, edges_ref=data.edge_labels,
)
assert np.isclose(loss, 1)
loss, _ = block_type_mape(
    nodes_pred=np.array([0., 1.]), nodes_ref=np.array([0., 1.]),
    edges_pred=np.ones(1), edges_ref=np.ones(1),
)
assert loss == 0
loss, _ = elementwise_mse(
    nodes_pred=np.ones((2, 2)), nodes_ref=np.zeros((2, 2)),
    edges_pred=np.array([[1., 1.], [99., 99.]]),
    edges_ref=np.array([[0., 0.], [np.nan, np.nan]]),
)
assert np.isclose(loss, 1)
'''


@pytest.mark.parametrize("mode", ["base", "torch_only", "broken_torch"])
def test_core_optional_torch(mode):
    result = subprocess.run(
        [sys.executable, "-c", SMOKE, mode], capture_output=True, text=True,
    )
    if mode == "broken_torch":
        assert result.returncode != 0
        assert "simulated missing dependency: sympy" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
```

- [x] **Step 2: Run the focused test before changing source.**

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/pytest -q tests/test_minimal_install.py
```

Expected: `base` fails at the unconditional Torch import in `metrics.py`; `torch_only` exercises the already-existing PyG fallback. The original no-Torch import failure was independently reproduced during planning.

- [x] **Step 3: Implement the optional imports and NumPy dispatch guard.**

In both core modules, replace the unconditional `import torch` with:

```python
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
```

In `metrics.py`, define the internal tensor detector before its first caller:

```python
def _is_torch_tensor(value):
    return torch is not None and torch.is_tensor(value)
```

Replace every other `torch.is_tensor(...)` call in that module with `_is_torch_tensor(...)`; retain `torch.is_tensor` inside the detector. This preserves every existing dispatch branch when Torch is installed. Keep all Torch-only calculations unchanged. In `processing.py`, `torch_predict` already imports Torch locally; add the same local import at the start of `irreps_from_data` so that requesting that Torch-specific operation still raises an import error when Torch is absent. Preserve the existing postponed annotations.

- [x] **Step 4: Add the base-only CI gate under `jobs`.**

```yaml
  pytest-minimal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install base package only
        run: python -m pip install . pytest
      - name: Verify base profile
        env:
          OMP_NUM_THREADS: "2"
          OPENBLAS_NUM_THREADS: "2"
        run: |
          python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"
          python -c "import graph2mat; assert graph2mat.MatrixDataProcessor"
          pytest -q tests/test_minimal_install.py -k base
```

Do not invoke `pytest --pyargs graph2mat` in this job: its package `conftest.py` intentionally imports the Torch bindings. The existing full-stack job should additionally run `pytest -q tests/test_minimal_install.py` so the Torch-present and broken-import branches remain covered in CI.

- [x] **Step 5: Run the focused test and full installed-stack suite.**

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/pytest -q tests/test_minimal_install.py --tb=short
PYTHONPATH=/tmp/g2m-server-tests.IcPjmO OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/pytest -q --disable-warnings --tb=short
git diff --check
```

Expected: three new regression cases pass; the existing 512 passing tests still pass. The temporary server dependency overlay is only used if it still exists; otherwise install the existing server test extras into an isolated temporary environment, not the training environment.

- [x] **Step 6: Review the scoped diff, update 3.5, and commit only these files.**

```bash
git add -- src/graph2mat/core/data/metrics.py src/graph2mat/core/data/processing.py tests/test_minimal_install.py .github/workflows/tests.yaml auditorias_mejoras/roadmap.md
git commit -m "fix: keep numpy core usable without torch"
```

Record the actual verification counts. Keep 3.5 partial: CLI startup and dependency/version reproducibility remain separate work. Request a read-only review before committing if a broader change is needed than the imports and dispatch guards above.

## Execution record (2026-10-07)

- Commit: `d7c87f4`, six files, including the necessary build prerequisite below.
- RED: the base subprocess failed at the unconditional Torch import; the other two characterization modes passed.
- GREEN: three subprocess modes pass; the full CPU suite passes **515 tests, 1 skipped** (62.80 s; 6023 pre-existing warnings).
- A fresh base-only wheel installation was also verified locally in a separate Python 3.12 venv: Torch truly absent, Graph2Mat loaded from site-packages, compiled label-reordering extension loaded, base smoke passed.
- Ruling: the fresh build exposed a duplicate Cython annotation in `core/modules/_labels_resort.py`; removed that annotation without changing the algorithm or dependency ranges. Wrong-fix risk was another build failure or a reordering regression; the actual build and a hand-derived transposed-block permutation both passed.
- The CI workflow was parsed locally; remote GitHub Actions have not been run. Roadmap 3.5 remains partial.
- Independent final review: no Critical, Important, or Minor findings. Existing Torch-specific coefficient/structure-weighted losses still require Torch (no backend rewrite); CLI and reproducible dependency ranges remain separate unfinished deliverables. These are explicit scope decisions, not claims of full base-profile feature parity.
- User chose retaining `g2m-v2` in the specified checkout; no merge, push, or PR was performed.
