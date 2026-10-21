"""Shared fixtures of the tools tests: small sisl Hamiltonians and a fake SIESTA."""
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import sisl

_FAKE_SIESTA = """\
import os, re, shutil, sys
text = sys.stdin.read()
print("fake siesta")
if os.environ.get("FAKE_SIESTA_FAIL"):
    sys.exit(1)
match = re.search(r"(?im)^\\s*SystemLabel\\s+(\\S+)", text)
label = match.group(1) if match else "siesta"
source = os.environ.get("FAKE_SIESTA_TSHS")
if source:
    shutil.copy(source, label + ".TSHS")
if os.environ.get("FAKE_SIESTA_FAIL_AFTER"):
    sys.exit(1)
"""


@pytest.fixture
def fake_siesta(tmp_path, monkeypatch):
    """Command of a fake SIESTA that copies ``$FAKE_SIESTA_TSHS`` to ``<SystemLabel>.TSHS``
    (``$FAKE_SIESTA_FAIL`` makes it exit with an error before, ``$FAKE_SIESTA_FAIL_AFTER``
    after writing it)."""
    script = tmp_path / "fake_siesta.py"
    script.write_text(_FAKE_SIESTA)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.delenv("FAKE_SIESTA_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SIESTA_TSHS", raising=False)
    monkeypatch.delenv("FAKE_SIESTA_FAIL_AFTER", raising=False)
    return f"{sys.executable} {script}"


def make_hamiltonian(geometry=None, onsite=0.1, hopping=-2.7, overlap=0.15, spin="unpolarized"):
    """Non-orthogonal nearest-neighbour Hamiltonian (one orbital per atom)."""
    if geometry is None:
        geometry = sisl.geom.graphene(atoms=sisl.Atom(6, R=1.6))
    H = sisl.Hamiltonian(geometry, orthogonal=False, spin=spin)
    n_spin = H.spin.size(H.dtype)
    on = [onsite + 0.05 * i for i in range(n_spin)]
    hop = [hopping + 0.1 * i for i in range(n_spin)]
    H.construct([(0.1, 1.6), ((*on, 1.0), (*hop, overlap))])
    return H


def write_fdf(directory: Path, geometry, label="sample", extra=""):
    """RUN.fdf with ``geometry`` and a fake pseudopotential next to it."""
    directory.mkdir(parents=True, exist_ok=True)
    geometry.write(directory / "RUN.fdf")
    text = (directory / "RUN.fdf").read_text()
    (directory / "RUN.fdf").write_text(f"SystemLabel {label}\nDM.Tolerance 1.d-4\n{extra}{text}")
    (directory / "C.psf").write_text("fake pseudo\n")
    return directory / "RUN.fdf"


@pytest.fixture
def hamiltonian_factory():
    return make_hamiltonian


@pytest.fixture
def rng():
    return np.random.default_rng(0)
