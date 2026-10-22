"""Running SIESTA for the label tools (H0 of Delta-learning, label noise, DM pilot).

The executable is the ``siesta`` argument, else the ``SIESTA`` environment variable
(split like a shell command, e.g. ``"mpirun -np 1 siesta"``), else ``siesta`` on the
``PATH``.
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

__all__ = [
    "replace_fdf_keys",
    "system_label",
    "find_pseudos",
    "siesta_command",
    "run_siesta",
    "run_siesta_into",
]


def _fdf_label(key: str) -> str:
    # fdf labels ignore case, '.', '-' and '_'.
    return re.sub(r"[._-]", "", key).lower()


def replace_fdf_keys(text: str, keys: Mapping[str, str]) -> str:
    """Drop the lines that set any of ``keys`` and append ``key  value`` for each of them."""
    _require_inline_fdf(text)
    labels = {_fdf_label(k) for k in keys}
    kept = [
        line
        for line in text.splitlines()
        if not (line.split() and _fdf_label(line.split()[0]) in labels)
    ]
    return "\n".join(kept + [f"{k}  {v}" for k, v in keys.items()]) + "\n"


def system_label(fdf: Path) -> str:
    """``SystemLabel`` of an fdf file (``"siesta"``, SIESTA's default, if not set)."""
    import sisl

    return sisl.get_sile(str(fdf)).get("SystemLabel", "siesta")


def _require_inline_fdf(text: str) -> None:
    """Text-only rewrites cannot resolve or stage external FDF settings safely."""
    for line in text.splitlines():
        setting = re.split(r"[#!]", line, maxsplit=1)[0]
        if re.match(r"\s*%include\b", setting, re.I) or "<" in setting:
            raise ValueError("Cannot rewrite external FDF settings from another file; resolve includes first")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hamiltonian(path, geometry, *, orbital_counts=None, spin=None, geometry_tol=1e-4):
    """Check readable, finite output and the structure/basis layout known by the caller."""
    import numpy as np
    import sisl

    result = sisl.get_sile(str(path)).read_hamiltonian()
    actual = result.geometry
    if (actual.na != geometry.na
            or not np.allclose(actual.xyz, geometry.xyz, rtol=0, atol=geometry_tol)
            or not np.allclose(actual.cell, geometry.cell, rtol=0, atol=geometry_tol)):
        raise RuntimeError("SIESTA output geometry differs from the requested geometry")
    if orbital_counts is not None and not np.array_equal(actual.orbitals, orbital_counts):
        raise RuntimeError("SIESTA output orbital layout differs from the requested basis")
    if spin is not None and result.spin.kind != spin.kind:
        raise RuntimeError("SIESTA output spin differs from the requested spin")
    for component in range(result.dim):
        if not np.isfinite(result.tocsr(component).data).all():
            raise ValueError("SIESTA output matrix contains non-finite values")


def find_pseudos(directory: Path) -> List[Path]:
    """Pseudopotentials (``*.psf``, ``*.psml``) next to a calculation or, as in some
    datasets, in one of its parent directories (the closest one that has any)."""
    directory = Path(directory)
    for d in (directory, *directory.parents):
        found = sorted([*d.glob("*.psf"), *d.glob("*.psml")])
        if found:
            return found
    raise RuntimeError(f"no pseudopotential (*.psf, *.psml) for {directory}")


def siesta_command(siesta: Optional[str] = None) -> List[str]:
    """Command that runs SIESTA (see the module docstring for the lookup order)."""
    command = siesta or os.environ.get("SIESTA") or shutil.which("siesta")
    if not command:
        raise FileNotFoundError(
            "SIESTA executable not found: pass it explicitly, set the SIESTA "
            "environment variable or put 'siesta' on the PATH."
        )
    return shlex.split(command)


def run_siesta(
    work_dir: Path,
    fdf_text: str,
    pseudos: Sequence[Path],
    *,
    siesta: Optional[str] = None,
    extra_files: Mapping[str, Path] = {},
    nice: int = 10,
) -> Path:
    """Run SIESTA in a fresh ``work_dir`` with ``fdf_text`` as ``RUN.fdf``.

    ``work_dir`` is removed first (only if empty or a previous run, with a ``RUN.fdf``).
    ``pseudos`` are copied in, and ``extra_files`` are copied under the given names. SIESTA reads ``RUN.fdf`` from stdin and writes
    ``RUN.out``; it runs on one OpenMP thread and with lower priority (``nice``).
    Raises ``subprocess.CalledProcessError`` if SIESTA fails. Returns ``work_dir``.
    """
    command = siesta_command(siesta)
    work_dir = Path(work_dir)
    if work_dir.exists() and any(work_dir.iterdir()) and not (work_dir / "RUN.fdf").exists():
        raise FileExistsError(f"refusing to empty {work_dir}: it is not a previous SIESTA run directory")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    (work_dir / "RUN.fdf").write_text(fdf_text)
    for pseudo in pseudos:
        shutil.copy(pseudo, work_dir)
    for name, path in extra_files.items():
        shutil.copy(path, work_dir / name)
    with (work_dir / "RUN.fdf").open() as stdin, (work_dir / "RUN.out").open("w") as stdout:
        subprocess.run(
            command,
            cwd=work_dir,
            stdin=stdin,
            stdout=stdout,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            preexec_fn=(lambda: os.nice(nice)) if nice else None,
        )
    return work_dir


def run_siesta_into(
    final_dir: Path,
    fdf_text: str,
    pseudos: Sequence[Path],
    expect: str,
    *,
    siesta: Optional[str] = None,
    extra_files: Mapping[str, Path] = {},
    nice: int = 10,
    validate: Optional[Callable[[Path], None]] = None,
) -> Path:
    """``final_dir / expect`` from a SIESTA run that only lands in ``final_dir`` if it succeeds.

    An existing result is reused only with a matching input/output hash receipt. Legacy
    or changed caches are rejected without deleting them. Otherwise SIESTA runs in a hidden
    sibling directory, which is renamed to ``final_dir`` only after SIESTA exits cleanly and
    wrote ``expect`` and passed ``validate``: a failed or interrupted run never leaves a result to be reused. A
    ``final_dir`` without ``expect`` is not touched (error).
    """
    final_dir = Path(final_dir)
    out = final_dir / expect
    request = {"fdf_sha256": hashlib.sha256(fdf_text.encode()).hexdigest(),
               "pseudos": {Path(p).name: _sha256(p) for p in pseudos},
               "extra_files": {name: _sha256(p) for name, p in extra_files.items()}}
    receipt_name = ".graph2mat-siesta.json"
    if out.exists():
        if validate is not None:
            validate(out)
        try:
            receipt = json.loads((final_dir / receipt_name).read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unverified SIESTA cache at {final_dir}; use a new directory") from exc
        if receipt != {"inputs": request, "output_sha256": _sha256(out)}:
            raise ValueError(f"Stale SIESTA cache at {final_dir}; inputs or output changed; use a new directory")
        return out
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} exists without {expect}: remove it to run SIESTA there")
    partial = final_dir.with_name(f".{final_dir.name}.partial-{os.getpid()}")
    run_siesta(partial, fdf_text, pseudos, siesta=siesta, extra_files=extra_files, nice=nice)
    if not (partial / expect).exists():
        raise RuntimeError(f"SIESTA wrote no {expect} in {partial}")
    if validate is not None:
        validate(partial / expect)
    (partial / receipt_name).write_text(json.dumps({"inputs": request, "output_sha256": _sha256(partial / expect)},
                                                 sort_keys=True) + "\n")
    os.replace(partial, final_dir)
    return out
