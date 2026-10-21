"""Running SIESTA for the label tools (H0 of Delta-learning, label noise, DM pilot).

The executable is the ``siesta`` argument, else the ``SIESTA`` environment variable
(split like a shell command, e.g. ``"mpirun -np 1 siesta"``), else ``siesta`` on the
``PATH``.
"""
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

__all__ = [
    "replace_fdf_keys",
    "system_label",
    "find_pseudos",
    "siesta_command",
    "run_siesta",
]


def _fdf_label(key: str) -> str:
    # fdf labels ignore case, '.', '-' and '_'.
    return re.sub(r"[._-]", "", key).lower()


def replace_fdf_keys(text: str, keys: Mapping[str, str]) -> str:
    """Drop the lines that set any of ``keys`` and append ``key  value`` for each of them."""
    labels = {_fdf_label(k) for k in keys}
    kept = [
        line
        for line in text.splitlines()
        if not (line.split() and _fdf_label(line.split()[0]) in labels)
    ]
    return "\n".join(kept + [f"{k}  {v}" for k, v in keys.items()]) + "\n"


def system_label(fdf: Path) -> str:
    """``SystemLabel`` of an fdf file (``"siesta"``, SIESTA's default, if not set)."""
    match = re.search(r"(?im)^\s*SystemLabel\s+(\S+)", Path(fdf).read_text())
    return match.group(1) if match else "siesta"


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

    ``work_dir`` is removed first. ``pseudos`` are copied in, and ``extra_files`` are
    copied under the given names. SIESTA reads ``RUN.fdf`` from stdin and writes
    ``RUN.out``; it runs on one OpenMP thread and with lower priority (``nice``).
    Raises ``subprocess.CalledProcessError`` if SIESTA fails. Returns ``work_dir``.
    """
    command = siesta_command(siesta)
    work_dir = Path(work_dir)
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
