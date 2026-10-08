"""To support the use of zipfiles for the data module,
we need to have the ability of applying glob patterns inside
the zipfile.

We want to use the glob.translate function, but this is only
available starting from Python 3.13. Until this package no
longer supports Python 3.12, we keep here a copy of the relevant
functions.

We have:
    - fnmatch.translate, because glob.translate depends on it
    - glob.translate, the function that we want to use.
"""

import functools

# ---------------------------------------------------
#   CODE COPIED FROM PYTHON 3.13 GLOB AND FNMATCH
# ---------------------------------------------------
import os
import re
import zipfile
from pathlib import Path
from typing import Union

_re_escape = functools.lru_cache(maxsize=512)(re.escape)
_re_setops_sub = re.compile(r"([&~|])").sub


def fnmatch_translate(pat, star, question_mark):
    res = []
    add = res.append
    star_indices = []

    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        i = i + 1
        if c == "*":
            # store the position of the wildcard
            star_indices.append(len(res))
            add(star)
            # compress consecutive `*` into one
            while i < n and pat[i] == "*":
                i += 1
        elif c == "?":
            add(question_mark)
        elif c == "[":
            j = i
            if j < n and pat[j] == "!":
                j = j + 1
            if j < n and pat[j] == "]":
                j = j + 1
            while j < n and pat[j] != "]":
                j = j + 1
            if j >= n:
                add("\\[")
            else:
                stuff = pat[i:j]
                if "-" not in stuff:
                    stuff = stuff.replace("\\", r"\\")
                else:
                    chunks = []
                    k = i + 2 if pat[i] == "!" else i + 1
                    while True:
                        k = pat.find("-", k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k + 1
                        k = k + 3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += "-"
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks) - 1, 0, -1):
                        if chunks[k - 1][-1] > chunks[k][0]:
                            chunks[k - 1] = chunks[k - 1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = "-".join(
                        s.replace("\\", r"\\").replace("-", r"\-") for s in chunks
                    )
                i = j + 1
                if not stuff:
                    # Empty range: never match.
                    add("(?!)")
                elif stuff == "!":
                    # Negated empty range: match any character.
                    add(".")
                else:
                    # Escape set operations (&&, ~~ and ||).
                    stuff = _re_setops_sub(r"\\\1", stuff)
                    if stuff[0] == "!":
                        stuff = "^" + stuff[1:]
                    elif stuff[0] in ("^", "["):
                        stuff = "\\" + stuff
                    add(f"[{stuff}]")
        else:
            add(_re_escape(c))
    assert i == n
    return res, star_indices


def translate(pat, *, recursive=False, include_hidden=False, seps=None):
    """Translate a pathname with shell wildcards to a regular expression.

    If `recursive` is true, the pattern segment '**' will match any number of
    path segments.

    If `include_hidden` is true, wildcards can match path segments beginning
    with a dot ('.').

    If a sequence of separator characters is given to `seps`, they will be
    used to split the pattern into segments and match path separators. If not
    given, os.path.sep and os.path.altsep (where available) are used.
    """
    if not seps:
        if os.path.altsep:
            seps = (os.path.sep, os.path.altsep)
        else:
            seps = os.path.sep
    escaped_seps = "".join(map(re.escape, seps))
    any_sep = f"[{escaped_seps}]" if len(seps) > 1 else escaped_seps
    not_sep = f"[^{escaped_seps}]"
    if include_hidden:
        one_last_segment = f"{not_sep}+"
        one_segment = f"{one_last_segment}{any_sep}"
        any_segments = f"(?:.+{any_sep})?"
        any_last_segments = ".*"
    else:
        one_last_segment = f"[^{escaped_seps}.]{not_sep}*"
        one_segment = f"{one_last_segment}{any_sep}"
        any_segments = f"(?:{one_segment})*"
        any_last_segments = f"{any_segments}(?:{one_last_segment})?"

    results = []
    parts = re.split(any_sep, pat)
    last_part_idx = len(parts) - 1
    for idx, part in enumerate(parts):
        if part == "*":
            results.append(one_segment if idx < last_part_idx else one_last_segment)
        elif recursive and part == "**":
            if idx < last_part_idx:
                if parts[idx + 1] != "**":
                    results.append(any_segments)
            else:
                results.append(any_last_segments)
        else:
            if part:
                if not include_hidden and part[0] in "*?":
                    results.append(r"(?!\.)")
                results.extend(fnmatch_translate(part, f"{not_sep}*", not_sep)[0])
            if idx < last_part_idx:
                results.append(any_sep)
    res = "".join(results)
    return rf"(?s:{res})\z"


# ---------------------------------------------------
# END OF CODE COPIED FROM PYTHON 3.13 GLOB AND FNMATCH
# ---------------------------------------------------


# Glob wrappers
def _glob_zipfile(path: zipfile.ZipFile, pattern: str):
    internal_path = Path(str(path)).relative_to(path.root.filename)
    internal_path = str(internal_path).removeprefix("./")
    if internal_path == ".":
        internal_path = ""
    if internal_path and not internal_path.endswith("/"):
        internal_path += "/"

    pattern = Path(internal_path + pattern)
    pattern = pattern.resolve().relative_to(Path().resolve())

    reg = translate(str(pattern))
    p = re.compile(reg.replace(r"\z", r"\Z"))

    fileslist = path.root.namelist()
    return iter(zipfile.Path(path.root, s) for s in fileslist if p.match(s))


def glob(path, pattern: str):
    """"""
    if isinstance(path, zipfile.Path):
        return _glob_zipfile(path, pattern)
    else:
        return Path(path).glob(pattern)


def maybe_zip_path(
    path: Union[Path, str], zipfile_mode: str = "r"
) -> Union[Path, zipfile.Path]:
    path = Path(path)

    for i, part in enumerate(path.parts):
        if part.endswith(".zip"):
            zip_path = Path(*path.parts[: i + 1])
            if zip_path.is_file():
                zipfile_mode = "r" if zipfile_mode.startswith("r") else "a"
                if zipfile_mode == "r":
                    # Open directly: Python 3.9 Path(ZipFile(...)) clones the
                    # reader and lets the temporary original close its file.
                    return zipfile.Path(zip_path, str(path.relative_to(zip_path)))
                root_zip = zipfile.ZipFile(zip_path, zipfile_mode)
                return zipfile.Path(root_zip, str(path.relative_to(zip_path)))

    return path


def maybe_clean_zip_path(path: Union[Path, zipfile.Path]) -> Union[Path, zipfile.Path]:
    """"""
    if isinstance(path, zipfile.Path):
        internal_path = (
            Path(str(path)).resolve().relative_to(Path(path.root.filename).resolve())
        )
        return zipfile.Path(path.root, str(internal_path))
    else:
        return path
