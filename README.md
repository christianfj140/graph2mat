graph2mat: Equivariant matrices meet machine learning
----------------------

![graph2mat_overview](https://raw.githubusercontent.com/BIG-MAP/graph2mat/main/docs/_static/images/graph2mat_overview.svg)

The aim of `graph2mat` is to pave your way into meaningful science by providing the **tools to interface to common machine learning frameworks** (e3nn, pytorch) **to learn equivariant matrices.**

**[Documentation](https://big-map.github.io/graph2mat/)**

It also provides a **set of tools** to facilitate the training and usage of the models created using the package:

- **Training tools**: It contains custom `pytorch_lightning` modules to train, validate and test the orbital matrix models.
- **Server**: A production ready server (and client) to serve predictions of the trained
    models. Implemented using `fastapi`.
- **Siesta**: A set of tools to interface the machine learning models with SIESTA. These include tools for input preparation, analysis of performance...

The package also implements a **command line interface** (CLI): `graph2mat`. The aim of this CLI is
to make the usage of `graph2mat`'s tools as simple as possible. It has two objectives:

- Make life easy for the model developers.
- Facilitate the usage of the models by non machine learning scientists, who just want
  good predictions for their systems.

Installation
------------

The base package supports NumPy processing and CLI help without the ML or server
extras:

```bash
pip install graph2mat
graph2mat --help
```

Install `graph2mat[siesta]` to generate SIESTA input templates,
`graph2mat[analysis]` for performance tables, or `graph2mat[server]` for the
prediction server and client. Adding the tools extra installs the dependencies
for the complete toolset.

The server extra includes the Torch/PyG/e3nn/Lightning/MACE stack needed to serve
checkpoints; the base package stays free of ML dependencies. HTTP-only clients
can use the base package with ``requests`` without importing that stack.

SOAP-based extrapolation additionally needs `graph2mat[extrapolation]`. It uses
the periodic boundary conditions of each configuration, including partial PBC.

```
pip install graph2mat[tools]
```

If you want to use `graph2mat` with e3nn you can also ask for the `e3nn` extra dependencies:

```
pip install graph2mat[tools,e3nn]
```

Reproducible CPU profiles
-------------------------

The repository contains complete runtime/test constraints for Linux x86_64:
`constraints/ci-py39.txt` (Python 3.9, tested ML floors) and
`constraints/ci-py312.txt` (Python 3.12, reference stack). The corresponding
`build-py39.txt` and `build-py312.txt` constrain the isolated Cython wheel build,
including its NumPy headers. Runtime constraints alone do not constrain builds.
They do not install optional packages into the base profile.

For a base-only Python 3.12 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install pip==25.3
.venv/bin/python -m pip install -c constraints/ci-py312.txt \
  --build-constraint constraints/build-py312.txt . pytest
.venv/bin/python -m pip check
.venv/bin/graph2mat --help
.venv/bin/python -m pytest -q tests/test_minimal_install.py -k base
```

For the complete CPU profile, including jointly resolved MACE and documentation:

```bash
.venv/bin/python -m pip install -c constraints/ci-py312.txt torch \
  --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -c constraints/ci-py312.txt \
  --build-constraint constraints/build-py312.txt \
  ".[test,tools,server,extrapolation,docs]" httpx
.venv/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest --pyargs graph2mat -q
.venv/bin/python -m pytest -q tests
```

For Python 3.9, use `python3.9` and replace both `py312` suffixes with `py39`.
The tested ML floors are Torch 2.4.1, PyG 2.4.0, Lightning 2.1.4, MACE 0.3.15
and e3nn 0.4.4. MACE's e3nn pin is respected, not overridden by a second install;
`TensorSquare` and the four-argument radial embedding are checked with real models.
Python 3.9 ZIP reads temporarily extract the archive because sisl's native ZIP
reader needs Python 3.10; temporary files are removed on success and failure.
Use Python >=3.10 for native ZIP I/O without that extra disk cost.

CI runs the pinned base and complete profiles on both Python versions and a
Python 3.12 job with latest compatible runtime packages. Configured CI is not
evidence that a remote workflow has run; local verification is recorded separately.

What is an equivariant matrix?
------------------------------

![water_equivariant_matrix](https://raw.githubusercontent.com/BIG-MAP/graph2mat/main/docs/_static/images/water_equivariant_matrix.png)


Contributions
--------------

We are very open to suggestions, contributions, discussions...

- If you have questions or want do discuss an idea, please [start a discussion](https://github.com/BIG-MAP/graph2mat/discussions)
- If you have a feature suggestion or bug report, please [open an issue](https://github.com/BIG-MAP/graph2mat/issues)

We are looking forward to your contributions!

The static gates use an isolated Python 3.12 environment:

```bash
python3.12 -m venv .venv-quality
.venv-quality/bin/python -m pip install -r requirements-quality.txt
.venv-quality/bin/pre-commit run ruff-check --all-files
.venv-quality/bin/pre-commit run mypy-core --all-files
```

Ruff checks semantic errors across package code and tests. Black and isort
retain formatting ownership with a common Black profile. Mypy checks the base
core against Python 3.9 and real NumPy types; optional scientific/ML dependencies
are opaque. Checker pins do not constrain the package's runtime environment.
Run the full test suite separately with the test/server/extrapolation extras.

The `graph2mat` package was originally created by Peter Bjørn Jorgensen (@peterbjorgensen) and Pol Febrer (@pfebrer) in the frame of a collaboration to machine learn density matrices. 

Since then, the following users have contributed to the code:

<a href="https://github.com/BIG-MAP/graph2mat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BIG-MAP/graph2mat" />
</a>

Citation
--------

If you use `graph2mat` for one of your works, please cite our original paper:

```
@article{febrer2025graph2mat,
  title={Graph2Mat: universal graph to matrix conversion for electron density prediction},
  author={Febrer, Pol and J{\o}rgensen, Peter Bj{\o}rn and Pruneda, Miguel and Garc{\'\i}a, Alberto and Ordej{\'o}n, Pablo and Bhowmik, Arghya},
  journal={Machine Learning: Science and Technology},
  volume={6},
  number={2},
  pages={025013},
  year={2025},
  publisher={IOP Publishing}
}
```

We'll be very happy to see what you have done with it :)
