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

SOAP-based extrapolation additionally needs `graph2mat[extrapolation]`. It uses
the periodic boundary conditions of each configuration, including partial PBC.

```
pip install graph2mat[tools]
```

If you want to use `graph2mat` with e3nn you can also ask for the `e3nn` extra dependencies:

```
pip install graph2mat[tools,e3nn]
```

What is an equivariant matrix?
------------------------------

![water_equivariant_matrix](https://raw.githubusercontent.com/BIG-MAP/graph2mat/main/docs/_static/images/water_equivariant_matrix.png)


Contributions
--------------

We are very open to suggestions, contributions, discussions...

- If you have questions or want do discuss an idea, please [start a discussion](https://github.com/BIG-MAP/graph2mat/discussions)
- If you have a feature suggestion or bug report, please [open an issue](https://github.com/BIG-MAP/graph2mat/issues)

We are looking forward to your contributions!

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
