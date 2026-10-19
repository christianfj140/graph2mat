Usage
------

### Server

To serve pretrained models, one can use the command `graph2mat serve`. You can do `graph2mat serve --help`
to check how it works. Serving models is as simple as:

```bash
graph2mat serve some.ckpt other.ckpt
```

This will serve the models in the checkpoint files. However, we recommend that you organize your
checkpoints into folders and then pass the names of the folders instead.

```bash
graph2mat serve first_model second_model
```

where `first_model` and `second_model` are folders that contain a `spec.yaml` file looking something like:

```yaml
description: |
    This model predicts single water molecules.
authors:
  - Pol Febrer (pol.febrer@icn2.cat)

files: # All the files related to this model.
  ckpt: best.ckpt
  basis: "*.ion.nc"
  structs: structs.xyz
  sample_metrics: sample_metrics.csv
  database: http://data.com/link/to/your/matrix_database
```

Once your server is running, you will get the url where the server is running, e.g. http://localhost:56000.
You can interact with it in multiple ways:
- Through the simple graphical interface included in the package, by opening `http://localhost:56000` in a browser.
- Through the `ServerClient` class in `graph2mat.tools.server.api_client`.
- By simply sending requests to the API of the server. These requests must be sent to `http://localhost:56000/api`. You
can check the documentation for the requests that the server understands under `http://localhost:56000/api/docs`.

Model metadata
--------------

``GET /api/models/{model_name}/info`` returns JSON-compatible processor fields
and the model description. ``basis_table.basis`` contains each point type,
radius (a number or list), canonical ``(multiplicity, l, parity)`` basis and
basis convention. ``node_attr_getters`` contains qualified names rather than
Python class/function objects. NumPy arrays/scalars are encoded as JSON
lists/numbers; neither the processor nor the basis table is modified.

Physical matrix output
----------------------

``/predict`` and ``local_write_predict`` use ``.DM`` for density matrices and
``.nc`` for Hamiltonians and energy-density matrices. The optional
``output_suffix`` query on ``/predict`` selects an explicitly supported format;
local output uses the requested filename's suffix. ``ServerClient`` selects the
matching format when returning a sisl object and requests the supplied suffix
when returning a path. Binary DM files omit geometry and basis: for a DM object,
pass a sisl geometry containing the model's orbital basis.

Hamiltonian models require ``fermi_level`` in their model specification (eV, in
the same energy gauge as their predictions). The output must contain explicit
overlap S, for example via ``h_and_overlap`` components ``(H, S)`` or
``(H_up, H_down, S)``. H-only predictions are not silently treated as a physical
non-orthogonal Hamiltonian. A genuinely orthogonal model can instead declare
``orthogonal_basis: true``. Non-finite entries, absent/non-positive overlap
diagonals and missing/non-finite Fermi levels produce HTTP 422 before writing.
This validates serialization prerequisites, not the accuracy or positive
definiteness of a learned overlap matrix.

NetCDF stores the supplied H and Ef. The sisl reader returns ``H - Ef*S``;
Graph2Mat applies no additional gauge shift. The TSHS writer does not preserve
a nonzero Ef, so ``.TSHS`` output is accepted only when the explicitly declared
``fermi_level`` is zero. A standalone EDM uses NetCDF in its supplied gauge,
without inventing a DM for an Ef-dependent correction. ``.EDM`` is not a sisl
writer format; paired TSDE output is not a standalone EDM format. Nonzero Ef
on a standalone EDM is rejected instead of silently ignored. See the
`sisl NetCDF API <https://sisl.readthedocs.io/en/latest/api/io/generated/sisl.io.siesta.ncSileSiesta.html>`_.

All server output is written to a private temporary file and atomically
published only after serialization succeeds. Local writes without overwrite
permission return HTTP 409 even if the destination appears during prediction.
Failed serialization preserves an existing destination and removes owned
temporaries. Successful download temporaries are removed after the response.
The client closes uploads and cleans its own files even on HTTP/read errors.

The reconstruction contract is described in :doc:`data_loading`. The CPU
oracles compare all generalized levels before/after physical output, not only
matrix-element errors: NetCDF levels read back in the Ef-relative gauge, and
TSHS with Ef=0 preserves the supplied levels. The Lightning ``MatrixWriter``
has its own HSX output path; it is not the server writer and does not inherit
all server guards. Use an explicit compatible geometry/basis when the reader
needs it to restore the orbital layout. These representation checks do not
demonstrate model accuracy, certify every k point of a learned S, or turn an
H-only output into a self-contained nonorthogonal Hamiltonian.

Extrapolation and SIESTA Lua
---------------------------

Standalone ``/api/extrapolation/extrapolate`` uses the same writer checks and
accepts ``fermi_level``/``orthogonal_basis`` explicitly. Matrix references must
match type, spin, orbital counts and periodic image ordering. When S is not a
predicted channel, it is excluded before computing a delta, rather than
broadcast into H/DM/EDM. An H-only delta can recover an explicit S from its
reference at the target geometry. When overlap is predicted explicitly, the
reference is added to the overlap delta once. Caller matrices are not modified.

The mounted extrapolation API inherits the server's ``local`` policy: with
``local=False`` every extrapolation request returns HTTP 403 before accessing
the filesystem or history. Standalone ``create_extrapolation_app`` keeps its
historical local-access default; set ``local=False`` to disable that API.
Extrapolation never overwrites an existing destination unless the request sets
``allow_overwrite=true`` (otherwise HTTP 409), including publication races.
The SIESTA Lua client explicitly requests replacement of its generated
``siesta.DM`` and requires a server launched with ``--local``.

The SIESTA MD Lua integration is **density-matrix-only**: it writes ``siesta.DM``
for ``DM.UseSaveDM``. Initialization declares ``matrix_type=density_matrix``;
H/EDM processors are rejected, including a later processor setup. Lua checks
curl's HTTP/exit failure and aborts initialization rather than continuing with
an invalid or stale DM. Standalone H/EDM extrapolation remains available using
their appropriate formats; no H-to-DM conversion is implied. Empty/unobserved
histories return HTTP 422.

Joint matrix models
-------------------

The generic HTTP endpoints do not accept a processor with ``matrix_targets``
enabled. Registration fails before inference/publication, rather than treating
the H/S/DM/EDM packing as spin channels of one matrix. Reconstruct and select
a target explicitly outside these endpoints; no joint HTTP API is added.
Single-target models and the existing writer/gauge checks are unchanged.
