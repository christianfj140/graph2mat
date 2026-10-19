Data loading and resource ownership
===================================

``MatrixDataModule.setup`` loads only the splits needed by the stage:
``fit`` uses train/val, ``validate`` uses val, ``test`` uses test, and
``predict`` uses predict. Passing ``None`` initializes every available split.
Repeated setup calls reuse existing datasets. Without explicit validation data,
validation can draw its subset lazily from training data; ``store_in_memory``
materializes only the splits requested by the current stage.

``store_in_memory=True`` reuses preprocessed CPU samples through ``InMemoryData``;
it is not a new disk cache. Materialization owns a deep snapshot, and each access
returns an independent copy of tensors and mutable metadata, including NumPy
arrays. In-place transforms or device transfers therefore cannot contaminate
later accesses. Rebuild the cache when input files, labels or preprocessing
change; there is no automatic invalidation. Stored ``data_objects`` are the
snapshots and must not be mutated directly. Explicit ``InMemoryData(size=0)``
loads no samples; negative or oversized limits raise before loading.

This safety costs a per-access copy and keeps the processed dataset in host RAM,
plus transient returned samples/batches and possible per-worker process copies. It is not
a VRAM reduction. A paired synthetic CPU test measured ~0.14–0.15ms per copied
sample and ~0.31–0.32ms extra (~91%) for cached two-sample collate versus the
previous unsafe shared-object cache. The lazy path also avoids formatting entire in-memory
configurations merely to check the FDF suffix. Its speedup on such inputs does
not establish faster file parsing or faster end-to-end training.

The automatic validation split accepts ``split_seed`` (default 32) and
``val_fraction`` (default 0.1, in [0, 1)). Explicit ``val_runs`` take priority;
an explicit empty list disables validation. The automatic split keeps at least
one training sample when data are available. For correlated trajectories, use
explicit validation runs from separate trajectories rather than a random split
of adjacent frames.

Warning-only split guardrail
---------------------------

Roadmap 3.12 adds a lightweight check during ``setup`` of labelled stages.
It compares only already-loaded/resolved train/val/test inputs, never loads
another split for the check, never parses a lazy matrix, and does nothing in
``setup("predict")``. It logs ordinary warnings, not Python warnings that
``-Werror`` could turn into exceptions. Alarms never repartition, remove samples,
change a seed/fraction or stop training. Unchanged reports are not logged again
on every setup. ``module.split_guardrail_report`` holds the latest report.

Evidence is deliberately limited to:

* Reused resolved source paths (including symlink aliases), or the same in-memory
  input object. ZIP identity includes both archive and exact member; a ZIP file
  or parent folder is **not** inferred to be a trajectory.
* Explicit matching ``sample_id``, ``sha256`` or ``group_id`` declarations.
  ``group_id`` should name the relevant correlated family, such as a trajectory
  or common parent structure. The caller defines that physical meaning.

For file inputs, an optional ``split_metadata`` map can be added to the existing
``runs_json`` file. Source keys are relative to ``root_dir`` (or absolute paths);
ordinary train/val/test path lists keep their existing meaning:

.. code-block:: json

    {
      "train": ["trajectory1/frame0/RUN.fdf"],
      "val": ["trajectory1/frame1/RUN.fdf"],
      "split_metadata": {
        "trajectory1/frame0/RUN.fdf": {"sample_id": "t1-f0", "group_id": "trajectory1"},
        "trajectory1/frame1/RUN.fdf": {"sample_id": "t1-f1", "group_id": "trajectory1"}
      }
    }

This example emits a shared-group alarm but still trains on those exact lists.
Lexical symlink names and original absolute source keys remain usable when
``copy_root_to_tmp=True``. A memory-backed ZIP can only be identified by its
archive object and exact member; separate unnamed readers are not assumed equal.
For in-memory configurations/prepared samples use
``sample.metadata["split_guardrail"] = {"sample_id": "frame0", "group_id": "MD1"}``.
Inline annotations take precedence over file annotations. IDs must be nonempty
strings; optional ``sha256`` is a 64-hex-digit **declared** content digest.
The check does not read/hash files or verify those declarations. In particular,
identical FDF text does not prove identical matrix labels, and copied/renamed
data without declared identity/hash can go undetected.

Malformed alarm annotations produce a diagnostic, not a loading error. Missing
group metadata yields ``not_evaluable`` (or a missing-information warning along
with detected overlaps), not a claim of contamination. Fully declared disjoint
groups yield ``no_known_overlap``, **not** certified independence. No geometric
similarity metric, distance threshold, distribution test, automatic manifest,
split generator or checkpoint policy is introduced. Custom/streaming datasets
that cannot be inspected without iteration are reported as unavailable.

Raw input identity is retained across normal in-memory/pool wrapping; all
parent samples of a rotating pool are checked, not only its active window.
Final ``teardown(None)`` clears the report and its raw-input references. Other
resource ownership and checkpoint/default behaviour remain unchanged.

Nonorthogonal Hamiltonian reconstruction
----------------------------------------

``matrix_component_policy="h_and_overlap"`` gives physical components an
explicit meaning: ``(H, S)`` for an unpolarized Hamiltonian, or
``(H_up, H_down, S)`` for collinear spin. H is in eV and S is dimensionless.
Reading and reconstruction preserve the atom/orbital order, cell, periodic
image layout and coefficients represented by the graph. This is a contract
on the graph's support, not recovery of blocks outside its cutoff. The
symmetric path reconstructs transposed partners at opposite images; the
directed path does not silently symmetrize its inputs.

For an existing ``sisl.Hamiltonian`` with compatible orbital basis:

.. code-block:: python

    from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor

    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(hamiltonian.atoms.atom),
        out_matrix="hamiltonian", sub_point_matrix=False,
        symmetric_matrix=True, n_matrix_components=2,
        matrix_component_policy="h_and_overlap",
    )
    data = BasisMatrixData.new(hamiltonian, processor)
    reconstructed = processor.matrix_from_data(data, threshold=None)
    levels = reconstructed.eigenvalue().eig  # solves H c = energy S c

The example is unpolarized and assumes Hermitian input on the represented
support. Collinear H+S needs three components. NumPy data take NumPy
predictions, Torch data take tensors; keep numerical precision explicit when
checking spectra. With ``predictions=None`` the labels supply H/S. With
``predictions={"node_labels": ..., "edge_labels": ...}`` both H and S come
from the supplied predictions; reconstruction must not mutate the labels.
PyG batches return one reconstructed matrix per structure without sharing
their coefficients.

H-only predictions remain H-only even if training labels include S for a
gauge loss. There is no implicit attachment of reference S or inference of
spin semantics from an arbitrary raw component count. ``raw_components``
storage does not authorize physical spin/SOC serialization. The generic
Hamiltonian processor neither constructs SIESTA's superposed-atomic-density
H0 nor automatically restores it: ``sub_point_matrix`` is a different,
isolated-point baseline and is disabled for Hamiltonians. Explicit matrix
references in extrapolation use their separate restoration contract.

Certification uses full-space generalized eigenvalues at Gamma and nonzero k,
an independent two-orbital spectrum, and H+cS controls. The latter shift every
level by c without changing energy differences. Singular S is not repaired
or truncated to make a spectrum pass; preservation of a predicted S does not
guarantee that S is positive definite, accurate, or well conditioned.
For file formats, gauge and explicit-reference rules, see :doc:`server`.

Spin-orbit Hamiltonians
-----------------------

``matrix_component_policy="spin_orbit"`` (opt-in, roadmap 8.1) loads SIESTA's
non-collinear spin-orbit Hamiltonian (sisl ``Spin("spin-orbit")``, 8 real
components per element, in sisl's order ``[Re uu, Re dd, Re ud, Im ud, Im uu,
Im dd, Re du, Im du]``) as 4 real time-reversal (TRS) channels
``[h0, gx, gy, gz]``. With TRS each spin block is
``h0 * s0 + i * (gx * sx + gy * sy + gz * sz)``: ``h0`` is an ordinary matrix
block and ``g`` an axial vector (``(l_i x l_j) x 1e``) stored in Cartesian
axes. The channels are ``h0 = (c0 + c1)/2``, ``gx = (c3 + c7)/2``,
``gy = (c2 - c6)/2`` and ``gz = (c4 - c5)/2``. ``g`` changes sign under
transposition, so the symmetric path applies ``SOC_TRANSPOSE_SIGN =
[1, -1, -1, -1]`` to the transposed partner; ``g`` is antisymmetric on-site.
The channels are fixed by the policy: ``n_matrix_components`` must be 1 or 4,
and the model and the command line interface set it to 4.

Loading checks that the data really have TRS. The four residuals ``Im h0``,
``Re hx``, ``Re hy`` and ``Re hz`` must be below ``SOC_TRS_RTOL = 1e-5`` of the
largest channel (SCF noise in Bi2Se3 is about 1e-6). Above that (a magnetic
Hamiltonian, for example) loading raises a ``ValueError`` that mentions
time-reversal symmetry; the residual is never dropped silently. Non-finite
values are rejected. Labels do not carry the overlap: for a non-orthogonal
(nine-component) Hamiltonian, S is left out and the eight Hamiltonian
components are selected through sisl's overlap index (``S_idx``), never by
position.

Reconstruction goes only to sisl formats (``out_format="sisl_H"``; other
formats raise because only the sisl path applies the sign of the transposed
copy). The result is an orthogonal ``spin-orbit`` Hamiltonian. Add the
reference overlap explicitly with ``attach_overlap``:

.. code-block:: python

    import sisl
    from graph2mat import AtomicTableWithEdges, BasisMatrixData, MatrixDataProcessor
    from graph2mat.core.data.spin_orbit import attach_overlap
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    geometry = sisl.get_sile("RUN.fdf").read_geometry()
    H = sisl.get_sile("bi2se3.TSHS").read_hamiltonian(geometry=geometry)
    S = sisl.Overlap.fromsp(H.geometry, H.tocsr(H.S_idx))   # reference S
    table = AtomicTableWithEdges(H.geometry.atoms.atom)

    processor = MatrixDataProcessor(
        basis_table=table, out_matrix="hamiltonian", sub_point_matrix=False,
        symmetric_matrix=True, matrix_component_policy="spin_orbit",
        n_matrix_components=4,
    )
    data = BasisMatrixData.new(H, processor)                # 4 TRS channels
    H_orth = processor.matrix_from_data(data, out_format="sisl_H", threshold=None)
    H_so = attach_overlap(H_orth, S)                        # non-orthogonal
    levels = H_so.eigh(k=[0, 0, 0])                         # H c = E S c

    # Training: the model flag must agree with the data policy.
    both_parities = "8x0e+8x0o+8x1o+8x1e+8x2e+8x2o+8x3o+8x3e"
    model = LitMACEMatrixModel(
        basis_table=table, symmetric_matrix=True, spin_orbit=True,
        hidden_irreps=both_parities, edge_hidden_irreps=both_parities, max_ell=3,
    )
    datamodule = MatrixDataModule(
        basis_table=table, out_matrix="hamiltonian", sub_point_matrix=False,
        symmetric_matrix=True, matrix_component_policy="spin_orbit",
        n_matrix_components=4, train_runs="train/*/RUN.fdf", val_runs="val/*/RUN.fdf",
    )

``attach_overlap`` requires an orthogonal spin-orbit ``sisl.Hamiltonian`` and
a single-component S carrier (such as ``sisl.Overlap``) with the same geometry
and ``nsc``; a Hamiltonian is rejected because its component 0 is not S.
With the command line interface the model flag is derived from the data
policy; ``on_fit_start`` raises if a model and a datamodule disagree, because a
plain model would fit ``[h0, gx, gy, gz]`` as four unrelated scalars.

``spin_orbit=True`` swaps the readout for ``E3nnSpinOrbitMatrixBlock``: ``h0``
as in the standard block plus ``g`` from a ``(l_i x l_j) x 1e`` product
(antisymmetric on-site), so the prediction is equivariant under rotations,
including spin rotations, and obeys TRS and Hermiticity by construction. The
losses are limited to ``elementwise_mse``, ``elementwise_mae``,
``block_type_mse`` and ``block_type_mae``, without physical weights. There is
no per-channel loss weighting yet. Rejected with spin-orbit, before any
training: magnetic spin-orbit, density and energy-density matrices,
``matrix_targets``, ``physical_labels``, ``spectral_window``,
``coefficient_training``, ``return_coefficients``, ``output_scaleshift``,
``long_range``, ``hamiltonian_local_context``, ``fuse_homonuclear``,
``basis_grouping`` other than ``"point_type"``, SK and low-rank readouts and
``n_output_components``. ``MatrixWriter`` and the HTTP server are not supported
yet and raise; ``PlotMatrixError`` works (it plots the channels as plain
components).

Known limits, measured on a real Bi2Se3 SIESTA Hamiltonian
(``auditorias_mejoras/soc_roundtrip_check.py``). The code is tested on CPU and
has not been scientifically validated by training on a SOC dataset.

- **Graph support.** The graph keeps blocks of atom pairs closer than the sum
  of their orbital radii. About 14k of the 72.5k nonzeros per component lie
  beyond it (Kleinman-Bylander projector tails, up to 1.8e-2 eV); truncating
  to the support shifts the eigenvalues by up to 12 eV at Gamma and 1.7 to
  3.4 eV at random k relative to the stored H. This is not a reconstruction
  error: a direct (non-delta) recipe needs longer readout edges (roadmap 8.2)
  or another treatment of the tails, whereas the delta recipe recovers them
  from H0. The roundtrip is exact only against the stored matrix cut to the
  support and projected onto TRS.
- **Reachable irreps.** MACE hidden irreps of natural parity only, for
  example ``8x0e+8x1o+8x2e+8x3o``, cannot reach the ``0o`` part of ``g``
  (about 1.2 % of the ``g`` coefficients for a Bi-like 2s2p2d basis: 8 of 630
  on-site and 16 of 1296 per edge). Include both parities, as above, in
  campaigns.

Optional training batches by declared cost
--------------------------------------------

Set ``max_batch_cost: 100656`` under ``data:`` in the Lightning CLI YAML (or
``MatrixDataModule(max_batch_cost=100656, ...)``) enables greedy training-only
batching. The default ``None`` retains the fixed-size loaders. The positive
integer budget counts ``C * (sum_i n_i**2 + sum_directed_edges n_i*n_j)``, where
``n_i`` is the number of basis orbitals and ``C`` is the component count. It
does not inspect label values. ``batch_size`` remains an additional limit on
the number of structures. Validation, test and prediction retain fixed-size
batching. No sample is dropped; one that exceeds the budget is emitted alone
with a warning. Ghost-only samples can have zero cost.

This proxy is **not a VRAM cap or an OOM guarantee**: it excludes backbone
activations, gradients and allocator overhead. An oversized singleton may still
fail. Smaller batches can increase CPU overhead and optimizer steps. Changing
grouping changes the optimization trajectory, so accuracy comparisons need
paired update budgets and seeds; equivalent predictions do not prove better
accuracy.

The first training loader reads the effective dataset once after the validation
split, retaining only integer costs, not another sample cache. This adds initial
I/O/conversion time. Costs are reused while the dataset object is unchanged,
rebuilt on replacement and cleared by ``teardown(None)``. Treat its contents,
geometry, basis and component count as immutable; rebuild the DataModule after
changing them. Dynamic ``rotating_pool_size`` and nonstandard samplers are
rejected for this option. ``pair_difference_mse`` is also rejected before the
training forward, including stage changes: this sampler cannot guarantee that
partners remain together. Custom losses requiring co-batched samples are the
caller's responsibility.

In Lightning, enabling cost batching also enables
``reload_dataloaders_every_n_epochs=1`` with a warning. This public setting is
needed to refresh the length when a later epoch packs differently; otherwise
Lightning can silently truncate it. Train/validation loaders are recreated,
but their datasets and integer costs are reused. Single-process sampling uses a
private generator seeded by the initial Torch seed plus the current epoch;
standard DDP sampling retains Lightning's seed and epoch handling.
The epoch plan remains reserved even if workers exhaust the sampler during
prefetch, so later length queries cannot reshuffle or truncate the active epoch.
Use ``Trainer.fit(model, datamodule=module)`` for this integration. Prebuilt
loaders passed separately to a Trainer do not receive the DataModule's epoch
reload handling or Lightning's constructor capture for sampler reinjection.

With standard ``DistributedSampler``, each rank packs its native shard. The
sampler computes the maximum batch count over all shards using the same Torch
sampler/seed/epoch, without collectives inside ``len``. Shorter ranks repeat
their local batches cyclically until all ranks take the same number of steps.
These extra batch repeats are logged by rank/epoch, in addition to the native
sampler's sample padding. Epoch metrics count **all actually processed
elements, including repeats**, not each original structure exactly once.
Use single-device evaluation when exact once-only coverage is required.

The paired synthetic CPU probe in
``auditorias_mejoras/bench_batching_coste.py`` measures proxy peaks, tensor bytes,
initial prepass/RAM and collate overhead. It is not an end-to-end GPU benchmark.

Dataset and worker resource ownership
-------------------------------------

With ``copy_root_to_tmp=True``, each process owns a unique temporary directory.
A ZIP root is copied as an archive, preserving any internal root subdirectory.
Other DDP ranks create their own copy during setup. This avoids shared deletion
races but duplicates copying and disk space per rank.

Normal stage teardown keeps temporary files for subsequent stages.
``datamodule.teardown(None)`` closes datasets and archives and removes its own
temporary directory. Normal process exit also removes that directory. Abrupt
termination such as SIGKILL cannot run Python cleanup hooks.

``RotatingPoolData.close()`` stops its transfer thread, reader process, queue,
and Manager. The DataModule calls it after fit, retaining the underlying dataset
so that a later fit can create a fresh pool. Repeated setup does not create extra
pools. Standalone users of a rotating pool must call ``close()`` when finished.
Reader errors, failed sample deserialization and unexpected reader exits are
reported to pool consumers and DataLoader workers; a failed pool never silently
continues training on its last cached samples.

The pool accepts ``seed=...``; by default its seed is drawn from NumPy's global
seeded generator, so ``seed_everything`` also controls its sampling order.
Asynchronous replacement timing remains scheduling-dependent: seeded sampling
does not make rotating-pool training exactly deterministic. Use a regular
dataset instead when exact reproducibility is required.

File-backed ZIP datasets reopen every archive independently after fork, even
when the parent has read a sample first. Their ``close()`` method closes only
readers they own, not archives supplied by the caller.

Completely absent sparse blocks are supplied lazily when labels are flattened,
without inserting them into the input matrix. They retain the converter's
missing-entry policy: zero for Hamiltonians and NaN for density/energy-density
matrices. A wholly empty multi-component matrix retains its component axis.

Sparse block coordinates are built without densifying the global matrix.
Reverse periodic edges keep the original value traversal order, including
rectangular blocks. Multi-component CSR conversion stops at the first unused
(negative) column of each row, keeps the last value of duplicate coordinates,
and retains block insertion order. Integer fill values no longer truncate
fractional components; real blocks retain float64 allocation and raw complex
components retain their imaginary parts. This does not authorize physical
Hamiltonian serialization of arbitrary raw channels or certify the scalar
complex/CUDA conversion paths.

Extrapolation history uses this same converter and component policy without
modifying the caller's sparse entries. ``MatrixTimeSeriesState.add_last_matrix``
with ``all_dims=True`` retains raw components, including explicit overlap;
raw storage does not authorize physical Hamiltonian serialization.
``HistoryCopy`` accepts the extrapolator's descriptor options without changing
its objective for complete histories. Histories retain both edge directions for
non-symmetric matrices and mark image blocks outside an older auxiliary cell as
missing instead of indexing a different image.

Incomplete histories align descriptors with the available observations of each
block. ``HistoryCopy`` retains its global matrix descriptor and existing
constant/order terms; it selects columns present in those available frames.
SOAP uses each frame's atom descriptors even when an onsite label is absent.
NaN/Inf descriptor features in the target and unusable historical rows are
excluded from the least-squares fit. Partially missing matrix labels are fitted
independently per component; a component with no observation contributes zero
correction. Entirely unknown DM/EDM blocks are omitted, while absent Hamiltonian
entries retain their known-zero policy. Empty histories, or histories without
any observed matrix block, raise ``ValueError`` instead of returning a fabricated
prediction. Inputs are not modified, and complete histories keep the original
least-squares objective and dtype.

SOAP descriptors require ``graph2mat[extrapolation]``. They use atomic numbers,
not arbitrary SIESTA atom tags, and honor each configuration's partial PBC.
Local image vectors are Cartesian vectors from integer cell offsets and the
cell of each historical configuration, including variable-cell trajectories.
The optional dependency and its public API were verified against DScribe 2.1.2;
see `DScribe SOAP documentation <https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html>`_.

Reproducibility and checkpoint provenance
--------------------------------------------

Use ``seed_everything: 0`` in the Lightning CLI YAML (or
``pytorch_lightning.seed_everything(0, workers=True)`` in Python) to seed the
model, data loading and rotating pool. ``trainer.deterministic: true`` requests
deterministic kernels, which can reduce throughput and may reject operations
without a deterministic implementation. It does not remove the rotating pool's
asynchronous scheduling dependence.

For script-based distributed CPU training, set ``trainer.accelerator: cpu``,
``trainer.devices: 2`` and ``trainer.strategy: ddp`` (Gloo). During distributed
training, parameters absent from a rank's matrix loss receive connected zero
gradients, including MACE's unused energy heads. The loss value is unchanged;
this permits ordinary DDP without unused-parameter detection. The optimizer still
excludes those energy heads. This adds zero-gradient materialization/communication
in DDP; the single-process path is unchanged. It does not by itself establish
distributed epoch-metric correctness, which is verified separately.

Epoch metrics for the basic block-type/elementwise MSE and MAE objectives,
node-only/edge-only MSE and block-type Huber/Smooth L1 use valid matrix-element
sums and counts, not averages of
batch losses. ``val_loss`` and ``hp_metric`` share that global objective. Per-step
training losses and their gradients are unchanged. Absolute-error means, sample
standard deviations and maxima also reduce across batches and distributed ranks;
standard deviation is zero with fewer than two valid elements.
Distributed samplers may pad uneven datasets with repeated
samples: the metrics count the samples actually evaluated, including those repeats.
Block-type Huber's per-branch ``node_smooth_l1`` and ``edge_smooth_l1`` statistics
use those same valid-element means, including scalar and spin labels.
The basic matrix losses ignore node and edge rows with any NaN label component,
using the same mask for relative-error references and epoch denominators. Empty
branches contribute zero with connected zero gradients. Empty threshold selections
have zero mean/fraction, and fractions count scalar elements, including channels.
Loss-side standard deviation retains its existing backend convention (sample std
in Torch, population std in NumPy) when there are at least two values, and is zero
otherwise. Physical-block Huber and the composite label-Huber component retain
block boundaries after masking; wholly missing blocks are excluded from the
block mean and per-type statistics. A wholly missing branch contributes zero.
Physical-block Huber and the complete composite objective also aggregate branch
means globally. Coefficient MSE/MAE diagnostics use valid coefficient elements
per operation; the total averages the observed operation means equally, as in
the per-step objective. Composite adds its configured coefficient weight to
the global label-Huber component. ``coefficient_blocks`` counts operations (its
historical meaning), and reports unique observed operations per epoch, not the
sum of batch counts. Missing operations contribute no weight; all ranks log the
same ordered schema, even with disjoint species/pairs. A projected block with a
NaN coefficient is unobserved as a whole: partial matrix labels cannot determine
its full irreducible coefficients. Empty/unobserved coefficients have connected
zero gradients. Coefficient targets and MSE/MAE also accept label rows with
multiple matrix components, provided their count matches the readout. Components
are concatenated within each coefficient row, not mixed with matrix-element
indices. Compatible per-interaction readouts share one target coordinate system;
their aggregated coefficients use the same epoch denominators. These extensions
do not change the single-component contract of the physical Huber/composite loss.
Coefficient projection/reconstruction follows the signed operation and physical
orbital orientation for ``point_type`` and ``basis_shape`` grouping. Points and
edges without basis functions do not require coefficients. Reconstruction with
``max`` grouping crops the padded blocks to their physical basis, including
directed edges and multiple components. Direct coefficient targets with genuine
padding are rejected: the labels do not uniquely determine unobserved grouped
coefficients, and zero-padding would impose additional targets. Use a matrix
loss until a separately certified masked coefficient objective is available.
``max`` without padding still admits the exact projection.
The CPU-prepared candidate in
``graph2mat.bindings.torch.data.coefficient_targets`` separates these diagnostics
from an element-weighted matrix MSE. ``CoefficientMatrixMSEProjector`` snapshots
the effective Q and local angular-pair groups without retaining the model; rebuild
the snapshot if Q changes. Local QR factors preserve Q roundoff (including a
float32-to-float64 cast), physical padding masks, component order and the target's
orthogonal residual. ``coefficient_matrix_mse`` weights both edge SSE and element
count by ``edge_weight`` and keeps its products outside external autocast.
Whole missing blocks have connected zero gradients; partial NaN blocks, unsupported
RTP groups or masks cutting a local support raise ``CoefficientMSEUnsupported``
to request the ordinary matrix route. Graph2Mat and MatrixMACE expose
``forward(..., coefficients_only=True)``: Graph2Mat returns node/edge/metadata
dictionaries, and MatrixMACE returns node_coefficients/edge_coefficients plus
metadata, without node/edge labels. Sum/mean per-interaction aggregation is
unchanged. Default forwards still return their original matrix labels.
MatrixMACE rejects coefficient-only calls with reentrant gradient checkpointing
or long-range label corrections; their callers must use matrix mode.
``TypeLayoutCollate`` accepts an optional CPU projector, and
``MatrixDataModule.set_coefficient_mse_projector`` sets it for subsequently created
training loaders only. Set it after restoring Q and before creating worker
iterators. Targets travel outside the PyG store, so ``get_example`` remains
unchanged; Lightning transfers the nested tensors separately. Directed batches
do not receive the symmetric type layout. Uncertified batches carry a fallback
reason instead of invented targets; invalid shapes still raise. The real-MSE
certificate rejects complex Q/labels before any cast that would lose imaginary
parts. The projector also requires fixed Q and a certified transpose map in
every per-interaction readout; trainable Q or incompatible maps request fallback.
Factors in prepared targets share the CPU snapshot and must be consumed read-only.

``LitMACEMatrixModel(coefficient_training=True, loss="elementwise_mse")``
opts into this route during training. In the Lightning CLI YAML, set
``model.coefficient_training: true``. The default is false; old checkpoints
without this flag retain matrix training. Normal ``MatrixDataModule`` fits
create the CPU snapshot after checkpoint restoration and output scaling, before
train worker iterators, and clear it after fit. Validation, test and prediction
keep matrix outputs. A manually supplied train DataLoader without this DataModule
uses matrix fallback rather than projecting labels on the accelerator.

Certified train steps return only ``loss``, not matrix labels. Global epoch loss
and node/edge RMSE use the physical SSE and element counts through the existing
TorchMetrics update/reset/DDP path. The objective remains element-weighted MSE,
not the equally weighted operation means of ``coefficient_space_mse``. Eligibility
is checked before each forward, including when training stages change the loss.

MAE/Huber/gauge, verbose matrix statistics, gradient checkpointing, long-range
corrections, training callbacks needing labels, custom batch-transfer or train
batch hooks and custom logging overrides use one ordinary matrix forward with
a warning once per reason. Validation/test-only MatrixWriter and samplewise
logging remain compatible. Real float32/float64 and finite nonnegative edge
weights are certified; external autocast additionally requires ``readout_fp32``.
Strategies restoring weights after setup fall back because the CPU snapshot
would otherwise precede the restored Q. Changing Q or labels after preparation
is outside this fixed-snapshot contract.

CPU fits and two-rank Gloo comparisons establish loss/gradient/epoch equivalence,
not a GPU speed, VRAM or accuracy improvement. End-to-end CUDA AMP remains
unverified; the basal MACE backbone itself fails under CPU BF16 autocast before
the readout, so its eligibility guard is not an end-to-end AMP certificate.

Gauge H and optional predicted S use independent valid-element means, with the
configured edge weight. Alignment ignores any H/S reference row with a NaN
component before subtraction; missing rows have zero gradients. Empty gauge
branches return connected zero. Shift magnitudes average structures/components,
including a zero shift for an entirely unobserved structure.
The optional spectral term reduces its own squared-error sum over
``sum(k_window**2)`` across batches/ranks, not the number of structures.
``spectral_elements`` reports that summed denominator; zero padding and empty
windows add no weight. The per-step spectral formula and gradients are unchanged.
Spectral projection requires complete H/S reference labels and finite
``spectral_C``: NaN inputs raise explicitly instead of inventing missing matrix
elements.

Pair training aggregates its base MSE by valid elements and its difference term
by observed pairs, then applies ``pair_weight``. ``n_pairs`` is summed and
``pair_rmse`` takes the root after the global reduction. The original step
formula, including the symmetric Gamma reconstruction and scalar-component
normalization, is retained. Paired reference matrices must be complete;
unpaired structures keep the base MSE's NaN mask.
Partners must remain together within a training batch/rank: splitting them
changes the step's physical signal, not just its aggregation. ``expected_pairs``
can fail explicitly when this contract is violated. No pairs are invented
across steps or ranks. Epoch values are invariant under partitions retaining
the same observed pairs. Validation/test are plain global MSE with no pairing
or training pair guard, as before, for any partition.

Relative MAPE/mix epoch objectives keep the original ``abs(ref)>1e-6`` selection
for relative errors, distinct from the valid-element selection used by their
absolute MAE/MSE terms. Each node/edge term has its own global denominator.
STD-mix uses global sample variance of relative errors (Torch's convention),
not a mean of batch standard deviations. Empty relative selections contribute
zero without removing observed absolute terms. Per-step definitions, weights
and gradients are unchanged.

Normalized density error is a non-differentiable sisl postprocessing diagnostic,
not a training loss. Its epoch objective averages ``integral(abs(delta_rho))/q0``
per structure; ``avg_error_percent`` instead divides the summed integrals by
the summed electron counts. Verbose counters sum across batches/ranks. Conversion
preserves the data's symmetry/component convention without adding the atomic
reference to errors. Missing reference rows contribute no observed density error;
this is not a reconstruction of their unknown density. Non-finite observed errors
and non-positive/non-finite electron counts raise explicitly. Empty batches and
zero error matrices return finite zero without a sisl empty reduction.

Size-domain epoch losses first average per-structure losses in each domain, then
apply its configured weight, normalized over domains present in the epoch.
``n_structures``/``n_large_structures`` are sums. As in the step objective, a
structure without observed labels contributes zero but remains in its domain.
Threshold objectives retain their original **sum**, rather than being changed
to element means: the epoch objective is the sum across batches/ranks. Their
above-threshold means count only selected scalar elements, and fractions count
all valid scalar elements, including spin channels. Empty selections contribute
zero. Smooth thresholds evaluate the same logistic function with stable
Torch/SciPy primitives, preventing overflow from poisoning gradients.

With the default ``SamplewiseMetricsLogger`` failure-tolerant policy, a failed
metric writes NaN to its sample rows and emits a warning on its first failure per
metric class and callback instance. Explicit metric lists retain the strict policy
that raises the original exception.

``PlotMatrixError`` preserves periodic image columns and averages each entry over
its valid observations, not the total number of matrices. Partially missing label
rows exclude all their components, consistently with the losses; unobserved entries
stay blank. It plots separate ``MAE_componentN``/``RMSE_componentN`` maps for
multi-component data instead of treating errors as a physical spin/overlap matrix.
Atom types, edge/image ordering and label shapes must match throughout the split;
an incompatible layout raises an error instead of summing unrelated entries.
MD coordinates may move. Each epoch/split resets the accumulated maps, and the
matrix processor's atomic reference is never added to an error map.

Checkpoints include a ``provenance`` dictionary with Python and installed
versions of Torch, PyG, Lightning, e3nn, MACE and sisl, plus the tracked source
checkout's Git commit and dirty flag. Missing optional distributions or a
checkout unavailable in an installed wheel are represented by ``None``.
The Git directory comes from the module, not the caller's working directory;
untracked files do not set the dirty flag. Provenance is informational: old
checkpoints without it still load, and prediction weights are unchanged.

``OrbitalMatrixCLI`` can run ``test``/``predict`` from ``--ckpt_path`` with the
stored model/data configuration, while explicit CLI options retain priority.
It uses Graph2Mat's existing checkpoint-config loader rather than Lightning's
additional weights-only reparse. These checkpoints contain Python objects:
load **only trusted files**. No process-wide Torch safe-global/loading policy
is changed. ``MatrixWriter``/``SamplewiseMetricsLogger`` split options accept a
string or sequence through the CLI as well as Python.

Programmatic ``AtomicTableWithEdges`` checkpoints record per-orbital cutoffs
separately from sisl's radial serialization. This preserves exact graph support
even without a radial function, and avoids cutoff changes from sampled radial
data; it does not invent radial functions or change interpolation. The original
file-content representation and legacy ``atoms`` states remain accepted. An old
atoms-only checkpoint that already lost a cutoff cannot recover that missing
information: use the original basis rather than guessing a radius.

Joint H/S/DM/EDM targets (opt-in)
--------------------------------

``matrix_targets=None`` retains the single-target path, parameter keys and old
checkpoint loading. Selecting ``hamiltonian`` and ``overlap`` enables joint
prediction; ``density_matrix`` and/or ``energy_density_matrix`` may be added.
One MACE backbone feeds the existing multicomponent readout. Its packed
columns are not spin channels of a single Hamiltonian: the version1 descriptor
fixes canonical H,S,DM,EDM order, slices, units and real spin1/2. The complete
set has four unpolarized or seven collinear columns; S is always scalar.

This mode learns direct matrices and requires ``sub_point_matrix=False``.
There is no implicit H0/DM0. Select whole matrices only when their labels exist,
even if a selected target has weight0. Real local FDF/TSHS/TSDE or a compatible
sisl bundle are supported; new-mode ZIP is rejected. The TSHS supplies actual
overlap; the TSDE reader's absent-overlap placeholder is never a target.
Geometry, basis/orbital order, cell/images, spin and H/EDM energy gauges must
match. H/EDM use eV; S is dimensionless; DM uses sisl occupation components
without another factor2. Fractional occupation is allowed, without idempotency.

The explicit ``matrix_target_weights`` and ``matrix_target_losses`` mappings
must cover precisely the selected names. Weights are finite/nonnegative with
at least one positive value. H selects ``elementwise_mse`` or
``gauge_invariant_mse``; S/DM/EDM select ``elementwise_mse`` initially. Each
target has its own denominator and conventional units. Weights are not an
automatic normalization or a scientifically tuned recipe. A target with zero
weight is not evaluated; an active target without observations in any sample
raises before optimization. Unknown DM/EDM coefficients are excluded by
component masks before arithmetic, not filled with observed zeros. Explicit
stored zeros remain observed. A sisl container with shared component support
cannot recover discarded per-spin support; an explicit validated
``observation_masks`` bundle can restrict that support, never invent it.
Archived DM/EDM support, including explicit zeros, must itself be Hermitian
across opposite images before compacting edges; asymmetric support is rejected.

Optional ``physical_labels`` can map target names to ``[charge, energy]``
selections. The existing list shorthand means charge for selected DM and
energy for selected H/DM/EDM. Matrix-specific loss kwargs
``electron_weight``, ``mulliken_weight`` and ``energy_weight`` remain zero by
default. Charge/Mulliken apply only to DM; energy requires certified full
H/S/DM/EDM partners and an explicit supported SCF mesh. Both gauge alignment
and these physical terms use S_ref, never S_pred. Energy is the perturbation
objective continuing the reference outside the graph, not an identity between
four predictions or the exact energy of the truncated exported matrix.

``MatrixMACE.forward`` additionally returns differentiable views at
``output["matrix_predictions"][target]["node_labels"|"edge_labels"]``.
Geometry-only prediction obtains its descriptor from configuration/checkpoint;
it does not read reference matrices as features. Batches may mix sizes/species,
not semantic layouts or spins. Checkpoints persist and validate the descriptor,
loss configuration and weights; matching column counts alone are insufficient.

The CPU YAML example is exercised by the real CLI checkpoint prediction test:

.. literalinclude:: examples/multitarget_predict.yaml
   :language: yaml

Supply a trusted matching checkpoint to ``predict --config <example.yaml>
--ckpt_path <checkpoint> --data.root_dir <directory>``. Its weights demonstrate
syntax only; no scientific accuracy is established. For fit, replace
``predict_structs`` with explicit train/validation FDF splits and retain the
matrix/layout/loss settings. This does not authorize or prescribe a campaign.

Reconstruction always selects a target and carries predicted overlap explicitly:

.. code-block:: python

    from graph2mat.bindings.torch.load import load_from_lit_ckpt
    from graph2mat.tools.lightning.callbacks import MatrixWriter

    model, processor = load_from_lit_ckpt(trusted_checkpoint, as_torch=True)
    model.eval()
    hamiltonian = processor.torch_predict(model, geometry, matrix_target="hamiltonian")
    writer = MatrixWriter("predicted.nc", matrix_target="density_matrix")

H returns ``sisl.Hamiltonian`` with S_pred; S returns ``sisl.Overlap``; DM/EDM
carry their physical channels and S_pred. No hidden S_ref, repair of non-SPD S,
eigenvalue clipping or recovery beyond graph support is performed. Numeric
``threshold`` affects the selected H/DM/EDM, not overlap. Spectra therefore
need their own S_pred validity check. Export initially supports explicit
target-specific NetCDF, in model energy gauge with serialized Ef0. S uses an
explicit-overlap carrier to bypass sisl's Overlap writer identity convention;
its coefficients are preserved. Invalid targets/formats/finite values and
within-batch path collisions fail before writing matrices. This is preflight
protection, not a transaction against arbitrary filesystem/IO failures.

``SamplewiseMetricsLogger`` requires a target and explicit supported MSE/H-gauge
metrics in joint mode; CSV includes target/unit. ``PlotMatrixError`` selects
one target and excludes unknown individual components. Legacy row masks stay
unchanged. CSV headers initialize on first use, after the joint-mode selector
guard, so constructing an invalid callback cannot truncate its output file.
A ``matrix_target`` selector on a legacy metrics logger is rejected before
opening the output; omit it for the unchanged single-target CSV schema.
Generic HTTP endpoints reject unsplit joint models rather than misreading spin.

Long-range correction, Hamiltonian context, coefficient-only training and
single-target loss replacement stages are rejected in joint mode. Standard
and low-rank readouts are CPU tested; SK retains its existing C1/C2 limit and
does not accept the complete4/7 packing. ``output_scaleshift=True`` requires
fully observed finite labels for coefficient fitting; partial observations
raise before fitting any buffers. Leave it disabled (the default) for partial
targets; the masked matrix losses still support them without imputation.
Compile backend=eager CPU preserves
outputs/gradients/keys, not evidence about CUDA/cuEq/VRAM. More outputs can
increase memory and runtime; CPU fixtures do not prove improved prediction.
