Data loading and resource ownership
==================================

``MatrixDataModule.setup`` loads only the splits needed by the stage:
``fit`` uses train/val, ``validate`` uses val, ``test`` uses test, and
``predict`` uses predict. Passing ``None`` initializes every available split.
Repeated setup calls reuse existing datasets. Without explicit validation data,
validation can draw its subset lazily from training data; ``store_in_memory``
materializes only the splits requested by the current stage.

The automatic validation split accepts ``split_seed`` (default 32) and
``val_fraction`` (default 0.1, in [0, 1)). Explicit ``val_runs`` take priority;
an explicit empty list disables validation. The automatic split keeps at least
one training sample when data are available. For correlated trajectories, use
explicit validation runs from separate trajectories rather than a random split
of adjacent frames.

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
----------------------------------------

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
standard deviation is zero with fewer than two valid elements. Pair objectives
still have separate pending aggregation work.
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
zero gradients. The existing flat single-component H-only contract is unchanged.
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
elements. Pair objectives still have separate missing-label requirements.

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
