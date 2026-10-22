# Hamiltonian Component Policy Design

## Summary

The current `hamiltonian-spin-colineal-support` branch preserves every component
present in a `sisl.SparseCSR.data` array. That is useful for experimental spin
work, but it also means a non-spin, non-orthogonal SIESTA Hamiltonian can expose
raw `(H, S)` data as two label components. Graph2Mat then treats both columns as
trainable matrix targets, so the overlap matrix `S` can be silently mixed into a
Hamiltonian training objective.

The proposed fix is to add an explicit `matrix_component_policy` to the data
path. The safe Hamiltonian default should train Hamiltonian components only and
exclude overlap unless overlap is explicitly requested.

No production code is changed by this document.

## Current Data Flow

```text
SIESTA run files
  -> OrbitalConfiguration.new(..., out_matrix="hamiltonian")
  -> _sisl_run_to_orbitalconfiguration
  -> sisl.get_sile(...).read_hamiltonian(...)
  -> sisl.Hamiltonian._csr / sisl.SparseCSR.data
  -> _sisl_to_orbital_configuration
  -> csr_to_block_dict
  -> _csr_to_block_dict_components when spmat.data.shape[1] > 1
  -> OrbitalMatrix / BasisMatrix block_dict
  -> BasisMatrix._flatten_block
  -> BasisMatrixData.point_labels / edge_labels
  -> TorchBasisMatrixData / MatrixDataModule
  -> CLI infer/link data.n_matrix_components -> model.n_matrix_components
  -> LitMACEMatrixModel / MatrixMACE
  -> block_type_mae and other metrics
  -> MatrixWriter
  -> MatrixDataProcessor.labels_to
  -> nodes_and_edges_to_sparse_orbital
  -> csr_to_sisl_sparse_orbital
  -> sisl.Hamiltonian written to HSX/TSHS-like output
```

## Where H And S Become Trainable Components

The mixing is not introduced in the loss. It is introduced earlier, during matrix
reading and label construction:

- `src/graph2mat/core/data/configuration.py`
  - `OrbitalConfiguration.new(..., out_matrix="hamiltonian")` dispatches path
    inputs to `from_run`.
  - `_sisl_run_to_orbitalconfiguration` uses
    `getattr(main_input, f"read_{out_matrix}")`, so `out_matrix="hamiltonian"`
    calls the sisl `read_hamiltonian` path.
  - `_sisl_to_orbital_configuration` passes `matrix._csr` directly to
    `csr_to_block_dict`.

- `src/graph2mat/core/data/sparse.py`
  - `csr_to_block_dict` takes `spmat.data[:, 0]` only for one-component data.
  - If `spmat.data.shape[1] > 1`, it calls `_csr_to_block_dict_components` with
    the whole `spmat.data`.
  - `_csr_to_block_dict_components` creates blocks shaped
    `(n_orb_i, n_orb_j, n_components)` and copies `data[ival, :]` into them.

- `src/graph2mat/core/data/matrices/basis_matrix.py`
  - `BasisMatrix._flatten_block` reshapes 3D blocks to
    `(n_elements, n_components)`.
  - `to_flat_nodes_and_edges` concatenates those arrays into `point_labels` and
    `edge_labels`.

- `src/graph2mat/core/data/processing.py`
  - `MatrixDataProcessor.get_labels_from_types_and_edges` calls
    `config.matrix.to_flat_nodes_and_edges` and returns the flattened labels.
  - `BasisMatrixDataBase.__init__` accepts 1D or 2D labels and only checks that
    node and edge labels have the same component count.

- `src/graph2mat/tools/lightning/data.py` and `src/graph2mat/tools/lightning/cli.py`
  - `MatrixDataModule` stores `n_matrix_components` and passes it to
    `MatrixDataProcessor`.
  - `_validate_n_matrix_components` checks that configured and loaded component
    counts match.
  - The CLI auto-infers component count from the first labeled sample and links
    `data.n_matrix_components` to `model.n_matrix_components`.

- `src/graph2mat/core/data/metrics.py`
  - `get_predictions_error` subtracts labels directly.
  - `block_type_mae` averages over all values it receives. If labels are `(H, S)`,
    it optimizes both as equivalent label channels.
  - `_spin_channel_stats` logs every 2D component as `spin{i}`, even if a channel
    is overlap.

- `src/graph2mat/core/data/sparse.py`
  - `csr_to_sisl_sparse_orbital` currently interprets two Hamiltonian CSR
    components as `spin="polarized"` and three or more as polarized spin plus
    overlap. That is ambiguous for non-spin `(H, S)`.

## Proposed API

Add a public data-processing config field:

```yaml
data:
  out_matrix: hamiltonian
  matrix_component_policy: h_only
```

Recommended name: `matrix_component_policy`.

Rationale:

- It matches existing public names such as `out_matrix`, `symmetric_matrix`, and
  `n_matrix_components`.
- It describes the label channels used for training and prediction, not only the
  file reader.
- It can be threaded through `MatrixDataModule`, `MatrixDataProcessor`,
  `OrbitalConfiguration.new`, and direct `OrbitalConfiguration.from_matrix`
  without a new top-level subsystem.

Recommended policy values:

- `h_only`
  - Safe default for `out_matrix="hamiltonian"`.
  - Keep Hamiltonian channels and exclude overlap.
  - Non-spin non-orthogonal Hamiltonian: keep `H` only.
  - Spin-collinear non-orthogonal Hamiltonian: keep spin Hamiltonian channels
    and exclude `S`.

- `spin_h_only`
  - Same component selection as `h_only`, but requires a spin-polarized
    Hamiltonian and fails clearly for non-spin targets.

- `raw_components`
  - Preserve current label-reading behavior.
  - Every raw `SparseCSR.data` component is a trainable label.
  - Required for debugging or legacy experiments that intentionally train raw
    sisl components.
  - Multi-component Hamiltonian serialization is rejected unless a semantic
    policy such as `h_only`, `spin_h_only`, or `h_and_overlap` is used.

- `h_and_overlap`
  - Explicit multi-task mode.
  - Keep Hamiltonian channel(s) plus overlap.
  - This must not rely on component count alone when writing back to sisl,
    because two components can mean either spin Hamiltonian channels or
    unpolarized `(H, S)`.
  - In the write path, two components mean `(H, S)` and three components mean
    `(H_up, H_down, S)`.

Possible future value:

- `component_indices`
  - A lower-level debug path such as `matrix_component_indices: [0, 2]`.
  - This should not be needed for the H2O fix and can be deferred.

## Default

Use the safe default for Hamiltonians: `h_only`.

This is intentionally not fully backward-compatible for experiments that relied
on raw multi-component Hamiltonian labels, because silently training overlap as
Hamiltonian is a scientific correctness issue. Backward compatibility should be
provided by an explicit `raw_components` policy and by clear mismatch errors that
mention the new policy.

For non-Hamiltonian targets, either leave behavior unchanged or accept only
`raw_components` until there is a defined physical component policy for those
matrix types.

## Smallest Safe Change Point

Filter components while converting a `sisl.Hamiltonian` into an
`OrbitalConfiguration`, before blocks and flat labels are produced.

Recommended implementation point:

1. Add `matrix_component_policy` to `MatrixDataProcessor`.
2. Pass it through `MatrixDataProcessor.get_config_kwargs`.
3. Add it to `OrbitalConfiguration.from_run` / `from_matrix` conversion calls.
4. In `_sisl_to_orbital_configuration`, inspect the `sisl.Hamiltonian` object and
   resolve the raw component indices to keep.
5. Pass optional `component_indices` into `csr_to_block_dict`, or slice the
   `matrix._csr.data` equivalent before `_csr_to_block_dict_components`.

Filtering here is safer than filtering in metrics because:

- `n_matrix_components` inference sees the actual training target.
- Metrics and losses operate on already-correct labels.
- Model output shape remains tied to target semantics.
- The raw overlap channel is not carried into labels accidentally.

Filtering here is also safer than filtering only in `BasisMatrix._flatten_block`
because `_sisl_to_orbital_configuration` still has access to `sisl.Hamiltonian`
metadata such as spin and overlap placement. A plain 3D block array has already
lost too much context.

## Component Semantics

Do not infer semantics from component count alone. Use Hamiltonian metadata where
sisl exposes it:

- `matrix.spin` distinguishes unpolarized and polarized Hamiltonians.
- `matrix.orthogonal` and/or `matrix.S_idx` identify whether overlap is stored.
- The overlap component is not a Hamiltonian spin channel.

The intended semantic representation is:

```text
raw component index -> role
0                   -> H, spin=None or spin=0
1                   -> H, spin=1 for polarized H, or S for unpolarized H+S
2                   -> S for polarized H+S
```

The exact resolver should use sisl metadata first, and only use count-based
fallbacks for legacy objects where metadata is unavailable. If metadata and
component count disagree, fail loudly with an error that asks for
`matrix_component_policy: raw_components`.

## Backward Compatibility

Backward-compatible escape hatch:

```yaml
data:
  out_matrix: hamiltonian
  matrix_component_policy: raw_components
  n_matrix_components: 2
```

Migration behavior:

- Existing raw experiments can keep their current label shapes by adding
  `matrix_component_policy: raw_components`.
- Existing non-spin H2O configs with `n_matrix_components: 2` should fail after
  the safe default because labels become one-component `H` labels. The error
  should say either set `n_matrix_components: 1` for Hamiltonian-only training or
  set `matrix_component_policy: raw_components` if the raw `(H, S)` target was
  intentional.
- CLI auto-inference should run after policy filtering, so omitted
  `n_matrix_components` resolves to the policy-selected component count.

I did not find an H2O config file with `n_matrix_components: 2` in this checkout.
The repository contains a hardcoded external H2O path in
`src/graph2mat/core/data/tests/test_h_dimension.py`, but that external directory
is not present in this environment.

## Spin-Collinear Representation

Spin-collinear Hamiltonians should be represented as multiple Hamiltonian
channels, not as raw sisl columns:

- Orthogonal spin-collinear: labels shaped `(n_elements, 2)` for `(H_up, H_down)`.
- Non-orthogonal spin-collinear: labels shaped `(n_elements, 2)` for
  `(H_up, H_down)` under `h_only`; overlap is excluded.
- If overlap is needed for spectral metrics, it should be supplied explicitly as
  overlap data, not silently trained as a Hamiltonian channel.

This preserves the current experimental spin-collinear support while avoiding
the false equivalence between spin channels and overlap.

## Writer And Prediction Semantics

The current write-back path also needs an explicit policy:

- `MatrixWriter` calls `matrix_data.convert_to(data_processor.default_out_format)`.
- `MatrixDataProcessor.labels_to` passes node and edge labels to
  `nodes_and_edges_to_sparse_orbital`.
- `csr_to_sisl_sparse_orbital` maps two Hamiltonian components to polarized spin
  and three components to polarized spin plus overlap.

That count-based mapping is no longer acceptable for policy-selected labels:

- `h_only` with two components means spin Hamiltonian channels.
- `h_and_overlap` with two components means unpolarized `H + S`.
- `raw_components` preserves labels for debugging but does not serialize
  multi-component Hamiltonians as a physical spin object.

Implementation should pass the resolved component roles or the
`matrix_component_policy` into the Hamiltonian conversion path. Do not make
`len(csr) == 2` globally mean spin.

## Test Plan

Tests that should fail before the fix and pass after it:

- `src/graph2mat/core/data/tests/test_spin_collinear.py`
  - Add non-spin non-orthogonal Hamiltonian fixture with raw `(H, S)`.
  - With default `matrix_component_policy="h_only"`, assert
    `OrbitalConfiguration.from_matrix(...).matrix` produces one-component blocks
    equal to `H`, not `S`.

- `src/graph2mat/core/data/tests/test_spin_collinear.py`
  - Add spin-collinear non-orthogonal fixture `(H_up, H_down, S)`.
  - With `h_only`, assert labels keep two components
    `(H_up, H_down)` and exclude `S`.

- `src/graph2mat/core/data/tests/test_spin_collinear.py`
  - With `raw_components`, assert the same fixtures preserve all raw components.
    This protects existing experimental behavior.

- `src/graph2mat/tools/lightning/tests/test_matrix_components.py`
  - Add datamodule/CLI inference test showing filtered H2O-like labels infer
    `n_matrix_components == 1`.
  - Add mismatch test where `n_matrix_components=2` plus default policy raises an
    error mentioning `matrix_component_policy`.

- `src/graph2mat/tools/lightning/tests/test_matrix_components.py`
  - Keep existing polarized tests passing with explicit `raw_components` where
    they are testing current raw conversion behavior.

- `src/graph2mat/core/data/tests/test_sparse.py`
  - Add write-back coverage so unpolarized `h_and_overlap` with two
    components writes a non-orthogonal unpolarized `sisl.Hamiltonian`, while
    spin-collinear `h_only` with two components writes polarized spin.

- `src/graph2mat/core/data/tests/test_metrics.py`
  - Add a direct `block_type_mae` sanity test showing it averages all label
    components it receives. This documents why the labels must be filtered
    before metrics.

## Implementation Plan

PR 1: component policy and label filtering

- Add `matrix_component_policy` to `MatrixDataProcessor`.
- Thread it through `MatrixDataModule`, `infer_n_matrix_components_from_data_inputs`,
  and the CLI autoconfiguration call.
- Thread it through `OrbitalConfiguration.from_run` and `from_matrix`.
- Add a small Hamiltonian component resolver in the data/configuration or sparse
  layer.
- Add optional component selection to `csr_to_block_dict`.
- Add tests for non-spin `H+S`, spin `H_up/H_down`, spin `H_up/H_down/S`, and
  `raw_components`.

PR 2: write-back semantics

- Pass component roles or policy to `nodes_and_edges_to_sparse_orbital` and
  `csr_to_sisl_sparse_orbital`.
- Remove or gate the global assumption that two Hamiltonian components always
  mean polarized spin.
- Add tests for unpolarized `H+S` output and polarized spin output.

PR 3: user-facing migration

- Update docs and examples.
- Improve `n_matrix_components` mismatch messages to mention
  `matrix_component_policy`.
- Add a warning or release-note entry for users who need
  `matrix_component_policy: raw_components`.

PR 4: explicit overlap consumers

- If spectral metrics need overlap, add a separate overlap source in the metric
  or callback path.
- Keep overlap out of the Hamiltonian training target unless
  `h_and_overlap` is explicitly selected.

## Risks And Open Questions

- The exact sisl metadata contract for all Hamiltonian spin modes should be
  verified in the test environment. The resolver should prefer `S_idx`, `spin`,
  and `orthogonal` over count heuristics.
- Existing raw multi-component experiments will need one config line:
  `matrix_component_policy: raw_components`.
- `h_and_overlap` needs careful loss weighting. Hamiltonian and overlap
  have different physical scales and should not be silently averaged as equal
  channels.
- The current `_spin_channel_stats` name is misleading for any non-spin
  multi-component label. After policy filtering, it is acceptable for spin-only
  labels, but raw/debug labels may still produce misleading metric names.
- Prediction-time HSX output for `h_only` non-orthogonal training will
  not contain a learned overlap. That is desirable for the training target, but
  spectral metrics must receive overlap explicitly.
- Non-collinear/SOC Hamiltonians are outside this design. They should be rejected
  or forced through `raw_components` until a separate semantic policy is defined.
