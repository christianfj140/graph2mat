# Changelog

## Unreleased — branch `g2m-v2` (relative to upstream `main`)

Work in progress on a fork; nothing here has been released. Every new model or data
feature is opt-in unless listed under "Changed defaults". Maturity levels:

- **stable**: tested on CPU and used in trained, compared recipes;
- **experimental**: tested and trained on one or two systems only;
- **untrained**: code and tests only, never trained on real data.

The research record behind each item (measurements, decisions, negative results) is
`auditorias_mejoras/roadmap.md` (in Spanish; its "Estado vigente" section is the
current state).

### Changed defaults

- `LitMACEMatrixModel(edge_block_readout=E3nnLinearEdgeBlock)`: a linear term next to
  the tensor product plus an on-site 0e bias; much lower H error in every system tried.
- `lean_mace=True`: MACE without energy readouts or position gradients (same output).
- The two directions of a symmetric edge block are computed in one call.
- The collate computes the per-type batch layout on the CPU workers (no device syncs).
- `MatrixDataModule` / CLI infer `symmetric_matrix` from the matrix type (H, DM, EDM).

The model hyperparameter defaults (`correlation=1`, `max_ell=3`, `num_interactions=2`,
`num_polynomial_cutoff=3`, `loss=elementwise_mse`) are unchanged. The recommended
recipe (fp32, `num_polynomial_cutoff=6`, correlation 3, lmax 3, 3 interactions,
`gauge_invariant_mse`, Δ-learning) is `docs/examples/benchmarks/delta_recipe.yaml`.

### Added — stable

- Δ-learning on the non-self-consistent H0 of the superposed atomic density:
  `graph2mat.tools.siesta.delta` and `graph2mat siesta delta {h0,label,add-h0}`.
- Overlap-aware training: (H, S) labels (`matrix_component_policy="h_and_overlap"`),
  `gauge_invariant_mse` (one c·S shift per structure, common to both spins), H-only
  prediction with S as a label (`n_output_components`).
- Spin-polarized (collinear) Hamiltonians: `matrix_component_policy="spin_h_only"`.
- Training stages (`training_stages`: per-stage loss, frozen/trained parts, learning
  rate), `selection_metric`, weight-decay groups.
- Options with measured but recipe-dependent effect: `pair_envelope`, `mp_cutoff`,
  `edge_message_mid_irreps`, `edge_update_layers`, `readout_per_interaction`,
  `simplify_node_irreps`, `keep_last_layer_irreps`, `output_scaleshift`,
  `basis_grouping`, `E3nnLowRankEdgeBlock`, `E3nnSlaterKosterEdgeBlock`,
  `E3nnEdgeBlockNodeMix`, `long_range` (`"charges"`, `"qeq"`),
  `hamiltonian_local_context`, `coefficient_training`.
- Physics losses and labels: first-order spectral term (`spectral_weight`,
  `MatrixDataModule(spectral_window=...)`), S null-space term, electron count,
  Mulliken and energy consistency (`physical_labels`).
- Performance options: `gradient_checkpointing`, `torch_compile`, `cueq`,
  `max_batch_cost`, `fuse_homonuclear`, tensor caching, vectorized CSR/COO
  reconstruction.
- Evaluation tools: `graph2mat.tools.analysis` (Γ and band/DOS metrics, gauge-aligned
  errors, error breakdown by shell/pair/l, OOD attribution, KB-tail control) and
  `graph2mat.tools.spectral_targets`.
- SIESTA helpers: `graph2mat.tools.siesta.runner` (atomic runs, hash receipts for
  cached results), `label_noise`, `dm_pilot`.

### Added — experimental

- Spin-orbit Hamiltonians without magnetism: `matrix_component_policy="spin_orbit"`
  (4 time-reversal-symmetric channels), `E3nnSpinOrbitMatrixBlock`,
  `LitMACEMatrixModel(spin_orbit=True)`, Δ recipe with TRS projection
  (`graph2mat siesta delta label --trs`). Trained on Bi2Se3 only.
- Longer readout edges for KB-projector blocks: `edge_cutoff_padding`.
- Multi-target H + S + DM + EDM (`matrix_targets`): works, no benefit measured.
- Training in the irreps space (coefficient path): works, no speed-up measured.

### Added — untrained

- Magnetic spin-orbit (`matrix_component_policy="spin_orbit_full"`, 8 channels) with
  per-atom magnetic moments as input (`magnetic_moments`, collinear or vector) and exact
  time reversal. `MatrixWriter` writes these predictions; the HTTP server refuses them.

### Changed behaviour: inputs that now raise instead of giving wrong results

- A species whose basis differs from the basis table (same Z, other size, other
  angular shells or shell order); model and data basis tables that differ.
- An explicit `nsc` different from the matrix's in `from_config`.
- Symmetric edge labels that cannot be matched to edge representatives (no guessing).
- Model options that would be silently ignored in the requested combination
  (for example `mp_cutoff` without a radial embedding for the readout).
- `torch_compile_mode="reduce-overhead"`.
- `lr_scheduler` in a CLI config with `optimizer: null` is now applied to the model's
  optimizer (it was dropped). Resuming a checkpoint trained without a scheduler with
  one in the config raises.
- SIESTA tools: fdf files with `%include` or `<` redirections in text rewrites; cached
  results without a matching receipt.
- Freezing training stages under DDP without `find_unused_parameters`.

### Fixed (also present upstream)

- `_labels_resort` index bug (reported upstream).
- Point reference subtraction for multi-component blocks; complex values in the
  sparse-to-block conversion; `load_labels=False` on prepared samples.
- Many correctness fixes in losses, metrics, callbacks, the server and data loading;
  see `auditorias_mejoras/auditoria_bugs_20261021.md`.

### Dependencies

- Version ranges pinned in `pyproject.toml` (torch ≥2.4.1, e3nn ≥0.4.4,<0.6,
  mace_torch ≥0.3.15,<0.4, pytorch-lightning ≥2.1.4, jsonargparse ≥4.48,
  sisl ≥0.16.2,<0.17, ...). New `extrapolation` extra (`dscribe`).
- Cython extensions changed (`_sparse`, `_labels_resort`): editable installs must be
  rebuilt.

### Known limitations

- The full test suite takes about 45 min on CPU; CI has not run on GitHub for this branch.
- A DM/EDM read from a `.DM` file without coordinates is not checked against the
  label geometry.
- Per-structure losses average structures that have no labels in the batch.
- Band/eigenvalue metrics are offline tools; they are not part of Lightning validation.
- Two species with the same Z and different basis are rejected, not supported.
- Multi-GPU is out of scope; DDP is tested on CPU only.
