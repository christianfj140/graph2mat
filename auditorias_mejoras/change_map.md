# Change map of `g2m-v2` against upstream `main`

Purpose: prepare reviewable pull requests. Every change of the branch's product code
(`src/`, `tests/`, `docs/`, packaging, CI) is grouped below into a proposed pull request, in
dependency order. Each group says what changes, why, which commits carry it, how a reviewer
verifies it, where the evidence is, and what changes for existing users.

- Commits are listed by short hash (`git show <hash>`); merges and research-only commits are
  left out. `git log main..g2m-v2 -- src tests` lists them all.
- "Evidence" points to `auditorias_mejoras/` (research record, in Spanish): the row of
  `roadmap.md` ("Estado vigente" first, then the row in its wave table) and the reports it
  links. Measurements are from the project's benchmark sets (graphene `w90` and `6x6`,
  bilayers, H2O, TMD, BaTiO3, ZnO, Bi2Se3).
- User-facing summary: `CHANGELOG.md`. Option guide: `docs/model_options.rst`.
- Test commands assume a full install (`pip install -e ".[test,tools,server,extrapolation]"`).
  `pytest -m "not slow"` skips the tests above 1 s.

| PR | Theme | Depends on |
|---|---|---|
| 1 | Fixes of inherited bugs | — |
| 2 | Packaging, optional dependencies, CI | — |
| 3 | Data pipeline: component policies, validation, loading | 1 |
| 4 | Losses and epoch-level metrics | 3 |
| 5 | Overlap-aware training: (H, S) labels, gauge-invariant and physical losses | 3, 4 |
| 6 | Lightning training control: stages, scheduler, selection, callbacks | 4 |
| 7 | Model and readout options | 3 |
| 8 | Long-range electrostatics | 5, 7 |
| 9 | Spin: collinear, spin-orbit, magnetic moments, long readout edges | 3, 7 |
| 10 | Joint H/S/DM/EDM targets | 3, 4, 7 |
| 11 | Server and extrapolation | 3 |
| 12 | SIESTA and analysis tools (Δ-learning workflow) | 5, 9 |
| 13 | Documentation | all |

---

## 1. Fixes of inherited bugs

**What.** Bugs present upstream, each with a test that fails before the fix.

| Fix | Commits | Test |
|---|---|---|
| Label resorting with repeated types (`_labels_resort`; already proposed upstream) | upstream PR, `a66b46c` | `core/modules/tests/` |
| Symmetric edge slicing across batched graphs | `9d8cc0c` | `core/data/tests/test_processing.py` |
| Point reference subtracted from every component of multi-component blocks | `ba9b440` | `core/data/tests/test_processing.py` |
| Fractional values kept in Python CSR → blocks; complex values kept | `bb01435`, `1c5229d` | `test_sparse_fill_precision.py`, `test_review_data_models_regressions.py` |
| sisl CSR rows read up to `ncol` only | `b5e9d51` | `test_sparse_ncol.py` |
| Non-default `sc_off` order remapped (images were swapped silently) | `928dee0` | `test_nonstandard_sc_off.py` |
| Periodic direction without a lattice vector refused | `498d0c2` | `test_neighborhood_degenerate_cell.py` |
| Orbital conventions pinned against independent harmonics; QE d/f refused | `477de7c` | `test_orbital_conventions.py` |
| `WaterDipole` / `WaterDipoleInv` equivariant / invariant | `e585e0f` | `test_node_feats_symmetry.py` |
| Same-species reverse edges of non-symmetric matrices in coefficient targets | `67aa8e0` | `core/data/tests/test_coefficient_*` |
| The 14 test failures inherited from the fork | `72b468c` | the suite |
| `load_labels=False` ignored for prepared samples | `1c5229d` | `test_review_data_models_regressions.py` |

**Evidence.** `auditoria_bugs_20261021.md`, `auditoria_codex_20261021/bugs.md` (DATA-01, -05, -06).
**Compatibility.** Results change only where they were wrong.

## 2. Packaging, optional dependencies, CI

**What.** `import graph2mat` works without torch; the CLI base help works without optional
dependencies; version ranges in `pyproject.toml`; reproducible CPU profiles
(`constraints/`); GitHub Actions with a static-check job (ruff, mypy on `core`), a minimal
install job and CPU test jobs; the `slow` marker.

**Commits.** `d7c87f4`, `024fecb`, `ee337e9`, `d4ad82b`, `75e4080`, `cae9955`, `4c294f6`,
`38fa1cb`, `17d7b43`, and the slow-test marker commit.
**Verify.** `tests/test_minimal_install.py`, the CI run of the branch.
**Evidence.** `roadmap.md` rows 0.5, 0.7, 3.5, 3.9.
**Compatibility.** Dependency ranges are now bounded (see `CHANGELOG.md`, "Dependencies").

## 3. Data pipeline

**What.**
- `matrix_component_policy` (`h_only`, `spin_h_only`, `h_and_overlap`, `raw_components`)
  and multi-component labels.
- Validation that turns silent label misplacement into errors: same-Z species with another
  basis (size, angular shells or shell order), explicit `nsc` different from the matrix's,
  ambiguous symmetric edge representatives (no subset-sum guess), an external geometry with
  other atoms.
- Loading: data filters preserved, only requested splits loaded, temporary roots isolated,
  configurable split seed and validation fraction, checkpoint provenance, warning-only split
  guardrail, `edge_cutoff_padding` in the graph, opt-in cost-aware batches
  (`max_batch_cost`), tensor cache, vectorized CSR/COO conversion, per-type batch layout in
  the collate.

**Commits.** Policies: `467ffb4`, `ec4f742`, `db74d63`, `535c01b`, `9fb7f82`, `1640c79`.
Validation: `7648951`, `de43c1a`, `f4fe6f7`, `d3d7fe5`, `1c5229d`, `7064654`.
Loading: `d7ddf67`, `6227edf`, `7fe54b5`, `ae7d6f9`, `69a8f08`, `bc54e91`, `cbe697a`,
`3a0af15`, `22029ca`, `0ef7ca9`, `3b47e34`, `19057a4`, `eb05ec7`, `8c15386`.
**Verify.** `core/data/tests/` (`test_same_z_different_basis.py`,
`test_data_filters_and_subgraphs.py`, `test_sparse_vectorization.py`, ...),
`tools/lightning/tests/test_datamodule_*`, `test_split_guardrail.py`, `test_cost_batching.py`.
**Evidence.** Rows 2.5, 2.6, 3.6, 3.7, 3.11, 3.12, 6.1–6.4, 8.3; GPU speed of the vectorized
reconstruction: −22 % (`w90`) and −67 % (`6x6`) per unit (row 6.2).
**Compatibility.** Inputs that misplaced labels now raise (`CHANGELOG.md`, "Changed behaviour").

## 4. Losses and epoch-level metrics

**What.** Epoch metrics aggregated over the whole epoch (global RMSE, valid denominators,
DDP-reduced) instead of averages of batch values; masks for missing labels; Huber, MAPE and
mixed losses without NaN; per-structure and per-domain losses; `pair_difference_mse`
(finite-difference pairs at Γ); `val_matrix_rmse`, a loss-independent metric.

**Commits.** `b7c612d`, `d106410`, `0f3529f`, `35585f7`, `6cf523f`, `234102b`, `0cdf723`,
`ac74c7f`, `0437bc9`, `b13ca5f`, `4920a55`, `bbde216`, `e664d02`, `30ee1a7`, `bf10967`,
`89c52ca`, `6c44eac`, `fe219a0`, `6b2b619`.
**Verify.** `tools/lightning/tests/test_*epoch_metrics.py`, `test_ddp_cpu.py`,
`core/data/tests/test_metrics_valid_labels.py`, `test_per_structure_losses.py`.
**Evidence.** Rows 0.3, 0.4, 3.1, 3.2.
**Compatibility.** Logged epoch values change (they are now the true epoch value).
Known limitation: per-structure losses average structures without labels in the batch.

## 5. Overlap-aware training

**What.** (H, S) labels and nonorthogonal reconstruction; `gauge_invariant_mse` (removes the
best `c·S` shift per structure, common to both spins), optionally with S only as a label
(`n_output_components=1`); first-order spectral term and S null-space term
(`spectral_window`); physical references and losses (electron count, Mulliken, energy);
`output_scaleshift` fitted on H; `MatrixWriter` attaches the label S to H-only predictions.

**Commits.** `7b97b7a`, `71c1afa`, `496d792`, `46bc937`, `15f98d0`, `f89adf7`, `75e75d2`,
`6509fad`, `5215fe7`, `1719d6a`, `ae04e79`, `78606d5`, `8cfbc85`, `371f5a2`, `1c5229d`
(common spin shift).
**Verify.** `core/data/tests/test_hamiltonian_training_losses.py`,
`test_physical_losses.py`, `test_periodic_hermiticity_oracles.py`,
`tools/lightning/tests/test_gauge_epoch_metrics.py`, `test_physical_training.py`,
`bindings/torch/tests/test_multitarget_loss.py`.
**Evidence.** Row 4.2: `6x6` aligned H 1.05 → 0.70 meV (5/5 seeds), H2O 29.6 → 5.6 meV,
gap 986 → 80 meV. Rows 4.1, 4.4, 4.5, 4.7 (`campana_nullspace_h2oclu_20261017.md`).
**Compatibility.** Opt-in.

## 6. Lightning training control

**What.** `training_stages` (loss, learning rate, frozen or trained parts per stage;
checkpoints restore the active stage's loss; DDP needs `find_unused_parameters` when
freezing); `selection_metric`; `lr_scheduler` attached to the model's optimizer (it was
dropped with `optimizer: null`); sorted run globs; model and data basis must agree;
per-rank sample metrics under DDP; `PlotMatrixError` refuses DDP.

**Commits.** `1a131f1` (stages introduced), `8cfbc85`, `81b18ca`, `1a0160f`, `2248f52`, `1c5229d`, `2caeb69`, `b52c73b`,
`aaa5eb8`, `edd27ff`.
**Verify.** `tools/lightning/tests/test_training_stages.py`, `test_lr_scheduler.py`,
`test_selection_metric.py`, `test_review_model_regressions.py`, `test_ddp_cpu.py`.
**Evidence.** Rows 1.15, 3.13; `auditoria_bugs_20261021.md` G2 (scheduler), Codex L1–L4.
**Compatibility.** A `lr_scheduler` in an existing config is now applied; resuming a
checkpoint trained without it raises instead of changing the run.

## 7. Model and readout options

**What.**
- Default edge readout `E3nnLinearEdgeBlock` (tensor product plus a linear term) and
  `lean_mace=True`.
- Opt-in: `mp_cutoff`, `keep_last_layer_irreps`, `edge_message_mid_irreps`,
  `edge_update_layers`, `pair_envelope`, `readout_per_interaction`, `simplify_node_irreps`,
  `gradient_checkpointing`, `torch_compile`, `cueq`, `output_scaleshift`, `basis_grouping`,
  `E3nnLowRankEdgeBlock`, `E3nnSlaterKosterEdgeBlock`, homonuclear weight fusion,
  `hamiltonian_local_context`, coefficient-space training.
- Options that would be ignored in a combination now raise.

**Commits.** `a0f6d18`, `a39439a`, `bd1d64d`, `a5e8134`, `cc261fb`, `762217c`, `4b5194b`,
`35d0172`, `fe65bfe`, `55dbbde`, `5d607b3`, `92e1ad3`, `80f3cf5`, `9c50ee5`, `2eda320`,
`4efeb1a`, `1a131f1`, `41a03f4`, `48a9f0f`, `ed975eb`, `5999120`, `beb0098`, `624101e`,
`26bdbed`, `e2d204a`.
**Verify.** `bindings/e3nn/modules/tests/` (equivariance and option tests for each block),
`tools/lightning/tests/test_low_rank_edge_model.py`, `test_slater_koster_model.py`,
`test_homonuclear_fusion_model.py`, `test_coefficient_*`, `test_model_guards.py`.
**Evidence.** Row 1.7: `w90` H 12.6 → 5.1 meV with the same VRAM. Rows 1.4–1.11, 2.1–2.8,
5.3, 6.7–6.9, 7.1–7.5 (most options measured without a general gain; kept opt-in).
**Compatibility.** The default edge readout changes (checkpoints keep their own).

## 8. Long-range electrostatics

**What.** `long_range="charges"` or `"qeq"`: learned charges and their Ewald potential,
added to H as `(V_i + V_j)/2 · S_ij`; total charge from `NetCharge`.

**Commits.** `6834407`, `88b3bee`, `5a4d900`, `a492fbf`, `fb236ca`, `e2d204a` (k-vectors).
**Verify.** `tools/lightning/tests/test_long_range.py`.
**Evidence.** Row 8.4 and "Banco de materiales": BaTiO3 gap and bands −25 to −45 % with
fixed charges, which break polar ZnO slabs; `qeq` does not.
**Compatibility.** Opt-in; requires `h_and_overlap` data.

## 9. Spin

**What.**
- Collinear spin (`spin_h_only`): stable.
- Spin-orbit without magnetism (`spin_orbit`, 4 time-reversal-symmetric channels,
  `E3nnSpinOrbitMatrixBlock`, `spin_orbit=True`): trained on Bi2Se3.
- Magnetic spin-orbit (`spin_orbit_full`) with per-atom moments and exact time reversal:
  code and tests, never trained.
- `edge_cutoff_padding`: readout edges beyond R_i + R_j for Kleinman–Bylander blocks.

**Commits.** Collinear: `1d52abb`, `2f08fa6`. SOC: `ee2f89a`, `463f1a8`, `db94112`,
`9f87f06`, `3c31d44`, `6df9fa8`, `e120ec6`, `65feb05`, `802409d`, `43b52b2`, `88c344e`,
`d32c955`, `e1e7e4b`. Magnetism: `4d8b56e`, `ec87d85`, `83fefa6`, `3388ab2`, `e1cb317`,
`da478db`. Padding: `a279b45`, `542659e`, `fd62690`, `37a5589`.
**Verify.** `core/data/tests/test_spin_collinear.py`, `test_spin_orbit.py`,
`test_spin_orbit_full.py`, `test_magnetic_moments.py`, `test_edge_cutoff_padding.py`,
`bindings/e3nn/modules/tests/test_spin_orbit_*`, `tools/lightning/tests/test_spin_orbit_model.py`,
`test_magnetic_model.py`, `test_edge_cutoff_padding_model.py`.
**Evidence.** Rows 8.1, 8.2; `informe_soc_f0_bi2se3_20261019.md`,
`informe_magnetismo_f0_fe_20261021.md`; `docs/data_loading.rst` (spin-orbit and magnetic
sections).
**Compatibility.** Opt-in; unsupported combinations raise.

## 10. Joint H/S/DM/EDM targets

**What.** `matrix_targets`: one backbone and readout for several matrices, with per-target
losses, weights and masks, reconstruction and callbacks.

**Commits.** `65d2253`, `714e68f`.
**Verify.** `core/data/tests/test_multitarget*.py`, `tools/lightning/tests/test_multitarget_*.py`.
**Evidence.** Row 7.6, `campana_multitarget_w90_20261019.md`: works; no gain over H alone
with the measured recipe.
**Compatibility.** Opt-in.

## 11. Server and extrapolation

**What.** Repaired `/test` and local writes, serialized processor metadata, physical matrix
formats, `HistoryCopy` pairs each label with the previous step's matrix, aligned incomplete
histories, percent-encoded Lua client queries; spin-orbit and magnetic processors refused.

**Commits.** `f84d29a`, `015fc98`, `187402e`, `2ca32dc`, `3172bc9`, `dd2fccf`, `c51be04`,
`1e12f10`, `2df88f8`, `e120ec6`.
**Verify.** `tools/server/tests/` (needs `fastapi`; `HistorySOAPs` tests need `dscribe`).
**Evidence.** Row 3.4.
**Compatibility.** `HistoryCopy` predictions change (they paired the wrong steps).

## 12. SIESTA and analysis tools

**What.** The Δ-learning workflow and evaluation tools, migrated from the research
repository: `graph2mat.tools.siesta` (`runner` with atomic runs and cache receipts,
`delta`: H0, Δ labels, TRS projection for spin-orbit, `add_h0`; `label_noise`; `dm_pilot`),
`graph2mat siesta delta {h0,label,add-h0}`, `graph2mat.tools.spectral_targets`,
`graph2mat.tools.analysis` (metrics, error breakdown, OOD attribution, KB-tail control),
and the example `docs/examples/benchmarks/`.

**Commits.** `d28c65b`, `a6f2222`, `7d41797`, `041e0ad`, `ba08711`, `7ee73f2`, `2c3b840`,
`eca05a8`, `c6c4850`, `17e246f`, `0a83cd2`.
**Verify.** `tools/siesta/tests/`, `tools/analysis/tests/`, `tools/tests/`,
`tests/test_benchmark_example.py` (a fake SIESTA runs the whole workflow).
**Evidence.** Rows 5.1, 4.3, 4.4, 4.7, 5.4, 7.7, 8.2; `informe_migracion_mdvs_20261021.md`
(identical results to the original scripts on real data). Row 5.1: H2O aligned H
5.6 → 1.0 meV, gap 80 → 9.6 meV.
**Compatibility.** New modules. fdf files with `%include` or `<` are refused by the text
rewrites; cached SIESTA results need a receipt.

## 13. Documentation

**What.** Parameter docstrings of `LitMACEMatrixModel`, `MatrixDataModule`,
`MatrixDataProcessor` and `MatrixMACE` (shown by the CLI `--help`); `docs/model_options.rst`;
`docs/data_loading.rst` sections; `CHANGELOG.md`.

**Commits.** `714e68f`, `802409d`, `e1e7e4b`, `c7cebdc`, `80614f3`.
**Verify.** `make -C docs html` (the notebooks need `pandoc`).

---

## Not for upstream

Research record and personal tooling, kept on this branch only: `auditorias_mejoras/`
(roadmap, audits, campaign runners and results), `Comparison/`, `docs/superpowers/`
(plans and specs), `.claude/`, `.agents/`, `agent-skills/`, `AGENTS.md`.
