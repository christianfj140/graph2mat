# H2O Coefficient-Space Diagnostics After Symmetric Edge Representative Fix

## Executive Verdict

The main old coefficient-space conclusions survive the symmetric-edge representative hardening:

- The target H2O H-only Hamiltonian remains representable in the current e3nn coefficient basis.
- The concrete shared-operation parameter-rank bottleneck remains: `174 / 276` coefficient directions.
- The full model remains rank-deficient in coefficient space: `196 / 276` for all trainable parameters at `1e-8`.
- The diagnostic dense readout still reaches full coefficient rank: `276 / 276`.
- With robust target edge representatives, the dense diagnostic readout still reconstructs one-sample H to numerical accuracy: `0.000783 meV` H MAE.

One important caveat remains: the production `coefficient_space_mse` audit did not complete on this real checkpoint because `_map_types` does not handle the `BasisTableWithEdges.group` type-map object exposed by this loaded readout. That is separate from the symmetric-edge representative fix, but it means the production coefficient-space loss path still needs a follow-up compatibility fix before using the loss audit as training evidence.

## Workspaces

- Graph2Mat repo: `/home/christian/repositorios/graph2mat`
- Branch: `hamiltonian-spin-colineal-support`
- H2O source workspace found: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01`
- Checkpoint used: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`
- Diagnostic workspace view: `Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view`
- `MD_vs_OnlyAtomDisplacement` was not present at the old hard-coded path, so the rerun used the available `MD_vs_AtomicDisplacement` H2O H-only workspace as read-only input.

The workspace view contains only a small config file and symlinks to the read-only sample, basis files, and checkpoint.

## Data Policy

The rerun workspace config is H-only:

- `out_matrix: hamiltonian`
- `matrix_component_policy: h_only`
- `n_matrix_components: 1`
- `symmetric_matrix: true`

Flat label sizes from the robust projection run:

| quantity | value |
|---|---:|
| node label values | 219 |
| edge label values | 155 |
| node coefficient blocks | 3 |
| edge coefficient blocks | 3 |
| total coefficient dimension | 276 |
| robust edge representative indices | `[0, 2, 4]` |
| robust edge representative types | `[0, 1, 1]` |

## Projection And Reconstruction

The dense probe was patched locally to use `MatrixDataProcessor._get_symmetric_unique_edge_mask` for target coefficient projection instead of the old external diagnostic helper's `edge_types_full[::2]` shortcut.

Target coefficients projected from H-only labels and mapped back to labels reconstruct the target to numerical precision:

| metric | value |
|---|---:|
| node label MAE | `1.637432e-07 eV` |
| node label max abs | `3.814698e-06 eV` |
| edge label MAE | `3.576502e-08 eV` |
| edge label max abs | `4.768372e-07 eV` |
| H reconstruction MAE | `0.000127 meV` |
| H reconstruction RMSE | `0.000322 meV` |
| H reconstruction max abs | `3.197941e-06 eV` |
| H allclose @1e-5 | `true` |

Output: `dense_readout_probe/dense_readout_probe_results.json`.

## Rank Comparison

### Concrete Joint Readout Rank

| operation | joint dim | old parameter rank | new parameter rank | old input rank | new input rank | old combined rank | new combined rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| `edge:H-H` | 25 | 17 | 17 | 25 | 25 | 25 | 25 |
| `edge:H-O` | 130 | 80 | 80 | 130 | 128 | 130 | 128 |
| `node:H` | 30 | 22 | 22 | 30 | 30 | 30 | 30 |
| `node:O` | 91 | 55 | 55 | 91 | 91 | 91 | 91 |
| total | 276 | 174 | 174 | 276 | 274 | 276 | 274 |

The fixed-input shared-operation parameter-rank bottleneck is unchanged. The H-O local-input and combined ranks are `128 / 130` in the rerun, not full-rank as in the old artifact, but this does not weaken the core conclusion that parameter controllability is the limiting path for the default shared readout.

Output: `concrete_joint_readout_rank/joint_rank_results.json`.

### Full-Model Coefficient-Space Jacobian Rank

| parameter group | old rank @1e-8 | new rank @1e-8 | dimension |
|---|---:|---:|---:|
| `readout` | 189 | 194 | 276 |
| `readout_blocks` | 190 | 194 | 276 |
| `edge_preprocessing` | 104 | 106 | 276 |
| `mace` | 190 | 191 | 276 |
| `all_trainable` | 190 | 196 | 276 |

The new full-model rank is slightly higher, but still strongly rank-deficient.

Output: `full_model_jacobian_rank/full_model_rank_results.json`.

### Dense Diagnostic Readout

| quantity | old | new |
|---|---:|---:|
| coefficient dimension | 276 | 276 |
| dense operation-rank sum @1e-8 | 276 | 276 |
| best coefficient MAE | `2.523707e-07` | `4.977370e-07` |
| final coefficient MAE | `3.320324e-07` | `6.925336e-07` |
| H MAE | `0.000289 meV` | `0.000783 meV` |
| H RMSE | `0.000678 meV` | `0.001547 meV` |

The dense diagnostic readout still reaches full independent coefficient rank and near-zero one-sample H reconstruction error.

Output: `dense_readout_probe/dense_readout_probe_results.json`.

## Completed Diagnostics

- Exact readout local-rank diagnostic: passed.
- Robust coefficient projection/reconstruction sanity: passed through patched dense diagnostic probe.
- Concrete joint readout rank: passed.
- Full-model coefficient-space Jacobian rank: passed.
- Dense diagnostic readout probe with robust target representatives: passed.

## Failed Or Skipped Diagnostics

- `coefficient_space_loss_audit.py`: failed after checkpoint/model load because production `coefficient_space_mse` calls `_map_types(readout.types_to_graph2mat, point_types)`, and this real checkpoint exposes `types_to_graph2mat` as a custom `BasisTableWithEdges.group` mapping object. The failure was:

```text
IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed
```

- Pytest validation was not run because the available H2O workspace virtualenv does not include `pytest`.

## Commands

```bash
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m py_compile \
  Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/coefficient_space_loss_audit.py \
  Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py \
  Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py \
  Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py \
  Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_probe.py
```

```bash
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/exact_readout_rank \
  --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

```bash
CUDA_VISIBLE_DEVICES= \
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py \
  --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement \
  --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank \
  --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

```bash
CUDA_VISIBLE_DEVICES= \
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py \
  --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement \
  --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank \
  --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

```bash
CUDA_VISIBLE_DEVICES= \
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_probe.py \
  --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement \
  --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view \
  --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/dense_readout_probe \
  --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt \
  --steps-per-stage 250
```

The first concrete-rank run without `CUDA_VISIBLE_DEVICES=` failed with a CPU/CUDA tensor mismatch; the CPU-only rerun passed.

## Remaining Uncertainty

- The rerun used the available `MD_vs_AtomicDisplacement` H2O workspace, not the missing old `MD_vs_OnlyAtomDisplacement` one-sample workspace. The verdict is stable, but not all numeric values are strictly apples-to-apples.
- The production `coefficient_space_mse` path remains unverified on this real checkpoint until the type-map compatibility issue is fixed.
- The dense probe target path is now robust, but some external MD diagnostic helpers still contain brittle `edge_types_full[::2]` assumptions and should not be used for future coefficient target extraction without patching.
