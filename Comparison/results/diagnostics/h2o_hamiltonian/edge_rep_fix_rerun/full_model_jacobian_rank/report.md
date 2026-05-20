# Full Model Jacobian Rank Diagnostic

Source workspace: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view`
Checkpoint: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`

## Verdict

**B_MACE_FEATURE_MANIFOLD_LIMIT**: The full trainable model remains rank-deficient even in independent coefficient space (196/276).

## Output Spaces

- label-space output dim: `374`
- coefficient-space output dim: `276`

Label space includes full flattened symmetric self blocks. Coefficient space removes that expected redundancy by using the irreps coefficients emitted by each readout operation.

## Rank By Parameter Group

| output | group | params | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `label_space` | `readout` | 524048 | `[374, 524048]` | 174 | 174 | 176 | 225 | 252 | 25.1538 | 1.03138e-08 | 2.43885e+09 |
| `label_space` | `readout_blocks` | 413200 | `[374, 413200]` | 174 | 174 | 175 | 226 | 253 | 10.4364 | 1.00171e-08 | 1.04187e+09 |
| `label_space` | `edge_preprocessing` | 110848 | `[374, 110848]` | 97 | 97 | 97 | 103 | 116 | 24.133 | 1.03119e-08 | 2.3403e+09 |
| `label_space` | `mace` | 214753 | `[374, 214753]` | 174 | 175 | 192 | 216 | 239 | 143.64 | 1.06147e-08 | 1.35321e+10 |
| `label_space` | `all_trainable` | 738801 | `[374, 738801]` | 174 | 175 | 192 | 225 | 242 | 143.887 | 1.01812e-08 | 1.41326e+10 |
| `coefficient_space` | `readout` | 524048 | `[276, 524048]` | 174 | 174 | 176 | 194 | 199 | 25.1733 | 1.12018e-08 | 2.24725e+09 |
| `coefficient_space` | `readout_blocks` | 413200 | `[276, 413200]` | 174 | 174 | 178 | 194 | 206 | 10.4364 | 1.01529e-08 | 1.02793e+09 |
| `coefficient_space` | `edge_preprocessing` | 110848 | `[276, 110848]` | 97 | 97 | 97 | 106 | 114 | 24.1555 | 1.00713e-08 | 2.39845e+09 |
| `coefficient_space` | `mace` | 214753 | `[276, 214753]` | 174 | 174 | 178 | 191 | 211 | 143.64 | 1.04109e-08 | 1.37971e+10 |
| `coefficient_space` | `all_trainable` | 738801 | `[276, 738801]` | 174 | 174 | 178 | 196 | 209 | 143.888 | 1.12387e-08 | 1.28029e+10 |

## Interpretation

The decisive comparison is coefficient-space `all_trainable`: it tests whether upstream MACE plus edge preprocessing plus readout parameters can move all independent H2O coefficients. The label-space rank should not be expected to reach 374 while self blocks are constrained to symmetric change-of-basis images.
 The small singular values are highly ill-conditioned near `1e-8`; the stable `1e-5`/`1e-6` coefficient-space rank is the same as the fixed-feature shared-readout rank. The conclusion is unchanged at every reported tolerance because `all_trainable` remains below the independent coefficient dimension.

## Output Files

- JSON: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank/full_model_rank_results.json`
- rank CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank/full_model_rank_by_parameter_group.csv`
- coefficient rank CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank/coefficient_space_full_model_rank.csv`

## Reproduction Command

```bash
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

## Validation Runs

- `PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/full_model_jacobian_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`: passed. Generated exact row-wise Jacobian ranks for label and coefficient spaces.

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `4efeb1a7e2151bdb8e2a30131a75c52b5ddb2bcd`
- Pipeline branch: `main`
- Pipeline commit: `dc921c15eb4cfad312b1dfaceb7b194611b775a9`

## Limitations

- The Jacobians are exact row-wise autograd Jacobians at the checkpoint and one concrete batch.
- Gradients are computed at the model's checkpoint dtype and accumulated into temporary NumPy memmaps before SVD of `J J^T`.
- Label-space rank includes expected symmetric-block redundancy; coefficient-space rank is the independent controllability test.
