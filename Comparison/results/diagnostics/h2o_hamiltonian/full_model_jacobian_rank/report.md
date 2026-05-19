# Full Model Jacobian Rank Diagnostic

Source workspace: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix`
Checkpoint: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt`

## Verdict

**B_MACE_FEATURE_MANIFOLD_LIMIT**: The full trainable model remains rank-deficient even in independent coefficient space (190/276).

## Output Spaces

- label-space output dim: `374`
- coefficient-space output dim: `276`

Label space includes full flattened symmetric self blocks. Coefficient space removes that expected redundancy by using the irreps coefficients emitted by each readout operation.

## Rank By Parameter Group

| output | group | params | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `label_space` | `readout` | 511904 | `[374, 511904]` | 174 | 174 | 176 | 230 | 250 | 38.2156 | 1.00413e-08 | 3.80582e+09 |
| `label_space` | `readout_blocks` | 406816 | `[374, 406816]` | 174 | 174 | 174 | 235 | 255 | 5.42693 | 1.00161e-08 | 5.41823e+08 |
| `label_space` | `edge_preprocessing` | 105088 | `[374, 105088]` | 97 | 97 | 98 | 106 | 118 | 38.0076 | 1.23408e-08 | 3.07983e+09 |
| `label_space` | `mace` | 184065 | `[374, 184065]` | 174 | 175 | 191 | 215 | 238 | 113.434 | 1.0751e-08 | 1.05511e+10 |
| `label_space` | `all_trainable` | 695969 | `[374, 695969]` | 174 | 175 | 193 | 219 | 238 | 114.385 | 1.05564e-08 | 1.08356e+10 |
| `coefficient_space` | `readout` | 511904 | `[276, 511904]` | 174 | 174 | 177 | 189 | 203 | 38.2156 | 1.05367e-08 | 3.6269e+09 |
| `coefficient_space` | `readout_blocks` | 406816 | `[276, 406816]` | 174 | 174 | 174 | 190 | 205 | 5.42693 | 1.00521e-08 | 5.3988e+08 |
| `coefficient_space` | `edge_preprocessing` | 105088 | `[276, 105088]` | 97 | 97 | 99 | 104 | 115 | 38.0076 | 1.16511e-08 | 3.26214e+09 |
| `coefficient_space` | `mace` | 184065 | `[276, 184065]` | 174 | 174 | 178 | 190 | 209 | 113.436 | 1.08756e-08 | 1.04303e+10 |
| `coefficient_space` | `all_trainable` | 695969 | `[276, 695969]` | 174 | 174 | 177 | 190 | 208 | 114.387 | 1.22759e-08 | 9.31802e+09 |

## Interpretation

The decisive comparison is coefficient-space `all_trainable`: it tests whether upstream MACE plus edge preprocessing plus readout parameters can move all independent H2O coefficients. The label-space rank should not be expected to reach 374 while self blocks are constrained to symmetric change-of-basis images. The small singular values are highly ill-conditioned near `1e-8`; the stable `1e-5`/`1e-6` coefficient-space rank is the same as the fixed-feature shared-readout rank. The conclusion is unchanged at every reported tolerance because `all_trainable` remains below the independent coefficient dimension.

## Output Files

- JSON: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_rank_results.json`
- rank CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_rank_by_parameter_group.csv`
- coefficient rank CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/coefficient_space_full_model_rank.csv`

## Reproduction Command

```bash
PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py
```

## Validation Runs

- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python -m py_compile Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py`: passed.
- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/full_model_jacobian_rank/full_model_jacobian_rank.py`: passed. Generated exact row-wise Jacobian ranks for label and coefficient spaces.

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `9fb7f82f15425a9039b55fbc672b947a3afb224a`
- Pipeline branch: `main`
- Pipeline commit: `4dd1944648624d8f44d841d02348c26ce0e11439`

## Limitations

- The Jacobians are exact row-wise autograd Jacobians at the checkpoint and one concrete batch.
- Gradients are computed at the model's checkpoint dtype and accumulated into temporary NumPy memmaps before SVD of `J J^T`.
- Label-space rank includes expected symmetric-block redundancy; coefficient-space rank is the independent controllability test.
