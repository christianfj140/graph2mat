# Concrete H2O Joint Readout Rank Diagnostic

Source workspace: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix`
Checkpoint: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt`

## Verdict

**B_SHARED_OPERATION_RANK_LIMIT**: At least one shared readout operation has a parameter Jacobian rank below its concrete joint H2O coefficient dimension.

## Interpretation

The parameter Jacobian is rank-deficient for every measured operation at the concrete checkpoint inputs. The local-input Jacobian and the combined parameter+input Jacobian are full-rank for every operation, so the coefficient maps are not limited by the e3nn change-of-basis or by the operation output space itself. The failure mode is specifically fixed-input shared-operation controllability: changing only the readout operation parameters cannot span all concrete H2O target coefficients.

## Scope

This diagnostic uses the actual one-sample H2O batch and the actual MACE readout inputs produced by the checkpoint. For each shared Graph2Mat readout operation it concatenates every coefficient block that uses that operation, then computes exact Jacobian ranks.

The reported coefficient ranks are irreps-space ranks. The symmetric node block-space ranks remain lower because a 5x5 H self block has 15 independent coefficients and a 13x13 O self block has 91.

## Joint Rank Summary

| operation | blocks | per-block coeffs | joint dim | params | local input dim | param rank @1e-8 | input rank @1e-8 | combined rank @1e-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge:H-H` | 1 | 25 | 25 | 9472 | 288 | 17 | 25 | 25 |
| `edge:H-O` | 2 | 65 | 130 | 17920 | 576 | 80 | 130 | 130 |
| `node:H` | 2 | 15 | 30 | 86592 | 1280 | 22 | 30 | 30 |
| `node:O` | 1 | 91 | 91 | 257248 | 640 | 55 | 91 | 91 |

## Detailed Operation Results

### edge:H-H

- module key: `(0, 0, 0)`
- operation class: `E3nnSimpleEdgeBlock`
- samples: `['H1->H2']`
- joint output shape: `[1, 25]`
- irreps_out: `5x0e+4x1o+1x1e+1x2e`
- block shape after change of basis: `[5, 5]`
- parameter count: `9472`
- local input shapes: `{'edge_messages': [[1, 144], [1, 144]]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[25, 9472]` | 17 | 17 | 17 | 17 | 17 | 0.611777 | 0.103962 | 5.88463 |
| local inputs | `[25, 288]` | 25 | 25 | 25 | 25 | 25 | 4.45231 | 0.370354 | 12.0218 |
| parameters + local inputs | `[25, 9760]` | 25 | 25 | 25 | 25 | 25 | 4.48138 | 0.370354 | 12.1003 |

### edge:H-O

- module key: `(0, 1, 1)`
- operation class: `E3nnSimpleEdgeBlock`
- samples: `['H2->O0', 'H1->O0']`
- joint output shape: `[2, 65]`
- irreps_out: `6x0e+7x1o+2x1e+1x2o+4x2e+1x3o`
- block shape after change of basis: `[5, 13]`
- parameter count: `17920`
- local input shapes: `{'edge_messages': [[2, 144], [2, 144]]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[130, 17920]` | 80 | 80 | 80 | 80 | 80 | 4.54121 | 0.00223995 | 2027.37 |
| local inputs | `[130, 576]` | 130 | 130 | 130 | 130 | 130 | 4.19286 | 0.202473 | 20.7082 |
| parameters + local inputs | `[130, 18496]` | 130 | 130 | 130 | 130 | 130 | 5.93332 | 0.202473 | 29.3043 |

### node:H

- module key: `0`
- operation class: `E3nnSimpleNodeBlock`
- samples: `['H1', 'H2']`
- joint output shape: `[2, 15]`
- irreps_out: `4x0e+2x1o+1x2e`
- block shape after change of basis: `[5, 5]`
- parameter count: `86592`
- local input shapes: `{'node_feats': [2, 640]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[30, 86592]` | 22 | 22 | 22 | 22 | 22 | 2.05253 | 0.0027175 | 755.299 |
| local inputs | `[30, 1280]` | 30 | 30 | 30 | 30 | 30 | 3.13761 | 1.09289 | 2.87093 |
| parameters + local inputs | `[30, 87872]` | 30 | 30 | 30 | 30 | 30 | 3.63051 | 1.09289 | 3.32194 |

### node:O

- module key: `1`
- operation class: `E3nnSimpleNodeBlock`
- samples: `['O0']`
- joint output shape: `[1, 91]`
- irreps_out: `7x0e+6x1o+1x1e+2x2o+6x2e+2x3o+1x4e`
- block shape after change of basis: `[13, 13]`
- parameter count: `257248`
- local input shapes: `{'node_feats': [1, 640]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[91, 257248]` | 55 | 55 | 55 | 55 | 55 | 5.42693 | 0.0136413 | 397.832 |
| local inputs | `[91, 640]` | 91 | 91 | 91 | 91 | 91 | 6.35123 | 0.0307388 | 206.619 |
| parameters + local inputs | `[91, 257888]` | 91 | 91 | 91 | 91 | 91 | 8.34306 | 0.0307388 | 271.418 |

## Input Feature Distances

| operation | sample i | sample j | label i | label j | euclidean | cosine | max abs delta | note |
|---|---:|---:|---|---|---:|---:|---:|---|
| `edge:H-H` | 0 | None | `H1->H2` | `` | None | None | None | single_sample_no_pairwise_distance |
| `edge:H-O` | 0 | 1 | `H2->O0` | `H1->O0` | 25.980864738420646 | 0.05328665619129781 | 7.525680065155029 |  |
| `node:H` | 0 | 1 | `H1` | `H2` | 22.074064875435273 | 0.385655719230108 | 5.239537239074707 |  |
| `node:O` | 0 | None | `O0` | `` | None | None | None | single_sample_no_pairwise_distance |

## Output Files

- JSON: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/joint_rank_results.json`
- Rank CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/joint_rank_by_operation.csv`
- Input feature distances CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/input_feature_distances.csv`

## Reproduction Command

```bash
PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py
```

## Validation Runs

- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python -m py_compile Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py`: passed.
- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py`: passed. Generated joint_rank_results.json, joint_rank_by_operation.csv, input_feature_distances.csv, and report.md.

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `9fb7f82f15425a9039b55fbc672b947a3afb224a`
- Pipeline branch: `main`
- Pipeline commit: `4dd1944648624d8f44d841d02348c26ce0e11439`

## Limitations

- This is a local Jacobian diagnostic at the checkpoint and concrete batch.
- Parameter ranks hold readout inputs fixed; they test shared operation controllability.
- Input ranks hold operation parameters fixed; they test sensitivity to local readout inputs, not the full upstream MACE parameter manifold.
