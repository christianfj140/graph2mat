# Concrete H2O Joint Readout Rank Diagnostic

Source workspace: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view`
Checkpoint: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`

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
| `edge:H-H` | 1 | 25 | 25 | 592 | 72 | 17 | 25 | 25 |
| `edge:H-O` | 2 | 65 | 130 | 1120 | 144 | 80 | 128 | 128 |
| `node:H` | 2 | 15 | 30 | 88736 | 2112 | 22 | 30 | 30 |
| `node:O` | 1 | 91 | 91 | 320528 | 1056 | 55 | 91 | 91 |

## Detailed Operation Results

### edge:H-H

- module key: `(0, 0, 0)`
- operation class: `E3nnSimpleEdgeBlock`
- samples: `['H1->H2']`
- joint output shape: `[1, 25]`
- irreps_out: `5x0e+4x1o+1x1e+1x2e`
- block shape after change of basis: `[5, 5]`
- parameter count: `592`
- local input shapes: `{'edge_messages': [[1, 36], [1, 36]]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[25, 592]` | 17 | 17 | 17 | 17 | 17 | 5.32112 | 0.502489 | 10.5895 |
| local inputs | `[25, 72]` | 25 | 25 | 25 | 25 | 25 | 4.43685 | 0.56604 | 7.8384 |
| parameters + local inputs | `[25, 664]` | 25 | 25 | 25 | 25 | 25 | 6.24407 | 0.56604 | 11.0311 |

### edge:H-O

- module key: `(0, 1, 1)`
- operation class: `E3nnSimpleEdgeBlock`
- samples: `['H1->O0', 'H2->O0']`
- joint output shape: `[2, 65]`
- irreps_out: `6x0e+7x1o+2x1e+1x2o+4x2e+1x3o`
- block shape after change of basis: `[5, 13]`
- parameter count: `1120`
- local input shapes: `{'edge_messages': [[2, 36], [2, 36]]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[130, 1120]` | 80 | 80 | 80 | 80 | 80 | 10.4364 | 0.0109934 | 949.334 |
| local inputs | `[130, 144]` | 128 | 128 | 128 | 128 | 128 | 7.71848 | 0.02744 | 281.286 |
| parameters + local inputs | `[130, 1264]` | 128 | 128 | 128 | 128 | 128 | 11.3372 | 0.02744 | 413.162 |

### node:H

- module key: `0`
- operation class: `E3nnSimpleNodeBlock`
- samples: `['H1', 'H2']`
- joint output shape: `[2, 15]`
- irreps_out: `4x0e+2x1o+1x2e`
- block shape after change of basis: `[5, 5]`
- parameter count: `88736`
- local input shapes: `{'node_feats': [2, 1056]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[30, 88736]` | 22 | 22 | 22 | 22 | 22 | 1.79305 | 0.0652872 | 27.4641 |
| local inputs | `[30, 2112]` | 30 | 30 | 30 | 30 | 30 | 4.02539 | 0.86039 | 4.67857 |
| parameters + local inputs | `[30, 90848]` | 30 | 30 | 30 | 30 | 30 | 4.31734 | 0.86039 | 5.01789 |

### node:O

- module key: `1`
- operation class: `E3nnSimpleNodeBlock`
- samples: `['O0']`
- joint output shape: `[1, 91]`
- irreps_out: `7x0e+6x1o+1x1e+2x2o+6x2e+2x3o+1x4e`
- block shape after change of basis: `[13, 13]`
- parameter count: `320528`
- local input shapes: `{'node_feats': [1, 1056]}`

| Jacobian | shape | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | min effective sv @1e-8 | condition @1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameters | `[91, 320528]` | 55 | 55 | 55 | 55 | 55 | 3.42441 | 0.0460789 | 74.3163 |
| local inputs | `[91, 1056]` | 91 | 91 | 91 | 91 | 91 | 8.18989 | 0.275003 | 29.7811 |
| parameters + local inputs | `[91, 321584]` | 91 | 91 | 91 | 91 | 91 | 8.86379 | 0.275003 | 32.2317 |

## Input Feature Distances

| operation | sample i | sample j | label i | label j | euclidean | cosine | max abs delta | note |
|---|---:|---:|---|---|---:|---:|---:|---|
| `edge:H-H` | 0 | None | `H1->H2` | `` | None | None | None | single_sample_no_pairwise_distance |
| `edge:H-O` | 0 | 1 | `H1->O0` | `H2->O0` | 20.724838855190832 | -0.17264254990614944 | 10.903398036956787 |  |
| `node:H` | 0 | 1 | `H1` | `H2` | 21.57125789551645 | 0.365296307111566 | 3.988045334815979 |  |
| `node:O` | 0 | None | `O0` | `` | None | None | None | single_sample_no_pairwise_distance |

## Output Files

- JSON: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank/joint_rank_results.json`
- Rank CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank/joint_rank_by_operation.csv`
- Input feature distances CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank/input_feature_distances.csv`

## Reproduction Command

```bash
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

## Validation Runs

- `PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python Comparison/results/diagnostics/h2o_hamiltonian/concrete_joint_readout_rank/concrete_joint_readout_rank.py --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/concrete_joint_readout_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`: passed. Generated joint_rank_results.json, joint_rank_by_operation.csv, input_feature_distances.csv, and report.md.

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `4efeb1a7e2151bdb8e2a30131a75c52b5ddb2bcd`
- Pipeline branch: `main`
- Pipeline commit: `dc921c15eb4cfad312b1dfaceb7b194611b775a9`

## Limitations

- This is a local Jacobian diagnostic at the checkpoint and concrete batch.
- Parameter ranks hold readout inputs fixed; they test shared operation controllability.
- Input ranks hold operation parameters fixed; they test sensitivity to local readout inputs, not the full upstream MACE parameter manifold.
