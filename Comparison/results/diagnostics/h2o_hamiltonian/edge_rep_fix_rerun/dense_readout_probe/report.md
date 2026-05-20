# Dense Diagnostic Readout Probe

Source workspace: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view`
Checkpoint: `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt`

## Verdict

**A_DENSE_READOUT_MEMORIZES**: The dense diagnostic readout reaches full independent coefficient rank and reconstructs the one-sample Hamiltonian to numerical accuracy.

## Readout API

- node readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock`
- edge readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock`
- default readouts are unchanged; this script passes the dense classes through `node_block_readout` and `edge_block_readout`.

## Fit Summary

- coefficient output dim: `276`
- dense readout operation-rank sum @1e-8: `276`
- final coefficient MAE: `3.5390931429901684e-07`
- final coefficient MSE: `2.928804659733941e-13`
- node label MAE: `4.4189443191380506e-07`
- edge label MAE: `2.744696640518434e-07`
- H MAE: `0.0003657852336346042` meV
- H RMSE: `0.0006903107746413614` meV
- H max abs: `5.302923536731896e-06` eV

## Exact Dense Readout Rank By Operation

| op | output dim | params | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | condition @1e-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge:(0, 0, 0)` | 25 | 7081 | 25 | 25 | 25 | 25 | 25 | 19.632 | 1.58018 |
| `edge:(0, 1, 1)` | 130 | 10001 | 130 | 130 | 130 | 130 | 130 | 128.016 | 26.2179 |
| `node:0` | 30 | 1132047 | 30 | 30 | 30 | 30 | 30 | 38.3529 | 1.90261 |
| `node:1` | 91 | 1212379 | 91 | 91 | 91 | 91 | 91 | 70.5228 | 1.25808 |

## Baseline Comparison

- old coefficient-space all-trainable rank @1e-8: `190` / `276`
- old fixed-feature readout-block rank @1e-8: `190` / `276`
- dense readout rank is computed operation-by-operation with disjoint operation parameters, so the summed rank is an exact coefficient-space rank lower bound for all trainable parameters.

## Training Details

- final-layer least-squares initialization: `{'node:0': {'status': 'fit', 'design_shape': [2, 1057], 'design_rank_1e_10': 2, 'target_shape': [2, 15]}, 'node:1': {'status': 'fit', 'design_shape': [1, 1057], 'design_rank_1e_10': 1, 'target_shape': [1, 91]}, 'edge:(0, 0, 0)': {'status': 'skipped', 'reason': 'symm_transpose_uses_forward_backward_block_average'}, 'edge:(0, 1, 1)': {'status': 'fit', 'design_shape': [2, 73], 'design_rank_1e_10': 2, 'target_shape': [2, 65]}}`
- optimizer schedule: `[0.01, 0.003, 0.001, 0.0003]`
- steps per stage: `250`
- total steps: `1000`
- runtime: `1.9221553090028465` seconds

## Output Files

- JSON: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/dense_readout_probe/dense_readout_probe_results.json`
- rank CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/dense_readout_probe/dense_readout_rank_by_operation.csv`
- coefficient errors CSV: `/home/christian/repositorios/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/dense_readout_probe/dense_readout_coefficient_errors.csv`

## Reproduction Command

```bash
PYTHONPATH=src:/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/scripts:/home/christian/repositorios/MD_vs_AtomicDisplacement/scripts/torch_serialization_compat /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_probe.py --pipeline-root /home/christian/repositorios/MD_vs_AtomicDisplacement --source-workspace Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/dense_readout_probe --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt --steps-per-stage 250
```

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `4efeb1a7e2151bdb8e2a30131a75c52b5ddb2bcd`
- Pipeline branch: `main`
- Pipeline commit: `dc921c15eb4cfad312b1dfaceb7b194611b775a9`

## Limitations

- This is a diagnostic, non-equivariant coefficient readout probe.
- MACE node features and edge preprocessing outputs are fixed from the checkpoint during coefficient fitting.
- Full all-parameter dense Jacobian is not materialized because full readout-block rank already proves the all-parameter rank lower bound is full.
