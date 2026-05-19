# Dense Diagnostic Readout Probe

Source workspace: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix`
Checkpoint: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt`

## Verdict

**A_DENSE_READOUT_MEMORIZES**: The dense diagnostic readout reaches full independent coefficient rank and reconstructs the one-sample Hamiltonian to numerical accuracy.

## Readout API

- node readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseNodeBlock`
- edge readout: `graph2mat.bindings.e3nn.E3nnDiagnosticDenseEdgeBlock`
- default readouts are unchanged; this script passes the dense classes through `node_block_readout` and `edge_block_readout`.

## Fit Summary

- coefficient output dim: `276`
- dense readout operation-rank sum @1e-8: `276`
- final coefficient MAE: `3.3203244242940855e-07`
- final coefficient MSE: `3.269984109877111e-13`
- node label MAE: `3.2969997163405895e-07`
- edge label MAE: `2.2429991420281216e-07`
- H MAE: `0.0002890360683773567` meV
- H RMSE: `0.0006779498474472067` meV
- H max abs: `7.671703528444596e-06` eV

## Exact Dense Readout Rank By Operation

| op | output dim | params | rank 1e-5 | rank 1e-6 | rank 1e-7 | rank 1e-8 | rank 1e-10 | max sv | condition @1e-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `edge:(0, 0, 0)` | 25 | 90457 | 25 | 25 | 25 | 25 | 25 | 14.5101 | 1.19043 |
| `edge:(0, 1, 1)` | 130 | 102017 | 130 | 130 | 130 | 130 | 130 | 52.1173 | 3.8203 |
| `node:0` | 30 | 419855 | 30 | 30 | 30 | 30 | 30 | 39.3015 | 2.52937 |
| `node:1` | 91 | 468571 | 91 | 91 | 91 | 91 | 91 | 74.2375 | 1.64194 |

## Baseline Comparison

- old coefficient-space all-trainable rank @1e-8: `190` / `276`
- old fixed-feature readout-block rank @1e-8: `190` / `276`
- dense readout rank is computed operation-by-operation with disjoint operation parameters, so the summed rank is an exact coefficient-space rank lower bound for all trainable parameters.

## Training Details

- final-layer least-squares initialization: `{'node:0': {'status': 'fit', 'design_shape': [2, 641], 'design_rank_1e_10': 2, 'target_shape': [2, 15]}, 'node:1': {'status': 'fit', 'design_shape': [1, 641], 'design_rank_1e_10': 1, 'target_shape': [1, 91]}, 'edge:(0, 0, 0)': {'status': 'skipped', 'reason': 'symm_transpose_uses_forward_backward_block_average'}, 'edge:(0, 1, 1)': {'status': 'fit', 'design_shape': [2, 289], 'design_rank_1e_10': 2, 'target_shape': [2, 65]}}`
- optimizer schedule: `[0.01, 0.003, 0.001, 0.0003]`
- steps per stage: `250`
- total steps: `1000`
- runtime: `43.42204685101751` seconds

## Output Files

- JSON: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_probe_results.json`
- rank CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_rank_by_operation.csv`
- coefficient errors CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_coefficient_errors.csv`

## Reproduction Command

```bash
PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/dense_readout_probe/dense_readout_probe.py
```

## Repository Context

- Graph2Mat branch: `hamiltonian-spin-colineal-support`
- Graph2Mat commit: `9fb7f82f15425a9039b55fbc672b947a3afb224a`
- Pipeline branch: `main`
- Pipeline commit: `4dd1944648624d8f44d841d02348c26ce0e11439`

## Limitations

- This is a diagnostic, non-equivariant coefficient readout probe.
- MACE node features and edge preprocessing outputs are fixed from the checkpoint during coefficient fitting.
- Full all-parameter dense Jacobian is not materialized because full readout-block rank already proves the all-parameter rank lower bound is full.
