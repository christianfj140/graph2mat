# Exact Readout Rank Diagnostic

## Scope

This report computes exact local Jacobians with PyTorch autograd for Graph2Mat e3nn readout blocks. Fresh rows use standalone blocks; checkpoint rows use the actual readout blocks loaded from the provided checkpoint. The report includes both the irreducible operation output and the final block after `E3nnIrrepsMatrixBlock.change_of_basis`.

## Reproduction Command

```bash
Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py --checkpoint /home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt --seeds 0 1 2
```

## Configuration

```json
{
  "checkpoint": "/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt",
  "checkpoint_exists": true,
  "command": "Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py --checkpoint /home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt --seeds 0 1 2",
  "device": "cpu",
  "dtype": "float64",
  "edge_messages_irreps": "4x0e+4x1o+4x2e",
  "h_basis": "2x0e+1x1o",
  "n_matrix_components": 1,
  "node_feats_irreps": "20x0e+20x1o+20x2e",
  "o_basis": "4x0e+3x1o",
  "relative_tolerances": [
    1e-05,
    1e-06,
    1e-07,
    1e-08
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "tolerances": [
    1e-05,
    1e-06,
    1e-07,
    1e-08,
    1e-10
  ],
  "vectorize": true
}
```

## Operation Verdicts

### fresh: node:H

- source: `fresh`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `4940`
- input dim: `180`
- irreps output dim: `15`
- block output dim: `25`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 15/25, 15/25, 15/25

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15/15 | 15/25 | 2.08293 | 0.966975 | 2.15407 |
| 1 | 15/15 | 15/25 | 1.98053 | 1.00656 | 1.96763 |
| 2 | 15/15 | 15/25 | 1.8719 | 1.02323 | 1.82941 |

### fresh: node:O

- source: `fresh`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `25740`
- input dim: `180`
- irreps output dim: `91`
- block output dim: `169`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 91/169, 91/169, 91/169

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91/91 | 91/169 | 2.71608 | 0.388578 | 6.98979 |
| 1 | 91/91 | 91/169 | 2.59237 | 0.394143 | 6.57724 |
| 2 | 91/91 | 91/169 | 2.48002 | 0.370936 | 6.68583 |

### fresh: edge:H-H

- source: `fresh`
- kind: `edge`
- operation class: `E3nnSimpleEdgeBlock`
- parameter count: `592`
- input dim: `72`
- irreps output dim: `25`
- block output dim: `25`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 25/25 | 25/25 | 2.70441 | 0.596341 | 4.53501 |
| 1 | 25/25 | 25/25 | 2.67095 | 0.520134 | 5.13512 |
| 2 | 25/25 | 25/25 | 2.37866 | 0.468821 | 5.07371 |

### fresh: edge:H-O

- source: `fresh`
- kind: `edge`
- operation class: `E3nnSimpleEdgeBlock`
- parameter count: `1456`
- input dim: `72`
- irreps output dim: `65`
- block output dim: `65`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65/65 | 65/65 | 2.63503 | 0.0656828 | 40.1175 |
| 1 | 65/65 | 65/65 | 3.44343 | 0.052526 | 65.5567 |
| 2 | 65/65 | 65/65 | 2.79837 | 0.0376011 | 74.4225 |

### checkpoint:best-298.ckpt: node:H

- source: `checkpoint:best-298.ckpt`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `86592`
- input dim: `640`
- irreps output dim: `15`
- block output dim: `25`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 15/25, 15/25, 15/25

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15/15 | 15/25 | 1.60187 | 1.20321 | 1.33133 |
| 1 | 15/15 | 15/25 | 1.73455 | 1.21564 | 1.42686 |
| 2 | 15/15 | 15/25 | 1.63348 | 1.08453 | 1.50616 |

### checkpoint:best-298.ckpt: node:O

- source: `checkpoint:best-298.ckpt`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `257248`
- input dim: `640`
- irreps output dim: `91`
- block output dim: `169`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 91/169, 91/169, 91/169

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91/91 | 91/169 | 1.9059 | 0.861908 | 2.21126 |
| 1 | 91/91 | 91/169 | 2.05081 | 0.932906 | 2.1983 |
| 2 | 91/91 | 91/169 | 2.01811 | 0.864701 | 2.33389 |

### checkpoint:best-298.ckpt: edge:H-H

- source: `checkpoint:best-298.ckpt`
- kind: `edge`
- operation class: `E3nnSimpleEdgeBlock`
- parameter count: `9472`
- input dim: `288`
- irreps output dim: `25`
- block output dim: `25`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 25/25 | 25/25 | 2.44443 | 1.06208 | 2.30155 |
| 1 | 25/25 | 25/25 | 2.66626 | 1.08548 | 2.4563 |
| 2 | 25/25 | 25/25 | 1.97126 | 0.896639 | 2.1985 |

### checkpoint:best-298.ckpt: edge:H-O

- source: `checkpoint:best-298.ckpt`
- kind: `edge`
- operation class: `E3nnSimpleEdgeBlock`
- parameter count: `17920`
- input dim: `288`
- irreps output dim: `65`
- block output dim: `65`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65/65 | 65/65 | 2.10309 | 0.746128 | 2.81867 |
| 1 | 65/65 | 65/65 | 2.21509 | 0.739623 | 2.99489 |
| 2 | 65/65 | 65/65 | 1.97347 | 0.675859 | 2.91995 |

## Limitations

- Fresh rows instantiate standalone readout blocks with random weights and random local inputs.
- Checkpoint rows use trained readout weights, but still probe random local readout inputs. They test the local readout operation itself, not whether MACE feature extraction collapses to a lower-dimensional manifold on a concrete H2O geometry.
- The irreps-space rank is the most direct test of the learned readout operation. The block-space rank is additionally bounded by the change-of-basis image for symmetric self blocks.

## Validation Runs

- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python -m py_compile Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py`: passed.
- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py --checkpoint /home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt --seeds 0 1 2`: passed; wrote `rank_results.json` and this report.
- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python -m pytest src/graph2mat/bindings/e3nn/modules/tests/test_e3nngraph2mat.py`: failed in the existing test suite before any production edits, with e3nn `wigner_D` dtype errors from integer `torch.tensor(0)` angle inputs.
- `PYTHONPATH=src /home/christian/graph2mat-env/bin/python -m pytest src/graph2mat/core/modules/tests/test_graph2mat.py`: 7 passed, 2 failed in existing multicomponent branch behavior (`np.concatenate` on empty edge outputs; multicomponent transpose expectation mismatch).
- `git diff --check`: passed.
