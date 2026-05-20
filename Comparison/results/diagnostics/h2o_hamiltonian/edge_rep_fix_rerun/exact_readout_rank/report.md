# Exact Readout Rank Diagnostic

## Scope

This report computes exact local Jacobians with PyTorch autograd for Graph2Mat e3nn readout blocks. Fresh rows use standalone blocks; checkpoint rows use the actual readout blocks loaded from the provided checkpoint. The report includes both the irreducible operation output and the final block after `E3nnIrrepsMatrixBlock.change_of_basis`.

## Reproduction Command

```bash
Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/exact_readout_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt
```

## Configuration

```json
{
  "checkpoint": "/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt",
  "checkpoint_exists": true,
  "command": "Comparison/results/diagnostics/h2o_hamiltonian/exact_readout_rank/exact_readout_rank.py --output-dir Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/exact_readout_rank --checkpoint /home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/workspaces/20260520_095922/md/md_dataset1_zzu9ln_sweep015_ep550_lr0p005_l3_c32_i3_corr2_b96_w4_s15_honly_huber_b0p01/training/lightning_logs/my_first_model/version_0/checkpoints/best-4788.ckpt",
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

### checkpoint:best-4788.ckpt: node:H

- source: `checkpoint:best-4788.ckpt`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `88736`
- input dim: `1056`
- irreps output dim: `15`
- block output dim: `25`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 15/25, 15/25, 15/25

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15/15 | 15/25 | 1.78163 | 1.2344 | 1.44332 |
| 1 | 15/15 | 15/25 | 1.71366 | 1.19258 | 1.43693 |
| 2 | 15/15 | 15/25 | 1.58414 | 1.20617 | 1.31336 |

### checkpoint:best-4788.ckpt: node:O

- source: `checkpoint:best-4788.ckpt`
- kind: `node`
- operation class: `E3nnSimpleNodeBlock`
- parameter count: `320528`
- input dim: `1056`
- irreps output dim: `91`
- block output dim: `169`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: rank deficient at abs tol 1e-8 across local-input seeds: 91/169, 91/169, 91/169

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91/91 | 91/169 | 1.88309 | 0.993421 | 1.89556 |
| 1 | 91/91 | 91/169 | 1.96745 | 0.980795 | 2.00598 |
| 2 | 91/91 | 91/169 | 1.86707 | 0.946167 | 1.9733 |

### checkpoint:best-4788.ckpt: edge:H-H

- source: `checkpoint:best-4788.ckpt`
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
| 0 | 25/25 | 25/25 | 2.96077 | 0.679707 | 4.35595 |
| 1 | 25/25 | 25/25 | 2.77466 | 0.560711 | 4.94847 |
| 2 | 25/25 | 25/25 | 2.16471 | 0.48797 | 4.43616 |

### checkpoint:best-4788.ckpt: edge:H-O

- source: `checkpoint:best-4788.ckpt`
- kind: `edge`
- operation class: `E3nnSimpleEdgeBlock`
- parameter count: `1120`
- input dim: `72`
- irreps output dim: `65`
- block output dim: `65`
- irreps-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds
- block-space verdict: full rank at abs tol 1e-8 in all 3 local-input seeds

| seed | irreps rank @1e-8 | block rank @1e-8 | max sv | min positive sv | condition estimate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 65/65 | 65/65 | 3.07536 | 0.0832084 | 36.9597 |
| 1 | 65/65 | 65/65 | 3.63919 | 0.0711045 | 51.1809 |
| 2 | 65/65 | 65/65 | 3.04902 | 0.102966 | 29.612 |

## Limitations

- Fresh rows instantiate standalone readout blocks with random weights and random local inputs.
- Checkpoint rows use trained readout weights, but still probe random local readout inputs. They test the local readout operation itself, not whether MACE feature extraction collapses to a lower-dimensional manifold on a concrete H2O geometry.
- The irreps-space rank is the most direct test of the learned readout operation. The block-space rank is additionally bounded by the change-of-basis image for symmetric self blocks.
