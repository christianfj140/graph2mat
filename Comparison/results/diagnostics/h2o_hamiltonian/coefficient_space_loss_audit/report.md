# Coefficient-Space Loss Audit

Source workspace: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix`
Checkpoint: `/home/christian/Escritorio/CINN/repositorios/MD_vs_OnlyAtomDisplacement/Comparison/results/diagnostics/h2o_hamiltonian/one_sample_overfit_h_only_after_yield_fix/training/logs/one_sample_h_only/version_0/checkpoints/best-298.ckpt`

## Verdict

**D_DEFAULT_RESCUED_BY_COEFF_LOSS**: Coefficient-space loss also rescues the default readout; dense label-space MSE also converges, so the failure is specific to the current MAE-style objective/optimization.

## Step-Zero And Parity

| state | coeff MSE | coeff MAE | label MAE | H MAE meV | H RMSE meV | spectral RMSE eV |
|---|---:|---:|---:|---:|---:|---:|
| `after_lstsq_before_optimizer` | 2.92472505569458 | 0.6139609217643738 | 0.4063791036605835 | 238.1427722472759 | 1051.549410061322 | 60.12795238176899 |
| `after_coefficient_space_fit_same_model` | 5.80192505150734e-10 | 4.357725629233755e-06 | 9.135822438111063e-06 | 0.005304594633387171 | 0.0282702865847776 | 0.00022007718604065229 |

## Objective Gradient Comparison

| objective | loss | grad norm | max abs grad | nonzero grad values |
|---|---:|---:|---:|---:|
| `coefficient_mse` | 35.709598541259766 | 8.672477896769802 | 0.8242826461791992 | 779841 |
| `coefficient_mae` | 2.9923110008239746 | 0.6153488849882197 | 0.01666666753590107 | 779802 |
| `label_mse` | 53.271812438964844 | 12.524320953615378 | 1.3700401782989502 | 779842 |
| `label_mae` | 5.021430969238281 | 1.097238073894266 | 0.029724514111876488 | 764931 |

Gradient cosine similarities:

- `coefficient_mse_vs_label_mae`: `0.3780352202117354`
- `coefficient_mse_vs_label_mse`: `0.8919979224399229`
- `coefficient_mae_vs_label_mae`: `0.7080178559498154`

## Controlled Normal-Loop Training

| experiment | loss | H MAE meV | H RMSE meV | coeff MAE | label MAE | spectral RMSE eV | support F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `dense_readout_coefficient_space_mse` | `coefficient_space_mse` | 0.02935275984940058 | 0.12675771955223578 | 1.3092943845549598e-05 | 5.0216771342093125e-05 | 0.006364218142823123 | 0.7611241217798596 |
| `dense_readout_label_space_mse` | `block_type_mse` | 0.0006385574685610014 | 0.0032259424459461995 | 6.28702764515765e-07 | 1.1661804819596e-06 | 9.631375806961061e-05 | 0.7629107981220657 |
| `dense_readout_block_type_mae` | `block_type_mae` | 4.223005905094715 | 7.667159765739295 | 0.005765518173575401 | 0.008953077718615532 | 0.22347617802447078 | 0.7611241217798596 |
| `default_readout_coefficient_space_mse` | `coefficient_space_mse` | 0.9193035037034956 | 3.4557034193548333 | 0.0014567130710929632 | 0.0016407094663009048 | 0.007094746640101605 | 1.0 |

## Interpretation

- The coefficient exposure path is internally consistent: after coefficient-space fitting, coefficient, label, and reconstructed-H errors are all near zero for the same model object.
- The failure is not label-space projection itself: dense `block_type_mse` reaches near-zero H error, while dense `block_type_mae` remains several meV away.
- On the same initialized model, `coefficient_mse` and `label_mse` gradients are strongly aligned, while `coefficient_mse` and `label_mae` are weakly aligned.
- The default readout also improves substantially with coefficient-space MSE, reaching sub-meV H MAE in this one-sample readout-ops run.

## Output Files

- JSON: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/coefficient_space_loss_audit.json`
- objective CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/objective_gradient_comparison.csv`
- training CSV: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/controlled_training_comparison.csv`
- report: `/home/christian/Escritorio/CINN/repositorios/grap2math_yo/graph2mat/Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/report.md`

## Reproduction Command

```bash
PYTHONPATH=src /home/christian/graph2mat-env/bin/python Comparison/results/diagnostics/h2o_hamiltonian/coefficient_space_loss_audit/coefficient_space_loss_audit.py
```

## Validation

- `py_compile coefficient_space_loss_audit.py: passed`
- `coefficient_space_loss_audit.py: passed`
