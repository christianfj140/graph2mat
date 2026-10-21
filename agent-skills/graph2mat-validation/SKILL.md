---
name: graph2mat-validation
description: Rigorous scientific validation protocol for Graph2Mat matrix predictions, including H/S error, blockwise metrics, sparse-support quality, spectral observables, extrapolation, seeds, cost, and fair model or dataset comparisons.
---

# Graph2Mat Validation

## Purpose

Use this skill to evaluate whether a Graph2Mat model, architecture, loss, dataset, or training change produces a scientifically meaningful improvement.

## Central rule

Do not equate lower matrix MAE with better physical predictions.

Validate in layers:

1. representation compatibility;
2. matrix accuracy;
3. physically meaningful decompositions;
4. downstream observables;
5. generalization;
6. uncertainty;
7. computational cost.

## Pre-registration

Before examining final results, define:

- primary claim;
- primary metric;
- test split;
- checkpoint-selection rule;
- allowed hyperparameter search;
- statistical replication plan;
- cost budget.

Do not select the winning metric post hoc.

## Mandatory comparability

Require identical or explicitly controlled:

- train/validation/test protocol;
- basis and orbital order;
- SIESTA settings;
- target component semantics;
- pseudopotentials;
- matrix support convention;
- model-selection rule;
- evaluation code and thresholds.

If comparing datasets, keep the model and training protocol fixed unless the experiment explicitly studies joint optimization.

## H and S validation

For H, report at minimum:

- MAE;
- RMSE;
- relative Frobenius error;
- p95/p99 absolute error;
- Hermiticity defect.

When relevant also report:

- onsite/offsite;
- species-pair;
- orbital-pair;
- distance-conditioned error;
- support precision/recall/F1.

For S, use analogous metrics when S is predicted.

If S is not predicted but is needed downstream, label reference-S evaluations as conditional diagnostics.

## Sparse-support policy

Declare whether metrics are computed on:

- reference support;
- prediction support;
- union support.

Do not hide false zeros or false nonzeros through one-sided support metrics.

Do not average over the full implicit zero matrix unless that choice is deliberate and justified.

## Delta-learning

If the model predicts a residual, evaluate both:

- residual accuracy, if scientifically useful;
- reconstructed physical matrix accuracy.

Do not compare residual MAE against another model's physical-H MAE.

## Spectral validation

Use the generalized eigenproblem when the basis is non-orthogonal.

Recommended metrics:

- eigenvalue RMSE at identical k-points;
- band error in specified energy windows;
- DOS error with identical broadening;
- errors near gaps, crossings, Fermi level, or other target regions.

Do not claim spectral improvement solely from H-MAE.

## Derivative-aware validation

If a change improves H but worsens `dH/dR`, treat that as a real trade-off until measurement artifacts are excluded.

Do not state that derivative accuracy is universally more important than H-MAE.

Instead:

- H-MAE measures static matrix fidelity;
- `dH/dR` is more directly related to local response;
- neither is a sufficient proxy for electron-phonon coupling.

If EPC is the final claim, require explicit EPC validation.

## Generalization axes

Separate and report distinct axes:

- displacement amplitude;
- number/fraction of active atoms;
- collective vs local distortions;
- supercell size;
- structural motif;
- composition;
- defects;
- bilayers/interfaces;
- moire or rotational mismatch, if applicable.

Do not merge all of these into one generic "OOD" score.

## Leakage controls

Group correlated structures before splitting.

Keep together:

- same parent structure;
- symmetry-equivalent perturbations when appropriate;
- nested variants of one sample;
- finite-difference stencil families;
- strongly correlated trajectory frames.

The final test set must not be used to choose sampler, amplitude, architecture, loss, or checkpoint.

## Statistical replication

Do not use a rigid universal seed count.

Use:

- at least 3 seeds for exploratory variance estimation;
- more seeds when effects are small relative to seed-to-seed variance;
- paired seeds/splits where possible.

Report effect size and uncertainty, not only mean values.

Do not call a difference meaningful if it is comparable to run-to-run variation.

## Cost accounting

Separate:

1. statistical sample efficiency;
2. real computational efficiency.

Report both:

`error vs N_valid_labels`

and

`error vs total_labeling_cost`.

The total labeling cost should include:

- successful SCF calculations;
- failed SCF calculations;
- retries;
- discarded structures;
- CPU-hours;
- wall time when relevant.

Also report training/inference GPU-hours separately.

A sampler can be statistically efficient but computationally inefficient.

## Fair dataset comparison

When comparing samplers, match two things separately:

- budget;
- physical target distribution.

Equal `N` does not make a fair comparison if one sampler covers a different displacement-amplitude or active-atom distribution.

Use a common frozen test distribution.

If the samplers intentionally target different physical distributions, state that the experiment compares task-specific design strategies rather than isolated sampling algorithms.

## Decision tiers

Use:

- Invalid: incompatible target, basis, split, units, or provenance.
- Diagnostic-only: useful trend but insufficient controls.
- Validation-ready: controlled internal comparison with uncertainty and frozen evaluation.
- Claim-ready: independent final test, uncertainty, physically relevant observables, and complete provenance.

## Output

Conclude with:

1. primary result;
2. uncertainty;
3. physical relevance;
4. failure modes;
5. cost trade-off;
6. generalization boundaries;
7. evidence tier;
8. highest-value next experiment.
