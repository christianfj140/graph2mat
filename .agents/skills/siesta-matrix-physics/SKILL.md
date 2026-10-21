---
name: siesta-matrix-physics
description: Physics and numerical-consistency checks for SIESTA atom-orbital matrices H and S, including non-orthogonal basis effects, orbital ordering, sparse structure, k-space reconstruction, Hermiticity, and basis/gauge conventions.
---

# SIESTA Matrix Physics

## Purpose

Use this skill when reasoning about SIESTA Hamiltonian or overlap matrices, their machine-learned predictions, or observables derived from them.

The goal is to prevent physically invalid conclusions caused by basis mismatch, component mismatch, sparse-support conventions, non-orthogonality, or incorrect matrix reconstruction.

## Core principle

A low numerical error on a stored matrix is not sufficient evidence of physical correctness.

Always distinguish:

- matrix representation correctness;
- numerical prediction error;
- downstream observable accuracy.

## Mandatory compatibility checks

Before comparing two matrices, verify:

1. same atomic structure and atom ordering;
2. same basis set and orbital ordering;
3. same pseudopotentials;
4. same SIESTA settings relevant to the matrix;
5. same matrix component semantics;
6. same spin/SOC convention;
7. same units;
8. same translation/block convention;
9. same sparse-support convention;
10. same overlap treatment.

If any required compatibility item is unknown, label the comparison diagnostic-only.

## Non-orthogonal basis

For a non-orthogonal basis, do not assume `S = I`.

Electronic eigenproblems are generally of the form

`H(k) c = epsilon S(k) c`.

If a downstream calculation uses only H while the representation requires S, state explicitly that the result is conditional or incomplete.

If predicted H is evaluated with reference S, label the result as a conditional diagnostic unless the scientific protocol explicitly permits it.

## Matrix semantics

Determine whether the stored matrix is:

- H only;
- S only;
- multi-component;
- spin-resolved;
- auxiliary channels plus H;
- a residual/delta target rather than the physical matrix itself.

Never interpret an auxiliary matrix channel as a physical Hamiltonian component without evidence.

For delta-learning, reconstruct the physical matrix explicitly:

`H = H0 + DeltaH`

before any physical comparison unless the requested metric is specifically about the residual.

## Hermiticity

Check Hermiticity before interpreting spectra.

Recommended diagnostic:

`||H - H^dagger||_F / max(||H||_F, eps)`

Apply the analogous diagnostic to S and to derivatives when relevant.

Do not silently symmetrize before reporting the raw defect. If symmetrization is used for a downstream calculation, report both:

- raw Hermiticity defect;
- result after the explicitly documented projection.

## Sparse matrices

Never average errors over implicit zeros without declaring that convention.

Distinguish at minimum:

- reference support;
- prediction support;
- union support.

Useful diagnostics include:

- MAE/RMSE on declared support;
- support precision/recall/F1;
- false-zero rate;
- false-nonzero rate;
- relative Frobenius error.

For sparse thresholds, test sensitivity to the threshold if conclusions could change.

## Blockwise analysis

When possible decompose error by:

- onsite vs offsite;
- species pair;
- orbital pair;
- interatomic distance;
- translation vector;
- matrix block norm.

A global MAE can hide a severe error in a physically important sparse subset.

## k-space reconstruction

When reconstructing `H(k)` and `S(k)` from real-space blocks:

- verify translation-vector sign convention;
- verify phase convention;
- verify cell indexing;
- verify Hermiticity in k-space;
- use identical k-points for reference and prediction.

Never compare spectra obtained from different k-grids or different broadening/energy windows without labeling the mismatch.

## Spectral validation

Use matrix-level metrics and spectral metrics together.

Recommended downstream checks:

- eigenvalue errors at identical k-points;
- band errors in physically relevant windows;
- DOS with identical broadening and normalization;
- Fermi-level-sensitive observables only when Fermi energies are valid and comparable.

Do not conclude that lower matrix MAE guarantees better bands or DOS.

## Moving basis and derivatives

For atom-centered moving bases, stored matrix-element derivatives are representation-dependent.

Do not assume that a raw `dH/dR` is automatically the complete physical electron-phonon perturbation.

When the physical observable requires derivative terms associated with the overlap or basis motion, require the formulation to state them explicitly.

## Output requirements

For any matrix-physics assessment, report:

1. target matrix and component semantics;
2. compatibility status;
3. support convention;
4. primary numerical metrics;
5. Hermiticity status;
6. downstream observable checks;
7. any conditional assumptions involving S or basis/gauge;
8. whether the result is diagnostic-only or scientifically comparable.
