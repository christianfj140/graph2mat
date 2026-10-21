---
name: hamiltonian-derivatives
description: Validation and interpretation of Hamiltonian derivatives dH/dR from SIESTA references and ML predictions, including finite-difference convergence, autograd cross-checks, sparse metrics, basis/gauge consistency, and leakage-safe stencil design.
---

# Hamiltonian Derivatives

## Purpose

Use this skill whenever `dH/dR`, finite-displacement Hamiltonians, derivative learning, autograd derivatives, or derivative-based physical observables are involved.

## Reference definition

For Cartesian coordinate `R_(a,alpha)`, a central finite-difference reference is

`dH/dR ~= [H(R0 + delta e_(a,alpha)) - H(R0 - delta e_(a,alpha))] / (2 delta)`.

A finite difference of forces, a force-constant matrix, a dynamical matrix, or a phonon matrix is not a substitute for `dH/dR`.

## Stencil integrity

Require that `R0`, `R+`, and `R-`:

- share atom ordering;
- share cell and basis;
- differ only in the intended displacement;
- use identical SIESTA settings;
- share orbital ordering;
- share matrix-component semantics.

Store explicit metadata for:

- atom index;
- Cartesian axis;
- sign;
- delta;
- parent structure;
- split;
- reference/prediction provenance.

## Split policy

A complete stencil family must remain in one split.

Never place `R+` and `R-` from one parent in different train/test partitions.

Group all derivative-related variants before splitting.

## Delta convergence

Never assume one `delta` is correct.

Use a delta sweep whenever possible.

The useful region is a plateau where:

- truncation error is sufficiently small;
- SCF/reference noise is not amplified;
- the derivative estimate is stable;
- conclusions do not depend sensitively on delta.

If smaller delta worsens stability, suspect reference noise or numerical cancellation.

If larger delta changes the result systematically, suspect nonlinear/truncation effects.

## Reference-noise checks

For a representative subset:

- tighten SCF tolerances;
- compare derivative estimates;
- inspect whether the delta plateau shifts.

Do not attribute all derivative disagreement to ML if the reference derivative is numerically unstable.

## Prediction derivative routes

Possible ML derivative routes include:

- finite differences of predicted H;
- autograd/Jacobian of predicted H;
- a separately trained derivative head.

If more than one route exists, compare them on identical geometries.

A disagreement between autograd and finite differences can indicate:

- implementation error;
- nondifferentiable preprocessing;
- incorrect coordinate mapping;
- hidden geometry canonicalization;
- numerical precision issues.

## Delta-learning

If the model predicts `DeltaH = H - H0`, the physical derivative is

`dH/dR = dH0/dR + d(DeltaH)/dR`.

Do not compare only the residual derivative against a full-H SIESTA derivative.

State explicitly whether `H0` depends on geometry and how its derivative is handled.

## Metrics

Recommended primary metrics:

- MAE in eV/Angstrom;
- RMSE in eV/Angstrom;
- relative Frobenius error.

Useful diagnostics:

- cosine similarity;
- correlation;
- p95/p99;
- maximum absolute error;
- onsite/offsite;
- species pair;
- orbital pair;
- atom;
- Cartesian axis;
- distance;
- Hermiticity defect.

For near-zero reference derivative norms, do not use unstable relative metrics as the primary score.

## Sparse-support policy

Declare derivative support convention:

- reference support;
- prediction support;
- union support.

Report false-zero/false-nonzero behavior where meaningful.

Do not let implicit zeros dominate the score.

## Basis and gauge

A stored matrix-element derivative in an atom-centered moving basis is representation-dependent.

Do not state that accurate raw `dH/dR` alone proves accurate physical EPC.

If the target physical formulation also depends on overlap/basis-motion terms, defer the physical claim to the EPC validation layer.

## Trade-off interpretation

If H-MAE improves while derivative error worsens, possible causes include:

- an objective that optimizes values but not local slopes;
- oversmoothing;
- local high-frequency error;
- insufficient perturbation coverage;
- numerical derivative noise;
- regularization-induced flattening.

Distinguish these experimentally.

A derivative-aware loss or paired `R+/R-` training can be a causal test, but improvement after adding such a term does not by itself prove the original model was physically wrong.

## Statistical design

Use paired comparisons where possible:

- same parent stencils;
- same delta values;
- same train/test split;
- matched seeds.

Do not impose a rigid fixed seed number.

Use more replications if the effect size is small relative to seed variance.

## Evidence tier

Derivative results are:

- Invalid if stencil/units/basis/component mapping is inconsistent.
- Diagnostic-only if delta convergence or uncertainty is missing.
- Validation-ready if the stencil protocol, delta, uncertainty, and representation are controlled.
- Claim-ready only if the derivative is directly relevant to the stated claim and downstream physical validation is also adequate.

## Output

Report:

1. derivative definition;
2. stencil integrity;
3. delta convergence;
4. reference noise;
5. prediction route;
6. primary metrics;
7. diagnostic decompositions;
8. basis/gauge caveats;
9. uncertainty;
10. evidence tier.
