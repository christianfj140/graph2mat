---
name: electron-phonon-validation
description: Rigorous validation of electron-phonon coupling derived from learned SIESTA Hamiltonians, including dH/dR and dS/dR completeness, phonon normalization, Fourier conventions, gauge/degeneracy handling, k+q mapping, and downstream observables.
---

# Electron-Phonon Validation

## Purpose

Use this skill for any claim involving electron-phonon coupling, `g(k,q,nu)`, mode-resolved couplings, EPC proxies, or observables derived from learned electronic matrices and atomic displacements.

## Non-negotiable rule

Accurate H does not imply accurate `dH/dR`.

Accurate `dH/dR` does not imply accurate electron-phonon coupling.

Never use `dH/dR` as a sufficient proxy for EPC.

EPC must be validated explicitly when EPC is the scientific claim.

## Validation ladder

Validate each layer independently:

1. equilibrium H and S;
2. Cartesian matrix derivatives;
3. overlap/basis-motion derivatives if required;
4. phonon frequencies and eigenvectors;
5. real-space to q-space transformation;
6. electronic generalized eigenproblem;
7. mode projection;
8. `g(k,q,nu)`;
9. integrated/downstream observable.

Do not skip failed lower layers.

## Physical formulation completeness

Before interpreting `g`, identify the exact formula implemented.

Determine whether the formulation includes all terms required by the chosen moving non-orthogonal basis.

If the implementation uses only something equivalent to

`C^dagger (dH/dR) C`

without establishing that overlap/basis-motion terms are unnecessary or already included, label the quantity an EPC proxy rather than a validated physical matrix element.

If `dS/dR` or Pulay-like/moving-basis terms are required by the formulation, they must be present and validated.

## Electronic-state normalization

For non-orthogonal bases, verify normalization and eigenproblem conventions involving S(k).

Check:

- eigenvector normalization;
- phase convention;
- band ordering;
- degeneracy handling;
- k and k+q correspondence.

Do not compare raw complex matrix elements across calculations unless gauge alignment is meaningful.

## Phonon layer

Verify:

- force-constant/dynamical-matrix provenance;
- phonon frequencies;
- eigenvector normalization;
- atomic masses;
- acoustic modes;
- degeneracies;
- sign/phase conventions.

Near degeneracies, compare subspaces or invariant norms rather than mode indices alone.

## Fourier conventions

Audit explicitly:

- real-space lattice-vector convention;
- sign in the exponential;
- normalization factor;
- reference cell;
- q-point convention;
- atom/image indexing;
- Hermitian relations.

A sign or translation mismatch can preserve some norms while corrupting phase-sensitive quantities.

## k+q mapping

Verify that the electronic final state corresponds to the intended `k+q` under the chosen reciprocal-lattice convention.

Check folding into the Brillouin zone and band indexing.

Do not silently compare `k+q` values represented in different reciprocal images unless the gauge/transformation is handled consistently.

## Coupling metrics

Prefer gauge-robust quantities when raw complex phases are arbitrary.

Useful metrics:

- absolute error in `|g|`;
- relative error in `|g|` where reference is not near zero;
- error in `|g|^2`;
- mode-resolved error;
- q-resolved error;
- k-resolved error;
- energy-window-conditioned error;
- strongest-coupling channels;
- subspace norms near degeneracies.

Do not use relative error as the primary metric for near-zero couplings.

## Reference hierarchy

For a strong physical claim, require an independent EPC reference on a controlled subset.

Possible hierarchy:

- internal consistency only;
- comparison to finite-difference matrix-based reconstruction;
- comparison to an independent validated EPC implementation;
- comparison of downstream observables.

Clearly label the level achieved.

## Downstream validation

When relevant, propagate to observables such as:

- mode-resolved coupling strengths;
- Eliashberg-related quantities;
- linewidths;
- scattering rates;
- other target EPC observables.

Use identical broadening, grids, temperatures, and integration conventions.

## Dataset claims

If comparing dataset designs, do not conclude that the sampler with the best H or `dH/dR` is automatically best for EPC.

Require explicit EPC evaluation on a common frozen test.

Match both:

- label budget;
- target physical distribution.

For phonon-oriented datasets, evaluate whether gains are confined to the sampled q/mode manifold.

## Error localization

If EPC is wrong, determine the first layer where reference and prediction diverge:

- H/S;
- Cartesian derivatives;
- overlap/basis terms;
- phonons;
- Fourier transform;
- electronic states;
- mode projection;
- final coupling.

Do not debug the final `g` aggregate first.

## Evidence tier

Use:

- Invalid: inconsistent basis, normalization, q/k mapping, units, or formula.
- Proxy-only: an incomplete coupling expression is evaluated.
- Diagnostic-only: formula is plausible but independent physical validation is missing.
- Validation-ready: all layers are controlled and compared on frozen data.
- Claim-ready: independent EPC reference and uncertainty support the stated physical conclusion.

## Output

Always state:

1. exact EPC quantity/formula;
2. whether it is complete for the basis representation;
3. lower-layer validation status;
4. gauge/degeneracy policy;
5. primary EPC metrics;
6. independent reference status;
7. downstream observable status;
8. evidence tier.
