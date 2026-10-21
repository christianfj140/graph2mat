---
name: materials-dataset-design
description: Scientific design and comparison of atomistic training datasets for matrix-learning models, especially Sobol, random Cartesian, phonon-mode, sparse, collective, and active-atom perturbations under fixed SCF budgets.
---

# Materials Dataset Design

## Purpose

Use this skill to design or compare atomistic datasets intended to minimize expensive electronic-structure labels while preserving accuracy in H, derivatives, EPC, or other downstream observables.

## Primary objective

Do not optimize "number of structures" in isolation.

Optimize the minimum real labeling cost that satisfies predefined accuracy targets.

Keep separate:

- statistical sample efficiency;
- computational labeling efficiency;
- downstream physical accuracy.

## Define the task first

Before comparing samplers, define:

- material/system family;
- target observables;
- target deployment distribution;
- displacement-amplitude range;
- active-atom fraction;
- local vs collective distortions;
- temperature/thermal distribution if relevant;
- supercell-size range;
- defects/interfaces/composition if relevant;
- maximum SCF budget;
- accuracy thresholds.

## Fair comparison requires two matches

A valid sampler comparison must consider both:

1. budget matching;
2. physical-distribution matching.

Equal `N` is not sufficient if the samplers generate systematically different amplitudes, active-atom counts, norms, or local environments.

Where possible, compare samplers under a common target distribution.

If the samplers intentionally target different distributions, frame the result as a task-specific design comparison, not a pure algorithmic comparison.

## Sampler families

Possible families include:

- random Cartesian;
- Sobol/quasi-random;
- Latin hypercube;
- sparse active-atom perturbations;
- local pair modes;
- radial/angular shells;
- normal-mode or phonon-mode sampling;
- thermal snapshots;
- active-learning-selected structures.

Do not assume any family is superior a priori.

## Sobol vs random Cartesian

For a causal comparison, keep fixed:

- dimensional parameterization;
- coordinate bounds;
- physical displacement mapping;
- active-atom policy;
- norm/amplitude distribution.

Change only the sequence generator where possible.

Document:

- dimension;
- scrambling;
- seed;
- skip/leap policy if used;
- physical transform from unit hypercube.

## Phonon-oriented designs

If sampling normal modes, cover explicitly:

- acoustic and optical branches;
- q-points;
- signs/phases;
- amplitudes;
- degenerate subspaces;
- mode combinations if deployment requires them.

A phonon-oriented dataset may be efficient for EPC while under-covering arbitrary Cartesian local distortions.

Treat that as specialization, not necessarily failure.

## Nested learning curves

Prefer nested subsets so that increasing N isolates the effect of more labels.

For each sampler report curves against:

- valid labeled structures;
- successful SCF count;
- total attempted SCF count;
- total labeling CPU-hours.

Do not hide failed or discarded calculations.

## Cost accounting

Separate:

`error vs N_successful`

from

`error vs total_cost`.

Total cost should include:

- successful SCF;
- failed SCF;
- retries;
- relaxations/preprocessing if sampler-specific;
- discarded structures.

Keep training GPU-hours separate from labeling cost.

## Frozen evaluation

Use a final frozen test independent of sampler selection.

Potential test strata:

- near-equilibrium interpolation;
- unseen amplitudes;
- local perturbations;
- collective perturbations;
- thermal configurations;
- derivative stencils;
- phonon modes/q-points;
- larger supercells;
- structural transfer.

The final test must not be used to tune sampler hyperparameters.

## Leakage prevention

Group before splitting:

- common parent structures;
- nested perturbations;
- stencil families;
- symmetry-equivalent variants where appropriate;
- neighboring correlated trajectory frames.

Do not split correlated variants as independent samples.

## Multi-objective selection

Do not collapse H, derivative, EPC, and cost into a post-hoc weighted scalar unless weights were specified before seeing results.

Prefer Pareto analysis across:

- H error;
- derivative error;
- EPC error;
- labeling cost;
- training cost;
- robustness across seeds.

If a hard scientific requirement exists, define threshold constraints first and minimize cost subject to them.

## Replication

Use paired experimental designs when possible.

At least 3 replicates can provide exploratory variance estimates, but the required number of seeds depends on:

- effect size;
- observed variance;
- decision stakes.

Small effects require more replication.

## Extrapolation claims

Treat separately:

- amplitude extrapolation;
- active-atom extrapolation;
- size extrapolation;
- structural extrapolation;
- composition extrapolation;
- mode/q extrapolation.

A larger supercell alone does not prove broad physical extrapolation.

## Recommended decision criterion

A strong operational target is:

"minimum total electronic-structure labeling cost such that all predefined H, dH/dR, and EPC thresholds are satisfied on an independent frozen test with acceptable uncertainty."

## Output

Report:

1. target deployment distribution;
2. sampler parameterization;
3. distribution-matching strategy;
4. nested budget levels;
5. leakage controls;
6. metrics and thresholds;
7. total labeling cost;
8. statistical replication;
9. Pareto/frontier result;
10. generalization boundaries.
