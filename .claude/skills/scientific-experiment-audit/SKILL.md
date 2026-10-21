---
name: scientific-experiment-audit
description: Fail-closed audit framework for computational-physics and ML experiments, separating software correctness from scientific validity and requiring explicit claim-evidence mapping, provenance, uncertainty, leakage controls, and reproducible decision gates.
---

# Scientific Experiment Audit

## Purpose

Use this skill to audit whether a computational experiment can support the scientific claim being made.

This skill is intentionally fail-closed.

A functioning pipeline does not imply a valid scientific result.

## Separate two questions

Always distinguish:

1. Is the software doing what it claims operationally?
2. Does the experiment support the scientific conclusion?

A test suite can establish the first without establishing the second.

## Start from claims

Construct a claim-evidence matrix.

For every important claim record:

- exact claim;
- primary metric;
- reference;
- test population;
- uncertainty requirement;
- cost definition;
- blocking assumptions;
- evidence status.

Do not begin by listing code improvements.

## Evidence hierarchy

Useful levels:

- implementation evidence;
- internal numerical consistency;
- controlled validation;
- independent test;
- downstream physical validation;
- external/independent reference where required.

The evidence level must match the strength of the claim.

## Fail-closed comparability

Block scientific comparison when any essential item differs or is unknown:

- material identity;
- geometry;
- basis;
- pseudopotentials;
- physical target;
- units;
- spin/component semantics;
- split;
- reference-generation protocol;
- evaluation code;
- model-selection protocol.

Do not repair missing provenance by assumption.

## Leakage audit

Inspect:

- train/validation/test lineage;
- parent-child structure relationships;
- derivative stencil families;
- near-duplicate structures;
- trajectory autocorrelation;
- symmetry-related samples;
- reused test data during model/sampler selection.

If the test influenced selection, it is not an independent final test.

## Metric audit

Check whether the metric directly measures the stated claim.

Examples:

- H-MAE is not EPC accuracy;
- training loss is not a scientific endpoint;
- matrix error is not automatically band accuracy;
- derivative error is not automatically complete physical coupling.

Require downstream validation when the claim is downstream.

## Statistical audit

Do not prescribe a universal fixed seed count.

Assess:

- paired vs unpaired design;
- seed-to-seed variance;
- effect size;
- confidence intervals or bootstrap uncertainty where appropriate;
- multiple comparisons;
- stopping/selection rules.

Three seeds may be adequate for an exploratory large effect and inadequate for a small effect.

## Cost audit

Distinguish:

- successful labels;
- attempted labels;
- failed/retried labels;
- CPU-hours;
- GPU-hours;
- wall time;
- preprocessing cost;
- storage/IO if material.

Report both sample efficiency and total computational efficiency.

Do not call a method "cheaper" based only on successful structure count.

## Sampler-comparison audit

For dataset-design claims, check both:

- equal budget;
- comparable target distribution.

Equal sample count does not isolate sampling strategy if amplitude, norm, active-atom count, or physical state distributions differ.

If distributions differ intentionally, phrase the claim as a comparison of complete dataset-design recipes.

## Reproducibility

Require enough information to recreate:

- input structures;
- SIESTA settings;
- pseudopotentials/basis;
- random seeds;
- splits;
- sampler parameters;
- software versions/commits;
- model configs;
- checkpoints;
- metric definitions;
- hardware for cost claims.

## Severity levels

Classify findings as:

- BLOCKER: invalidates the main claim.
- MAJOR: can materially change the conclusion.
- MODERATE: limits interpretation or reproducibility.
- MINOR: low scientific impact.
- OPPORTUNITY: improvement not required for validity.

## Evidence tiers

Use:

- Invalid.
- Diagnostic-only.
- Validation-ready.
- Claim-ready.

Do not upgrade a result because it "looks convincing."

## Recommended audit order

1. State the claim.
2. Identify required evidence.
3. Verify provenance/comparability.
4. Check leakage.
5. Check reference quality.
6. Check metric validity.
7. Check uncertainty.
8. Check cost accounting.
9. Check downstream physical evidence.
10. Assign evidence tier.
11. Recommend the smallest experiment that resolves the largest uncertainty.

## Output format

Produce:

### Claim
Exact scientific statement.

### Evidence required
Minimum evidence necessary.

### Current evidence
What is actually available.

### Findings
Each with severity and consequence.

### Valid conclusions
What can be stated now.

### Invalid/unsupported conclusions
What must not be stated.

### Highest-value next experiment
The smallest experiment that most reduces decision uncertainty.

## Hard prohibitions

Do not:

- declare a winner with incompatible provenance;
- use the final test for model or sampler selection;
- infer EPC quality from H or dH/dR alone;
- call a method cheaper from successful-label count alone;
- hide failed SCF attempts from cost comparisons;
- treat a fixed seed count as universally sufficient;
- treat equal N as a fair sampler comparison when physical distributions differ.
