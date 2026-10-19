# Six authorized w90 fits — Native execution

Authority: user «Lanzalos», 2026-10-19, after the v2 pilot and its cost estimate.
This executes the already-described six fits, not a new architecture or tuning
proposal. Work in the current dirty g2m-v2 checkout; no commits/worktrees.

## Binding protocol

Reuse `pilot_multitarget_w90.py` factories, frozen sources and the successful v2
CPU certificate. Do not modify that script, product, data, historical results,
defaults or autorun. H-only and H/S/DM/EDM direct, seeds 0/1/2, 4000 Adam updates
each, fixed weights, fp32/TF32 off, batch32, same backbone and sample order per
seed. Same cosine scheduler over4000 updates. Validate150 every200 updates;
select minimum H gauge-MSE (`val_loss` or `val/target/hamiltonian/loss`), never
joint total loss. No test-based selection. Save standard Lightning checkpoints.

One physical Lightning epoch contains the precomputed4000 batches; logical
dataset epochs remain38 batches with the historical final14-sample batch.
No epoch-dependent training stages are enabled. Reset diagnostic metrics at
logical epoch boundaries; do not reset optimizer or scheduler.

Sequence h_s0,joint_s0,h_s1,joint_s1,h_s2,joint_s2, separate worker processes.
Own detached supervisor, no automatic retries. Stop the sequence on any failure.
Adopt conservative caps20min per fit and90min total; never extend automatically.
Reuse v2 GPU guards: allocator<=55% total and30% launch-time free reserve;
start<75C, abort>=80C/free<25%/foreign compute PID/telemetry failure. Sample0.5s.
This targets>=20% free but cannot exclude sudden external allocations.
Preserve artifacts/started markers; do not restart an existing campaign.

The v2 certificate and k-points are reused only after validating hashes and all
13512 original source hashes plus product/helper hashes. Generate fresh seed
initializations, not trained pilot checkpoints. Freeze new runner/tests/configs,
orders and initial states too; check integrity before/after each fit.

## Task 1: Prepare and prove the training boundary

Write CPU tests first. Extend via a new experimental script only. Prove both
real small-model Lightning routes, update/order/LR continuity, validation and
H-only selector, checkpoint reload, exact shared backbone, no-clobber and hash
failure. Expected: focal tests pass with no external data or CUDA access.

## Task 2: Supervise and verify

Reuse bounded telemetry/PID release helpers, own process groups and immutable
six-job list. Test sequence abort and no automatic retry, timeout and ownership
boundaries. Run focal pilot and multitarget regressions. One fresh read-only
Native reviewer before GPU; fix Important/Critical once with RED→GREEN.

## Task 3: Prepare real artifacts and launch

Verify references/certificate, prepare frozen six-job directory, detach own
supervisor. Verify first actual updates/checkpoint and state; document launch
in roadmap/handoff. Estimate~1h train/validation; scientific CPU evaluation
after fits, not an accuracy claim. Do not wait for six fits to claim launch.

## Review Focus

Check actual optimizer-update budget and scheduler continuity, real sample IDs,
target-specific H selector vs joint total, checkpoint reload semantics, frozen
provenance, exclusive start, detached lifetime, bounded cleanup/telemetry,
foreign PID and stale own PID handling, and stopping pending jobs on failure.
No unrelated dirty files, source edits, extra arms or new datasets authorized.
