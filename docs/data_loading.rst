Data loading and resource ownership
==================================

``MatrixDataModule.setup`` loads only the splits needed by the stage:
``fit`` uses train/val, ``validate`` uses val, ``test`` uses test, and
``predict`` uses predict. Passing ``None`` initializes every available split.
Repeated setup calls reuse existing datasets. Without explicit validation data,
validation can draw its subset lazily from training data; ``store_in_memory``
materializes only the splits requested by the current stage.

The automatic validation split accepts ``split_seed`` (default 32) and
``val_fraction`` (default 0.1, in [0, 1)). Explicit ``val_runs`` take priority;
an explicit empty list disables validation. The automatic split keeps at least
one training sample when data are available. For correlated trajectories, use
explicit validation runs from separate trajectories rather than a random split
of adjacent frames.

With ``copy_root_to_tmp=True``, each process owns a unique temporary directory.
A ZIP root is copied as an archive, preserving any internal root subdirectory.
Other DDP ranks create their own copy during setup. This avoids shared deletion
races but duplicates copying and disk space per rank.

Normal stage teardown keeps temporary files for subsequent stages.
``datamodule.teardown(None)`` closes datasets and archives and removes its own
temporary directory. Normal process exit also removes that directory. Abrupt
termination such as SIGKILL cannot run Python cleanup hooks.

``RotatingPoolData.close()`` stops its transfer thread, reader process, queue,
and Manager. The DataModule calls it after fit, retaining the underlying dataset
so that a later fit can create a fresh pool. Repeated setup does not create extra
pools. Standalone users of a rotating pool must call ``close()`` when finished.

The pool accepts ``seed=...``; by default its seed is drawn from NumPy's global
seeded generator, so ``seed_everything`` also controls its sampling order.
Asynchronous replacement timing remains scheduling-dependent: seeded sampling
does not make rotating-pool training exactly deterministic. Use a regular
dataset instead when exact reproducibility is required.

File-backed ZIP datasets reopen every archive independently after fork, even
when the parent has read a sample first. Their ``close()`` method closes only
readers they own, not archives supplied by the caller.

Reproducibility and checkpoint provenance
----------------------------------------

Use ``seed_everything: 0`` in the Lightning CLI YAML (or
``pytorch_lightning.seed_everything(0, workers=True)`` in Python) to seed the
model, data loading and rotating pool. ``trainer.deterministic: true`` requests
deterministic kernels, which can reduce throughput and may reject operations
without a deterministic implementation. It does not remove the rotating pool's
asynchronous scheduling dependence.

For script-based distributed CPU training, set ``trainer.accelerator: cpu``,
``trainer.devices: 2`` and ``trainer.strategy: ddp`` (Gloo). During distributed
training, parameters absent from a rank's matrix loss receive connected zero
gradients, including MACE's unused energy heads. The loss value is unchanged;
this permits ordinary DDP without unused-parameter detection. The optimizer still
excludes those energy heads. This adds zero-gradient materialization/communication
in DDP; the single-process path is unchanged. It does not by itself establish
distributed epoch-metric correctness, which is verified separately.

Checkpoints include a ``provenance`` dictionary with Python and installed
versions of Torch, PyG, Lightning, e3nn, MACE and sisl, plus the tracked source
checkout's Git commit and dirty flag. Missing optional distributions or a
checkout unavailable in an installed wheel are represented by ``None``.
The Git directory comes from the module, not the caller's working directory;
untracked files do not set the dirty flag. Provenance is informational: old
checkpoints without it still load, and prediction weights are unchanged.
