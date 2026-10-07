Data loading and resource ownership
==================================

``MatrixDataModule.setup`` loads only the splits needed by the stage:
``fit`` uses train/val, ``validate`` uses val, ``test`` uses test, and
``predict`` uses predict. Passing ``None`` initializes every available split.
Repeated setup calls reuse existing datasets. Without explicit validation data,
validation can draw its subset lazily from training data; ``store_in_memory``
materializes only the splits requested by the current stage.

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
