"""Existing 4.7 loss in train, unchanged gauge validation/checkpoint selection."""
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

from graph2mat.core.data.metrics import gauge_invariant_mse


def projection_columns(S, padded_columns):
    S = np.asarray(S, dtype=np.float64)
    if (S.ndim != 2 or S.shape[0] != S.shape[1] or not np.isfinite(S).all()
            or np.linalg.norm(S-S.T)/max(np.linalg.norm(S), 1e-30) > 1e-8):
        raise ValueError("nonfinite/nonsquare/non-Hermitian overlap")
    S = (S+S.T)/2
    values, vectors = np.linalg.eigh(S)
    if values[0] < 1e-3 or np.min(abs(values-1e-2)) < 1e-10:
        raise ValueError("overlap floor active or ambiguous low-space boundary")
    low = values < 1e-2
    if type(padded_columns) is not int or padded_columns < int(low.sum()):
        raise ValueError("padding cannot truncate low-space directions")
    C = np.zeros((len(S), padded_columns), dtype=np.float64)
    C[:, :low.sum()] = vectors[:, low]/np.sqrt(values[low])
    np.testing.assert_allclose(C.T@S@C, np.diag(np.arange(padded_columns) < low.sum()), atol=1e-10, rtol=0)
    return C


class NullspaceTrainingCheckpoint(ModelCheckpoint):
    """One scientific checkpoint callback, compatible with frozen RecoveryCallback."""
    def __init__(self, spectral_weight=1e-4, **kwargs):
        kwargs.setdefault("monitor", "val_loss")
        kwargs.setdefault("mode", "min")
        kwargs.setdefault("save_top_k", 1)
        if (spectral_weight != 1e-4 or kwargs["monitor"] != "val_loss"
                or kwargs["mode"] != "min" or kwargs["save_top_k"] != 1):
            raise ValueError("approved fixed weight and best-val_loss/min policy required")
        super().__init__(**kwargs)
        self.spectral_weight = spectral_weight
        self._validation_loss = None

    def _check(self, model):
        loss = model.loss_fn
        if (type(loss) is not gauge_invariant_mse or model.training_stages_config
                or loss is not model.default_loss_fn
                or loss.metric_kwargs.get("spectral_weight") != self.spectral_weight):
            raise ValueError("native gauge loss with fixed train weight, without stages, required")
        return loss

    def on_fit_start(self, trainer, pl_module):
        self._check(pl_module)

    def on_validation_start(self, trainer, pl_module):
        if self._validation_loss is not None:
            raise ValueError("nested validation would overwrite train-weight restoration")
        self._validation_loss = self._check(pl_module)
        self._validation_loss.metric_kwargs["spectral_weight"] = 0.

    def _restore(self):
        if self._validation_loss is not None:
            self._validation_loss.metric_kwargs["spectral_weight"] = self.spectral_weight
            self._validation_loss = None

    def on_validation_end(self, trainer, pl_module):
        try:
            super().on_validation_end(trainer, pl_module)
        finally:
            self._restore()

    def on_exception(self, trainer, pl_module, exception):
        self._restore()
        super().on_exception(trainer, pl_module, exception)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Resumed CLI serializes its in-memory basis as an unusable string;
        # keep the original None/glob constructor policy and the real top-level table.
        for key in ("hyper_parameters", "datamodule_hyper_parameters"):
            if key in checkpoint:
                checkpoint[key]["basis_table"] = None
