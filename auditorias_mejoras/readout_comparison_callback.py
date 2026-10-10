"""Experiment-only paired initialization and synchronized CPU/CUDA telemetry."""

import hashlib
import json
from pathlib import Path
import time

import graph2mat  # bootstrap e3nn's safe globals before optional imports
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from graph2mat.tools.lightning.data import MatrixDataModule


def tensor_hash(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PairedDataModule(MatrixDataModule):
    """Pair shuffled sample order independently of readout constructor RNG usage."""

    shuffle_training = True

    def train_dataloader(self):
        return self._with_type_layout(DataLoader(self.train_dataset,
            batch_size=self.batch_size, shuffle=self.shuffle_training, drop_last=False,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
            **self._loader_kwargs()), coefficient_projector=self._coefficient_mse_projector)


class FrozenProbeDataModule(PairedDataModule):
    """One deterministic real batch without shuffling."""

    shuffle_training = False


class ComparisonCallback(pl.Callback):
    def __init__(self, shared_path: str, output_path: str, seed: int, warmup: int = 10,
                 probe: bool = False):
        self.shared_path, self.output_path = Path(shared_path), Path(output_path)
        self.seed, self.warmup, self.probe = seed, warmup, probe
        self.initial_hashes = {}
        self.raw_ms = {key: [] for key in ("forward", "backward", "step")}
        self.seen = 0
        self.cuda = False
        self.batch_hash = None
        self.gradient_checked = False

    def on_fit_start(self, trainer, pl_module):
        if trainer.ckpt_path:
            raise ValueError("campaign starts from paired fresh initialization, not a resumed checkpoint")
        modules = {"backbone": pl_module.model.mace,
                   "onsite": pl_module.model.matrix_readouts.self_interactions}
        if self.shared_path.exists():
            shared = torch.load(self.shared_path, map_location="cpu", weights_only=True)
            for name, module in modules.items():
                module.load_state_dict(shared[name], strict=True)
        else:
            self.shared_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.shared_path.with_suffix(".tmp")
            torch.save({name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
                        for name, module in modules.items()}, tmp)
            tmp.replace(self.shared_path)
        self.initial_hashes = {name: tensor_hash(module.state_dict().items()) for name, module in modules.items()}

    def on_train_start(self, trainer, pl_module):
        # Readout constructors consume different RNG counts; pair sample order explicitly.
        pl.seed_everything(self.seed, workers=True)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.cuda = next(pl_module.parameters()).is_cuda
        if self.probe:
            batch_hash = tensor_hash((key, value) for key, value in batch.items() if torch.is_tensor(value))
            if self.batch_hash is not None and batch_hash != self.batch_hash:
                raise ValueError("probe batch changed")
            self.batch_hash = batch_hash
        self.sync()
        if self.cuda and self.seen == self.warmup:
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_before_backward(self, trainer, pl_module, loss):
        self.sync()
        self.forward_end = time.perf_counter()

    def on_after_backward(self, trainer, pl_module):
        self.sync()
        self.backward_end = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.sync()
        end = time.perf_counter()
        if self.seen >= self.warmup:
            self.raw_ms["forward"].append((self.forward_end-self.start)*1000)
            self.raw_ms["backward"].append((self.backward_end-self.forward_end)*1000)
            self.raw_ms["step"].append((end-self.start)*1000)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss).all():
            raise ValueError("nonfinite training loss")
        self.seen += 1
        # Lightning zeroes gradients before validation; check outside timed hooks.
        gradients = [(name, p.grad) for name, p in pl_module.named_parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for _, g in gradients):
            raise ValueError("missing or nonfinite gradient")
        for branch in ("sk_net", "radial_gates"):
            selected = [g for name, g in gradients if branch in name]
            if selected and not any(g.abs().sum() > 0 for g in selected):
                raise ValueError("inactive gradient branch " + branch)
        self.gradient_checked = True

    def on_train_end(self, trainer, pl_module):
        if not self.gradient_checked:
            raise ValueError("missing gradient verification")
        result = {"initial_hashes": self.initial_hashes, "raw_ms": self.raw_ms,
                  "steps": self.seen, "device": "cuda" if self.cuda else "cpu",
                  "batch_hash": self.batch_hash,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.cuda else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.cuda else None,
                  "torch_version": torch.__version__, "model_dtype": str(next(pl_module.parameters()).dtype),
                  "cuda_device": torch.cuda.get_device_name() if self.cuda else None,
                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                  "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "scope": "synchronized wall-clock hooks; forward includes loss, step includes Adam; excludes H2D/collate; fresh process, memory reset after warmup"}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2)+"\n")
