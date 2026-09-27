"""A package to generate and manipulate sparse equivariant matrices.

The package contains a `core` submodule implementing the main functionality,
and other submodules that implement specific functionalities.
"""

from .__version__ import __version__

try:
    import torch.serialization as _torch_serialization

    # e3nn 0.4.4 (pinned by mace-torch) torch.load's its Wigner constants, which contain
    # slice objects that torch>=2.6 rejects by default (weights_only=True).
    if hasattr(_torch_serialization, "add_safe_globals"):
        _torch_serialization.add_safe_globals([slice])
except ImportError:
    pass

from .core import *
