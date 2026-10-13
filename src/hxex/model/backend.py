# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the engine computes: device and floating-point precision, as an explicit setting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

_DTYPES = ("auto", "float32", "float64")


@dataclass(frozen=True)
class Backend:
    """Device and dtype of every tensor the engine creates (nothing is set globally).

    ``device``: ``'auto'`` (CUDA if available, else CPU), ``'cpu'``, ``'cuda'`` or ``'cuda:N'``.
    ``dtype``: ``'auto'`` (float32, on the GPU and on the CPU), ``'float32'`` or ``'float64'``.
    The prototype used float64 on the CPU; float32 is about 1.5 times faster there and selects the
    same K for all but a few measurements in a thousand (docs/HISTORY.md item 46), so float64 is
    the setting to choose for the last digits.
    """

    device: str = "auto"
    dtype: str = "auto"

    def __post_init__(self):
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if not (self.device in ("auto", "cpu") or self.device.split(":")[0] == "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {self.device!r}"
            )

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {self.device!r} requested but CUDA is not available")
        return torch.device(self.device)

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "auto":
            return torch.float32
        return getattr(torch, self.dtype)

    def tensor(self, x: Any) -> torch.Tensor:
        """``x`` (array, list or tensor) as a tensor of this backend's dtype on its device."""
        dtype = self.torch_dtype()
        if isinstance(x, torch.Tensor):
            return x.to(device=self.torch_device(), dtype=dtype)
        np_dtype = np.float32 if dtype == torch.float32 else np.float64
        return torch.tensor(np.asarray(x, dtype=np_dtype), dtype=dtype, device=self.torch_device())

    def to_dict(self) -> dict[str, str]:
        return {"device": self.device, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Backend:
        return cls(**data)
