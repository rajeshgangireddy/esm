"""Device dispatch helpers, so cuda/xpu/cpu are handled the same way.

Keep this separate from "is a CUDA-only library available" (flash-attn,
xformers, transformer_engine, cuequivariance) -- that's a different question
and stays gated on `torch.cuda.is_available()` directly, wherever it's
checked.
"""

from __future__ import annotations

import torch

_ACCELERATED_AUTOCAST_TYPES = ("cuda", "xpu")


def resolve_default_device() -> torch.device:
    """Best available accelerator, else CPU."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def autocast_device_type(x: torch.Tensor | torch.device | torch.nn.Module) -> str:
    """device_type string for torch.autocast, from a tensor/device/module."""
    if isinstance(x, torch.device):
        return x.type
    if isinstance(x, torch.nn.Module):
        return next(x.parameters()).device.type
    return x.device.type


def supports_amp_autocast(device_type: str) -> bool:
    """Whether device_type supports bf16/fp16 autocast (cuda, xpu do; mps/cpu don't the same way)."""
    return device_type in _ACCELERATED_AUTOCAST_TYPES


def empty_cache(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()


def synchronize(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()


def manual_seed_all(device_type: str, seed: int) -> None:
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device_type == "xpu":
        torch.xpu.manual_seed_all(seed)


def get_rng_state_all(device_type: str) -> list[torch.Tensor] | None:
    if device_type == "cuda":
        return torch.cuda.get_rng_state_all()
    elif device_type == "xpu":
        return torch.xpu.get_rng_state_all()
    return None


def set_rng_state_all(device_type: str, states: list[torch.Tensor]) -> None:
    if device_type == "cuda":
        torch.cuda.set_rng_state_all(states)
    elif device_type == "xpu":
        torch.xpu.set_rng_state_all(states)
