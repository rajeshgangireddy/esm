"""Device dispatch helpers for CUDA, XPU, and CPU."""

from __future__ import annotations

import torch

_ACCELERATED_AUTOCAST_TYPES = ("cuda", "xpu")


def resolve_default_device() -> torch.device:
    """Prefer CUDA, then XPU, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def supports_amp_autocast(device_type: str) -> bool:
    """Use CUDA/XPU autocast while preserving CPU/MPS precision."""
    return device_type in _ACCELERATED_AUTOCAST_TYPES


def empty_cache(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "xpu":
        torch.xpu.empty_cache()


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
