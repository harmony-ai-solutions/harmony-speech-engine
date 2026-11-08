"""Canonical device resolution (Phase 11-0).

The engine accepts a device per model entry (``device_config.device``) and
previously passed those strings around unchanged. That is ambiguous on CUDA:
``torch.device("cuda")`` is the *unindexed* device (index ``None``) while a
loaded model reports ``cuda:0``; the two compare **unequal**. Phase 9's AR
runner used that comparison to decide whether a pool arena could serve a model,
so every CUDA model silently fell back to the sequential loop.

This module makes the resolution explicit and canonical:

* ``None`` / ``"auto"`` → ``cuda:<active>`` when CUDA is available, else ``cpu``;
* ``"cuda"`` (no index) → ``cuda:<active>`` — the process's active device. That
  is the single-device default (device 0 after ``CUDA_VISIBLE_DEVICES`` is
  applied); a mixed-device deployment should spell explicit indices for every
  secondary device so resolution never depends on construction order;
* ``"cuda:N"`` → ``cuda:N``, **preserved**: explicit placement is honoured
  everywhere (pool arena, runner registration, probe, VRAM accounting);
* ``"cpu"`` → ``cpu``.

Everything is pure torch (no CUDA initialisation beyond the query itself), so a
CPU-only test host resolves ``cpu`` without touching the driver.
"""

from __future__ import annotations

import torch

__all__ = [
    "cuda_index",
    "device_key",
    "devices_match",
    "resolve_device",
]


def _cuda_available() -> bool:
    """``torch.cuda.is_available()`` that never raises on odd driver states."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive on broken driver installs
        return False


def _active_cuda_index() -> int:
    """Index of the process's active CUDA device (0 on a fresh process)."""
    return int(torch.cuda.current_device())


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Return the canonical, indexed device for ``device``.

    Idempotent: resolving a resolved device returns it unchanged, and
    ``resolve_device(resolve_device(x)) == resolve_device(x)``.

    An unindexed ``"cuda"`` becomes ``cuda:<active>`` only when CUDA is
    available; on a CPU-only host it stays ``torch.device("cuda")`` so callers
    keep their existing "configured for CUDA, unavailable" handling (the
    Whisper loader's cuDNN guard, the AR pool's CPU fallback).
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        if _cuda_available():
            return torch.device("cuda", _active_cuda_index())
        return torch.device("cpu")

    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type != "cuda" or resolved.index is not None:
        return resolved
    if not _cuda_available():
        return resolved
    return torch.device("cuda", _active_cuda_index())


def device_key(device: torch.device | str | None) -> str:
    """Stable string key for per-device accounting maps (``"cuda:1"``, ``"cpu"``)."""
    return str(resolve_device(device))


def cuda_index(device: torch.device | str | None) -> int | None:
    """CUDA index of ``device`` (``None`` for CPU, or unindexed CUDA without a driver)."""
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return None
    return None if resolved.index is None else int(resolved.index)


def devices_match(left: torch.device | str | None, right: torch.device | str | None) -> bool:
    """True when both spellings resolve to the same device.

    This is the comparison Phase 9's runner registration was missing:
    ``devices_match("cuda", "cuda:0")`` is ``True`` on a single-device host,
    while ``devices_match("cuda:0", "cuda:1")`` stays ``False`` so a
    cross-device arena is still refused.
    """
    return resolve_device(left) == resolve_device(right)
