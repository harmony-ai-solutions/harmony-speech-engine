"""S3Gen mixed-precision experiment seam (Phase 11-2).

``HSE_S3GEN_AUTOCAST`` selects whether the S3Gen vocoder hop (flow-matching
mel decoder + HiFT waveform decoder) runs under CUDA ``torch.autocast``:

* ``off`` (default) — no autocast, byte-identical to the pre-experiment build.
* ``bfloat16`` — autocast to ``torch.bfloat16``.
* ``float16`` — autocast to ``torch.float16``.

Why this is not the Phase 4 fp16-weights policy
-----------------------------------------------
Phase 4 cast S3Gen's **weights** to float16 and measured audible degradation
(``docs/models.md`` §Precision): the flow/vocoder noise flipped sampled tokens
after the AR stage. This experiment is a different shape — fp32 master weights,
per-op casting, with norms/softmax/reductions staying fp32 — and it is
**falsifiable**: ``scripts/audio_regression.py``'s ``dtype:fp32-vs-autocast``
axis compares seeded A/B renders per family, and a clean negative closes the
question.

Where it is applied
-------------------
* **Solo upstream paths** — :func:`install_inference_autocast` decorates one
  loaded ``S3Gen`` instance's ``inference`` method (the seam the loader already
  uses for dtype hygiene). Every upstream vocode call goes through it: the
  sequential English/Turbo loops, multilingual, VC's ``generate``, and the AR
  runner's per-row fallback. Wrapping ``inference`` (not ``generate``) keeps the
  AR backbones (T3) out of the autocast scope — only S3Gen is the experiment.
* **Batched flow** — :func:`s3gen_autocast` is entered surgically inside
  :func:`~harmonyspeech.modeling.models.chatterbox.s3gen_batch._flow_batched`,
  around the encoder and CFM decoder forwards only; ODE state updates, masking,
  unpadding and ``mel2wav`` stay fp32.

Device policy
-------------
Autocast is a **CUDA-only** experiment (the T3 dtype policy is CUDA-only for the
same reason: CPU half precision is neither measured nor faster here). On any
non-CUDA device the context manager is a no-op, so CPU unit tests keep fp32 and
the default-off path cannot change behaviour.
"""

from __future__ import annotations

import contextlib
import os

import torch
from loguru import logger

__all__ = [
    "S3GEN_AUTOCAST_ENV",
    "autocast_enabled",
    "install_inference_autocast",
    "s3gen_autocast",
    "s3gen_autocast_dtype",
]

#: Environment variable selecting the experiment's precision.
S3GEN_AUTOCAST_ENV = "HSE_S3GEN_AUTOCAST"

#: Accepted spellings, normalised to the torch dtype.
_AUTOCAST_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}

#: Spellings that mean "off" (the default when the variable is absent).
_DISABLED = {"", "off", "none", "false", "0", "disabled"}

#: Unknown values already warned about, so a request loop logs once, not per call.
_WARNED: set[str] = set()


def s3gen_autocast_dtype(environ=None) -> torch.dtype | None:
    """Resolve ``HSE_S3GEN_AUTOCAST`` to a torch dtype, or ``None`` for off.

    Read on every call (not cached at import) so an operator — or a test — can
    flip the gate without reloading the model. An unknown value logs one warning
    and behaves as ``off``: a typo must never silently change the numerics.
    """
    source = os.environ if environ is None else environ
    raw = str(source.get(S3GEN_AUTOCAST_ENV, "")).strip().lower()
    if raw in _DISABLED:
        return None
    dtype = _AUTOCAST_DTYPES.get(raw)
    if dtype is None:
        if raw not in _WARNED:
            _WARNED.add(raw)
            logger.warning(
                f"Unknown {S3GEN_AUTOCAST_ENV}={raw!r}; expected off|bfloat16|float16 "
                "— S3Gen stays float32"
            )
        return None
    return dtype


def _autocast_device_type(device) -> str | None:
    """``"cuda"`` when the device supports the experiment, else ``None``.

    Separate from :func:`s3gen_autocast` so the unit suite can drive the branch
    on a CPU-only host (the task forbids bf16 CPU inference in the default
    suite, not a branch check).
    """
    try:
        kind = torch.device(device).type
    except (TypeError, RuntimeError):  # pragma: no cover - defensive on odd stand-ins
        kind = str(device)
    return "cuda" if kind == "cuda" else None


def autocast_enabled(device) -> bool:
    """Whether S3Gen autocast would actually engage on ``device`` right now."""
    return s3gen_autocast_dtype() is not None and _autocast_device_type(device) is not None


@contextlib.contextmanager
def s3gen_autocast(device, dtype: torch.dtype | None = None):
    """Enter S3Gen autocast for ``device``; yields the active dtype (or ``None``).

    ``None`` is yielded both when the gate is off and when the device is not
    CUDA, so callers can cast tensors back to the model dtype only when a cast
    actually happened (that keeps the default path bit-identical).
    """
    resolved = s3gen_autocast_dtype() if dtype is None else dtype
    if resolved is None:
        yield None
        return
    device_type = _autocast_device_type(device)
    if device_type is None:
        yield None
        return
    with torch.autocast(device_type=device_type, dtype=resolved):
        yield resolved


def _restore_float32(value):
    """Cast any half-precision tensor inside an ``inference`` result to float32.

    The upstream callers assume fp32 (``.numpy()`` rejects bfloat16, and the
    watermark/trim paths are fp32); only the autocast-enabled path ever produces
    a half tensor, so the off path returns the original objects untouched.
    """
    if torch.is_tensor(value):
        if value.is_floating_point() and value.dtype in (torch.float16, torch.bfloat16):
            return value.float()
        return value
    if isinstance(value, tuple):
        return tuple(_restore_float32(item) for item in value)
    if isinstance(value, list):
        return [_restore_float32(item) for item in value]
    return value


def install_inference_autocast(s3gen):
    """Wrap one loaded S3Gen's ``inference`` with the autocast seam (idempotent).

    The wrapper checks the env gate on every call, so installing it is safe even
    when the experiment is off (one extra Python call, identical numerics). It
    covers every solo vocode path — including upstream ``generate`` methods that
    call ``s3gen.inference`` internally — without touching the installed package
    (the same instance-level seam discipline the loader already uses for dtype).
    """
    if s3gen is None or not hasattr(s3gen, "inference"):
        return s3gen
    if getattr(s3gen, "_hse_autocast_seam", False):
        return s3gen

    original = s3gen.inference

    def inference(*args, **kwargs):
        dtype = s3gen_autocast_dtype()
        if dtype is None:
            return original(*args, **kwargs)
        with s3gen_autocast(getattr(s3gen, "device", None), dtype=dtype) as active:
            result = original(*args, **kwargs)
        return _restore_float32(result) if active is not None else result

    s3gen.inference = inference
    s3gen._hse_autocast_seam = True
    return s3gen
