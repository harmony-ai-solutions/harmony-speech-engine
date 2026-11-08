"""S3Gen mixed-precision seam (Phase 11-2 experiment, Phase 11-3 config knob).

Two inputs select whether the S3Gen vocoder hop (flow-matching mel decoder +
HiFT waveform decoder) runs under CUDA ``torch.autocast``:

* the per-model ``ModelConfig.s3gen_precision`` key — ``"fp32"`` (default,
  byte-identical to the pre-experiment build) or ``"fp16"`` (Phase 11-3: the
  approved promotion of the 11-2 experiment); and
* the ``HSE_S3GEN_AUTOCAST`` environment variable — ``off`` | ``bfloat16`` |
  ``float16``. **The environment variable overrides the config value** when it
  is set (an explicit ``off`` forces fp32); it is kept as the bench/testing
  escape hatch the 11-2 probe and the audio harness drive.

Why this is not the Phase 4 fp16-weights policy
-----------------------------------------------
Phase 4 cast S3Gen's **weights** to float16 and measured audible degradation
(``docs/models.md`` §Precision): the flow/vocoder noise flipped sampled tokens
after the AR stage. This is a different shape — fp32 master weights, per-op
casting, with norms/softmax/reductions staying fp32 — and it is **falsifiable**:
``scripts/audio_regression.py``'s ``dtype:fp32-vs-autocast`` axis compares
seeded A/B renders per family. The 11-2 verdict (fp16 PASS, bf16 rejected) is
what made ``"fp16"`` an approved config value.

Where it is applied
-------------------
* **Solo upstream paths** — :func:`install_inference_autocast` decorates one
  loaded ``S3Gen`` instance's ``inference`` method (the seam the loader already
  uses for dtype hygiene) and stores that model's configured precision on the
  instance (``_hse_s3gen_precision``). Every upstream vocode call goes through
  it: the sequential English/Turbo loops, multilingual, VC's ``generate``, and
  the AR runner's per-row fallback. Wrapping ``inference`` (not ``generate``)
  keeps the AR backbones (T3) out of the autocast scope — only S3Gen is the
  experiment.
* **Batched flow** — :func:`s3gen_autocast` is entered surgically inside
  :func:`~harmonyspeech.modeling.models.chatterbox.s3gen_batch._flow_batched`,
  around the encoder and CFM decoder forwards only; ODE state updates, masking,
  unpadding and ``mel2wav`` stay fp32.

Device policy
-------------
Autocast is a **CUDA-only** feature (the T3 dtype policy is CUDA-only for the
same reason: CPU half precision is neither measured nor faster here). On any
non-CUDA device the context manager is a no-op, so CPU unit tests keep fp32 and
the default-fp32 path cannot change behaviour.
"""

from __future__ import annotations

import contextlib
import os

import torch
from loguru import logger

__all__ = [
    "S3GEN_AUTOCAST_ENV",
    "S3GEN_PRECISION_FP16",
    "S3GEN_PRECISION_FP32",
    "VALID_S3GEN_PRECISIONS",
    "autocast_enabled",
    "effective_s3gen_precision",
    "install_inference_autocast",
    "s3gen_autocast",
    "s3gen_autocast_dtype",
]

#: Environment variable selecting the experiment's precision (overrides config).
S3GEN_AUTOCAST_ENV = "HSE_S3GEN_AUTOCAST"

#: Config spellings (``ModelConfig.s3gen_precision``).
S3GEN_PRECISION_FP32 = "fp32"
S3GEN_PRECISION_FP16 = "fp16"
VALID_S3GEN_PRECISIONS = (S3GEN_PRECISION_FP32, S3GEN_PRECISION_FP16)

#: Accepted env spellings, normalised to the torch dtype.
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


def _configured_dtype(configured) -> torch.dtype | None:
    """Map a config ``s3gen_precision`` value to a torch dtype (or ``None``).

    ``ModelConfig`` only admits ``fp32``/``fp16``; the aliases here keep a
    hand-built config or a probe honest without widening the shipped surface.
    An unknown value is treated as fp32 (a typo must never change numerics).
    """
    if configured is None:
        return None
    raw = str(configured).strip().lower()
    if raw in ("", S3GEN_PRECISION_FP32, "float32", "none", "off"):
        return None
    if raw in (S3GEN_PRECISION_FP16, "float16", "half"):
        return torch.float16
    if raw in ("bfloat16", "bf16"):
        return torch.bfloat16
    return None


def s3gen_autocast_dtype(environ=None, configured=None) -> torch.dtype | None:
    """Resolve the effective S3Gen autocast dtype, or ``None`` for off/fp32.

    Precedence: **``HSE_S3GEN_AUTOCAST`` overrides ``configured``** when the
    variable is present (including an explicit ``off``, which forces fp32 even
    if the config says ``fp16`` — the bench/testing escape hatch). When the
    variable is absent, ``configured`` (the model's ``s3gen_precision``)
    decides: ``fp16`` → ``torch.float16``, ``fp32``/``None`` → ``None``.

    Read on every call (not cached at import) so an operator — or a test — can
    flip either input without reloading the model. An unknown env value logs
    one warning and behaves as ``off``.
    """
    source = os.environ if environ is None else environ
    if S3GEN_AUTOCAST_ENV in source:
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
    return _configured_dtype(configured)


def effective_s3gen_precision(configured=S3GEN_PRECISION_FP32, environ=None) -> str:
    """Return the effective precision name for observability (``/health``).

    The config knob only names ``fp32``/``fp16``; an env override may name
    ``bfloat16`` (the rejected experiment) or ``off``. This reports exactly
    what the seam will do, not merely what the config asked for.
    """
    dtype = s3gen_autocast_dtype(environ=environ, configured=configured)
    if dtype is None:
        return S3GEN_PRECISION_FP32
    if dtype == torch.bfloat16:
        return "bfloat16"
    return S3GEN_PRECISION_FP16


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


def autocast_enabled(device, precision=None) -> bool:
    """Whether S3Gen autocast would actually engage on ``device`` right now."""
    return s3gen_autocast_dtype(configured=precision) is not None and _autocast_device_type(device) is not None


@contextlib.contextmanager
def s3gen_autocast(device, dtype: torch.dtype | None = None, *, precision=None):
    """Enter S3Gen autocast for ``device``; yields the active dtype (or ``None``).

    ``precision`` is the model's configured ``s3gen_precision``; the env gate
    still overrides it. ``None`` is yielded both when the effective precision is
    fp32/off and when the device is not CUDA, so callers can cast tensors back
    to the model dtype only when a cast actually happened (that keeps the
    default path bit-identical).
    """
    resolved = s3gen_autocast_dtype(configured=precision) if dtype is None else dtype
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


def install_inference_autocast(s3gen, precision=None):
    """Wrap one loaded S3Gen's ``inference`` with the autocast seam (idempotent).

    ``precision`` is the model's configured ``s3gen_precision`` (Phase 11-3),
    stored on the instance as ``_hse_s3gen_precision`` so the batched flow and
    the AR runner can resolve the same per-model value. The wrapper checks the
    env gate first (which overrides the config) on every call, so installing it
    is safe even when precision is fp32 (one extra Python call, identical
    numerics). It covers every solo vocode path — including upstream
    ``generate`` methods that call ``s3gen.inference`` internally — without
    touching the installed package (the same instance-level seam discipline the
    loader already uses for dtype).
    """
    if s3gen is None or not hasattr(s3gen, "inference"):
        return s3gen
    if precision is not None:
        raw = str(precision).strip().lower()
        s3gen._hse_s3gen_precision = (
            raw if raw in VALID_S3GEN_PRECISIONS else S3GEN_PRECISION_FP32
        )
    if getattr(s3gen, "_hse_autocast_seam", False):
        return s3gen

    original = s3gen.inference

    def inference(*args, **kwargs):
        configured = getattr(s3gen, "_hse_s3gen_precision", S3GEN_PRECISION_FP32)
        dtype = s3gen_autocast_dtype(configured=configured)
        if dtype is None:
            return original(*args, **kwargs)
        with s3gen_autocast(getattr(s3gen, "device", None), dtype=dtype) as active:
            result = original(*args, **kwargs)
        return _restore_float32(result) if active is not None else result

    s3gen.inference = inference
    s3gen._hse_autocast_seam = True
    return s3gen
