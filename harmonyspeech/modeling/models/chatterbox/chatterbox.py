"""Chatterbox model wrappers for Harmony Speech Engine.

This module provides wrapper classes for the Chatterbox TTS and VC models,
exposing a consistent interface for model loading and inference.
"""

import os
import types

import numpy as np
import torch

# Import Chatterbox models from the chatterbox library
from chatterbox import ChatterboxMultilingualTTS, ChatterboxTTS, ChatterboxVC
from chatterbox.models.t3 import T3
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as _CHATTERBOX_SUPPORTED_LANGUAGES
from chatterbox.tts_turbo import ChatterboxTurboTTS as _ChatterboxTurboTTS
from huggingface_hub import snapshot_download
from loguru import logger

from harmonyspeech.modeling.models.chatterbox.t3_ar import ChatterboxT3AR

# HuggingFace repo ID for ChatterboxTurboTTS (mirrors tts_turbo.py REPO_ID)
_CHATTERBOX_TURBO_REPO_ID = "ResembleAI/chatterbox-turbo"

# ---------------------------------------------------------------------------
# dtype policy (Phase 4-1)
# ---------------------------------------------------------------------------
# `dtype` selects the precision of the autoregressive T3 backbone only. The
# measured rollout (docs/benchmarks/audio-regression.md) showed that casting
# S3Gen — its flow-matching decoder, the 16 kHz speech tokenizer, or the
# HiFiGAN vocoder — to float16 audibly degrades synthesis: vocoder/flow noise
# flips sampled tokens *after* the AR stage, producing different words even
# with identical reference audio (upstream itself keeps an `estimator_dtype =
# "fp32"` hint in s3gen.py). The VoiceEncoder's LSTM path is float32-only by
# construction (chatterbox.py casts its inputs back to float32).
#
# Hence: T3 -> configured dtype, S3Gen -> float32, VoiceEncoder -> float32.
# This keeps the ~1 GiB/role T3 saving without touching the quality-critical
# vocoder. CPU loading also forces float32 (the CPU executor rewrites fp16 to
# bf16, and bf16 sampling in T3 is neither tested nor faster here).
#
# Phase 11-2 layers an *experiment* on top of that policy: the S3Gen weights
# stay float32, but `HSE_S3GEN_AUTOCAST` (off by default) may run its forwards
# under CUDA `torch.autocast`. See `s3gen_precision.py` and docs/models.md.
_S3GEN_DTYPE = torch.float32
_VE_DTYPE = torch.float32


def _resolve_load_dtype(device: str, dtype: torch.dtype | None) -> torch.dtype:
    """Resolve the effective T3 dtype for a device (CPU is always float32).

    Any CUDA spelling keeps the requested dtype (Phase 11-0: ``cuda:0`` and an
    explicit ``cuda:1`` are CUDA just like the unindexed ``cuda`` — a string
    equality test here silently demoted a resolved secondary device to float32).
    """
    if dtype is None:
        return torch.float32
    try:
        device_type = torch.device(device).type
    except (TypeError, RuntimeError):  # pragma: no cover - defensive on odd stand-ins
        device_type = str(device)
    if device_type != "cuda":
        return torch.float32
    return dtype


def _submodule_dtypes(model) -> dict[str, str]:
    """Return the effective dtype names of the precision-relevant submodules."""
    result: dict[str, str] = {}
    for name in ("t3", "s3gen", "ve"):
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            result[name] = str(param.dtype).replace("torch.", "")
            break
    return result


def apply_model_dtype(model, dtype: torch.dtype | None, s3gen_precision: str = "float32") -> None:
    """Apply the project dtype policy to a freshly loaded Chatterbox model.

    Casts the T3 backbone to ``dtype`` (on CUDA only) and leaves S3Gen / the
    VoiceEncoder in float32. The Perth watermark network is also pinned to
    float32: it is constructed inside the loader's default-dtype context, so a
    float16 request would otherwise leave it with half-precision conv weights
    that reject the float32 mel input. Logs the effective per-submodule dtypes
    so an operator can see what actually got loaded.

    ``s3gen_precision`` (Phase 11-3) is the model's configured S3Gen vocode
    precision; it is stored on the S3Gen seam so the solo, batched and AR
    vocode paths all resolve the same per-model value.
    """
    device = str(getattr(model, "device", "cpu"))
    effective = _resolve_load_dtype(device, dtype)
    # Upstream `from_local` constructs modules inside the loader's
    # `torch.set_default_dtype(model_config.dtype)` context, so a float16
    # request would otherwise leave S3Gen / VE (and every other module created
    # there) silently half-precision even though `load_state_dict` loads float32
    # weights. Pin the policy explicitly.
    s3gen = getattr(model, "s3gen", None)
    if isinstance(s3gen, torch.nn.Module):
        model.s3gen = s3gen.to(_S3GEN_DTYPE)
    ve = getattr(model, "ve", None)
    if isinstance(ve, torch.nn.Module):
        model.ve = ve.to(_VE_DTYPE)
    t3 = getattr(model, "t3", None)
    if isinstance(t3, torch.nn.Module):
        model.t3 = t3.to(effective)
    # The watermarker is not part of the synthesis stack; keep it float32.
    watermarker = getattr(model, "watermarker", None)
    perth_net = getattr(watermarker, "perth_net", None)
    if isinstance(perth_net, torch.nn.Module):
        watermarker.perth_net = perth_net.float()
    # Phase 11-2: install the S3Gen autocast seam (idempotent, env-gated and a
    # no-op while `HSE_S3GEN_AUTOCAST` is unset). Wrapping `s3gen.inference`
    # here — rather than each caller — covers every solo vocode path including
    # upstream `generate()` methods, while keeping the AR backbones out of scope.
    if s3gen is not None:
        from harmonyspeech.modeling.models.chatterbox.s3gen_precision import (
            install_inference_autocast,
        )

        install_inference_autocast(s3gen, precision=s3gen_precision)
    logger.info(
        "Chatterbox effective submodule dtypes "
        f"(requested={getattr(dtype, '__str__', lambda: dtype)()}, "
        f"s3gen_precision={s3gen_precision}): {_submodule_dtypes(model)}"
    )

# ---------------------------------------------------------------------------
# Monkey-patch for ChatterboxMultilingualTTS to support CPU loading
# ---------------------------------------------------------------------------
# The upstream ChatterboxMultilingualTTS.from_local doesn't pass map_location
# to torch.load, causing failures on CPU when the checkpoint was saved on CUDA.
# This patch adds proper CPU mapping when device="cpu" is specified.
import chatterbox.mtl_tts as _mtl_module

_original_torch_load = _mtl_module.torch.load


def _patched_torch_load(*args, map_location=None, **kwargs):
    """Patched torch.load that defaults to CPU map_location when device is cpu."""
    if map_location is None:
        # Check if we're being called from the multilingual model loading path
        # by inspecting the first argument (the file path)
        if args and isinstance(args[0], (str, object)) and hasattr(args[0], "__str__"):
            # Default to CPU if no map_location specified
            map_location = torch.device("cpu")
    kwargs["map_location"] = map_location
    return _original_torch_load(*args, **kwargs)


# Apply the patch to the chatterbox.mtl_tts module
_mtl_module.torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Monkey-patch for perth package to handle missing PerthImplicitWatermarker
# ---------------------------------------------------------------------------
# The perth package may fail to import PerthImplicitWatermarker on some platforms
# (e.g., missing optional dependencies), causing it to be None. When chatterbox
# tries to instantiate it, we get "TypeError: 'NoneType' object is not callable".
# This patch provides a fallback that uses DummyWatermarker when the real one is unavailable.
import perth as _perth_module

if _perth_module.PerthImplicitWatermarker is None:

    class _FallbackPerthImplicitWatermarker:
        """Fallback watermarker that uses DummyWatermarker when PerthImplicitWatermarker is unavailable."""

        def __init__(self, *args, **kwargs):
            # Use DummyWatermarker as fallback
            self._dummy = _perth_module.DummyWatermarker(*args, **kwargs)

        def __getattr__(self, name):
            # Delegate all attribute access to the dummy
            return getattr(self._dummy, name)

        def apply_watermark(self, *args, **kwargs):
            return self._dummy.apply_watermark(*args, **kwargs)

        def detect_watermark(self, *args, **kwargs):
            return self._dummy.detect_watermark(*args, **kwargs)

    _perth_module.PerthImplicitWatermarker = _FallbackPerthImplicitWatermarker


def _ensure_chatterbox_t3_ar(model) -> None:
    """Graft HSE's cleaned decode loop onto a freshly loaded Chatterbox T3.

    Models built through the sharing factory already get ``ChatterboxT3AR`` (see
    ``factory._AR_T3_ROLES``); this covers the direct ``from_pretrained`` paths
    (which upstream builds with plain ``T3``) so both entry points decode the same
    way. A genuine Turbo T3 (GPT-2 topology, ``inference_turbo``) is left alone.
    """
    t3 = getattr(model, "t3", None)
    if isinstance(t3, T3) and not isinstance(t3, ChatterboxT3AR):
        t3.__class__ = ChatterboxT3AR


class ChatterboxTTSModel:
    """Wrapper for ChatterboxTTS model.

    ChatterboxTTS provides text-to-speech synthesis with voice cloning support.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        s3gen_precision: str = "float32",
        **kwargs,
    ) -> "ChatterboxTTS":
        """Load ChatterboxTTS model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            dtype: Requested precision for the T3 backbone (Phase 4-1). S3Gen
                and the VoiceEncoder always stay float32 — see the module
                docstring for the measured rationale.
            s3gen_precision: S3Gen vocode precision (Phase 11-3): ``"float32"``
                (default) or ``"float16"`` (CUDA autocast forwards).
            **kwargs: Additional arguments passed to ChatterboxTTS.from_pretrained.

        Returns:
            ChatterboxTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        model = ChatterboxTTS.from_pretrained(device=device, **kwargs)
        _ensure_chatterbox_t3_ar(model)
        apply_model_dtype(model, dtype, s3gen_precision)
        return model


class ChatterboxTurboTTSModel:
    """Wrapper for ChatterboxTurboTTS model.

    ChatterboxTurboTTS is a separate, faster TTS model from chatterbox.tts_turbo.
    It supports top_k and norm_loudness parameters and does NOT support
    exaggeration, cfg_weight, or min_p.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        s3gen_precision: str = "float32",
        **kwargs,
    ) -> "_ChatterboxTurboTTS":
        """Load ChatterboxTurboTTS model from pretrained weights.

        Workaround: The upstream `_ChatterboxTurboTTS.from_pretrained()` contains a bug
        where it calls `snapshot_download(token=os.getenv("HF_TOKEN") or True)`.
        When HF_TOKEN is not set, `None or True` evaluates to `True`, which instructs
        huggingface_hub to require a locally-cached token — causing a
        `LocalTokenNotFoundError` even though the model repo is public.

        We bypass this by calling `snapshot_download` ourselves with
        `token=os.getenv("HF_TOKEN") or None` (so unauthenticated downloads work for
        public repos), then delegating to `from_local`.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            dtype: Requested precision for the T3 backbone (Phase 4-1).
            s3gen_precision: S3Gen vocode precision (Phase 11-3).
            **kwargs: Additional keyword arguments (unused; kept for API compatibility).

        Returns:
            ChatterboxTurboTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        # Use the caller-supplied token if available, otherwise None (allow anonymous
        # access to this public repo rather than forcing token=True).
        # Also treat empty string as None to avoid 'Illegal header value b'Bearer '' errors.
        hf_token = os.getenv("HF_TOKEN") or None
        if hf_token == "":
            hf_token = None

        local_path = snapshot_download(
            repo_id=_CHATTERBOX_TURBO_REPO_ID,
            token=hf_token,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )

        model = _ChatterboxTurboTTS.from_local(local_path, device)

        # Workaround: librosa.load / librosa.resample inside prepare_conditionals
        # return float64 numpy arrays by default. Two downstream consumers both
        # crash when receiving float64 input:
        #
        # 1. s3tokenizer.log_mel_spectrogram — does torch.from_numpy(audio) preserving
        #    float64, then tries `_mel_filters (float32) @ magnitudes (float64)`:
        #      RuntimeError: expected scalar type Float but found Double
        #
        # 2. voice_encoder.forward (via ve.embeds_from_wavs) — LSTM rejects float64:
        #      ValueError: input must have the type torch.float32, got type torch.float64
        #
        # Fix: monkey-patch both consumers on the loaded model instance to cast
        # numpy arrays / tensors to float32 at entry.

        # --- Patch 1: s3tokenizer.log_mel_spectrogram ---
        _orig_log_mel = model.s3gen.tokenizer.log_mel_spectrogram.__func__

        def _log_mel_float32(self_tok, audio, padding=0):
            if isinstance(audio, np.ndarray):
                audio = audio.astype(np.float32)
            elif torch.is_tensor(audio) and audio.dtype != torch.float32:
                audio = audio.float()
            return _orig_log_mel(self_tok, audio, padding)

        model.s3gen.tokenizer.log_mel_spectrogram = types.MethodType(_log_mel_float32, model.s3gen.tokenizer)

        # --- Patch 2: voice_encoder.forward (mels input to LSTM) ---
        _orig_ve_forward = model.ve.forward

        def _ve_forward_float32(mels, *args, **kwargs):
            if torch.is_tensor(mels) and mels.dtype != torch.float32:
                mels = mels.float()
            return _orig_ve_forward(mels, *args, **kwargs)

        model.ve.forward = _ve_forward_float32

        apply_model_dtype(model, dtype, s3gen_precision)
        return model


class ChatterboxMultilingualTTSModel:
    """Wrapper for ChatterboxMultilingualTTS model.

    ChatterboxMultilingualTTS provides TTS synthesis supporting multiple languages.
    """

    # 23 supported language codes mirroring upstream chatterbox library.
    # Used by serving_engine for LanguageOptions registration and API validation.
    SUPPORTED_LANGUAGES: dict = _CHATTERBOX_SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        s3gen_precision: str = "float32",
        **kwargs,
    ) -> "ChatterboxMultilingualTTS":
        """Load ChatterboxMultilingualTTS model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            dtype: Requested precision for the T3 backbone (Phase 4-1).
            s3gen_precision: S3Gen vocode precision (Phase 11-3).
            **kwargs: Additional arguments passed to ChatterboxMultilingualTTS.from_pretrained.

        Returns:
            ChatterboxMultilingualTTS model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        model = ChatterboxMultilingualTTS.from_pretrained(device=device, **kwargs)

        # Phase 12-7: keep Llama_520M's native SDPA and use HSE's cleaned decode
        # loop. The alignment analyzer (which needs eager attention) is opt-in via
        # ``HSE_CHATTERBOX_MTL_ANALYZER`` and toggles attention per call.
        _ensure_chatterbox_t3_ar(model)

        apply_model_dtype(model, dtype, s3gen_precision)
        return model


class ChatterboxVCModel:
    """Wrapper for ChatterboxVC (Voice Conversion) model.

    ChatterboxVC provides voice conversion functionality to transform
    speech from one voice to another while preserving content.
    """

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        s3gen_precision: str = "float32",
        **kwargs,
    ) -> "ChatterboxVC":
        """Load ChatterboxVC model from pretrained weights.

        Args:
            device: Device to load the model on (e.g., "cpu", "cuda").
            dtype: Requested precision. VC has no T3 backbone, so this is
                accepted for API symmetry and only S3Gen's float32 policy
                applies (Phase 4-1).
            s3gen_precision: S3Gen vocode precision (Phase 11-3).
            **kwargs: Additional arguments passed to ChatterboxVC.from_pretrained.

        Returns:
            ChatterboxVC model instance.
        """
        if device is None:
            device = "cpu"
        elif isinstance(device, torch.device):
            device = str(device)

        model = ChatterboxVC.from_pretrained(device=device, **kwargs)
        apply_model_dtype(model, dtype, s3gen_precision)
        return model


__all__ = [
    "ChatterboxTTSModel",
    "ChatterboxTurboTTSModel",
    "ChatterboxMultilingualTTSModel",
    "ChatterboxVCModel",
    "apply_model_dtype",
]
