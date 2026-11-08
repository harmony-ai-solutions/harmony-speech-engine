"""Startup max-shape memory probe for CTranslate2 Whisper configs (Phase 10-3).

CTranslate2 allocates outside PyTorch's caching allocator, so
``torch.cuda.memory_allocated/reserved`` cannot see Whisper at all. This probe
runs a synthetic **max-shape** transcription against an already-loaded model
(``num_workers`` concurrent workers, ``window_batch`` windows each, the longest
admitted audio) and records the device-level ``torch.cuda.mem_get_info`` delta as
the config's ``peak_activation_bytes``.

The probe is deliberately best-effort and never fails boot:

* CPU-only builds / no CUDA / a CPU-placed model / a missing optional import all
  return a skip record with a ``reason`` instead of raising;
* it never downloads weights — it only uses the model the engine already
  loaded;
* an unexpected runtime error is logged and reported as a skip, because a
  missing attribution is honest while a guessed number is not.

Phase 11-0 fixed the probe's own input: all-zero audio made the batched pipeline
raise ``No clip timestamps found`` for any ≥30 s shape, leaving
``per_window_bytes = 0`` and the runtime governor inert. The probe now builds
deterministic speech-like audio and passes explicit ≤30 s ``clip_timestamps``
(no VAD dependence), so the max shape really runs and the measured cost is the
served one. It also measures on the **model's** device (``cuda:1`` stays 1).

The benchmark host fills the real numbers into
``docs/benchmarks/vram-profiles.md``; unit tests exercise the delta math with
mocked CUDA calls.
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from harmonyspeech.engine.vram_budget import (
    DEFAULT_WHISPER_BEAM_SIZE,
    DEFAULT_WHISPER_MAX_AUDIO_SECONDS,
    per_window_bytes,
)

_SAMPLE_RATE = 16000
#: faster-whisper chunks audio into 30 s clips before batching.
_CLIP_SECONDS = 30.0
#: Device-memory sampling interval used to catch the transient activation peak.
_SAMPLE_INTERVAL_S = 0.02


def _probe_audio(max_audio_seconds: float) -> "np.ndarray":
    """Deterministic speech-like probe audio for the max-shape run.

    All-zero audio used to make the batched pipeline raise
    ``No clip timestamps found`` for any audio ≥30 s (Phase 11-0 finding 3).
    Silence also under-measures the decoder side: Whisper stops after a token
    or two on it. A harmonic stack amplitude-modulated at syllable rate gives
    the encoder a normal spectrogram and the decoder a normal-length transcript,
    so the measured peak is the served shape rather than a degenerate one.
    """
    import numpy as np

    samples = max(1, int(_SAMPLE_RATE * float(max_audio_seconds)))
    t = np.arange(samples, dtype=np.float64) / _SAMPLE_RATE
    # Pitch glides like speech (not a stationary tone, which Whisper hallucinates on).
    f0 = 120.0 + 10.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / _SAMPLE_RATE
    signal = np.zeros(samples, dtype=np.float64)
    for harmonic in range(1, 8):
        signal += np.sin(harmonic * phase) / harmonic
    # 4 Hz syllable envelope plus a little noise for consonant-like onsets.
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    signal += 0.02 * np.random.default_rng(0).standard_normal(samples)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (0.25 * signal / peak).astype(np.float32)


def _probe_clip_timestamps(max_audio_seconds: float) -> list[dict]:
    """Explicit ≤30 s clip windows covering the probe audio.

    ``BatchedInferencePipeline.transcribe`` raises for audio ≥30 s when neither
    VAD nor ``clip_timestamps`` is supplied; synthesising the timestamps here
    keeps the probe independent of VAD behaviour (silence has no speech to
    detect) and exercises exactly the number of windows the served max shape
    forms.
    """
    clips: list[dict] = []
    offset = 0.0
    while offset < float(max_audio_seconds):
        end = min(offset + _CLIP_SECONDS, float(max_audio_seconds))
        clips.append({"start": offset, "end": end})
        offset = end
    return clips


def _skip_record(
    reason: str,
    *,
    workers: int,
    window_batch: int,
    beam_size: int,
    max_audio_seconds: float,
) -> dict:
    return {
        "measured": False,
        "reason": reason,
        "peak_activation_bytes": 0,
        "per_window_bytes": 0,
        "workers": workers,
        "window_batch": window_batch,
        "beam_size": beam_size,
        "max_audio_seconds": max_audio_seconds,
        "windows_probed": 0,
        "method": None,
    }


def probe_whisper_peak(
    model,
    *,
    workers: int = 1,
    window_batch: int = 1,
    beam_size: int = DEFAULT_WHISPER_BEAM_SIZE,
    max_audio_seconds: float = DEFAULT_WHISPER_MAX_AUDIO_SECONDS,
    device_index: int | None = None,
    sample_peak: bool = True,
) -> dict:
    """Measure a max-shape synthetic forward's device-memory delta.

    Returns a record shaped like::

        {
          "measured": bool,
          "reason": str | None,
          "peak_activation_bytes": int,
          "per_window_bytes": int,
          "workers": int,
          "window_batch": int,
          "beam_size": int,
          "max_audio_seconds": float,
          "windows_probed": int,
          "method": "mem_get_info_delta" | None,
        }

    ``sample_peak=False`` measures only the retained before/after delta; it keeps
    the unit test deterministic. Production uses the sampled peak so transient
    CTranslate2 activation memory is captured even when the allocator frees it
    before the run returns.
    """
    workers = max(1, int(workers or 1))
    window_batch = max(1, int(window_batch or 1))
    beam_size = max(1, int(beam_size or DEFAULT_WHISPER_BEAM_SIZE))
    try:
        max_audio_seconds = float(max_audio_seconds)
    except (TypeError, ValueError):
        max_audio_seconds = DEFAULT_WHISPER_MAX_AUDIO_SECONDS
    if max_audio_seconds <= 0:
        max_audio_seconds = DEFAULT_WHISPER_MAX_AUDIO_SECONDS

    shape = {
        "workers": workers,
        "window_batch": window_batch,
        "beam_size": beam_size,
        "max_audio_seconds": max_audio_seconds,
    }

    if model is None:
        return _skip_record("no-model", **shape)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency in practice
        return _skip_record(f"torch-import-failed: {type(exc).__name__}", **shape)

    try:
        if not torch.cuda.is_available():
            return _skip_record("cuda-unavailable", **shape)
    except Exception as exc:  # pragma: no cover - defensive on odd driver states
        return _skip_record(f"cuda-check-failed: {type(exc).__name__}", **shape)

    native = getattr(model, "model", None)
    model_device = str(getattr(native, "device", "") or "").lower()
    if not model_device.startswith("cuda"):
        return _skip_record(f"model-device:{model_device or 'unknown'}", **shape)

    try:
        import numpy as np
        from faster_whisper import BatchedInferencePipeline
    except Exception as exc:  # pragma: no cover - faster-whisper is a hard dependency in practice
        return _skip_record(f"faster-whisper-import-failed: {type(exc).__name__}", **shape)

    audio = _probe_audio(max_audio_seconds)
    clip_timestamps = _probe_clip_timestamps(max_audio_seconds)

    def _run_one(_index: int) -> None:
        # One pipeline per worker: it is a thin view over the shared model and
        # carries a per-instance timestamp cursor, so sharing one across threads
        # would race (the Phase 8-5 handler resets it per request for the same
        # reason).
        pipeline = BatchedInferencePipeline(model=model)
        pipeline.last_speech_timestamp = 0.0
        segments, _info = pipeline.transcribe(
            audio,
            batch_size=window_batch,
            beam_size=beam_size,
            best_of=beam_size,
            temperature=0.0,
            # Phase 11-0: explicit clips (not VAD) so a ≥30 s probe never raises
            # "No clip timestamps found" and the window count is exact.
            vad_filter=False,
            clip_timestamps=clip_timestamps,
            without_timestamps=True,
        )
        for _segment in segments:
            pass

    try:
        if device_index is None:
            # The model's own device wins over the process's active one: with
            # explicit secondary-device placement (``cuda:1``) the two differ.
            native_index = getattr(native, "device_index", None)
            if isinstance(native_index, (list, tuple)) and native_index:
                device_index = int(native_index[0])
            elif isinstance(native_index, int):
                device_index = native_index
            else:
                device_index = int(torch.cuda.current_device())
        free_before, total = torch.cuda.mem_get_info(device_index)
        free_before = int(free_before)
        total = int(total)
        baseline_used = max(0, total - free_before)

        stop = threading.Event()
        peak_used = baseline_used

        def _sample() -> None:
            nonlocal peak_used
            while not stop.wait(_SAMPLE_INTERVAL_S):
                try:
                    free_now, _total_now = torch.cuda.mem_get_info(device_index)
                except Exception:  # pragma: no cover - sampler must never crash the probe
                    return
                peak_used = max(peak_used, total - int(free_now))

        sampler = None
        if sample_peak:
            sampler = threading.Thread(target=_sample, name="whisper-vram-probe", daemon=True)
            sampler.start()
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="whisper-probe") as pool:
                list(pool.map(_run_one, range(workers)))
        finally:
            if sampler is not None:
                stop.set()
                sampler.join(timeout=2.0)

        with contextlib.suppress(Exception):
            torch.cuda.synchronize(device_index)
        free_after, _total_after = torch.cuda.mem_get_info(device_index)
        used_after = max(0, total - int(free_after))
    except Exception as exc:
        logger.warning(
            f"Whisper max-shape probe failed ({type(exc).__name__}: {exc}); "
            "VRAM attribution stays unmeasured."
        )
        return _skip_record(f"probe-failed: {type(exc).__name__}: {exc}", **shape)

    windows_per_worker = min(window_batch, len(clip_timestamps))
    windows_probed = workers * windows_per_worker
    if sample_peak:
        peak_bytes = max(0, max(peak_used, used_after) - baseline_used)
    else:
        peak_bytes = max(0, used_after - baseline_used)

    record = {
        "measured": True,
        "reason": None,
        "peak_activation_bytes": peak_bytes,
        "per_window_bytes": per_window_bytes(peak_bytes, windows_probed),
        "workers": workers,
        "window_batch": window_batch,
        "beam_size": beam_size,
        "max_audio_seconds": max_audio_seconds,
        "windows_probed": windows_probed,
        "method": "mem_get_info_delta",
    }
    logger.info(
        f"Whisper max-shape probe: {workers} worker(s) x {windows_per_worker} window(s) x "
        f"{max_audio_seconds:.0f}s -> peak activation {peak_bytes / 2**20:.1f} MiB "
        f"({record['per_window_bytes'] / 2**20:.1f} MiB/window)"
    )
    return record


__all__ = ["probe_whisper_peak"]
