"""Small decodable audio payloads for unit tests.

Phase 1 routes every boundary audio field through the centralized decoder, so
tests that previously used arbitrary placeholder bytes (``b"fake_wav_bytes"``)
need real container bytes to keep exercising the behaviour they target.
"""

from __future__ import annotations

import base64
import functools
import io

import numpy as np
import soundfile as sf


@functools.lru_cache(maxsize=4)
def tiny_wav_bytes(sample_rate: int = 16000, seconds: float = 0.25) -> bytes:
    """Return a tiny mono PCM-16 WAV (default 250 ms at 16 kHz)."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@functools.lru_cache(maxsize=4)
def tiny_wav_b64(sample_rate: int = 16000, seconds: float = 0.25) -> str:
    return base64.b64encode(tiny_wav_bytes(sample_rate, seconds)).decode("ascii")
