"""Boundary conversion of executor audio output into the requested wire format.

Executors always emit base64-encoded WAV at the model's native sample rate
(that is the internal contract). This module is the only place that translates
that internal payload into what the caller asked for via ``output_options``:

* no options, or ``format="wav"`` without a ``sample_rate`` -> the payload is
  returned **byte-identical** and no decoding happens (zero cost, backward
  compatible);
* otherwise the WAV is decoded once, optionally resampled with soxr, and
  re-encoded (or turned into raw s16le for ``pcm``).

Validation lives here too so every audio-returning endpoint rejects
unsupported ``format``/``sample_rate`` combinations at admission time (400)
instead of silently ignoring them or failing after synthesis.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from harmonyspeech.common import audio_codec
from harmonyspeech.endpoints.openai.protocol import AudioOutputOptions

__all__ = [
    "EncodedAudio",
    "apply_output_options",
    "normalize_format",
    "validate_output_options",
]


@dataclass(frozen=True)
class EncodedAudio:
    """Result of applying output options to an executor payload."""

    data: str
    format: str
    #: Effective sample rate in Hz when the engine re-encoded/resampled the
    #: audio; ``None`` for a plain passthrough (rate is whatever the model
    #: produced and the payload is unchanged).
    sample_rate: int | None


def normalize_format(fmt: str | None) -> str:
    return (fmt or "wav").strip().lower()


def validate_output_options(opts: AudioOutputOptions | None) -> str | None:
    """Return a human-readable validation error, or ``None`` when valid."""
    if opts is None:
        return None

    fmt = normalize_format(opts.format)
    if fmt not in audio_codec.SUPPORTED_OUTPUT_FORMATS:
        return (
            f"Unsupported output format '{opts.format}'. "
            f"Supported formats in this runtime: {', '.join(sorted(audio_codec.SUPPORTED_OUTPUT_FORMATS))}."
        )

    sample_rate = opts.sample_rate
    if sample_rate is not None:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            return f"Invalid output sample_rate '{sample_rate}'. It must be a positive integer in Hz."
        if sample_rate not in audio_codec.SUPPORTED_SAMPLE_RATES:
            supported = ", ".join(str(rate) for rate in sorted(audio_codec.SUPPORTED_SAMPLE_RATES))
            return (
                f"Unsupported output sample_rate '{sample_rate}'. "
                f"Supported sample rates: {supported}."
            )

    return None


def apply_output_options(b64_wav: str, opts: AudioOutputOptions | None) -> EncodedAudio:
    """Convert an executor WAV payload according to ``output_options``.

    Raises :class:`ValueError` when the options are invalid; serving layers
    validate at admission, so this is a defence-in-depth check only.
    """
    fmt = normalize_format(opts.format) if opts is not None else "wav"
    target_sr = opts.sample_rate if opts is not None else None

    # Default / plain-WAV-without-resample path: byte-identical passthrough.
    if opts is None or (fmt == "wav" and target_sr is None):
        return EncodedAudio(data=b64_wav, format="wav", sample_rate=None)

    error = validate_output_options(opts)
    if error is not None:
        raise ValueError(error)

    waveform, source_sr = audio_codec.decode_wav_b64(b64_wav)
    effective_sr = int(target_sr) if target_sr is not None else source_sr
    encoded = audio_codec.encode_audio(waveform, source_sr, fmt, target_sr=target_sr)
    return EncodedAudio(
        data=base64.b64encode(encoded).decode("ascii"),
        format=fmt,
        sample_rate=effective_sr,
    )
