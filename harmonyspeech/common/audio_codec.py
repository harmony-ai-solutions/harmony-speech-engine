"""Centralized audio codec helpers for the engine boundary.

This module owns the *wire contract* for audio: which container/codec formats
the engine accepts on the way in, how they are decoded, and which formats it can
produce on the way out.

Decoding order (see :func:`decode_audio`):

1. Magic-byte sniffing (:func:`sniff_format`) so error messages and logs can
   name what the caller actually sent.
2. ``soundfile`` / libsndfile — WAV, FLAC, OGG/Vorbis and (with libsndfile
   >= 1.1) MP3. No third-party dependencies beyond the parsed container.
3. An ``ffmpeg`` subprocess fallback for anything libsndfile cannot read
   (AAC/ADTS, M4A/MP4, WebM, AMR, …). Optional: only used when ``ffmpeg`` is on
   ``PATH`` (``FFMPEG_AVAILABLE``).
4. A typed :class:`AudioDecodeError` naming the probed format and listing what
   was tried. It subclasses :class:`ValueError`, so existing per-request error
   handling turns it into a 400-style response instead of a 500 crash.

Everything is mono ``float32`` at a known sample rate so downstream model code
never has to think about containers. ``soxr`` does the resampling when the
requested sample rate differs from the source rate (``librosa.resample`` with
``res_type="soxr_hq"`` is used only if the ``soxr`` package is unavailable).

On the output side :func:`encode_audio` produces WAV/FLAC/OGG/MP3/OGG-Opus/PCM
through libsndfile (or AAC through ffmpeg). The set of formats the runtime can
actually produce is probed once at import into
``SUPPORTED_OUTPUT_FORMATS`` so validation never advertises a format the build
cannot encode.
"""

from __future__ import annotations

import base64
import binascii
import io
import shutil
import subprocess

import numpy as np
import soundfile as sf
from loguru import logger

__all__ = [
    "AudioDecodeError",
    "FFMPEG_AVAILABLE",
    "SUPPORTED_INPUT_FORMATS",
    "SUPPORTED_OUTPUT_FORMATS",
    "SUPPORTED_SAMPLE_RATES",
    "decode_audio",
    "decode_wav_b64",
    "encode_audio",
    "ensure_librosa_readable_bytes",
    "refresh_output_formats",
    "resample",
    "sniff_format",
    "to_wav_bytes",
]

# ``shutil.which`` is enough for both POSIX and Windows (ffmpeg.exe).
FFMPEG_BIN: str | None = shutil.which("ffmpeg")
FFMPEG_AVAILABLE: bool = FFMPEG_BIN is not None

#: Subprocess guard so a malformed input can never wedge a worker thread.
_FFMPEG_TIMEOUT_S = 120

#: Sample rates we accept for ``output_options.sample_rate``. This is a policy
#: whitelist (common telephony/consumer rates) rather than "anything soxr can
#: do"; it keeps the response contract predictable and cheap to validate.
SUPPORTED_SAMPLE_RATES: frozenset[int] = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})

# libsndfile major-format name -> our public format token.
_WRITE_TARGETS: dict[str, tuple[str, str | None]] = {
    "wav": ("WAV", "PCM_16"),
    "flac": ("FLAC", "PCM_16"),
    "ogg": ("OGG", "VORBIS"),
    "mp3": ("MP3", "MPEG_LAYER_III"),
    "opus": ("OGG", "OPUS"),
}

# Formats libsndfile can generally read. MP3 is added only if the linked
# libsndfile build reports support (>= 1.1).
_LIBSNDFILE_INPUT_FORMATS = {"wav", "flac", "ogg", "aiff", "caf", "au"}

# Human-readable list used in error messages.
_INPUT_FORMAT_HINT = (
    "wav, flac, ogg, aiff (native); mp3 (libsndfile >= 1.1); "
    "aac, m4a, webm, amr and other containers via the optional ffmpeg fallback"
)


class AudioDecodeError(ValueError):
    """Input audio could not be decoded. Lists what was tried."""


def _probe_output_formats() -> set[str]:
    """Return the output formats this runtime can actually encode."""
    formats: set[str] = {"pcm"}  # always available (raw s16le)
    for token, (major, subtype) in _WRITE_TARGETS.items():
        try:
            if sf.check_format(major, subtype):
                formats.add(token)
        except Exception:  # pragma: no cover - defensive against odd libsndfile builds
            logger.debug(f"soundfile capability probe failed for {token}", exc_info=True)
    if FFMPEG_AVAILABLE:
        formats.add("aac")
    return formats


#: Capability-aware whitelist of output format tokens (probed once at import).
SUPPORTED_OUTPUT_FORMATS: set[str] = _probe_output_formats()


def refresh_output_formats() -> set[str]:
    """Re-probe the output format whitelist (used after tests change capabilities)."""
    global SUPPORTED_OUTPUT_FORMATS
    SUPPORTED_OUTPUT_FORMATS = _probe_output_formats()
    return SUPPORTED_OUTPUT_FORMATS


def _soundfile_input_formats() -> set[str]:
    formats = set(_LIBSNDFILE_INPUT_FORMATS)
    try:
        if "MP3" in sf.available_formats():
            formats.add("mp3")
    except Exception:  # pragma: no cover - defensive
        pass
    return formats


SUPPORTED_INPUT_FORMATS: frozenset[str] = frozenset(
    _soundfile_input_formats() | ({"aac", "m4a"} if FFMPEG_AVAILABLE else set())
)


def sniff_format(data: bytes) -> str | None:
    """Best-effort magic-byte probe of an audio payload.

    Returns a lowercase format token (``wav``, ``flac``, ``ogg``, ``mp3``,
    ``aac``, ``m4a``, ``aiff``, ``webm``, ``amr``, ``caf``) or ``None`` when the
    bytes do not look like a known audio container.
    """
    if not data:
        return None
    head = bytes(data[:12])

    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"RF64" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"caff":
        return "caf"
    if head[:4] == b"#!AM":
        return "amr"
    if head[:3] == b"ID3":
        return "mp3"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"

    # Raw MPEG/ADTS frame sync (0xFFF...). Layer bits 00 => ADTS (AAC);
    # layer bits non-zero => MPEG audio (MP3).
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        if (head[1] & 0x06) == 0x00:
            return "aac"
        return "mp3"
    return None


def resample(waveform: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample a mono float waveform. ``soxr`` HQ when available."""
    if sr_from == sr_to or waveform.size == 0:
        return np.asarray(waveform, dtype=np.float32)
    try:
        import soxr

        return np.asarray(soxr.resample(waveform, sr_from, sr_to, quality="HQ"), dtype=np.float32)
    except ImportError:  # pragma: no cover - soxr is a declared dependency
        import librosa

        return np.asarray(
            librosa.resample(
                np.asarray(waveform, dtype=np.float32),
                orig_sr=sr_from,
                target_sr=sr_to,
                res_type="soxr_hq",
            ),
            dtype=np.float32,
        )


def _read_with_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(data)) as handle:
        samplerate = int(handle.samplerate)
        frames = handle.read(dtype="float32", always_2d=True)
    if frames.shape[1] > 1:
        frames = frames.mean(axis=1)
    else:
        frames = frames[:, 0]
    return np.asarray(frames, dtype=np.float32), samplerate


def _read_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Decode through ffmpeg to a mono 16-bit WAV, then read that with soundfile."""
    assert FFMPEG_BIN is not None  # guarded by the caller
    result = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "-ac",
            "1",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT_S,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited with {result.returncode}: {stderr or 'no error output'}")
    if not result.stdout:
        raise RuntimeError("ffmpeg produced no audio")
    return _read_with_soundfile(result.stdout)


def decode_audio(data: bytes, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes to a float32 mono waveform.

    Order: soundfile (WAV/FLAC/OGG/MP3 with libsndfile >= 1.1) -> ffmpeg
    subprocess fallback (AAC, MP4/M4A, WebM, exotic containers) -> typed
    :class:`AudioDecodeError`.

    :param data: raw audio container bytes.
    :param target_sr: when given, resample the decoded waveform to this rate.
    :returns: ``(waveform, sample_rate)`` with ``sample_rate == target_sr`` when
        a resample happened.
    """
    if target_sr is not None and (not isinstance(target_sr, int) or target_sr <= 0):
        raise ValueError(f"target_sr must be a positive integer, got {target_sr!r}")
    if not data:
        raise AudioDecodeError(
            "Input audio is empty. Supported input formats: " + _INPUT_FORMAT_HINT + "."
        )

    probed = sniff_format(data)
    attempts: list[str] = []

    try:
        waveform, samplerate = _read_with_soundfile(data)
    except Exception as exc:
        attempts.append(f"libsndfile: {exc}")
    else:
        if waveform.size == 0:
            raise AudioDecodeError(
                f"Input audio (probed format: {probed or 'unknown'}) contains no samples. "
                "Supported input formats: " + _INPUT_FORMAT_HINT + "."
            )
        if target_sr is not None and target_sr != samplerate:
            waveform = resample(waveform, samplerate, target_sr)
            samplerate = target_sr
        return waveform, samplerate

    if FFMPEG_AVAILABLE:
        try:
            waveform, samplerate = _read_with_ffmpeg(data)
        except Exception as exc:
            attempts.append(f"ffmpeg: {exc}")
        else:
            if waveform.size == 0:
                raise AudioDecodeError(
                    f"Input audio (probed format: {probed or 'unknown'}) decoded to zero samples. "
                    "Supported input formats: " + _INPUT_FORMAT_HINT + "."
                )
            if target_sr is not None and target_sr != samplerate:
                waveform = resample(waveform, samplerate, target_sr)
                samplerate = target_sr
            return waveform, samplerate
    else:
        attempts.append("ffmpeg: not available on PATH")

    raise AudioDecodeError(
        f"Unable to decode input audio (probed format: {probed or 'unrecognised'}). "
        f"Tried {'; '.join(attempts)}. Supported input formats: {_INPUT_FORMAT_HINT}. "
        "Install ffmpeg on PATH to extend the accepted format set."
    )


def decode_wav_b64(b64: str) -> tuple[np.ndarray, int]:
    """Decode a base64 audio payload (executor output is always WAV) to float32 mono."""
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 audio payload: {exc}") from exc
    return decode_audio(raw)


def _to_s16le(waveform: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


def _encode_aac(waveform: np.ndarray, samplerate: int) -> bytes:
    if not FFMPEG_AVAILABLE:
        raise ValueError("AAC output requires ffmpeg on PATH, which is not available in this runtime.")
    assert FFMPEG_BIN is not None
    pcm = np.asarray(waveform, dtype="<f4").tobytes()
    result = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "f32le",
            "-ar",
            str(samplerate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-f",
            "adts",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT_S,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"ffmpeg failed to encode AAC: {stderr or result.returncode}")
    return result.stdout


def encode_audio(waveform: np.ndarray, sr: int, fmt: str, target_sr: int | None = None) -> bytes:
    """Encode a float32 mono waveform into the requested wire format.

    ``fmt`` must be one of ``SUPPORTED_OUTPUT_FORMATS``. ``pcm`` returns raw
    little-endian signed 16-bit samples (no container); ``aac`` shells out to
    ffmpeg. Raises :class:`ValueError` listing the supported formats otherwise.
    """
    token = (fmt or "wav").strip().lower()
    if token not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{fmt}'. Supported formats: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}."
        )

    samplerate = int(sr)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr is not None and int(target_sr) != samplerate:
        audio = resample(audio, samplerate, int(target_sr))
        samplerate = int(target_sr)

    if token == "pcm":
        return _to_s16le(audio)
    if token == "aac":
        return _encode_aac(audio, samplerate)

    major, subtype = _WRITE_TARGETS[token]
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate, format=major, subtype=subtype)
    return buffer.getvalue()


def to_wav_bytes(data: bytes, target_sr: int | None = None) -> bytes:
    """Decode arbitrary audio bytes and re-encode them as a PCM-16 WAV byte string.

    Used where a third-party component insists on a file-like/path input
    (Chatterbox conditionals, for example) but callers may upload any accepted
    format.
    """
    waveform, samplerate = decode_audio(data, target_sr=target_sr)
    if waveform.size == 0:
        raise AudioDecodeError("Input audio contains no samples.")
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def ensure_librosa_readable_bytes(data: bytes) -> bytes:
    """Return bytes that ``librosa.load``/libsndfile can read.

    Containers libsndfile handles are passed through untouched (zero cost);
    anything else (AAC, M4A, …) is decoded and re-wrapped as WAV.
    """
    probed = sniff_format(data)
    if probed in _soundfile_input_formats():
        return data
    return to_wav_bytes(data)
