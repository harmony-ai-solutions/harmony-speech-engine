"""Streaming wire contract: envelope selection, headers and byte-level primitives.

Phase 14-1 defined the *contract* (which envelope, which formats, which
headers); Phase 14-2 adds the pure, I/O-free primitives both envelopes share:

* :func:`wav_stream_header` — the 44-byte RIFF header with ``0xFFFFFFFF`` in
  both size fields (OpenAI/Kokoro-compatible streaming WAV);
* :func:`pcm16_frame` — one canonical container/float32 → s16le conversion
  built on :mod:`harmonyspeech.common.audio_codec` (never a second decoder);
* :class:`StreamResampler` — the stateful ``soxr.ResampleStream`` wrapper that
  keeps resampling continuous across chunk boundaries (per-chunk one-shot
  resampling clicks and drifts);
* the SSE event builders (:func:`sse_event`, :func:`delta_payload`,
  :func:`done_payload`, :func:`error_payload`, :func:`keepalive`) used by 14-4.

Everything here is synchronous and pure: no FastAPI, no engine calls, no
network. The serving layer calls the encoders from a worker thread (14-3).
Both envelopes and their tests share this one source of truth:

* :func:`resolve_stream_format` — "did this request ask to stream, and with
  which envelope?" ``stream_format`` selects streaming on its own (OpenAI's
  behaviour: the body streams once a format is chosen); the deprecated
  ``stream: true`` alias maps to ``"audio"``.
* :data:`STREAMING_AUDIO_FORMATS` — the only output formats the streaming
  envelopes can emit (whole-buffer encoders such as mp3/flac cannot be framed
  incrementally).
* the canonical response-header names and :func:`streaming_response_headers`,
  emitted by 14-3/14-4 and pinned by the 14-5 contract tests.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from harmonyspeech.common import audio_codec

__all__ = [
    "HEADER_ACCEL_BUFFERING",
    "HEADER_AUDIO_BITS",
    "HEADER_AUDIO_CHANNELS",
    "HEADER_AUDIO_FORMAT",
    "HEADER_AUDIO_SAMPLE_RATE",
    "HEADER_CACHE_CONTROL",
    "HEADER_REQUEST_ID",
    "ResampleStream",
    "SSE_DONE_FRAME",
    "SSE_EVENT_DELTA",
    "SSE_EVENT_DONE",
    "SSE_EVENT_ERROR",
    "SSE_KEEPALIVE",
    "SSE_MEDIA_TYPE",
    "STREAMING_AUDIO_BITS",
    "STREAMING_AUDIO_CHANNELS",
    "STREAMING_AUDIO_FORMATS",
    "STREAMING_RESPONSE_HEADERS",
    "STREAM_FORMATS",
    "StreamResampler",
    "WAV_STREAM_HEADER_BYTES",
    "delta_payload",
    "done_payload",
    "error_payload",
    "keepalive",
    "pcm16_frame",
    "resolve_stream_format",
    "sse_event",
    "streaming_response_headers",
    "wav_stream_header",
]

#: The two streaming envelopes accepted on the wire (``AudioOutputOptions``
#: validates this set with a ``Literal``).
STREAM_FORMATS: frozenset[str] = frozenset({"audio", "sse"})

#: Output containers the streaming path can emit. ``wav`` is a 44-byte
#: streaming RIFF header followed by s16le frames; ``pcm`` is headerless s16le.
#: Every other format is a whole-buffer encoder and is rejected at admission.
STREAMING_AUDIO_FORMATS: frozenset[str] = frozenset({"wav", "pcm"})

#: Streamed audio is always mono signed 16-bit little-endian PCM. These are
#: wire-contract values (the ``X-Audio-*`` headers), not tuning knobs.
STREAMING_AUDIO_CHANNELS = 1
STREAMING_AUDIO_BITS = 16

#: Canonical response-header names for both envelopes (Phase 14-1).
#: ``X-Audio-Sample-Rate`` is the **effective/target** rate: the requested
#: ``output_options.sample_rate`` when the 14-2 resampler is in play, otherwise
#: the model's native rate.
HEADER_AUDIO_FORMAT = "X-Audio-Format"
HEADER_AUDIO_SAMPLE_RATE = "X-Audio-Sample-Rate"
HEADER_AUDIO_CHANNELS = "X-Audio-Channels"
HEADER_AUDIO_BITS = "X-Audio-Bits"
HEADER_REQUEST_ID = "X-Request-Id"
HEADER_ACCEL_BUFFERING = "X-Accel-Buffering"
HEADER_CACHE_CONTROL = "Cache-Control"

#: The complete header set every streaming response carries, in the order the
#: contract tests assert. Streaming hygiene (``X-Accel-Buffering: no`` so nginx
#: does not buffer the body, ``Cache-Control`` so intermediaries never cache or
#: transform a stream) is part of the set, not an optional extra.
STREAMING_RESPONSE_HEADERS: tuple[str, ...] = (
    HEADER_AUDIO_FORMAT,
    HEADER_AUDIO_SAMPLE_RATE,
    HEADER_AUDIO_CHANNELS,
    HEADER_AUDIO_BITS,
    HEADER_REQUEST_ID,
    HEADER_ACCEL_BUFFERING,
    HEADER_CACHE_CONTROL,
)


def resolve_stream_format(output_options) -> str | None:
    """Return the requested streaming envelope, or ``None`` for buffered output.

    Accepts either the pydantic :class:`~harmonyspeech.endpoints.openai.protocol.AudioOutputOptions`
    or the plain :class:`~harmonyspeech.common.inputs.TextToSpeechAudioOutputOptions`
    (attribute access only, so this module stays dependency-free).

    ``stream_format`` selects streaming whenever it is set — OpenAI's contract:
    choosing a format is what makes the body stream. The deprecated
    ``stream: true`` alias means ``"audio"``; when both are present the explicit
    ``stream_format`` wins.
    """
    if output_options is None:
        return None

    stream_format = getattr(output_options, "stream_format", None)
    if stream_format is not None:
        return str(stream_format).strip().lower()

    if getattr(output_options, "stream", False):
        return "audio"
    return None


def streaming_response_headers(
    *,
    audio_format: str,
    sample_rate: int,
    request_id: str,
    channels: int = STREAMING_AUDIO_CHANNELS,
    bits: int = STREAMING_AUDIO_BITS,
) -> dict[str, str]:
    """Build the canonical streaming response headers for either envelope.

    :param audio_format: the emitted container, ``"wav"`` or ``"pcm"``.
    :param sample_rate: the effective/target rate in Hz (never the source rate
        when a ``sample_rate`` was requested).
    :param request_id: the request id, mirroring the buffered response ``id``.
    :param channels: streamed channel count (mono).
    :param bits: sample width in bits (s16le).
    """
    normalized = str(audio_format).strip().lower()
    if normalized not in STREAMING_AUDIO_FORMATS:
        raise ValueError(
            f"Streaming supports only the 'wav' and 'pcm' formats, got '{audio_format}'."
        )

    return {
        HEADER_AUDIO_FORMAT: normalized,
        HEADER_AUDIO_SAMPLE_RATE: str(int(sample_rate)),
        HEADER_AUDIO_CHANNELS: str(int(channels)),
        HEADER_AUDIO_BITS: str(int(bits)),
        HEADER_REQUEST_ID: str(request_id),
        HEADER_ACCEL_BUFFERING: "no",
        HEADER_CACHE_CONTROL: "no-cache, no-transform",
    }


# ---------------------------------------------------------------------------
# Binary envelope primitives (Phase 14-2)
# ---------------------------------------------------------------------------

#: Size of :func:`wav_stream_header` — the canonical PCM RIFF header length.
WAV_STREAM_HEADER_BYTES = 44

#: ``0xFFFFFFFF`` marks a WAV whose length is unknown up front (streaming).
#: OpenAI and Kokoro ship the same sentinel in both the RIFF and data sizes.
_UNKNOWN_SIZE = 0xFFFFFFFF

#: Headerless s16le carries no rate, but ``encode_audio`` takes one. It is only
#: used to satisfy the signature: the ``pcm`` branch never resamples.
_PCM_FRAME_DEFAULT_RATE = 24000


def _require_int(name: str, value, *, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return int(value)


def wav_stream_header(
    sample_rate: int,
    channels: int = STREAMING_AUDIO_CHANNELS,
    bits: int = STREAMING_AUDIO_BITS,
) -> bytes:
    """Return the 44-byte RIFF/WAVE header for a streamed PCM body.

    Both size fields (the RIFF chunk size at offset 4 and the ``data`` chunk
    size at offset 40) hold ``0xFFFFFFFF``: the total length is unknown while
    the body streams. That is what OpenAI and Kokoro emit, and every streaming
    consumer (browsers, ``soundfile``/libsndfile, ``ffprobe``) reads the
    ``data`` chunk until EOF.

    Layout, field by field::

        "RIFF" | 0xFFFFFFFF | "WAVE"
        "fmt " | 16 | 1 (PCM) | channels | sample_rate | byte_rate | block_align | bits
        "data" | 0xFFFFFFFF

    with ``byte_rate = sample_rate * channels * bits // 8`` and
    ``block_align = channels * bits // 8``.

    .. warning::
       Known wart: Python's stdlib :mod:`wave` trusts the declared sizes, so
       ``wave.open(...).getnframes()`` reports ``0xFFFFFFFF / block_align``
       frames (2147483647 at s16le) instead of the real duration. That is a
       reader limitation, not a malformed header — ``soundfile``, ``ffprobe``
       and browsers report the streamed duration correctly. Do not "fix" it by
       rewriting the size fields; a seekable length defeats streaming.

    The streaming envelopes only ever emit mono s16le (the ``X-Audio-*``
    contract), which is the default; the other parameters exist so tests can
    pin the arithmetic field by field.
    """
    rate = _require_int("sample_rate", sample_rate, minimum=1, maximum=0xFFFFFFFF)
    channel_count = _require_int("channels", channels, minimum=1, maximum=0xFFFF)
    width = _require_int("bits", bits, minimum=8, maximum=32)
    if width % 8:
        raise ValueError(f"bits must be a whole number of bytes (8/16/24/32), got {bits!r}")

    block_align = channel_count * width // 8
    byte_rate = rate * block_align
    if byte_rate > 0xFFFFFFFF:  # pragma: no cover - unreachable within the validated bounds
        raise ValueError(f"byte rate {byte_rate} does not fit the 32-bit RIFF field")

    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", _UNKNOWN_SIZE),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,  # fmt chunk size (PCM)
                1,  # audio format: PCM
                channel_count,
                rate,
                byte_rate,
                block_align,
                width,
            ),
            b"data",
            struct.pack("<I", _UNKNOWN_SIZE),
        )
    )


def _mono_waveform(audio) -> np.ndarray:
    """Normalize a frame source to a contiguous 1-D float32 waveform.

    Accepts the typed artifact (``AudioArtifact``, whose :attr:`numpy` is the
    serving layer's per-chunk representation), a raw numpy array, or a
    tensor-like object exposing ``.numpy()`` (e.g. a CPU ``torch.Tensor``).
    Container payloads are decoded by :func:`pcm16_frame` before this runs.
    """
    numpy = getattr(audio, "numpy", None)
    if numpy is not None:
        if callable(numpy):  # torch.Tensor.numpy is a method, AudioArtifact.numpy a property
            numpy = numpy()
        array = np.asarray(numpy, dtype=np.float32)
    else:
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 2:
        raise ValueError(f"Expected a mono waveform, got shape {array.shape}")
    if array.ndim == 2:
        # (frames, channels) -> mono, matching audio_codec.encode_audio.
        array = array.reshape(-1) if 1 in array.shape else array.mean(axis=1)
    return np.ascontiguousarray(array, dtype=np.float32)


def pcm16_frame(audio, sample_rate: int | None = None) -> bytes:
    """Convert one frame source to headerless s16le PCM bytes.

    Verified against the serving layer (2026-11-10): the chunked path holds a
    typed :class:`~harmonyspeech.common.artifacts.AudioArtifact` per chunk
    (float32 mono ``[samples]``), so this is the single canonical conversion —
    :func:`harmonyspeech.common.audio_codec.encode_audio` with ``fmt="pcm"``,
    the same helper the buffered path uses. Nothing is re-implemented here.

    For the legacy pre-artifact payload (base64 WAV text) or container bytes
    (WAV/FLAC/OGG/...), the input is decoded through the codec layer's
    :func:`~harmonyspeech.common.audio_codec.decode_wav_b64` /
    :func:`~harmonyspeech.common.audio_codec.decode_audio` — the one decoder,
    including its optional ffmpeg fallback for exotic containers. Call it from
    a worker thread when the input may be a container (14-3 does).

    :param audio: an ``AudioArtifact``-like object (anything with ``.numpy``),
        a numpy/tensor-like waveform, container ``bytes``, or a base64 ``str``.
    :param sample_rate: rate of a raw waveform when it cannot be inferred. It
        does not change the output: headerless PCM is rate-less.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        waveform, rate = audio_codec.decode_audio(bytes(audio))
        rate = int(rate)
    elif isinstance(audio, str):
        waveform, rate = audio_codec.decode_wav_b64(audio)
        rate = int(rate)
    else:
        waveform = _mono_waveform(audio)
        rate = sample_rate
        if rate is None:
            rate = getattr(audio, "sample_rate", None)
        rate = int(rate) if rate is not None else _PCM_FRAME_DEFAULT_RATE
    return audio_codec.encode_audio(waveform, rate, "pcm")


def _empty_samples(channels: int) -> np.ndarray:
    if channels == 1:
        return np.empty(0, dtype=np.float32)
    return np.empty((0, channels), dtype=np.float32)


class StreamResampler:
    """Stateful sample-rate converter that stays continuous across chunks.

    Thin wrapper over :class:`soxr.ResampleStream` (verified: python-soxr
    1.0.0, ``ResampleStream(in_rate, out_rate, num_channels, dtype="float32",
    quality="HQ")`` / ``resample_chunk(x, last=False)``). Resampling each chunk
    with the one-shot :func:`~harmonyspeech.common.audio_codec.resample` would
    click at every boundary and drift in length; this keeps one filter state
    for the whole stream, so ``feed()`` output concatenated with ``flush()``
    is sample-identical to one-shot resampling of the concatenation.

    The stream is float32 end to end — the serving layer resamples between the
    watermark and the PCM encoder (14-3), and :func:`pcm16_frame` is the single
    place that quantizes. Feed s16le bytes here only via
    ``np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0``; doing it here
    would quantize twice.

    Invariants:

    * ``concatenate(feed(c) for c in chunks) + flush()`` equals
      ``soxr.resample(concatenate(chunks), in_rate, out_rate, quality="HQ")``
      for realistic chunk lengths (the continuity gate; soxr's edge handling
      for 1–2 sample *total* inputs is its own business, not a chunk seam);
    * :attr:`emitted_samples` is the number of output frames returned so far
      (sum of ``feed()``/``flush()`` lengths) — the counter 14-4 divides by the
      rate for ``duration_s``/``usage.audio_seconds``. Never recompute it from
      wire bytes: resampled length ≠ source length;
    * ``in_rate == out_rate`` is a pass-through (no filter, exact samples);
    * ``feed()`` after ``flush()`` raises: soxr's stream is finished, and a new
      request needs a new :class:`StreamResampler`.

    Mono input may be ``(frames,)`` or ``(frames, 1)``; output is ``(frames,)``.
    For ``channels > 1`` input/output are ``(frames, channels)``.
    """

    def __init__(
        self,
        in_rate: int | float,
        out_rate: int | float,
        channels: int = STREAMING_AUDIO_CHANNELS,
        *,
        quality: str = "HQ",
    ):
        self.in_rate = float(self._positive_rate("in_rate", in_rate))
        self.out_rate = float(self._positive_rate("out_rate", out_rate))
        self.channels = _require_int("channels", channels, minimum=1)
        self.quality = str(quality)

        self._closed = False
        self._input_samples = 0
        self._emitted_samples = 0
        self._passthrough = self.in_rate == self.out_rate
        self._stream = None
        if not self._passthrough:
            import soxr  # local import: the contract helpers stay import-light

            self._stream = soxr.ResampleStream(
                self.in_rate, self.out_rate, self.channels, dtype="float32", quality=self.quality
            )

    @staticmethod
    def _positive_rate(name: str, value) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")
        return float(value)

    def _as_chunk(self, chunk) -> np.ndarray:
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim == 0:
            array = array.reshape(1)
        if self.channels == 1:
            if array.ndim == 2:
                if array.shape[1] != 1:
                    raise ValueError(
                        f"StreamResampler is mono but the chunk has {array.shape[1]} channels"
                    )
                array = array.reshape(-1)
            elif array.ndim != 1:
                raise ValueError(f"Expected a 1-D mono chunk, got shape {array.shape}")
        elif array.ndim != 2 or array.shape[1] != self.channels:
            raise ValueError(
                f"Expected a (frames, {self.channels}) chunk, got shape {array.shape}"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def feed(self, chunk) -> np.ndarray:
        """Resample one chunk and return the output frames that are final now."""
        if self._closed:
            raise RuntimeError("feed() called after flush(); create a new StreamResampler per stream")
        array = self._as_chunk(chunk)
        if array.shape[0] == 0:
            return _empty_samples(self.channels)

        self._input_samples += int(array.shape[0])
        if self._passthrough:
            output = array
        else:
            assert self._stream is not None
            output = np.asarray(self._stream.resample_chunk(array), dtype=np.float32)
        self._emitted_samples += int(output.shape[0])
        return output

    def flush(self) -> np.ndarray:
        """Return the filter's remaining output once; later calls return empty."""
        if self._closed:
            return _empty_samples(self.channels)
        self._closed = True
        if self._stream is None:
            return _empty_samples(self.channels)

        empty = np.empty((0, self.channels), dtype=np.float32) if self.channels > 1 else np.empty(
            0, dtype=np.float32
        )
        tail = np.asarray(self._stream.resample_chunk(empty, last=True), dtype=np.float32)
        self._emitted_samples += int(tail.shape[0])
        return tail

    @property
    def emitted_samples(self) -> int:
        """Output frames returned so far (``feed()`` + ``flush()`` lengths)."""
        return self._emitted_samples

    @property
    def input_samples(self) -> int:
        """Input frames accepted so far."""
        return self._input_samples

    @property
    def closed(self) -> bool:
        """True once :meth:`flush` ran (or the pass-through stream ended)."""
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"StreamResampler({self.in_rate:g}->{self.out_rate:g} Hz, "
            f"channels={self.channels}, emitted={self._emitted_samples})"
        )


#: 14-2 task-file name for :class:`StreamResampler` (the wrapper over
#: ``soxr.ResampleStream``), kept discoverable for the serving-layer tasks.
ResampleStream = StreamResampler


# ---------------------------------------------------------------------------
# SSE envelope primitives (Phase 14-2)
# ---------------------------------------------------------------------------

#: OpenAI's speech-streaming event names (vLLM-Omni's precedent; the delta
#: payload field is ``audio``, never ``delta``).
SSE_EVENT_DELTA = "speech.audio.delta"
SSE_EVENT_DONE = "speech.audio.done"
SSE_EVENT_ERROR = "speech.audio.error"

#: A comment line keeps proxies/browsers from timing out during first-chunk
#: synthesis (SSE ignores lines starting with ``:``).
SSE_KEEPALIVE = ": keepalive\n\n"

#: Media type of the SSE envelope. Starlette appends ``; charset=utf-8`` to a
#: ``text/*`` media type that lacks one; the charset is stated here so the
#: exact wire value is a contract constant rather than a framework default.
SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"

#: ``data: [DONE]`` is the chat-completions-familiar final frame, emitted by
#: 14-4 *after* ``speech.audio.done``. Clients must not require it.
SSE_DONE_FRAME = "data: [DONE]\n\n"


def _json_data(payload) -> str:
    """Serialize a payload as one strict JSON line (newlines escaped)."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sse_event(event_type: str, payload) -> str:
    """Render one SSE frame: ``event:`` line + one ``data:`` JSON line + blank.

    ``json.dumps`` escapes newlines/quotes inside string values, so the frame
    stays exactly two lines for any payload and a strict line parser (split on
    ``\\n``, terminate on the blank line) always recovers the JSON. The event
    type is validated: a CR/LF there would inject a second frame.
    """
    event = str(event_type)
    if not event:
        raise ValueError("SSE event type must not be empty")
    if "\n" in event or "\r" in event:
        raise ValueError(f"SSE event type must be a single line, got {event_type!r}")
    return f"event: {event}\ndata: {_json_data(payload)}\n\n"


def delta_payload(audio_b64: str, chunk_index: int) -> dict:
    """Build a ``speech.audio.delta`` payload.

    The audio field is ``audio`` (base64 s16le PCM), matching OpenAI's
    contract — vLLM-Omni initially shipped ``delta`` and corrected it.
    ``chunk_index`` is additive (KittenML precedent) so clients can detect a
    dropped frame.
    """
    index = _require_int("chunk_index", chunk_index, minimum=0)
    return {
        "type": SSE_EVENT_DELTA,
        "audio": str(audio_b64),
        "chunk_index": index,
    }


def done_payload(
    id: str,
    model: str,
    created: int,
    sample_rate: int,
    format: str,
    chunks: int,
    duration_s: float,
    usage=None,
) -> dict:
    """Build the terminal ``speech.audio.done`` payload.

    ``duration_s`` must come from the frame generator's own emitted-sample
    counter (sum of resampler ``feed()``/``flush()`` lengths ÷ rate), never
    from wire bytes: the resampled length differs from the source length.

    ``usage`` is copied (never mutated) and extended additively with
    ``audio_seconds`` (from ``duration_s``) and ``input_characters`` (taken
    from ``usage`` when the caller supplied it; ``0`` otherwise). The additive
    fields follow the KittenML/AssemblyAI precedent and never replace an
    existing OpenAI usage object.
    """
    seconds = round(float(duration_s), 6)
    merged = dict(usage or {})
    merged["audio_seconds"] = seconds
    merged.setdefault("input_characters", 0)
    return {
        "type": SSE_EVENT_DONE,
        "id": str(id),
        "model": str(model),
        "created": int(created),
        "sample_rate": int(sample_rate),
        "format": str(format),
        "chunks": int(chunks),
        "duration_s": seconds,
        "usage": merged,
    }


def error_payload(err) -> dict:
    """Build a ``speech.audio.error`` payload shaped ``{error: {message, type, code}}``.

    ``err`` may be an exception (type = class name, ``code`` from a ``.code``
    attribute when present), a plain message string (type ``server_error``), or
    an already-shaped mapping/object with ``message``/``type``/``code`` keys.
    """
    message: str
    error_type = "server_error"
    code = None

    if isinstance(err, BaseException):
        message = str(err) or err.__class__.__name__
        error_type = err.__class__.__name__
        code = getattr(err, "code", None)
    elif isinstance(err, dict):
        inner = err.get("error") if isinstance(err.get("error"), dict) else err
        message = str(inner.get("message", "") or "")
        error_type = str(inner.get("type", error_type) or error_type)
        code = inner.get("code", code)
    else:
        inner_message = getattr(err, "message", None)
        message = str(inner_message if inner_message is not None else err)
        error_type = str(getattr(err, "type", error_type) or error_type)
        code = getattr(err, "code", code)

    return {"error": {"message": message, "type": error_type, "code": code}}


def keepalive() -> str:
    """Return the SSE keepalive comment frame (``: keepalive``)."""
    return SSE_KEEPALIVE
