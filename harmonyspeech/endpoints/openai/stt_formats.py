"""Speech-to-text response formats (Phase 10-4).

The transcription and translation endpoints return one of four representations
of the same executor payload (``result_data`` = ``{"text", "segments", "info"}``):

* ``json`` — the existing :class:`~harmonyspeech.endpoints.openai.protocol.SpeechToTextResponse`
  object (byte-identical to the pre-10-4 wire shape);
* ``text`` — the plain transcript;
* ``srt`` — SubRip subtitles rendered from ``result_data["segments"]``;
* ``vtt`` — WebVTT subtitles rendered from the same segments.

Everything here is pure: no model, no I/O, no engine state. Validation follows
the Phase 1-2 ``output_options`` pattern (the serving layer asks for a message
and turns it into a 400 at admission), so an unknown format never queues an
inference.

Timestamp rendering
-------------------
Both formats use ``HH:MM:SS`` with a millisecond fraction (``,`` for SRT, ``.``
for WebVTT), so a one-hour-plus segment renders as ``01:02:03,004``. Fractions
are rounded half-up to the nearest millisecond through :class:`~decimal.Decimal`
(``str(seconds)`` avoids float multiplication artefacts); negative or
non-finite values clamp to zero.

Cue text
--------
Cue text keeps internal newlines (both formats allow multi-line cues) but is
normalised to ``\\n`` line endings, stripped of leading/trailing blank lines and
has the cue-timing arrow ``-->`` replaced with ``- ->``. Neither format may
carry ``-->`` inside cue text: a parser would otherwise read it as a timing
line (WebVTT forbids it outright).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

__all__ = [
    "RESPONSE_FORMATS",
    "normalize_response_format",
    "segments_to_srt",
    "segments_to_vtt",
    "validate_response_format",
]

#: Supported ``response_format`` values, in documentation order.
RESPONSE_FORMATS: tuple[str, ...] = ("json", "text", "srt", "vtt")

_RESPONSE_FORMAT_SET = frozenset(RESPONSE_FORMATS)

#: WebVTT files start with this header followed by a blank line.
_VTT_HEADER = "WEBVTT\n"


def normalize_response_format(value: Any) -> str:
    """Return the canonical (lower-case) response format; absent means ``json``."""
    if value is None:
        return "json"
    return str(value).strip().lower() or "json"


def validate_response_format(value: Any) -> str | None:
    """Return a human-readable validation error, or ``None`` when valid."""
    fmt = normalize_response_format(value)
    if fmt not in _RESPONSE_FORMAT_SET:
        return f"Unsupported response_format '{value}'. Supported formats: {', '.join(RESPONSE_FORMATS)}."
    return None


def _milliseconds(seconds: Any) -> int:
    """Render a timestamp in whole milliseconds (half-up, non-negative)."""
    try:
        value = Decimal(str(seconds))
    except (InvalidOperation, TypeError, ValueError):
        return 0
    if not value.is_finite() or value < 0:
        return 0
    return int((value * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _timestamp(seconds: Any, separator: str) -> str:
    total_ms = _milliseconds(seconds)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{separator}{milliseconds:03d}"


def _cue_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Neither format may contain the cue-timing arrow inside cue text.
    text = text.replace("-->", "- ->")
    return text.strip("\n")


def _segments(segments: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [segment for segment in (segments or []) if isinstance(segment, Mapping)]


def segments_to_srt(segments: Iterable[Mapping[str, Any]] | None) -> str:
    """Render segments as a SubRip (``.srt``) document.

    Empty input renders an empty document (``""``), not an error.
    """
    blocks = []
    for index, segment in enumerate(_segments(segments), start=1):
        start = _timestamp(segment.get("start", 0.0), ",")
        end = _timestamp(segment.get("end", 0.0), ",")
        text = _cue_text(segment.get("text", ""))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def segments_to_vtt(segments: Iterable[Mapping[str, Any]] | None) -> str:
    """Render segments as a WebVTT (``.vtt``) document.

    Empty input renders a valid empty WebVTT document (the header only), not an
    error.
    """
    cues = []
    for segment in _segments(segments):
        start = _timestamp(segment.get("start", 0.0), ".")
        end = _timestamp(segment.get("end", 0.0), ".")
        text = _cue_text(segment.get("text", ""))
        cues.append(f"{start} --> {end}\n{text}")
    if not cues:
        return _VTT_HEADER
    return _VTT_HEADER + "\n" + "\n\n".join(cues) + "\n"
