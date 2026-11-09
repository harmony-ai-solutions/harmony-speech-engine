"""Model-agnostic text/audio helpers for long-form synthesis (Phase 12-1/12-3).

Everything here is pure: the module imports numpy at load time and pulls in
librosa lazily inside :func:`time_stretch` only. It must stay importable without
an engine, FastAPI or any model checkpoint so the serving layer (12-2) and unit
tests can use it on a bare CPU interpreter.

Waveform helpers cover the sentence splitter, the equal-power crossfade
stitcher (with stitcher-internal edge conditioning, Phase 12-5), peak
normalization, the phase-vocoder time-stretch, and the opt-in Phase B output
filters (DC offset removal, lead/trail silence trim and internal silence
collapse; Phase 12-3).

Edge conditioning and the ``trim_silence`` filter are deliberately separate:
conditioning is internal to the stitcher and only normalises chunk joins (it
also bounds the first/last chunk's outer edges to a guard pad), while
``trim_silence`` is an explicit whole-output option applied after stitching.

Text splitting is a port of the reference implementation in
``Chatterbox-TTS-Server/utils.py`` (MIT), extended with the two upstream
regressions that shipped after issue #144:

* bullets must be at column 0 *and* be followed by a space/tab, and a list is
  only a list with at least two items — stray narrative dashes never become
  bullets;
* em/en dashes and the Unicode ellipsis ``…`` are sentence-internal; only
  ``.!?`` (plus CJK ``。！？``) end a sentence.

Parenthetical prose (``(even the boldest ones)``) and inline paralinguistic
tags (``[laugh]``, ``[sigh]``, …) are ordinary sentence text: they are never
split out into standalone segments. When sentences are merged, no ASCII space
is inserted after CJK text (``MTLTokenizer`` would turn it into a ``[SPACE]``
token), so CJK chunks stay space-free.

Ours beyond upstream: CJK sentence enders for the multilingual voices, and a
character-budget merge with an explicit oversize-sentence fallback (clause
split, then hard wrap).

Waveform convention: ``(samples,)`` is mono and ``(samples, channels)`` is
multichannel. The repo's :class:`~harmonyspeech.common.artifacts.AudioArtifact`
exposes 1-D ``.numpy``; callers with ``[1, samples]`` tensors squeeze first.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "CrossfadeStitcher",
    "DEFAULT_CHUNK_CROSSFADE_MS",
    "DEFAULT_CHUNK_SENTENCE_PAUSE_MS",
    "DEFAULT_EDGE_GUARD_MS",
    "DEFAULT_EDGE_MAX_TRIM_MS",
    "DEFAULT_EDGE_SILENCE_THRESHOLD_DB",
    "DEFAULT_FILTER_FRAME_MS",
    "DEFAULT_FILTER_HOP_MS",
    "DEFAULT_INTERNAL_SILENCE_MAX_MS",
    "DEFAULT_INTERNAL_SILENCE_MIN_MS",
    "DEFAULT_PEAK_CEILING",
    "DEFAULT_SILENCE_THRESHOLD_DB",
    "DEFAULT_TRIM_PADDING_MS",
    "condition_edges",
    "crossfade_stitch",
    "fix_internal_silence",
    "peak_normalize",
    "remove_dc_offset",
    "split_text_into_chunks",
    "time_stretch",
    "trim_silence",
]

#: Default peak ceiling for :func:`peak_normalize` (leaves 1 % headroom).
DEFAULT_PEAK_CEILING = 0.99

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

#: Abbreviations whose trailing period is not a sentence boundary (upstream set).
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "rev.",
        "hon.",
        "st.",
        "etc.",
        "e.g.",
        "i.e.",
        "vs.",
        "approx.",
        "apt.",
        "dept.",
        "fig.",
        "gen.",
        "gov.",
        "inc.",
        "jr.",
        "sr.",
        "ltd.",
        "no.",
        "p.",
        "pp.",
        "vol.",
        "op.",
        "cit.",
        "ca.",
        "cf.",
        "ed.",
        "esp.",
        "et.",
        "al.",
        "ibid.",
        "id.",
        "inf.",
        "sup.",
        "viz.",
        "sc.",
        "fl.",
        "d.",
        "b.",
        "r.",
        "c.",
        "v.",
        "u.s.",
        "u.k.",
        "a.m.",
        "p.m.",
        "a.d.",
        "b.c.",
    }
)

#: Numbers such as ``3.14`` / ``.5``; a period inside one is not a boundary.
_NUMBER_DOT_NUMBER = re.compile(r"(?<!\d\.)\d*\.\d+")
#: Versions such as ``v1.2.3`` / ``2.3.4``; periods inside are not boundaries.
_VERSION = re.compile(r"[vV]?\d+(?:\.\d+)+")
#: Latin sentence enders: punctuation, optional closing quote, then whitespace/EOS.
_POTENTIAL_END = re.compile(r"([.!?])([\"']?)(\s+|$)")
#: CJK sentence enders and trailing closing brackets/quotes (no whitespace needed).
_CJK_END = re.compile(r"([。！？]+)([」』）】》〉”’\"']*)")
#: A piece ending in CJK text (ideographs, kana or fullwidth punctuation) is
#: joined without an ASCII space; ``MTLTokenizer`` would tokenise that space.
_CJK_JOIN_NO_SPACE = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u4e00-\u9fff\uff00-\uffef]")
#: A bullet list item: column 0, bullet char/number, then a space or tab.
#: Both constraints are the upstream issue #144 fixes: indented dashes and
#: dashes followed by a newline are narrative text, not list syntax.
_BULLET = re.compile(r"(?:^|\n)([-•*]|\d+\.)[ \t]+")
#: Clause punctuation used only by the oversize-sentence fallback.
_CLAUSE_BREAK = re.compile(r"[,;，；]")

#: Language tags whose scripts use CJK sentence punctuation.
_CJK_LANGUAGE_ROOTS = frozenset({"zh", "ja", "ko", "yue", "wuu"})


def _is_cjk_language(language: str | None) -> bool:
    """Return whether ``language`` names a CJK-script language (base tag only)."""
    if not language:
        return False
    base = language.replace("_", "-").split("-", 1)[0].lower()
    return base in _CJK_LANGUAGE_ROOTS


def _is_valid_sentence_end(text: str, period_index: int) -> bool:
    """Return whether the period at ``period_index`` really ends a sentence.

    Mirrors the upstream heuristic: the word ending at the period is checked
    against :data:`ABBREVIATIONS`, then the surrounding window is checked for a
    numeric/version match that contains the period.
    """
    word_start = period_index - 1
    scan_limit = max(0, period_index - 10)
    while word_start >= scan_limit and not text[word_start].isspace():
        word_start -= 1
    if text[word_start + 1 : period_index + 1].lower() in ABBREVIATIONS:
        return False

    context_start = max(0, period_index - 10)
    context_end = min(len(text), period_index + 10)
    context = text[context_start:context_end]
    relative_period = period_index - context_start

    for pattern in (_NUMBER_DOT_NUMBER, _VERSION):
        for match in pattern.finditer(context):
            if not (match.start() <= relative_period < match.end()):
                continue
            is_last_char = relative_period == match.end() - 1
            followed_by_space = period_index + 1 == len(text) or text[period_index + 1].isspace()
            if not (is_last_char and followed_by_space):
                return False
    return True


def _sentence_boundaries(text: str, cjk_enabled: bool) -> list[tuple[int, int]]:
    """Return sorted ``(sentence_end, next_start)`` offsets for ``text``."""
    boundaries: list[tuple[int, int]] = []

    for match in _POTENTIAL_END.finditer(text):
        period_index = match.start(1)
        if text[period_index] == ".":
            if (period_index > 0 and text[period_index - 1] == ".") or (
                period_index + 1 < len(text) and text[period_index + 1] == "."
            ):
                # Part of an ASCII ellipsis / dot run, never a boundary.
                continue
            if not _is_valid_sentence_end(text, period_index):
                continue
        end = match.start(1) + 1 + len(match.group(2) or "")
        boundaries.append((end, match.end()))

    if cjk_enabled:
        for match in _CJK_END.finditer(text):
            end = match.end(1) + len(match.group(2) or "")
            boundaries.append((end, match.end()))

    boundaries.sort(key=lambda item: item[0])
    return boundaries


def _split_text_by_punctuation(text: str, cjk_enabled: bool) -> list[str]:
    """Split ``text`` into sentences on Latin and (optionally) CJK enders."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    last = 0
    for end, next_start in _sentence_boundaries(text, cjk_enabled):
        if end <= last:
            # Boundary already consumed by an earlier (longer) slice.
            continue
        sentence = text[last:end].strip()
        if sentence:
            sentences.append(sentence)
        last = max(last, next_start)

    tail = text[last:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def _segment_plain(text: str, cjk_enabled: bool) -> list[str]:
    """Split plain text into sentences.

    Parenthetical prose and inline square-bracket tags are ordinary sentence
    text (the Dia-TTS carry-over cue regex was removed in Phase 12-6), so the
    whole string is handed to the punctuation splitter unchanged.
    """
    return _split_text_by_punctuation(text.strip(), cjk_enabled)


def _join_pieces(left: str, right: str) -> str:
    """Join two pieces, omitting the space when ``left`` ends in CJK text."""
    if not left:
        return right
    if _CJK_JOIN_NO_SPACE.match(left[-1]):
        return f"{left}{right}"
    return f"{left} {right}"


def _join_len(left: str, right: str) -> int:
    """Length of :func:`_join_pieces` without building the joined string."""
    if not left:
        return len(right)
    if _CJK_JOIN_NO_SPACE.match(left[-1]):
        return len(left) + len(right)
    return len(left) + 1 + len(right)


def _bullet_segments(text: str, cjk_enabled: bool) -> list[str]:
    """Split a detected bullet list, keeping each marker on its first sentence."""
    matches = list(_BULLET.finditer(text))
    segments: list[str] = []
    current = 0

    for index, match in enumerate(matches):
        if index == 0 and match.start() > current:
            segments.extend(_segment_plain(text[current : match.start()], cjk_enabled))

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        marker = match.group(1)
        body_start = match.end() - match.start()
        body = text[match.start() : next_start][body_start:].strip()
        body_segments = _segment_plain(body, cjk_enabled) if body else []
        if body_segments:
            # Keep the marker as part of the text: TTS prosody treats "- x" and
            # "1. x" as list intonation; stripping it is a normalisation call
            # the serving layer has not asked for.
            body_segments[0] = f"{marker} {body_segments[0]}"
            segments.extend(body_segments)
        else:
            segments.append(marker)
        current = next_start

    if current < len(text):
        segments.extend(_segment_plain(text[current:], cjk_enabled))
    return segments


def _hard_wrap(text: str, budget: int) -> list[str]:
    """Wrap ``text`` into pieces of at most ``budget`` characters.

    Prefers the last space inside the window so words stay intact; CJK text
    (which has no spaces) is cut exactly at the budget. This is the last-resort
    fallback for a single sentence with no usable clause punctuation.
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > budget:
        window = remaining[:budget]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = budget
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_oversize(segment: str, budget: int) -> list[str]:
    """Split one over-budget sentence on clauses, then hard-wrap what remains."""
    clause_pieces: list[str] = []
    start = 0
    for match in _CLAUSE_BREAK.finditer(segment):
        piece = segment[start : match.end()].strip()
        if piece:
            clause_pieces.append(piece)
        start = match.end()
    tail = segment[start:].strip()
    if tail:
        clause_pieces.append(tail)

    pieces: list[str] = []
    for clause in clause_pieces or [segment]:
        if len(clause) <= budget:
            pieces.append(clause)
        else:
            pieces.extend(_hard_wrap(clause, budget))

    # Re-merge adjacent pieces that still fit, so "A," is not emitted alone.
    merged: list[str] = []
    for piece in pieces:
        if merged and _join_len(merged[-1], piece) <= budget:
            merged[-1] = _join_pieces(merged[-1], piece)
        else:
            merged.append(piece)
    return merged


def split_text_into_chunks(
    text: str,
    chunk_size: int | None = None,
    language: str | None = None,
) -> list[str]:
    """Split ``text`` into sentence-aware chunks of at most ``chunk_size`` chars.

    ``chunk_size=None`` means "no merge budget": every sentence becomes its own
    chunk. A positive ``chunk_size`` merges neighbouring sentences greedily
    while the joined length fits; a sentence that alone exceeds the budget is
    split on clause punctuation first and hard-wrapped only as the last resort.
    Sentences are never split when they fit. Parenthetical prose and inline
    square-bracket tags stay inside their sentence: they are not segmented out
    by the splitter (see Phase 12-6).

    ``language`` is an advisory BCP-47-ish hint: CJK enders are enabled for
    ``zh``/``ja``/``ko`` roots and auto-detected from the text otherwise, so
    mixed-language input works without a hint. Chunks are stripped and joined
    with a single ASCII space — except after CJK text, where no space is
    inserted so the tokenizer does not emit ``[SPACE]`` tokens. The return value
    is empty for blank input.

    Raises:
        ValueError: if ``chunk_size`` is provided and not positive.
    """
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cjk_enabled = _is_cjk_language(language) or _CJK_END.search(normalized) is not None

    if len(_BULLET.findall(normalized)) >= 2:
        segments = _bullet_segments(normalized, cjk_enabled)
    else:
        segments = _segment_plain(normalized, cjk_enabled)

    if chunk_size is None:
        return [segment for segment in segments if segment]

    chunks: list[str] = []
    current = ""
    for segment in segments:
        if not segment:
            continue
        pieces = (
            _split_oversize(segment, chunk_size) if len(segment) > chunk_size else [segment]
        )
        for piece in pieces:
            if not current:
                current = piece
            elif _join_len(current, piece) <= chunk_size:
                current = _join_pieces(current, piece)
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------


def _as_waveform(wav: np.ndarray) -> np.ndarray:
    """Return ``wav`` as float32, validating the ``(samples[, channels])`` shape."""
    array = np.asarray(wav, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise ValueError(f"waveform must be 1-D or (samples, channels), got shape {array.shape}")
    return array


def _silence(samples: int, trailing_shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros((samples, *trailing_shape), dtype=np.float32)


def _empty_waveform(trailing_shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros((0, *trailing_shape), dtype=np.float32)


def _equal_power_curves(samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fade_out, fade_in)`` with ``cos² + sin² == 1`` (constant power)."""
    t = np.linspace(0, np.pi / 2, samples, dtype=np.float32)
    return np.cos(t) ** 2, np.sin(t) ** 2


def _crossfade_regions(tail: np.ndarray, head: np.ndarray, samples: int) -> np.ndarray:
    """Equal-power overlap-sum of ``tail`` and ``head`` (both length ``samples``)."""
    fade_out, fade_in = _equal_power_curves(samples)
    if tail.ndim == 2:
        fade_out = fade_out[:, None]
        fade_in = fade_in[:, None]
    return tail * fade_out + head * fade_in


# ---------------------------------------------------------------------------
# Stitcher-internal edge conditioning (Phase 12-5)
# ---------------------------------------------------------------------------
#
# Synthesis-native leading/trailing silence is not uniform: two chunks that
# sound the same length can carry 10 ms of silence on one edge and 300 ms on
# the other, so a plain butt splice (or a crossfade over that silence) produces
# the ragged "hiccup" the owner heard on the 2026-11-09 listening pass. The
# stitcher therefore normalises each chunk's edges *before* joining them: trim
# near-silence down to a small guard pad, bounded in absolute terms, with the
# gate relative to the chunk's own peak.
#
# This is separate from the user-facing :func:`trim_silence` filter: edge
# conditioning only makes joins uniform (and also bounds the first/last chunk's
# outer edges), while ``trim_silence`` is an explicit whole-output option.

#: A sample is silent when its magnitude is this many dB below the chunk peak.
DEFAULT_EDGE_SILENCE_THRESHOLD_DB = 45.0
#: Maximum silence removed from each chunk edge (ms) — never over-trim.
DEFAULT_EDGE_MAX_TRIM_MS = 150.0
#: Silence kept at each edge after conditioning (ms), so onsets stay natural.
DEFAULT_EDGE_GUARD_MS = 10.0

#: Crossfade between stitched chunks when the serving layer is chunking and the
#: client left ``crossfade_ms`` unset (Phase 12-5 tuned default, 2026-11-09).
#: 40 ms is long enough to hide a synthesis-native click/step and short enough
#: to stay inside actual speech rather than eating the sentence pause.
DEFAULT_CHUNK_CROSSFADE_MS = 40
#: Sentence pause when chunking is on and ``sentence_pause_ms`` is unset.
#: 120 ms is a natural inter-sentence gap; the crossfade consumes part of it,
#: leaving ~80 ms of audible silence by default.
DEFAULT_CHUNK_SENTENCE_PAUSE_MS = 120


def condition_edges(
    wav: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = DEFAULT_EDGE_SILENCE_THRESHOLD_DB,
    max_trim_ms: float = DEFAULT_EDGE_MAX_TRIM_MS,
    guard_ms: float = DEFAULT_EDGE_GUARD_MS,
) -> np.ndarray:
    """Trim a chunk's leading/trailing near-silence to a small guard pad.

    The gate is relative to the chunk's **own** sample peak: a sample is silent
    when its magnitude (channel maximum, so one live channel keeps the edge
    alive in stereo) is more than ``threshold_db`` below that peak. Up to
    ``max_trim_ms`` of silence is removed from each edge and ``guard_ms`` is
    kept, so a short natural onset/offset is never clipped and a chunk with
    several seconds of padding cannot lose more than the bound.

    All-quiet chunks (digital silence, peak ``0``) are returned unchanged: there
    is no relative gate to apply, and silently deleting a chunk would make the
    join behaviour undefined. A chunk with no near-silence is returned as an
    unchanged copy. Mono and ``(samples, channels)`` are both supported; the
    same sample range is applied to every channel.

    This is stitcher-internal (Phase 12-5). It is independent of the opt-in
    :func:`trim_silence` whole-output filter; see the module docs.
    """
    array = _as_waveform(wav)
    _require_positive("sample_rate", sample_rate)
    _require_positive("threshold_db", threshold_db)
    _require_non_negative("max_trim_ms", max_trim_ms)
    _require_non_negative("guard_ms", guard_ms)

    samples = array.shape[0]
    if samples == 0:
        return array.copy()

    magnitude = _sample_magnitude(array)
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        # Digital silence: honest no-op, the join stays defined.
        return array.copy()

    floor = peak * (10.0 ** (-threshold_db / 20.0))
    active = magnitude >= floor
    # ``lead``/``trail`` are the counts of near-silent samples at each edge
    # (``argmax`` on the boolean array is the first active index; on the
    # reversed array it is the number of silent samples at the end).
    lead = int(np.argmax(active))
    trail = int(np.argmax(active[::-1]))

    max_trim = int(round(sample_rate * max_trim_ms / 1000.0))
    guard = int(round(sample_rate * guard_ms / 1000.0))

    start = max(0, min(lead - guard, max_trim))
    end = samples - max(0, min(trail - guard, max_trim))
    if end <= start:  # pragma: no cover - the bound keeps one active region
        return array.copy()
    return array[start:end].copy()


class CrossfadeStitcher:
    """Chunk-by-chunk equal-power stitcher with a held-back tail.

    Designed for both the whole-buffer path (12-2) and a streaming consumer
    (Phase 14): :meth:`feed` returns only samples that can no longer change,
    withholding at most one fade of the most recent chunk's un-crossfaded tail
    until the next :meth:`feed` (where it is crossfaded into the next chunk's
    head) or :meth:`flush` (where it is returned unchanged).

    Every fed chunk is **edge-conditioned** first (Phase 12-5): synthesis-native
    leading/trailing near-silence is trimmed to a guard pad (bounded, gate
    relative to the chunk's own peak), so every join is uniform regardless of
    how much silence the model happened to emit. The held-back carry is faded
    **out** to silence (Phase 12-6) before the pause is inserted, and the next
    chunk's head is faded **in** from that silence. Because the pause is
    inserted before the crossfade, the join is speech-end → guard pad →
    ``cos²`` fade-out → pause → ``sin²`` fade-in → next chunk's onset (the fade
    consumes the last part of the pause). With no pause the two edges overlap in
    a direct equal-power crossfade instead. The carry always holds conditioned
    audio.

    Invariants:

    * feeding one chunk and flushing is byte-identical to the input when the
      chunk has no near-silence to trim (the serving layer never enters the
      stitcher for a single chunk anyway);
    * for ``N`` chunks longer than the fade, feeding all and flushing yields
      ``sum(len(conditioned_chunk)) + pause_samples * (N - 1)
      - fade_samples * (N - 1)`` samples per channel;
    * the inter-chunk pause is inserted before the crossfade, so the audible
      gap is ``sentence_pause_ms - crossfade_ms`` (clamped at zero) plus the
      two guard pads — the fade consumes part of the pause instead of adding
      to it;
    * there is never a trailing pause: the pause is only materialised when a
      following chunk actually arrives.
    """

    def __init__(
        self,
        crossfade_ms: int | None = 0,
        sentence_pause_ms: int | None = 0,
        sample_rate: int = 44100,
        *,
        edge_conditioning: bool = True,
    ):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
        crossfade_ms = 0 if crossfade_ms is None else crossfade_ms
        sentence_pause_ms = 0 if sentence_pause_ms is None else sentence_pause_ms
        if crossfade_ms < 0:
            raise ValueError(f"crossfade_ms must be >= 0, got {crossfade_ms!r}")
        if sentence_pause_ms < 0:
            raise ValueError(f"sentence_pause_ms must be >= 0, got {sentence_pause_ms!r}")

        self.sample_rate = int(sample_rate)
        self.crossfade_ms = int(crossfade_ms)
        self.sentence_pause_ms = int(sentence_pause_ms)
        self.fade_samples = int(round(self.sample_rate * self.crossfade_ms / 1000))
        self.pause_samples = int(round(self.sample_rate * self.sentence_pause_ms / 1000))
        self.edge_conditioning = bool(edge_conditioning)

        self._carry: np.ndarray | None = None
        self._started = False

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        """Add one chunk and return the samples that are now final."""
        array = _as_waveform(chunk)
        trailing_shape = array.shape[1:]
        if array.shape[0] == 0:
            return _empty_waveform(trailing_shape)
        if self.edge_conditioning:
            array = condition_edges(array, self.sample_rate)
            if array.shape[0] == 0:  # pragma: no cover - conditioning never empties
                return _empty_waveform(trailing_shape)

        if not self._started:
            self._started = True
            if self.fade_samples <= 0:
                self._carry = _empty_waveform(trailing_shape)
                return array.copy()
            self._carry = array[-self.fade_samples :].copy()
            return array[: -self.fade_samples].copy()

        assert self._carry is not None
        bridge = self._carry
        if self.pause_samples > 0:
            if self.fade_samples > 0 and bridge.shape[0] > 0:
                # Phase 12-6: fade Chunk i's held-back tail *into* silence so
                # it tapers to 0.0 (cos²) before the gap instead of hard-cutting
                # from the last sample. The paused bridge keeps the same length
                # (``len(carry) + pause``), so the length invariant below is
                # unchanged; the fade-in of Chunk i+1's head is still the
                # ``_crossfade_regions(bridge[-fade:], ...)`` step (where
                # ``bridge[-fade:]`` is now silence, not un-faded audio).
                fade_out = min(self.fade_samples, bridge.shape[0])
                bridge = np.concatenate(
                    [
                        bridge[:-fade_out],
                        _crossfade_regions(
                            bridge[-fade_out:],
                            _silence(fade_out, trailing_shape),
                            fade_out,
                        ),
                    ],
                    axis=0,
                )
            bridge = np.concatenate([bridge, _silence(self.pause_samples, trailing_shape)], axis=0)

        fade = min(self.fade_samples, bridge.shape[0], array.shape[0])
        if fade <= 0:
            self._carry = _empty_waveform(trailing_shape)
            return np.concatenate([bridge, array], axis=0)

        # ``array[:fade]`` is consumed by the crossfade; only ``array[fade:]``
        # is new audio. Hold back at most one fade of *that* tail — holding back
        # the last fade of the whole chunk would re-emit samples already inside
        # the crossfade when the chunk is shorter than two fades.
        rest = array[fade:]
        hold = min(self.fade_samples, rest.shape[0])
        emitted = np.concatenate(
            [
                bridge[:-fade],
                _crossfade_regions(bridge[-fade:], array[:fade], fade),
                rest[:-hold] if hold else rest,
            ],
            axis=0,
        )
        self._carry = rest[-hold:].copy() if hold else _empty_waveform(trailing_shape)
        return emitted

    def flush(self) -> np.ndarray:
        """Return the held-back tail once; later calls return an empty array."""
        if self._carry is None:
            return _empty_waveform(())
        carry = self._carry
        self._carry = None
        self._started = False
        return carry.copy()


def crossfade_stitch(
    chunks,
    crossfade_ms: int | None = 0,
    sentence_pause_ms: int | None = 0,
    sample_rate: int = 44100,
    *,
    edge_conditioning: bool = True,
) -> np.ndarray:
    """Stitch ``chunks`` with equal-power crossfades and optional sentence pauses.

    Whole-buffer convenience fold over :class:`CrossfadeStitcher`. With two or
    more non-empty chunks every chunk is edge-conditioned first (Phase 12-5);
    each join fades the outgoing tail out to silence and the incoming head in
    from silence (Phase 12-6) when a pause is present, and the result length is
    ``sum(len(conditioned_chunk)) + pause_samples * (N - 1)
    - fade_samples * (N - 1)`` (fades clamp on short chunks).

    A single non-empty chunk is returned **byte-identical** to the input: no
    join exists, so neither edge conditioning nor fade/pause is materialised.
    That is the short-text byte-identity gate for 12-2.
    """
    arrays: list[np.ndarray] = []
    trailing_shape: tuple[int, ...] = ()
    for chunk in chunks:
        array = _as_waveform(chunk)
        if array.shape[0] == 0:
            continue
        trailing_shape = array.shape[1:]
        arrays.append(array)
    if not arrays:
        return _empty_waveform(trailing_shape)

    stitcher = CrossfadeStitcher(
        crossfade_ms=crossfade_ms,
        sentence_pause_ms=sentence_pause_ms,
        sample_rate=sample_rate,
        edge_conditioning=edge_conditioning,
    )
    if len(arrays) == 1:
        # No join exists, so neither edge conditioning nor fade/pause is
        # materialised (the 12-2 single-chunk byte-identity gate).
        return arrays[0].copy()

    emitted: list[np.ndarray] = []
    for array in arrays:
        piece = stitcher.feed(array)
        if piece.shape[0]:
            emitted.append(piece)

    tail = stitcher.flush()
    if tail.shape[0]:
        emitted.append(tail)
    if not emitted:
        return _empty_waveform(trailing_shape)
    return np.concatenate(emitted, axis=0)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def peak_normalize(wav: np.ndarray, ceiling: float = DEFAULT_PEAK_CEILING) -> np.ndarray:
    """Scale ``wav`` so its absolute peak equals ``ceiling``.

    One gain is computed over all channels, so stereo imaging is preserved.
    Silent input (peak 0) is returned unchanged. ``ceiling`` must be in
    ``(0, 1]``; 0.99 leaves 1 % headroom for resampling/encoding overshoot.
    """
    if not 0.0 < ceiling <= 1.0:
        raise ValueError(f"ceiling must be in (0, 1], got {ceiling!r}")
    array = _as_waveform(wav)
    if array.size == 0:
        return array.copy()
    peak = float(np.max(np.abs(array)))
    if peak <= 0.0:
        return array.copy()
    return np.asarray(array * (ceiling / peak), dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase B output filters (Phase 12-3)
# ---------------------------------------------------------------------------
#
# Three opt-in, model-agnostic filters. They change samples by construction, so
# every one is default-off at the request layer; with all three disabled the
# serving layer never calls into this section and output stays byte-identical.
#
# All energy gates are *relative to the loudest analysis frame* (the upstream
# ``top_db`` convention): a frame is silent when its RMS is more than
# ``DEFAULT_SILENCE_THRESHOLD_DB`` below the loudest frame. Digital silence (all
# samples zero) is therefore always silent, while uniform low-level content is
# not mistaken for silence merely because it is quiet.

#: Short-time analysis frame length (ms). 32 ms spans a few pitch periods at
#: speech rates and is the standard short-time-energy window.
DEFAULT_FILTER_FRAME_MS = 32.0
#: Hop between analysis frames (ms). 75 % overlap keeps a frame boundary from
#: landing exactly on a speech onset/offset sample.
DEFAULT_FILTER_HOP_MS = 8.0
#: A frame is silent when its RMS is this many dB below the loudest frame.
DEFAULT_SILENCE_THRESHOLD_DB = 40.0
#: Silence kept at each end after :func:`trim_silence` (ms).
DEFAULT_TRIM_PADDING_MS = 50.0
#: Only internal pauses longer than this are shortened (ms).
DEFAULT_INTERNAL_SILENCE_MIN_MS = 700.0
#: An overlong internal pause is shortened to at most this (ms).
DEFAULT_INTERNAL_SILENCE_MAX_MS = 300.0


def _require_positive(name: str, value: float) -> float:
    if value is None or float(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return float(value)


def _require_non_negative(name: str, value: float) -> float:
    if value is None or float(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


def _sample_magnitude(array: np.ndarray) -> np.ndarray:
    """Per-sample magnitude, collapsing channels with ``max`` (mono-safe)."""
    magnitude = np.abs(array)
    if magnitude.ndim == 2:
        magnitude = magnitude.max(axis=1)
    return magnitude


def _frame_rms(
    array: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, int, int]:
    """Return ``(rms_per_frame, frame_length, hop_length)`` for ``array``.

    Uses a cumulative-sum integral image so the whole buffer costs ``O(n)`` time
    and memory regardless of frame overlap. Channels are averaged for the energy
    (a stereo pair is gated as one signal), which keeps the gate stable when one
    channel is quieter.
    """
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    power = np.square(array, dtype=np.float64)
    if power.ndim == 2:
        power = power.mean(axis=1)

    samples = power.shape[0]
    if samples == 0:
        return np.zeros(0, dtype=np.float64), frame_length, hop_length
    if samples <= frame_length:
        return np.sqrt(np.array([power.mean()], dtype=np.float64)), frame_length, hop_length

    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, samples - frame_length + 1, hop_length)
    sums = cumulative[starts + frame_length] - cumulative[starts]
    return np.sqrt(sums / frame_length), frame_length, hop_length


def _silent_frame_mask(rms: np.ndarray, threshold_db: float) -> np.ndarray:
    """True where a frame is more than ``threshold_db`` below the loudest frame."""
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 0.0:
        return np.ones(rms.shape, dtype=bool)
    floor = peak * (10.0 ** (-threshold_db / 20.0))
    return rms < floor


def remove_dc_offset(wav: np.ndarray) -> np.ndarray:
    """Remove the constant (DC) component by subtracting each channel's mean.

    Mean subtraction is chosen over a one-pole high-pass because it is exact,
    stateless and phase-free: it removes *only* the 0 Hz term and leaves every
    other bin untouched. (A high-pass would also attenuate legitimate low
    frequencies and shift phase.) Per-channel means preserve the stereo image
    when the two channels carry different offsets. Empty input is returned
    unchanged.

    This is the per-chunk filter in the chunked synthesis path: each synthesis
    run can carry a different offset, so the mean is removed *before* stitching
    (a whole-buffer mean would smear a boundary between chunks).
    """
    array = _as_waveform(wav)
    if array.shape[0] == 0:
        return array.copy()
    mean = array.mean(axis=0, keepdims=True, dtype=np.float64)
    return np.asarray(array - mean, dtype=np.float32)


def trim_silence(
    wav: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    padding_ms: float = DEFAULT_TRIM_PADDING_MS,
    frame_ms: float = DEFAULT_FILTER_FRAME_MS,
    hop_ms: float = DEFAULT_FILTER_HOP_MS,
) -> np.ndarray:
    """Trim leading and trailing silence from ``wav`` (energy-gated).

    A short-time RMS gate marks frames more than ``threshold_db`` below the
    loudest frame as silent; the first and last non-silent frames bound the
    output, refined to the exact sample where the signal crosses the gate's
    sample floor. ``padding_ms`` of audio is kept before the first and after the
    last active sample (default 50 ms, matching the upstream reference).

    Returns an empty waveform for all-silent input (digital silence has nothing
    to keep), and an unchanged copy when no silence is detected. Mono
    ``(samples,)`` and ``(samples, channels)`` are both accepted; a single sample
    range is applied to every channel so the stereo image is preserved.

    Defaults: 40 dB below the loudest frame, 32 ms frames, 8 ms hop, 50 ms
    padding. These are function-level knobs for tests and future callers; the
    request layer exposes only the on/off boolean.
    """
    array = _as_waveform(wav)
    _require_positive("sample_rate", sample_rate)
    _require_positive("threshold_db", threshold_db)
    _require_non_negative("padding_ms", padding_ms)
    _require_positive("frame_ms", frame_ms)
    _require_positive("hop_ms", hop_ms)

    samples = array.shape[0]
    if samples == 0:
        return array.copy()

    rms, frame_length, hop_length = _frame_rms(array, sample_rate, frame_ms, hop_ms)
    silent = _silent_frame_mask(rms, threshold_db)
    active = np.flatnonzero(~silent)
    if active.size == 0:
        return _empty_waveform(array.shape[1:])

    magnitude = _sample_magnitude(array)
    sample_peak = float(np.max(magnitude))
    floor = sample_peak * (10.0 ** (-threshold_db / 20.0)) if sample_peak > 0.0 else 0.0

    first, last = int(active[0]), int(active[-1])

    # Refine the frame-quantised boundary to the exact crossing sample.
    start = first * hop_length
    window_end = min(samples, start + frame_length)
    loud = np.flatnonzero(magnitude[start:window_end] >= floor)
    if loud.size:
        start += int(loud[0])

    if last == silent.size - 1:
        # The last analysis frame is active, so the buffer ends in audio that no
        # silent frame covers; keep the real tail rather than the frame end.
        end = samples
    else:
        end = min(samples, last * hop_length + frame_length)
        window_start = last * hop_length
        loud = np.flatnonzero(magnitude[window_start:end] >= floor)
        if loud.size:
            end = window_start + int(loud[-1]) + 1

    padding = int(round(sample_rate * padding_ms / 1000.0))
    start = max(0, start - padding)
    end = min(samples, end + padding)
    if end <= start:
        return _empty_waveform(array.shape[1:])
    return array[start:end].copy()


def fix_internal_silence(
    wav: np.ndarray,
    sample_rate: int,
    *,
    max_pause_ms: float = DEFAULT_INTERNAL_SILENCE_MAX_MS,
    min_silence_ms: float = DEFAULT_INTERNAL_SILENCE_MIN_MS,
    threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    frame_ms: float = DEFAULT_FILTER_FRAME_MS,
    hop_ms: float = DEFAULT_FILTER_HOP_MS,
) -> np.ndarray:
    """Collapse *internal* pauses longer than ``min_silence_ms`` to ``max_pause_ms``.

    Only silence strictly between two active regions is touched: leading and
    trailing silence are left to :func:`trim_silence` (a chunk boundary or the
    start/end of the buffer must never be mistaken for an internal pause). A
    silent run is shortened only when it lasts at least ``min_silence_ms``
    (default 700 ms, the upstream reference); the first ``max_pause_ms`` of the
    run are kept (default 300 ms) and the rest is removed.

    Run boundaries are refined to the exact sample crossing the same 40 dB
    energy gate used by :func:`trim_silence`, so a short pause is not trimmed
    merely because it overlaps an analysis frame. Returns an unchanged copy when
    there is nothing to fix (including all-silent and all-active input). Mono and
    ``(samples, channels)`` are supported; the same sample range is removed from
    every channel.
    """
    array = _as_waveform(wav)
    _require_positive("sample_rate", sample_rate)
    _require_positive("max_pause_ms", max_pause_ms)
    _require_positive("min_silence_ms", min_silence_ms)
    _require_positive("threshold_db", threshold_db)
    _require_positive("frame_ms", frame_ms)
    _require_positive("hop_ms", hop_ms)
    if max_pause_ms > min_silence_ms:
        raise ValueError(
            f"max_pause_ms ({max_pause_ms}) must be <= min_silence_ms ({min_silence_ms})"
        )

    samples = array.shape[0]
    if samples == 0:
        return array.copy()

    rms, frame_length, hop_length = _frame_rms(array, sample_rate, frame_ms, hop_ms)
    silent = _silent_frame_mask(rms, threshold_db)
    frame_count = silent.size
    if frame_count < 3 or not silent.any() or silent.all():
        return array.copy()

    magnitude = _sample_magnitude(array)
    sample_peak = float(np.max(magnitude))
    if sample_peak <= 0.0:
        return array.copy()
    floor = sample_peak * (10.0 ** (-threshold_db / 20.0))

    max_pause = int(round(sample_rate * max_pause_ms / 1000.0))
    min_silence = int(round(sample_rate * min_silence_ms / 1000.0))

    # Run-length encode the silent frames: pairs of ``[start, end)`` frame
    # indices. ``padded`` lets a run that touches either edge still close.
    padded = np.concatenate(([False], silent, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = zip(edges[::2], edges[1::2], strict=True)

    pieces: list[np.ndarray] = []
    cursor = 0
    changed = False
    for run_start, run_end in runs:
        if run_start == 0 or run_end == frame_count:
            # Leading/trailing silence is trim_silence's job.
            continue

        # Conservative bounds: end of the last active frame before the run, and
        # start of the first active frame after it. Refine to the exact sample
        # crossing inside those two frames so the pause length is honest.
        safe_start = min(samples, (run_start - 1) * hop_length + frame_length)
        safe_end = min(samples, run_end * hop_length)

        window_start = max(cursor, safe_start - frame_length)
        loud = np.flatnonzero(magnitude[window_start:safe_start] >= floor)
        true_start = window_start + int(loud[-1]) + 1 if loud.size else safe_start
        true_start = max(true_start, cursor)

        window_end = min(samples, safe_end + frame_length)
        loud = np.flatnonzero(magnitude[safe_end:window_end] >= floor)
        true_end = safe_end + int(loud[0]) if loud.size else safe_end

        if true_end - true_start < min_silence:
            continue

        cut_start = true_start + max_pause
        if cut_start >= true_end:
            continue
        pieces.append(array[cursor:cut_start])
        cursor = true_end
        changed = True

    if not changed:
        return array.copy()
    pieces.append(array[cursor:])
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def time_stretch(wav: np.ndarray, rate: float, sample_rate: int = 44100) -> np.ndarray:
    """Phase-vocoder time-stretch by ``rate`` (``>1`` faster/shorter).

    Phase-preserving (librosa's phase vocoder), CPU-only, mono or
    ``(samples, channels)``. Output length is ``len(wav) / rate`` to within the
    STFT frame rounding. ``rate == 1.0`` returns an unchanged copy without
    importing librosa. ``sample_rate`` selects the analysis window (2048 at
    >= 32 kHz, 1024 below) so the window spans a similar duration at both rates.
    """
    if rate is None or rate <= 0:
        raise ValueError(f"rate must be positive, got {rate!r}")
    array = _as_waveform(wav)
    if array.shape[0] == 0 or rate == 1.0:
        return array.copy()

    # Lazy: importing librosa pulls numba/scipy and costs ~1 s at import time.
    import librosa

    n_fft = 2048 if sample_rate >= 32000 else 1024
    if array.ndim == 1:
        stretched = librosa.effects.time_stretch(array, rate=float(rate), n_fft=n_fft)
        return np.asarray(stretched, dtype=np.float32)

    # librosa treats the *last* axis as time, so feed it (channels, samples).
    stretched = librosa.effects.time_stretch(array.T, rate=float(rate), n_fft=n_fft)
    return np.ascontiguousarray(np.asarray(stretched, dtype=np.float32).T)
