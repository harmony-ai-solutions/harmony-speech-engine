"""Unit tests for the Phase 12-1 long-form audio utilities.

Covers the sentence splitter (including both upstream issue #144 regressions and
CJK enders), the equal-power crossfade stitcher (whole-buffer and streaming),
peak normalization and the phase-vocoder time-stretch.
"""

import time

import numpy as np
import pytest

from harmonyspeech.common.audio_utils import (
    DEFAULT_CHUNK_CROSSFADE_MS,
    DEFAULT_CHUNK_SENTENCE_PAUSE_MS,
    DEFAULT_EDGE_GUARD_MS,
    DEFAULT_EDGE_MAX_TRIM_MS,
    DEFAULT_EDGE_SILENCE_THRESHOLD_DB,
    DEFAULT_INTERNAL_SILENCE_MAX_MS,
    DEFAULT_INTERNAL_SILENCE_MIN_MS,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_TRIM_PADDING_MS,
    CrossfadeStitcher,
    condition_edges,
    crossfade_stitch,
    fix_internal_silence,
    peak_normalize,
    remove_dc_offset,
    split_text_into_chunks,
    time_stretch,
    trim_silence,
)

# ---------------------------------------------------------------------------
# Splitter: happy path, budgets and degenerate input
# ---------------------------------------------------------------------------


def test_split_sentences_without_budget():
    text = "Hello world. This is a test. Done!"
    assert split_text_into_chunks(text) == ["Hello world.", "This is a test.", "Done!"]


def test_split_blank_input():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\t\r\n ") == []


def test_split_merges_sentences_within_budget():
    assert split_text_into_chunks("One. Two. Three.", chunk_size=10) == ["One. Two.", "Three."]
    assert split_text_into_chunks("One. Two. Three.", chunk_size=6) == ["One.", "Two.", "Three."]


def test_split_never_splits_a_fitting_sentence():
    text = "A sentence that is comfortably inside the budget. Another one."
    chunks = split_text_into_chunks(text, chunk_size=len(text))
    assert chunks == [text]


def test_split_oversize_sentence_uses_clause_fallback():
    text = "This is a very long sentence, with several clauses, that must be split into pieces."
    chunks = split_text_into_chunks(text, chunk_size=30)
    assert all(len(chunk) <= 30 for chunk in chunks)
    assert len(chunks) > 1
    assert " ".join(chunks) == " ".join(text.split())


def test_split_oversize_sentence_hard_wraps_without_clauses():
    text = "abcdefghij" * 5  # 50 chars, no spaces or clause punctuation
    chunks = split_text_into_chunks(text, chunk_size=10)
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == text


def test_split_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="chunk_size"):
        split_text_into_chunks("Hello.", chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        split_text_into_chunks("Hello.", chunk_size=-5)


# ---------------------------------------------------------------------------
# Splitter: abbreviations, numbers, versions, quotes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr. Smith went home.", ["Mr. Smith went home."]),
        ("Dr. Jones and Prof. Lee arrived.", ["Dr. Jones and Prof. Lee arrived."]),
        ("Use e.g. this example.", ["Use e.g. this example."]),
        ("That is i.e. the point.", ["That is i.e. the point."]),
        ("Bring pens, books, etc. and paper.", ["Bring pens, books, etc. and paper."]),
        ("Pi is 3.14 approximately. Use v1.2.3 today.", ["Pi is 3.14 approximately.", "Use v1.2.3 today."]),
        ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
        ("Wait... what happened?", ["Wait... what happened?"]),
    ],
)
def test_split_latin_edge_cases(text, expected):
    assert split_text_into_chunks(text) == expected


# ---------------------------------------------------------------------------
# Splitter: bullets and the two upstream issue #144 regressions
# ---------------------------------------------------------------------------


def test_split_bullet_list():
    text = "- First item.\n- Second item.\n- Third item."
    assert split_text_into_chunks(text) == ["- First item.", "- Second item.", "- Third item."]


def test_split_numbered_list():
    assert split_text_into_chunks("1. First\n2. Second") == ["1. First", "2. Second"]


def test_regression_dash_without_space_is_not_a_bullet():
    """f0afcc6: a dash followed by text/newline is narrative, not a list item."""
    text = "-Within the Tunnels of Alexander-\n\nWithin the tunnels we walked."
    assert split_text_into_chunks(text) == [text]


def test_regression_dash_after_punctuation_is_not_a_bullet():
    """f0afcc6: the dash is not at column 0, so it cannot start a list item."""
    text = "wait that sounds lewd)-\n\nWithin the tunnels we walked."
    assert split_text_into_chunks(text) == [text]


def test_regression_indented_dashes_are_not_bullets():
    """450de15: indented dashes (``  ---TITLE---``) are narrative text."""
    text = "  ---TITLE---\n\nSome text."
    assert split_text_into_chunks(text) == ["---TITLE---\n\nSome text."]


def test_regression_single_bullet_does_not_trigger_list_mode():
    """450de15: a list needs 2+ items; one dash stays in the sentence stream."""
    text = "-Within the tunnels we walked.\nA normal paragraph follows."
    chunks = split_text_into_chunks(text)
    assert chunks[0].startswith("-Within the tunnels we walked.")
    assert not any(chunk.startswith("- ") for chunk in chunks)


def test_bullet_list_still_splits_with_two_or_more_items():
    assert split_text_into_chunks("- one\n- two") == ["- one", "- two"]


# ---------------------------------------------------------------------------
# Splitter: narrative dashes (em/en dash are sentence-internal)
# ---------------------------------------------------------------------------


def test_regression_narrative_em_dash_does_not_split():
    """450de15: em dashes are sentence-internal, never a boundary."""
    text = "The plan — which was bold — failed. Next."
    assert split_text_into_chunks(text) == ["The plan — which was bold — failed.", "Next."]


def test_regression_narrative_dash_between_words():
    assert split_text_into_chunks("It was—surprisingly—fine.") == ["It was—surprisingly—fine."]


# ---------------------------------------------------------------------------
# Splitter: parenthetical prose, inline tags, ellipsis and CJK joining
# ---------------------------------------------------------------------------


def test_parenthetical_prose_stays_in_its_sentence():
    """Regression (Phase 12-6): ``(like this)`` is text, not a cue segment."""
    text = "The Gryffindors (even the boldest ones) retracted their fangs. Next sentence."
    assert split_text_into_chunks(text) == [
        "The Gryffindors (even the boldest ones) retracted their fangs.",
        "Next sentence.",
    ]


@pytest.mark.parametrize("tag", ["[gasp]", "[laugh]", "[sigh]"])
def test_inline_square_bracket_tags_stay_in_their_sentence(tag):
    """Turbo tags are inline paralinguistic hints, never standalone segments."""
    text = f"{tag} Okay, we have fifteen minutes left! Let's think."
    assert split_text_into_chunks(text) == [
        f"{tag} Okay, we have fifteen minutes left!",
        "Let's think.",
    ]


def test_inline_tags_all_stay_attached_to_one_sentence():
    text = "[gasp] Okay, we have fifteen minutes left! [sigh] Let's think."
    assert split_text_into_chunks(text) == [
        "[gasp] Okay, we have fifteen minutes left!",
        "[sigh] Let's think.",
    ]


def test_unicode_ellipsis_does_not_split_english_text():
    """Regression (Phase 12-6): ``…`` is sentence-internal like ASCII ``...``."""
    assert split_text_into_chunks("She waited… and then spoke. Then left.") == [
        "She waited… and then spoke.",
        "Then left.",
    ]
    # ``…`` alone never enables CJK splitting or ends a sentence.
    assert split_text_into_chunks("She waited… and then spoke.") == [
        "She waited… and then spoke."
    ]


@pytest.mark.parametrize("language", ["zh", "ja", "ko", "zh-CN", "ja_JP"])
def test_split_cjk_enders_with_language_hint(language):
    assert split_text_into_chunks("你好。世界！再见？", language=language) == ["你好。", "世界！", "再见？"]


def test_split_cjk_enders_auto_detected_without_hint():
    assert split_text_into_chunks("你好。世界！") == ["你好。", "世界！"]
    assert split_text_into_chunks("Hello. 你好。") == ["Hello.", "你好。"]


def test_split_cjk_closing_bracket_stays_with_sentence():
    assert split_text_into_chunks("「你好。」「再见。」") == ["「你好。」", "「再见。」"]


def test_split_cjk_ellipsis_is_not_a_boundary():
    """``…`` is sentence-internal in CJK too (only ``。！？`` end a sentence)."""
    assert split_text_into_chunks("等等……然后呢？") == ["等等……然后呢？"]


def test_merging_cjk_sentences_does_not_insert_an_ascii_space():
    chunks = split_text_into_chunks("第一句。第二句。", chunk_size=100)
    assert chunks == ["第一句。第二句。"]
    assert all(" " not in chunk for chunk in chunks)


def test_merging_latin_sentences_still_inserts_a_space():
    assert split_text_into_chunks("One. Two.", chunk_size=100) == ["One. Two."]


def test_oversize_cjk_clause_merge_does_not_insert_an_ascii_space():
    text = "一二三，四五六，七八九。"
    chunks = split_text_into_chunks(text, chunk_size=9)
    assert chunks == ["一二三，四五六，", "七八九。"]
    assert all(" " not in chunk for chunk in chunks)
    assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# Stitcher: equal-power math and length invariants
# ---------------------------------------------------------------------------


def test_crossfade_equal_power_hand_computed():
    # fade = 5 samples at 1 kHz. cos²/sin² over [0, pi/2] is constant-power.
    a = np.array([0, 1, 2, 3, 4, 10, 20, 30, 40, 50], dtype=np.float32)
    b = np.array([1, 2, 3, 4, 5, 100, 101, 102, 103, 104], dtype=np.float32)
    expected_region = np.array(
        [
            10.0,
            20.0 * 0.8535533905932737 + 2.0 * 0.1464466094067262,
            30.0 * 0.5 + 3.0 * 0.5,
            40.0 * 0.1464466094067262 + 4.0 * 0.8535533905932737,
            5.0,
        ],
        dtype=np.float32,
    )
    expected = np.concatenate([a[:-5], expected_region, b[5:]])

    out = crossfade_stitch([a, b], crossfade_ms=5, sentence_pause_ms=0, sample_rate=1000)

    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-5)


def test_crossfade_length_invariant_a_plus_b_minus_fade():
    rng = np.random.default_rng(1234)
    a = rng.standard_normal(100).astype(np.float32)
    b = rng.standard_normal(80).astype(np.float32)
    out = crossfade_stitch([a, b], crossfade_ms=20, sentence_pause_ms=0, sample_rate=1000)
    assert out.shape == (100 + 80 - 20,)


def test_crossfade_single_chunk_is_byte_identical():
    rng = np.random.default_rng(7)
    chunk = rng.standard_normal(321).astype(np.float32)
    for fade, pause in ((0, 0), (10, 0), (10, 50)):
        out = crossfade_stitch([chunk], crossfade_ms=fade, sentence_pause_ms=pause, sample_rate=1000)
        np.testing.assert_array_equal(out, chunk)


def test_crossfade_empty_input():
    assert crossfade_stitch([], crossfade_ms=10, sample_rate=1000).shape == (0,)


def test_crossfade_pause_insertion_without_fade():
    a = np.ones(100, dtype=np.float32)
    b = np.full(100, 0.5, dtype=np.float32)
    out = crossfade_stitch([a, b], crossfade_ms=0, sentence_pause_ms=50, sample_rate=1000)
    assert out.shape == (250,)
    np.testing.assert_array_equal(out[:100], a)
    np.testing.assert_array_equal(out[100:150], np.zeros(50, dtype=np.float32))
    np.testing.assert_array_equal(out[150:], b)


def test_crossfade_pause_and_fade_compose():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    out = crossfade_stitch([a, b], crossfade_ms=10, sentence_pause_ms=50, sample_rate=1000)
    # pause + two chunks - one fade; the fade consumes 10 ms of the 50 ms pause.
    assert out.shape == (100 + 50 + 100 - 10,)


def test_pause_fades_outgoing_tail_and_incoming_head():
    """Phase 12-6: with a pause, both chunk edges taper instead of hard-cutting."""
    sr = 1000
    fade = 40
    pause = 120
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    out = crossfade_stitch(
        [a, b],
        crossfade_ms=fade,
        sentence_pause_ms=pause,
        sample_rate=sr,
        edge_conditioning=False,
    )
    assert out.shape == (100 + 100 + pause - fade,)

    # Layout at 1 kHz: 60 ones | 40 cos² taper | 80 silence | 40 sin² fade-in |
    # 60 ones (20 held back mid-feed + 40 from flush).
    tail = out[60:100]
    silence = out[100:180]
    head = out[180:220]
    # The outgoing tail monotonically tapers from 1.0 to 0.0 — no hard step.
    assert tail[0] == pytest.approx(1.0, abs=1e-5)
    assert tail[-1] == pytest.approx(0.0, abs=1e-5)
    assert np.all(np.diff(tail) <= 1e-6)
    assert float(np.max(np.abs(np.diff(tail)))) < 0.1
    np.testing.assert_allclose(silence, 0.0, atol=1e-6)
    # The incoming head monotonically fades up from 0.0 to 1.0.
    assert head[0] == pytest.approx(0.0, abs=1e-5)
    assert head[-1] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.diff(head) >= -1e-6)
    assert float(np.max(np.abs(np.diff(head)))) < 0.1
    np.testing.assert_allclose(out[220:], 1.0, atol=1e-6)

    # Streaming must stay sample-for-sample identical to the whole-buffer fold.
    stitcher = CrossfadeStitcher(
        crossfade_ms=fade,
        sentence_pause_ms=pause,
        sample_rate=sr,
        edge_conditioning=False,
    )
    streamed = np.concatenate([stitcher.feed(a), stitcher.feed(b), stitcher.flush()])
    np.testing.assert_array_equal(streamed, out)


def test_crossfade_stereo_channels_are_independent():
    a = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    b = np.linspace(1.0, -1.0, 100, dtype=np.float32)
    stereo_a = np.stack([a, -a], axis=1)
    stereo_b = np.stack([b, -b], axis=1)
    mono = crossfade_stitch([a, b], crossfade_ms=20, sample_rate=1000)
    stereo = crossfade_stitch([stereo_a, stereo_b], crossfade_ms=20, sample_rate=1000)
    assert stereo.shape == (mono.shape[0], 2)
    np.testing.assert_allclose(stereo[:, 0], mono, atol=1e-6)
    np.testing.assert_allclose(stereo[:, 1], -mono, atol=1e-6)


def test_crossfade_fade_not_shorter_than_chunks_is_clamped():
    a = np.ones(10, dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    out = crossfade_stitch([a, b], crossfade_ms=20, sample_rate=1000)  # 20 samples > both chunks
    assert out.shape == (10,)
    assert out[0] == pytest.approx(1.0, abs=1e-6)
    assert out[-1] == pytest.approx(0.0, abs=1e-6)

    short_a = np.ones(4, dtype=np.float32)
    long_b = np.ones(10, dtype=np.float32)
    out = crossfade_stitch([short_a, long_b], crossfade_ms=10, sample_rate=1000)
    assert out.shape == (10,)


def test_streaming_stitcher_matches_whole_buffer():
    rng = np.random.default_rng(99)
    chunks = [
        rng.standard_normal(37).astype(np.float32),
        rng.standard_normal(5).astype(np.float32),
        rng.standard_normal(80).astype(np.float32),
        rng.standard_normal(20).astype(np.float32),
    ]
    for fade, pause in ((0, 0), (7, 0), (0, 30), (7, 30), (100, 10)):
        expected = crossfade_stitch(chunks, crossfade_ms=fade, sentence_pause_ms=pause, sample_rate=1000)
        stitcher = CrossfadeStitcher(crossfade_ms=fade, sentence_pause_ms=pause, sample_rate=1000)
        emitted = [stitcher.feed(chunk) for chunk in chunks]
        emitted.append(stitcher.flush())
        actual = np.concatenate(emitted) if emitted else np.zeros(0, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_streaming_flush_is_idempotent_and_tail_is_stable():
    stitcher = CrossfadeStitcher(crossfade_ms=10, sentence_pause_ms=0, sample_rate=1000)
    first = stitcher.feed(np.arange(50, dtype=np.float32))
    tail = stitcher.flush()
    assert stitcher.flush().shape == (0,)
    np.testing.assert_array_equal(np.concatenate([first, tail]), np.arange(50, dtype=np.float32))


def test_stitcher_rejects_invalid_timing():
    with pytest.raises(ValueError, match="sample_rate"):
        CrossfadeStitcher(sample_rate=0)
    with pytest.raises(ValueError, match="crossfade_ms"):
        CrossfadeStitcher(crossfade_ms=-1)
    with pytest.raises(ValueError, match="sentence_pause_ms"):
        CrossfadeStitcher(sentence_pause_ms=-1)
    # The whole-buffer fold validates even when a single chunk short-circuits
    # the join (no silent acceptance of invalid timing).
    with pytest.raises(ValueError, match="crossfade_ms"):
        crossfade_stitch([np.ones(10, dtype=np.float32)], crossfade_ms=-1, sample_rate=1000)


# ---------------------------------------------------------------------------
# Phase 12-5: stitcher-internal edge conditioning
# ---------------------------------------------------------------------------


def _padded(lead: int, trail: int, active: int = 100, value: float = 0.5) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(lead, dtype=np.float32),
            np.full(active, value, dtype=np.float32),
            np.zeros(trail, dtype=np.float32),
        ]
    )


def test_edge_conditioning_trims_pads_to_the_guard():
    # 1 kHz: guard 10 ms = 10 samples, max trim 150 ms = 150 samples.
    chunk = _padded(60, 90)

    out = condition_edges(chunk, 1000)

    assert out.shape == (10 + 100 + 10,)
    np.testing.assert_array_equal(out[:10], np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(out[10:110], np.full(100, 0.5, dtype=np.float32))
    np.testing.assert_array_equal(out[110:], np.zeros(10, dtype=np.float32))


def test_edge_conditioning_respects_the_max_trim_bound():
    # 1 s of lead silence cannot lose more than the 150 ms bound.
    chunk = _padded(1000, 0)

    out = condition_edges(chunk, 1000)

    assert out.shape == (1000 - 150 + 100,)
    # The bound leaves silence on purpose: never over-trim.
    assert float(np.max(np.abs(out[:850]))) == 0.0


def test_edge_conditioning_no_silence_is_unchanged():
    chunk = np.linspace(-0.5, 0.5, 80, dtype=np.float32)
    out = condition_edges(chunk, 1000)
    np.testing.assert_array_equal(out, chunk)


def test_edge_conditioning_all_silence_degrades_honestly():
    """An all-quiet chunk has no relative gate; it is left intact (not deleted)."""
    chunk = np.zeros(500, dtype=np.float32)
    out = condition_edges(chunk, 1000)
    np.testing.assert_array_equal(out, chunk)


def test_edge_conditioning_is_stereo_safe():
    """One live channel keeps an edge; both padded edges are trimmed together."""
    active = np.full(100, 0.5, dtype=np.float32)
    # Left channel padded on both edges, right channel live on both: the
    # channel-maximum magnitude keeps the edges alive, so nothing is trimmed.
    left = _padded(60, 60)
    right = np.concatenate(
        [
            np.full(60, 0.5, dtype=np.float32),
            active,
            np.full(60, 0.5, dtype=np.float32),
        ]
    )
    out = condition_edges(np.stack([left, right], axis=1), 1000)
    assert out.shape[0] == len(left)
    # Both channels padded: one range applies to both channels.
    both = np.stack([_padded(60, 60), _padded(60, 60)], axis=1)
    out = condition_edges(both, 1000)
    assert out.shape == (10 + 100 + 10, 2)
    assert float(np.max(np.abs(out[:10]))) == 0.0
    assert float(np.max(np.abs(out[-10:]))) == 0.0
    np.testing.assert_array_equal(out[:, 0], out[:, 1])


def test_edge_conditioning_rejects_invalid_parameters():
    chunk = _padded(10, 10)
    with pytest.raises(ValueError, match="sample_rate"):
        condition_edges(chunk, 0)
    with pytest.raises(ValueError, match="threshold_db"):
        condition_edges(chunk, 1000, threshold_db=0)
    with pytest.raises(ValueError, match="max_trim_ms"):
        condition_edges(chunk, 1000, max_trim_ms=-1)
    with pytest.raises(ValueError, match="guard_ms"):
        condition_edges(chunk, 1000, guard_ms=-1)


def test_edge_conditioning_constants_match_the_documented_tuning():
    assert DEFAULT_EDGE_SILENCE_THRESHOLD_DB == 45.0
    assert DEFAULT_EDGE_MAX_TRIM_MS == 150.0
    assert DEFAULT_EDGE_GUARD_MS == 10.0
    assert DEFAULT_CHUNK_CROSSFADE_MS == 40
    assert DEFAULT_CHUNK_SENTENCE_PAUSE_MS == 120


def _speech_gap(waveform: np.ndarray, threshold: float = 1e-4) -> int:
    """Samples between the last active sample and the next active sample."""
    active = np.flatnonzero(np.abs(waveform) > threshold)
    assert active.size, "no active samples"
    # The two speech regions are separated by the join; find the largest gap.
    breaks = np.flatnonzero(np.diff(active) > 1)
    assert breaks.size, "expected two speech regions"
    split = int(breaks[0])
    return int(active[split + 1] - active[split] - 1)


@pytest.mark.parametrize("fade_ms", [0, 40])
def test_join_gap_is_uniform_across_ragged_padded_silence(fade_ms):
    """The owner's stutter regression: gap depends on pause, not chunk pads.

    Chunks carry different amounts of synthesis-native leading/trailing silence;
    after edge conditioning the inter-speech gap is fixed by the pause setting
    (plus the two guard pads, minus the fade that consumes part of the pause).
    """
    pause_ms = 120
    guard = 10  # samples at 1 kHz
    fade = fade_ms  # samples at 1 kHz
    gaps = set()
    for lead_a, trail_a, lead_b, trail_b in (
        (20, 20, 20, 20),
        (60, 140, 90, 30),
        (140, 60, 30, 120),
    ):
        a = _padded(lead_a, trail_a)
        b = _padded(lead_b, trail_b)
        out = crossfade_stitch(
            [a, b], crossfade_ms=fade_ms, sentence_pause_ms=pause_ms, sample_rate=1000
        )
        gaps.add(_speech_gap(out))

    assert len(gaps) == 1, gaps
    (gap,) = gaps
    # Fade 0: pause + guard on each side. Fade > 0: the fade consumes part of
    # the pause (existing stitcher semantics), so subtract it.
    assert gap == pause_ms - fade + 2 * guard


def test_crossfade_length_invariant_with_edge_conditioning():
    """Result length = sum(conditioned chunks) + pauses - fades."""
    a = _padded(60, 90)
    b = _padded(30, 120)
    out = crossfade_stitch([a, b], crossfade_ms=20, sentence_pause_ms=50, sample_rate=1000)
    expected = (
        len(condition_edges(a, 1000))
        + len(condition_edges(b, 1000))
        + 50
        - 20
    )
    assert out.shape == (expected,)


def test_streaming_stitcher_matches_whole_buffer_with_conditioned_edges():
    rng = np.random.default_rng(2026)
    chunks = [
        _padded(80, 20, active=120),
        _padded(30, 200, active=60),
        np.concatenate([np.zeros(15, dtype=np.float32), rng.standard_normal(90).astype(np.float32)]),
    ]
    for fade, pause in ((0, 0), (40, 120), (0, 60), (25, 90)):
        expected = crossfade_stitch(chunks, crossfade_ms=fade, sentence_pause_ms=pause, sample_rate=1000)
        stitcher = CrossfadeStitcher(crossfade_ms=fade, sentence_pause_ms=pause, sample_rate=1000)
        emitted = [stitcher.feed(chunk) for chunk in chunks]
        emitted.append(stitcher.flush())
        actual = np.concatenate(emitted)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_crossfade_single_chunk_is_not_edge_conditioned():
    """A single chunk has no join, so the byte-identity gate stays exact."""
    chunk = _padded(200, 200)
    out = crossfade_stitch([chunk], crossfade_ms=40, sentence_pause_ms=120, sample_rate=1000)
    np.testing.assert_array_equal(out, chunk)



# ---------------------------------------------------------------------------
# Peak normalization
# ---------------------------------------------------------------------------


def test_peak_normalize_scales_to_ceiling():
    wav = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(wav, ceiling=0.5)
    np.testing.assert_allclose(out, [0.25, -0.5, 0.125], rtol=0, atol=1e-7)
    assert float(np.max(np.abs(out))) == pytest.approx(0.5)


def test_peak_normalize_raises_quiet_audio():
    out = peak_normalize(np.array([0.01, -0.005], dtype=np.float32), ceiling=0.99)
    assert float(np.max(np.abs(out))) == pytest.approx(0.99, rel=1e-6)


def test_peak_normalize_silence_is_unchanged():
    wav = np.zeros(64, dtype=np.float32)
    np.testing.assert_array_equal(peak_normalize(wav), wav)


def test_peak_normalize_stereo_uses_one_gain():
    wav = np.array([[1.0, -0.5], [0.25, -1.0]], dtype=np.float32)
    out = peak_normalize(wav, ceiling=0.5)
    assert float(np.max(np.abs(out))) == pytest.approx(0.5)
    np.testing.assert_allclose(out, wav * 0.5, rtol=0, atol=1e-7)


@pytest.mark.parametrize("ceiling", [0.0, -0.1, 1.1])
def test_peak_normalize_rejects_invalid_ceiling(ceiling):
    with pytest.raises(ValueError, match="ceiling"):
        peak_normalize(np.ones(8, dtype=np.float32), ceiling=ceiling)


# ---------------------------------------------------------------------------
# Time stretch
# ---------------------------------------------------------------------------


def test_time_stretch_rate_one_is_identity():
    rng = np.random.default_rng(3)
    wav = rng.standard_normal(1000).astype(np.float32)
    np.testing.assert_array_equal(time_stretch(wav, rate=1.0), wav)


@pytest.mark.parametrize("rate", [0.8, 1.25, 1.5, 2.0])
def test_time_stretch_length_is_inverse_rate(rate):
    sample_rate = 16000
    wav = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    out = time_stretch(wav, rate=rate, sample_rate=sample_rate)
    assert abs(len(out) - len(wav) / rate) <= 2
    assert np.isfinite(out).all()


def test_time_stretch_stereo_preserves_channels():
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    left = np.sin(2 * np.pi * 220 * t)
    right = np.sin(2 * np.pi * 330 * t)
    wav = np.stack([left, right], axis=1).astype(np.float32)
    out = time_stretch(wav, rate=1.5, sample_rate=sample_rate)
    assert out.ndim == 2
    assert out.shape[1] == 2
    assert abs(out.shape[0] - sample_rate / 1.5) <= 2


def test_time_stretch_rtf_sanity():
    sample_rate = 16000
    wav = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    # Warm the numba/STFT path so the RTF measurement is not first-call JIT.
    time_stretch(wav[:1600], rate=1.25, sample_rate=sample_rate)
    started = time.perf_counter()
    out = time_stretch(wav, rate=1.25, sample_rate=sample_rate)
    elapsed = time.perf_counter() - started
    assert out.size > 0
    assert elapsed < 5.0, f"time_stretch RTF is implausibly high: {elapsed:.3f}s for 1s audio"


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_time_stretch_rejects_non_positive_rate(rate):
    with pytest.raises(ValueError, match="rate"):
        time_stretch(np.ones(100, dtype=np.float32), rate=rate)


# ---------------------------------------------------------------------------
# Phase B filters: DC offset removal (Phase 12-3)
# ---------------------------------------------------------------------------

SR = 1000  # 1 sample/ms keeps the synthetic fixtures readable


def _alternating(samples: int, amplitude: float = 1.0) -> np.ndarray:
    """Zero-mean AC signal (alternating +/-) of ``samples`` samples."""
    pattern = np.array([amplitude, -amplitude], dtype=np.float32)
    return np.tile(pattern, samples // 2 + 1)[:samples].copy()


def test_remove_dc_offset_constant_signal_becomes_zero():
    wav = np.full(200, 0.5, dtype=np.float32)
    out = remove_dc_offset(wav)
    assert out.shape == wav.shape
    np.testing.assert_allclose(out, np.zeros_like(wav), atol=1e-7)


def test_remove_dc_offset_preserves_ac_content():
    ac = _alternating(400, 0.8)
    out = remove_dc_offset((ac + np.float32(0.25)).astype(np.float32))
    np.testing.assert_allclose(out, ac, rtol=0, atol=1e-6)
    assert abs(float(out.mean())) < 1e-6


def test_remove_dc_offset_stereo_is_per_channel():
    left = np.full(100, 0.3, dtype=np.float32)
    right = np.full(100, -0.2, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    out = remove_dc_offset(stereo)
    assert out.shape == stereo.shape
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-6)
    # Per-channel means, not a single global mean (which would shift both).
    np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-6)


def test_remove_dc_offset_empty_and_silence_are_unchanged():
    empty = np.zeros((0,), dtype=np.float32)
    assert remove_dc_offset(empty).shape == (0,)
    silence = np.zeros(64, dtype=np.float32)
    np.testing.assert_array_equal(remove_dc_offset(silence), silence)


# ---------------------------------------------------------------------------
# Phase B filters: lead/trail silence trim (Phase 12-3)
# ---------------------------------------------------------------------------


def _silence_audio(lead: int, active: int, trail: int, amplitude: float = 1.0) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(lead, dtype=np.float32),
            np.full(active, amplitude, dtype=np.float32),
            np.zeros(trail, dtype=np.float32),
        ]
    )


def test_trim_silence_all_silence_returns_empty():
    out = trim_silence(np.zeros(500, dtype=np.float32), SR)
    assert out.shape == (0,)


def test_trim_silence_no_silence_is_unchanged():
    wav = _alternating(500)
    out = trim_silence(wav, SR)
    np.testing.assert_array_equal(out, wav)


def test_trim_silence_removes_lead_and_trail():
    wav = _silence_audio(lead=200, active=600, trail=200)
    out = trim_silence(wav, SR, padding_ms=0)
    assert out.shape == (600,)
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_trim_silence_keeps_documented_padding():
    wav = _silence_audio(lead=200, active=600, trail=200)
    out = trim_silence(wav, SR)
    padding = int(SR * DEFAULT_TRIM_PADDING_MS / 1000)
    assert out.shape == (600 + 2 * padding,)


def test_trim_silence_stereo_preserves_shape_and_channels():
    mono = _silence_audio(lead=200, active=600, trail=200)
    stereo = np.stack([mono, -mono], axis=1)
    out = trim_silence(stereo, SR, padding_ms=0)
    assert out.shape == (600, 2)
    np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], -1.0, atol=1e-6)


def test_trim_silence_threshold_edge_uses_relative_gate():
    loud = np.full(400, 1.0, dtype=np.float32)
    # 30 dB below the loudest frame: still audible, kept.
    quiet_30 = loud * np.float32(10 ** (-30 / 20))
    kept = trim_silence(np.concatenate([loud, quiet_30]), SR, padding_ms=0)
    assert kept.shape == (800,)
    # 50 dB below: past the default 40 dB gate, trimmed.
    quiet_50 = loud * np.float32(10 ** (-50 / 20))
    trimmed = trim_silence(np.concatenate([loud, quiet_50]), SR, padding_ms=0)
    assert trimmed.shape == (400,)
    assert DEFAULT_SILENCE_THRESHOLD_DB == 40.0


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_trim_silence_rejects_non_positive_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        trim_silence(np.ones(10, dtype=np.float32), sample_rate)


def test_trim_silence_rejects_negative_padding():
    with pytest.raises(ValueError, match="padding_ms"):
        trim_silence(np.ones(10, dtype=np.float32), SR, padding_ms=-1)


# ---------------------------------------------------------------------------
# Phase B filters: internal silence (Phase 12-3)
# ---------------------------------------------------------------------------


def test_fix_internal_silence_collapses_a_long_pause():
    wav = np.concatenate(
        [
            np.ones(100, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
    )
    out = fix_internal_silence(wav, SR)
    # 100 active + 300 kept pause + 100 active.
    assert out.shape == (500,)
    np.testing.assert_allclose(out[:100], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[100:400], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[400:], 1.0, atol=1e-6)


def test_fix_internal_silence_collapses_every_long_pause():
    wav = np.concatenate(
        [
            np.ones(100, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
    )
    out = fix_internal_silence(wav, SR)
    assert out.shape == (900,)


def test_fix_internal_silence_leaves_short_pause_alone():
    wav = np.concatenate(
        [
            np.ones(100, dtype=np.float32),
            np.zeros(500, dtype=np.float32),  # below the 700 ms minimum
            np.ones(100, dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(fix_internal_silence(wav, SR), wav)


def test_fix_internal_silence_never_touches_lead_or_trail_silence():
    wav = np.concatenate(
        [
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
        ]
    )
    # Leading/trailing silence is trim_silence's job; this filter must not
    # mistake a chunk edge for an internal pause.
    np.testing.assert_array_equal(fix_internal_silence(wav, SR), wav)


def test_fix_internal_silence_all_silence_and_all_active_are_unchanged():
    silence = np.zeros(2000, dtype=np.float32)
    active = np.ones(2000, dtype=np.float32)
    np.testing.assert_array_equal(fix_internal_silence(silence, SR), silence)
    np.testing.assert_array_equal(fix_internal_silence(active, SR), active)


def test_fix_internal_silence_stereo_preserves_channels():
    mono = np.concatenate(
        [
            np.ones(100, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
    )
    stereo = np.stack([mono, -mono], axis=1)
    out = fix_internal_silence(stereo, SR)
    assert out.shape == (500, 2)
    np.testing.assert_allclose(out[:, 0], -out[:, 1], atol=1e-6)


def test_fix_internal_silence_chunk_boundary_pause_is_internal_not_trailing():
    # A stitched buffer: two chunks with a short sentence pause between them.
    # The pause is internal and shorter than the 700 ms threshold, so untouched.
    wav = np.concatenate(
        [
            np.ones(100, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(fix_internal_silence(wav, SR), wav)


def test_fix_internal_silence_rejects_max_pause_above_min_silence():
    with pytest.raises(ValueError, match="max_pause_ms"):
        fix_internal_silence(
            np.ones(100, dtype=np.float32), SR, max_pause_ms=800, min_silence_ms=700
        )


def test_filter_defaults_match_documented_thresholds():
    assert DEFAULT_SILENCE_THRESHOLD_DB == 40.0
    assert DEFAULT_TRIM_PADDING_MS == 50.0
    assert DEFAULT_INTERNAL_SILENCE_MIN_MS == 700.0
    assert DEFAULT_INTERNAL_SILENCE_MAX_MS == 300.0


# ---------------------------------------------------------------------------
# Phase B filters: composition (the serving layer's order)
# ---------------------------------------------------------------------------


def test_filters_compose_dc_then_trim_then_internal():
    """DC removal, then trim, then internal collapse, on one synthetic buffer.

    The AC content is zero-mean, so subtracting the global mean (the DC value)
    also restores the silent stretches to true silence — the realistic model of
    a per-run offset. Expected length: 300 active + 300 kept pause + 300 active
    = 900 after the 800 ms internal gap is collapsed and 200+200 outer silence
    is trimmed.
    """
    active = _alternating(300)
    base = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            active,
            np.zeros(800, dtype=np.float32),
            active,
            np.zeros(200, dtype=np.float32),
        ]
    )
    wav = (base + np.float32(0.4)).astype(np.float32)

    dc_removed = remove_dc_offset(wav)
    assert abs(float(dc_removed.mean())) < 1e-6
    trimmed = trim_silence(dc_removed, SR, padding_ms=0)
    assert trimmed.shape == (1400,)
    collapsed = fix_internal_silence(trimmed, SR)
    assert collapsed.shape == (900,)
    assert abs(float(collapsed.mean())) < 1e-6
