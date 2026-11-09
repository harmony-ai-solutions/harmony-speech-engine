# Phase 12-6: Text Splitter & Crossfade Stitcher Fixes (TTS-Server Carry-Over & Pause-Fade Bugs)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio (audit fix)
> **Depends on:** 12-1, 12-2, 12-5
> **Codebase mapping consulted:** `.planning/codebase/{ARCHITECTURE,CONVENTIONS,TESTING}.md`
> **State reference:** verified 2026-11-09 — `harmonyspeech/common/audio_utils.py` (`_CJK_END` `:140`, `_NON_VERBAL_CUE` `:146`, `_segment_plain` `:240-252`, `split_text_into_chunks` `:338-399`, `CrossfadeStitcher.feed` `:591-635`).

## Objective

Fix four bugs in `harmonyspeech/common/audio_utils.py` — two carried over from `Chatterbox-TTS-Server/utils.py` and two introduced during 12-1/12-5 — so that:

1. `CrossfadeStitcher` fades out the outgoing chunk's tail **and** fades in the incoming chunk's head across a sentence pause (instead of leaving the outgoing chunk's tail completely un-faded and crossfading digital zeros into the incoming chunk).
2. Normal parenthetical text `(like this)` and inline paralinguistic tags `[laugh]` are never split out of their surrounding sentence by the Dia-TTS carry-over regex `_NON_VERBAL_CUE`.
3. Unicode ellipsis `…` (`U+2026`) is treated as sentence-internal like ASCII `...`, never as a CJK sentence terminator.
4. Merging CJK sentences does not inject ASCII spaces that `MTLTokenizer` turns into `[SPACE]` tokens.

## False assumptions from earlier phases — DO NOT repeat

1. **False assumption (carried over from `Chatterbox-TTS-Server/utils.py:215, 985-1023`):** *"Parenthetical expressions matching `(\([\w\s'-]+\))` are atomic non-verbal TTS cues like `(laughs)` or `(whispers)` that must be split out into standalone segments."*
   - **Reality:** `Chatterbox-TTS-Server` copied that regex from a Dia TTS server. Base Chatterbox and Multilingual Chatterbox have **no** `(laughs)` tag support (passing `"(laughs)"` alone makes the model pronounce the word *"laughs"*), and `ChatterboxTurboTTS` uses **square-bracket** tags (`[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[cough]`, `[clear throat]`, `[sniff]`, `[groan]`, `[shush]` — see `Chatterbox-TTS-Server/engine.py:72-82`) **inline inside sentences**. Splitting on `(\([\w\s'-]+\))` breaks any sentence containing normal parenthetical prose — e.g. `"The Gryffindors (even the boldest ones) retracted their fangs."` — into 3 segments (`"The Gryffindors"`, `"(even the boldest ones)"`, `"retracted their fangs."`). At a chunk boundary (or when `chunk_size=None`), those fragments are synthesized as separate chunks where `punc_norm` appends a fake period `.` to `"The Gryffindors."` and the stitcher inserts a 120 ms pause mid-sentence.
2. **False assumption (12-1 `audio_utils.py:140, 367`):** *"Unicode ellipsis `…` (`U+2026`) can be grouped into `_CJK_END = r'([。！？…]+)...'`."*
   - **Reality:** `split_text_into_chunks` auto-enables `cjk_enabled` whenever `_CJK_END.search(normalized)` matches (`audio_utils.py:367`). Any English sentence containing a Unicode ellipsis `…` (e.g. `"She waited… and then spoke."`) therefore turns on CJK mode and splits the sentence right at `…` (even without trailing whitespace), contradicting `audio_utils.py:199` where ASCII `...` is explicitly kept sentence-internal.
3. **False assumption (12-1 / 12-5 `CrossfadeStitcher.feed`, `audio_utils.py:610-635`):** *"Concatenating `bridge = concat(self._carry, _silence(pause_samples))` and then calling `_crossfade_regions(bridge[-fade:], array[:fade], fade)` crossfades across the pause."*
   - **Reality:** `self._carry` has length `fade_samples` (40 ms by default) and `pause_samples` is 120 ms by default. Thus `bridge` is `[40 ms of un-faded Chunk i tail] + [120 ms of 0.0 silence]`. `bridge[:-fade]` emits the 40 ms of Chunk $i$'s tail **completely un-faded**, followed by a hard step to `0.0` for 80 ms, while `_crossfade_regions(bridge[-fade:], array[:fade], fade)` crossfades **the last 40 ms of digital zeros (`0.0`)** into Chunk $i+1$'s head (`0.0 * cos²(t) + array[:fade] * sin²(t)`). Chunk $i$'s tail never gets a fade-out at all! Compare `Chatterbox-TTS-Server/server.py:1217-1226`, which fades Chunk $i$'s tail **into** silence (`chunk_a -> silence`) and then fades silence **into** Chunk $i+1$'s head (`silence -> chunk_b`).

## Implementation steps

1. **Remove parenthetical `_NON_VERBAL_CUE` splitting (`harmonyspeech/common/audio_utils.py`):**
   - In `_segment_plain(text, cjk_enabled)` (`audio_utils.py:240-252`) and `split_text_into_chunks` (`audio_utils.py:384`), stop splitting on `_NON_VERBAL_CUE`. Pass `text.strip()` directly to `_split_text_by_punctuation(text.strip(), cjk_enabled)` so parenthetical prose `(even the boldest ones)` and inline square-bracket tags `[gasp]` remain inside their sentence.
   - Replace the old "non-verbal cue is isolated into its own segment" unit tests in `tests/unit/common/test_audio_utils.py` with regression tests proving that:
     - parenthetical prose inside a sentence stays in the same sentence chunk (`"The Gryffindors (even the boldest ones) retracted their fangs. Next sentence."` → 2 sentences, not 4);
     - inline Turbo tags (`"[gasp] Okay, we have fifteen minutes left! [sigh] Let's think."`) stay attached to their sentences.

2. **Fix `_CJK_END` and CJK segment joining (`harmonyspeech/common/audio_utils.py`):**
   - Remove `…` from `_CJK_END` (`_CJK_END = re.compile(r"([。！？]+)([」』）】》〉”’\"']*)")`), and in `_sentence_boundaries` also treat `…` like ASCII `...` (never a sentence boundary).
   - Add a helper `_join_pieces(left: str, right: str) -> str`: when `left` ends with a CJK character or CJK fullwidth punctuation (`\u3000-\u303f`, `\u3040-\u30ff`, `\u4e00-\u9fff`, `\uff00-\uffef`), join as `f"{left}{right}"` without inserting an ASCII space; otherwise join as `f"{left} {right}"`. Use `_join_pieces` in `_split_oversize` (`:332`) and `split_text_into_chunks` (`:392`).

3. **Fix `CrossfadeStitcher.feed()` when `pause_samples > 0` (`harmonyspeech/common/audio_utils.py:591-635`):**
   - Keep the zero-pause path (`pause_samples == 0`) unchanged: direct equal-power overlap `_crossfade_regions(self._carry[-fade:], array[:fade], fade)` of length `len(a) + len(b) - fade`.
   - When `pause_samples > 0` and `fade_samples > 0`:
     - Fade out the held-back tail `self._carry` into silence using the equal-power fade-out curve `cos²(t)` (`_crossfade_regions(self._carry[-fade_out_len:], zeros, fade_out_len)`),
     - Insert the audible silence gap,
     - Fade in `array[:fade_in_len]` from silence using the equal-power fade-in curve `sin²(t)` (`_crossfade_regions(zeros, array[:fade_in_len], fade_in_len)`),
     - Preserve the exact total length contract `sum(len(conditioned_chunk)) + pause_samples * (N - 1) - fade_samples * (N - 1)` (or document and pin the exact sample count across whole-buffer `crossfade_stitch` and streaming `CrossfadeStitcher.feed` + `flush`) so both chunk edges are smoothly tapered with zero step discontinuity at either side of the silence gap.
   - Verify that `crossfade_stitch(chunks, ...)` and streaming `feed()` + `flush()` remain sample-for-sample identical to each other.

4. **Update unit & integration tests:**
   - `tests/unit/common/test_audio_utils.py`:
     - Add a test proving that when `crossfade_ms > 0` and `sentence_pause_ms > 0` (e.g. 40 ms / 120 ms) on a constant-value chunk `np.ones(...)`, the outgoing chunk's tail smoothly tapers to `0.0` before the silence gap (no hard step from `1.0` to `0.0`) AND the incoming chunk's head smoothly tapers up from `0.0`.
     - Add tests for parenthetical prose staying inside its sentence, inline `[laugh]`/`[gasp]` tags staying inside their sentence, Unicode `…` in English prose not splitting sentences, and CJK sentence merging without ASCII space injection.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | Fix `CrossfadeStitcher.feed()` pause fade-out/fade-in, remove `_NON_VERBAL_CUE` splitting, remove `…` from `_CJK_END`, CJK-aware piece joining |
| `tests/unit/common/test_audio_utils.py` | Add pause fade-out taper assertion, parenthetical prose & `[tag]` tests, Unicode `…` test, CJK join test |
| `tests/integration/test_tts_chunking.py` | Verify stitch length & join tests pass with the fixed pause fade |

## Tests / validation

- Unit suite (`tests/unit/common/test_audio_utils.py`) and integration chunking suite (`tests/integration/test_tts_chunking.py`) green.
- Whole-buffer `crossfade_stitch` and streaming `CrossfadeStitcher` (`feed` + `flush`) produce bit-identical arrays across all `crossfade_ms` × `sentence_pause_ms` combinations.

## Checklist

- [x] `CrossfadeStitcher.feed()` fades out Chunk $i$ into silence and fades in Chunk $i+1$ from silence when `sentence_pause_ms > 0`
- [x] `_NON_VERBAL_CUE` parenthetical splitting removed; parenthetical prose and `[tag]` cues stay in-sentence
- [x] Unicode ellipsis `…` removed from `_CJK_END`; CJK chunk merging does not insert ASCII spaces
- [x] Unit and integration tests updated and green; `summary.md` updated

## Verification log

Dated 2026-11-09 (Phase 12-6 implementation).

**Verified**

- `_NON_VERBAL_CUE` regex and all cue-splitting branches removed; `_segment_plain`
  now hands the whole (stripped) text to `_split_text_by_punctuation`. Parenthetical
  prose and inline `[gasp]`/`[laugh]`/`[sigh]` tags stay inside their sentence.
- `_CJK_END` no longer contains `…` (now `([。！？]+)(...)`), so a Latin sentence
  containing `U+2026` neither auto-enables CJK mode nor splits at the ellipsis.
  `…` stays sentence-internal in CJK too (only `。！？` end a sentence).
- Added `_join_pieces` / `_join_len`: pieces are joined without an ASCII space when
  the left piece ends in CJK/fullwidth text (`\u3000-\u303f`, `\u3040-\u30ff`,
  `\u4e00-\u9fff`, `\uff00-\uffef`), otherwise with `" "`. Both `_split_oversize`
  and `split_text_into_chunks` use the pair, so CJK chunks stay space-free.
- `CrossfadeStitcher.feed()` fades `self._carry` out to silence with
  `_crossfade_regions(self._carry[-fade_out:], _silence(fade_out), fade_out)` before
  appending the pause when `pause_samples > 0`. `bridge.shape[0]` is unchanged
  (`len(carry) + pause`), so the existing `bridge[-fade:]`-from-silence fade-in and
  the length invariant `sum(len) + pause*(N-1) - fade*(N-1)` are preserved exactly.
- `tests/unit/common/test_audio_utils.py` (104 tests) and
  `tests/integration/test_tts_chunking.py` (37 tests): **141 passed**
  (`... -m pytest tests/unit/common/test_audio_utils.py tests/integration/test_tts_chunking.py`).
  New regressions cover parenthetical prose, inline tags, `…` in English, CJK
  merge without ASCII spaces (both sentence merge and oversize clause merge), and
  the paused join's monotonic `cos²` fade-out / `sin²` fade-in plus
  whole-buffer/streaming sample equality.

**Deviations / notes**

- The GitNexus index was 10 commits behind HEAD and predates `audio_utils.py`
  (Phase 12-1), so `gitnexus_impact` could not resolve these symbols; blast radius
  was assessed by grep instead. Public callers of the changed functions are
  `serving_text_to_speech.py` (splitter + stitcher) and the two benchmark scripts
  (splitter); every other reference is module-internal or a test.
- The fade-in remains the pre-existing `_crossfade_regions(bridge[-fade:], ...)`
  step (where `bridge[-fade:]` is silence for the default `pause >= fade`). This
  is what keeps the exact sample count; no separate fade-in length is introduced.
