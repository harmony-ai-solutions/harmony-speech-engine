# Phase 12-1: Audio Utilities + Protocol Fields (long-form core)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio (chunking, stitching, Phase A+B filters)
> **Depends on:** nothing (first Phase 12 task); supersedes the March plan's 1-1/2-1 sub-task files (`.current_work/chatterbox-audio-enhancements/`)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; upstream prior art read 2026-11-08 (see below)
> **State reference:** verified 2026-11-08 — `speed` and `norm_loudness` already exist (`endpoints/openai/protocol.py:59,91`, `common/inputs.py:314,325`, `speed_function` computed at `common/inputs.py:350-362`); **no** `audio_utils` module, **no** chunking/crossfade fields anywhere; serving path returns complete JSON per request (commented-out `stream` scaffolding only).

## Objective

Model-agnostic audio/text utilities + opt-in request fields: a sentence-aware text splitter (with every upstream-discovered edge case encoded as a day-one test), an equal-power crossfade stitcher with pause insertion, peak normalization, and a phase-preserving time-stretch for model types without native `speed`. Pure functions, no engine changes — 12-2 wires them.

## Owner decisions (2026-11-08)

- Phase A **and** Phase B filters are in scope for Phase 12 (B lands as 12-3).
- **`norm_loudness` keeps its name** — the March plan's rename to `normalize_audio` is dropped (wire-stable; the field's meaning already matches).

## Upstream prior art (read, do not rediscover — `.current_work/Chatterbox-TTS-Server` @ `915ae28`, MIT)

Their `utils.py` sentence splitter is the reference: abbreviation set (`:125`), `NUMBER_DOT_NUMBER` (`:199`), `VERSION` (`:202`), `POTENTIAL_END` = `([.!?])(["']?)(\s+|$)` (`:206`), bullet pattern (`:213`), non-verbal cues like `(whispers)` (`:215`), `_is_valid_sentence_end` (`:847`), `_split_text_by_punctuation` (`:882`). Their crossfade lives in `server.py:255+`: **equal-power curves** (`cos²/sin²` over the fade, perceptually constant loudness) and overlap-summing (`len(out) = len(a) + len(b) − fade_samples`).

**Two bugfixes they needed after shipping — our day-one regression tests:**

1. `f0afcc6` (issue #144): stray narrative dashes were parsed as bullets. Fix: the bullet regex requires a **space/tab after the bullet char** — `(?:^|\n)[ \t]*([-•*]|\d+\.)[ \t]+` — so `-\n\nNext paragraph` is not a bullet.
2. `450de15` (issue #144): narrative dashes caused false sentence splits — read the commit diff and encode its cases (em-dash `—` is sentence-internal, not a boundary).

**Ours beyond upstream:** CJK sentence enders `。！？` for `ChatterboxMultilingualTTS` and melo CJK voices — upstream is EN-only.

> ⚠️ **Post-implementation audit corrections (2026-11-09 — see [12-6-TextSplitterAndStitcherFixes.md](12-6-TextSplitterAndStitcherFixes.md)):**
> Four flaws in the 12-1 port (two carried over from `Chatterbox-TTS-Server`, two introduced here) were identified during code review and **must be fixed in 12-6**:
> 1. **Carried-over design flaw — `NON_VERBAL_CUE_PATTERN = r"(\([\w\s'-]+\))"`:** `Chatterbox-TTS-Server/utils.py` inherited this regex from a Dia TTS server. Splitting on it breaks any sentence containing normal parenthetical prose `(like this)` into three separate segments, causing mid-sentence chunk splits, fake periods via `punc_norm`, and artificial pauses. Chatterbox does not use `(laughs)` in parentheses at all — `ChatterboxTurboTTS` uses square-bracket tags (`[laugh]`, `[chuckle]`, `[gasp]`, `[sigh]`) **inline** inside sentences. Do **not** split on parenthetical cues.
> 2. **Unicode ellipsis `…` (`U+2026`) in `_CJK_END`:** Including `…` in `_CJK_END` caused any English/Latin sentence containing `…` to auto-enable `cjk_enabled = True` and split mid-sentence at `…` (contradicting the ASCII `...` rule). `…` is sentence-internal, not a CJK sentence ender.
> 3. **CJK chunk merge separator:** Merging CJK sentences with `f"{current} {piece}"` inserts ASCII spaces that `MTLTokenizer.encode` turns into `[SPACE]` tokens.
> 4. **`CrossfadeStitcher.feed()` broken fade-out when `sentence_pause_ms > 0`:** Concatenating `self._carry` (un-faded tail of Chunk $i$) with `_silence(pause_samples)` into `bridge` and then only crossfading `bridge[-fade:]` with `array[:fade]` means `bridge[-fade:]` is **digital silence (`0.0`)** whenever `pause_samples >= fade_samples`! Chunk $i$'s tail (`self._carry`) is emitted in `bridge[:-fade]` **completely un-faded** (hard-cutting to `0.0`), and only Chunk $i+1$'s head is faded in against zeros. Contrast `Chatterbox-TTS-Server/server.py:1217-1226`, which fades Chunk $i$ **into** silence and then fades silence **into** Chunk $i+1$. See [12-6](12-6-TextSplitterAndStitcherFixes.md) for the fix.

## Implementation steps

1. `harmonyspeech/common/audio_utils.py` (NEW): `split_text_into_chunks(text, chunk_size=None, language=None)` — sentence-first, sentence-aware merge into `chunk_size` bounds (character budget), never splitting mid-sentence except an oversize-sentence fallback (split on commas/semicolons, then hard-wrap as last resort); `crossfade_stitch(chunks, crossfade_ms, sentence_pause_ms, sample_rate)` — equal-power overlap crossfade + optional inter-chunk silence; `peak_normalize(wav, ceiling)`; `time_stretch(wav, rate, sample_rate)` — phase-preserving (librosa is already an engine dependency, used by the VC decode path; keep CPU-side numpy/torch only elsewhere).
2. Protocol fields (opt-in, defaults preserve byte-identical behavior): `split_text: bool = False`, `chunk_size: int | None = None`, `crossfade_ms: int | None = None`, `sentence_pause_ms: int | None = None` in `GenerationOptions` (`protocol.py`) + `common/inputs.py` + `task_handler/inputs.py` pass-through. `norm_loudness` untouched.
3. Unit tests `tests/unit/common/test_audio_utils.py`: splitter (happy path; abbreviations `Mr./Dr./e.g./i.e./etc.`; `3.14`, `v1.2.3`; quotes before whitespace; bullets incl. numbered; **dash-as-bullet regression**; **narrative-dash regression**; non-verbal cues; CJK `。！？`; oversize-sentence fallback; empty/whitespace input), stitcher (equal-power math vs hand-computed fixture; length invariant `a+b−fade`; pause insertion; mono/stereo; edge `fade ≥ chunk length`), normalize, stretch (RTF sanity, length ∝ 1/rate).
4. Docs: `docs/api.md` request-field reference (fields are inert until 12-2 wires them — document as "reserved, effective from 12-2" or land docs with 12-2; pick one, stay honest).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | NEW — splitter, stitcher, normalize, stretch |
| `harmonyspeech/endpoints/openai/protocol.py`, `harmonyspeech/common/inputs.py`, `harmonyspeech/task_handler/inputs.py` | opt-in fields (no behavior change yet) |
| `tests/unit/common/test_audio_utils.py` | NEW — incl. both upstream bugfix regressions + CJK |
| `CHANGELOG.md` | entry with 12-2 (single user-visible line for the feature) |

## Tests / validation

- Full `tests/unit` + `tests/integration` + `tests/e2e/wire_contract` green (fields default-off ⇒ wire contracts unchanged).
- Splitter/stitcher property checks seeded and deterministic.

## Checklist

- [x] `audio_utils.py` landed (splitter + stitcher + normalize + stretch)
- [x] Both upstream bugfix regressions + CJK cases pinned green
- [x] Opt-in protocol fields landed, default-off, wire contracts unchanged
- [ ] Suites green; `summary.md` ticked

## Deviations / verification log

Dated 2026-11-09 (Phase 12-1 implementation).

**Verified**

- `speed`/`norm_loudness` exist at `protocol.py:59,91`, `common/inputs.py:314,325`;
  `speed_function` at `common/inputs.py:350-362`. No `audio_utils.py` and no
  chunk/crossfade/pause fields existed before this task.
- `serving_text_to_speech.py` returns one complete `TextToSpeechResponse`;
  `stream: true` is not commented-out scaffolding but an active HTTP-400
  ("Stream output is not yet supported") plus a TODO (`:42-46`). State reference
  slightly stale, intent confirmed (no streaming path).
- Upstream `f0afcc6` (space/tab after bullet char; repro `-Within ...-\n\n`) and
  `450de15` (column-0 bullets, 2+ items; repro `  ---TITLE---`, `...lewd)-\n\n`)
  read and encoded as regression tests; `utils.py:125,199,202,206,213,215,847,882`
  and `server.py:255+` ported (abbreviation/number/version guards, equal-power
  `cos²/sin²`, `a+b-fade` overlap invariant). Em/en dashes are sentence-internal.
- Purity: `audio_utils.py` imports only `re` + numpy at load time; librosa is
  lazy inside `time_stretch` (never imported for `rate == 1.0`). Pre-existing
  caveat: `harmonyspeech/__init__.py` imports `HarmonySpeechEngine`, so any
  `harmonyspeech.*` import loads torch; the module itself has no engine imports.
  librosa 0.11.0 is a hard engine dependency (OpenVoice mel/VC decode,
  Chatterbox `prepare_conditionals`, HarmonySpeech mel, MeloTTS, VoiceFixer,
  `model_runner_base.py:3029`). Its `time_stretch` length is `N/rate` and it
  treats the **last** axis as time, hence the transpose for `(samples, channels)`.
- CJK `…` included as an ender (spec left it optional); runs collapse to one
  boundary.

**Deviations**

- `task_handler/inputs.py` needed **no code change**: `from_openai` passes the
  pydantic `GenerationOptions` object through whole and `prepare_*` reads only
  named fields, so the new fields already reach every prepare function and the
  serving layer. Adding them to per-family return tuples would change every
  runner contract (12-2 needs no runner changes). Recorded, not missing.
- No extra pydantic constraints (`ge=0`): `protocol.py` has no validator
  convention; `audio_utils` rejects negative timings/budgets at wiring time.
- `chunk_size=None` is policy-free (one sentence per chunk); 12-2 owns the
  default budget. Bullet items are sentence-split with the marker on the first
  sentence (upstream keeps items atomic); issue #144 detection is unaffected.
- CHANGELOG `Unreleased` house style is undated bold lines (no dated entries),
  so no date was added; the entry is honest (fields reserved/inert until 12-2).
  Docs landed now and mark the fields **Reserved**. `summary.md` untouched per
  task scope, hence the fourth checklist box stays unticked (suites are green).
- Extra `tests/unit/wire_contract/test_long_form_fields.py` (protocol defaults +
  pass-through) beyond the spec's unit file, matching the repo's wire suite.

**API notes for 12-2**

- `split_text_into_chunks(text, chunk_size=None, language=None)`: blank → `[]`;
  sentence-per-chunk when `chunk_size=None`; CJK auto-detect plus `zh/ja/ko`
  hint; cues atomic; oversize → clause split, then hard wrap.
- `CrossfadeStitcher(crossfade_ms, sentence_pause_ms, sample_rate)`:
  `feed(chunk) -> np.ndarray` returns final samples, withholds at most one fade
  of the un-crossfaded tail; `flush()` returns the tail. `crossfade_stitch` is a
  thin fold over it; one chunk round-trips byte-identically.
- Waveforms: numpy `(samples,)` or `(samples, channels)`; `AudioArtifact.numpy`
  is 1-D. Pause goes before the fade: audible gap = `pause - crossfade` (≥ 0),
  no trailing pause. `peak_normalize` is true normalization (raises quiet audio);
  `time_stretch` is lazy librosa phase-vocoder, `rate > 1` = faster.

**Full-suite result:** `1386 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, 2026-11-09; baseline 1320 + 66 new tests).
