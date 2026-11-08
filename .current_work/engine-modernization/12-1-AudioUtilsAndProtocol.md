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

**Ours beyond upstream:** CJK sentence enders `。！？` (+ `…`?) for `ChatterboxMultilingualTTS` and melo CJK voices — upstream is EN-only.

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

- [ ] `audio_utils.py` landed (splitter + stitcher + normalize + stretch)
- [ ] Both upstream bugfix regressions + CJK cases pinned green
- [ ] Opt-in protocol fields landed, default-off, wire contracts unchanged
- [ ] Suites green; `summary.md` ticked
