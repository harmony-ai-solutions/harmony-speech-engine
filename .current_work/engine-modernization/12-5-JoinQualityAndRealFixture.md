# Phase 12-5: Join-Quality Tuning + Real-Text Benchmark Fixture

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio (listening-gate follow-up)
> **Depends on:** 12-1..12-4 landed (uncommitted in tree); owner listening feedback 2026-11-09
> **State reference:** verified 2026-11-09 — unset `crossfade_ms`/`sentence_pause_ms` default to **0/0** (`serving_text_to_speech.py:275-276`) — default chunked joins are hard butt-splices; the stitcher does no per-chunk edge conditioning, so joins carry uncontrolled synthesis-native leading/trailing silence; listening samples used explicit 20 ms/80 ms (effective gap 60 ms + ragged edge silence).

## Owner listening verdict (2026-11-09)

1. **Chatterbox chunked length is CORRECT behavior**: unchunked long-form "loses it towards the end and blabbles before breaking up" — the OFF baseline is degenerate on long text (also explains OFF audio being shorter and multilingual's variable 6–29 s outputs). Record: chunking is a *quality/coherence* win for AR long-form, not only throughput. No code change — dated notes in 12-4 log + throughput.md method.
2. **Joins stutter** ("small hiccup like silence stuttering where clips get connected") — crossfade needs fine tuning: fix below.
3. **Word-salad fixtures are unlistenable** — benchmarks must use real sentences; the owner supplied an extract from *The Villainess Malfoy* (below) as the long-form fixture text.

## The Villain Malfoy fixture text (owner-provided, 2026-11-09)

Normalize hard line-wraps to single spaces (the paste wraps mid-sentence); keep em-dashes, quotes and the apostrophe exactly — the text deliberately exercises the em-dash narrative rule (450de15) and multi-sentence structure:

> Always destined to stand out, and always destined to shine the brightest—her name is Carina Malfoy. One could easily call her the true protagonist of today's Hogwarts. Even the Gryffindors, who normally bare their teeth at the mere sight of Slytherin, momentarily retract their fangs whenever she appears. When she offers an angelic smile, even the most notoriously difficult professors bend over backward to accommodate her. She doesn't even need to lift a finger. Hogwarts is crawling with fools desperate to touch so much as the tip of a Malfoy's sleeve. Her sycophants flock around her like decorative leaves adorn a flower, while ordinary pests hover about, hoping to catch even a passing whiff of her scent. Tiresome chores are handed off to the pests, and filthy matters are swept away by the leaves.

## Implementation steps

1. **Join-quality fix (`harmonyspeech/common/audio_utils.py`)** — per-chunk **edge conditioning** inside the stitcher: before crossfading, conservatively trim each chunk's leading/trailing near-silence (energy-gated: threshold relative to the chunk's own peak, e.g. −45 dBFS; bounded max trim, e.g. 150 ms per edge; keep a small guard pad, e.g. 10 ms, so onsets stay natural). This is stitcher-internal and INDEPENDENT of the user-facing `trim_silence` filter (document the interplay: edge conditioning normalizes joins; the filter is an explicit whole-output option). All joins then become speech-end → equal-power fade → optional pause → fade → speech-start, uniform by construction.
2. **Tuned defaults** when `split_text` is on and the fields are unset: `crossfade_ms` default ~40 ms (fade over actual speech, not silence) and `sentence_pause_ms` default ~120 ms (natural inter-sentence gap). Justify with a join-gap uniformity test (synthetic chunks carrying padded silence: joined gap ≈ pause setting exactly, independent of the padded silence length) and, if cheap, measured edge-silence stats from a real model render. Explicit fields still win; validation and wire docs updated.
3. **Real-text fixture (`scripts/bench_throughput.py`)**: replace the word-index generator for `--scenario longform` with the normalized Malfoy text (a module constant — even more deterministic than arithmetic generation); update the pinned tests (SHA-256, chunk count at the 300-char budget — assert whatever it actually is, and note it in the fixture report); `--print-longform-scenarios` stays truthful. The audio_regression axes keep their synthetic literals (numeric purpose, never listened to) — record that decision in the log.
4. **Length invariant update**: edge conditioning changes stitch math (result length = Σ trimmed chunks − fades + pauses); update the 12-1 length-invariant and streaming-equivalence tests, and the 12-2 byte-identity gates stay green (single-chunk path untouched).
5. **Tests**: edge-trim unit cases (no silence, all-silence chunk → honest fallback: no trim of an all-quiet chunk beyond the bound; stereo; guard pad), join-gap uniformity, new defaults pinned (incl. wire-contract default-off behavior unchanged for non-chunked requests), fixture SHA/chunks re-pinned, existing chunking integration suite green.
6. **Docs**: `docs/api.md` (new defaults + edge conditioning described honestly), CHANGELOG entry update, `docs/benchmarks/throughput.md` §Phase 12 method note (fixture switched to real prose 2026-11-09, owner decision; matrix re-measured), 12-4 log: the three owner verdict bullets verbatim-ish.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | edge conditioning + default constants |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | default crossfade/pause when chunked |
| `scripts/bench_throughput.py` | Malfoy fixture constant + fixture report fields |
| `tests/unit/common/test_audio_utils.py`, `tests/unit/benchmarks/test_bench_longform_scenarios.py`, integration chunking tests | re-pinned + new gates |
| `docs/api.md`, `CHANGELOG.md`, `docs/benchmarks/throughput.md`, 12-4/12-5 task files | honest docs + dated verdicts |

## Checklist

- [x] Edge conditioning landed; join-gap uniformity pinned
- [x] Defaults tuned (40/120) + validated; single-chunk byte-identity green
- [x] Real-text fixture landed; pinned tests updated; scenarios truthful
- [x] Docs/CHANGELOG updated; full suite green (1512 passed, 5 skipped)
- [ ] `summary.md` ticked (after owner re-listen)

## Deviations / verification log

Dated 2026-11-09 (Phase 12-5 code half; owner re-listen and host matrix re-run are
the orchestrator's follow-up).

**Edge conditioning semantics (implemented, `harmonyspeech/common/audio_utils.py`)**

- `condition_edges(wav, sample_rate)` runs inside `CrossfadeStitcher.feed` before
  a chunk is split into carry/crossfade audio, so the carry always holds
  conditioned samples (Phase 14 streaming semantics are unchanged).
- Gate: per-sample magnitude (channel **maximum**, so one live channel keeps an
  edge alive in stereo) relative to that chunk's **own** sample peak; silent
  when more than **−45 dBFS** below the peak (`DEFAULT_EDGE_SILENCE_THRESHOLD_DB
  = 45.0`). A per-sample crossing was chosen over a frame gate because the
  stitcher is O(n) pure numpy and the refinement is exact; the conservatism
  (keeping a low noise floor) is intentional.
- Bound: at most **150 ms** removed per edge (`DEFAULT_EDGE_MAX_TRIM_MS`), with
  **10 ms** kept (`DEFAULT_EDGE_GUARD_MS`). A 1 s pad loses 150 ms, not 1 s.
- All-digital-silence chunks (peak 0) are returned unchanged: no relative gate
  exists, and deleting a chunk would make the join undefined. Chunks with no
  near-silence are returned as an unchanged copy. Mono and `(samples, channels)`
  are supported; one sample range is applied to all channels. numpy-only, no new
  dependencies.
- DC note: a chunk whose DC offset exceeds ~0.56 % of its peak can defeat the
  magnitude gate (the offset sits above the relative floor). The per-chunk
  `remove_dc_offset` filter (12-3) removes it when a family needs it; the
  conditioning is deliberately conservative rather than adding a high-pass.
- Interplay with `trim_silence` (documented in `docs/api.md`): conditioning is
  stitcher-internal and normalises joins (and bounds the first/last chunk's
  outer edges to the guard); `trim_silence` remains an explicit whole-output
  option. On chunked output the 10 ms guard is shorter than the filter's 32 ms
  analysis frame, so `trim_silence` usually has nothing to remove at the outer
  edges. The integration test that used to prove "trim runs on the final buffer"
  was rewritten to pin this interaction (the original discriminator is no longer
  observable once every chunk edge is conditioned).

**Tuned defaults (implemented)**

- `DEFAULT_CHUNK_CROSSFADE_MS = 40`, `DEFAULT_CHUNK_SENTENCE_PAUSE_MS = 120`
  live in `audio_utils` (the canonical tuning constants). The serving layer
  resolves them in `_resolve_join_timing(options)`, called only from
  `_create_chunked_response` — i.e. only when chunking produced 2+ chunks.
- Unset (`None`) gets the defaults; an explicit value **including 0** wins.
  Wire defaults stay `None` (`GenerationOptions`), so non-chunked requests and
  the 0/1-chunk fast path are byte-identical to before. Validation is unchanged
  (`>= 0`).
- Whole-buffer `crossfade_stitch` returns a single non-empty chunk
  **byte-identically** (no join → no conditioning, no fade/pause), preserving
  the 12-2 single-chunk gate. The streaming `CrossfadeStitcher` conditions every
  fed chunk (a streaming consumer cannot know whether a join is coming); this is
  stated in the class docstring for Phase 14.

**Join-gap uniformity (the owner's stutter regression)**

- `test_join_gap_is_uniform_across_ragged_padded_silence` stitches synthetic
  chunks whose leading/trailing pads vary (20–140 ms) and asserts a single
  measured inter-speech gap for every combination. Closed form:
  `gap = pause_ms - fade_ms + 2 * guard` (with fade 0 that is `pause + 2 *
  guard`), because the existing stitcher inserts the pause *before* the fade and
  the fade consumes part of it. The gap is independent of the pad lengths — the
  regression the owner heard. Tolerance is exact because the fixtures are
  sample-exact.

**Length invariants / existing tests updated**

- Stitcher and `crossfade_stitch` docstrings now state
  `sum(len(conditioned_chunk)) + pause * (N-1) - fade * (N-1)`.
- Added length-invariant and streaming-equivalence cases with padded chunks
  (conditioned result equals the streaming fold).
- Integration tests that pinned exact stitch math now pass **explicit**
  `crossfade_ms=0`/`sentence_pause_ms=0` where they mean "no fade/pause"
  (the fade now defaults to 40 ms and consumes part of a short pause; a chunk
  shorter than the fade is clamped). Affected: crossfade-shortens, ordered-stitch,
  kitten lane pairing, native/non-native speed ratio, per-family duration growth.
- `test_per_family_chunked_wav_and_duration_growth[harmonyspeech]` changed
  direction: that pipeline appends a **150 ms silence break after every chunk**,
  so conditioning (150 ms → 10 ms guard) removes more than the 50 ms pause adds.
  The test now asserts that normalisation direction for that family and keeps the
  growth assertion for the others. This is a real audible improvement worth the
  orchestrator's listening pass.
- Added `test_unset_join_fields_use_the_tuned_40_120_defaults` (full serving path:
  length = sum(chunks) + 120 ms − 40 ms) and a wire-contract unit test for
  `_resolve_join_timing` (defaults, explicit 0, mixed fields).

**Real-text fixture (`scripts/bench_throughput.py`)**

- Replaced the word-index generator (pools + `_longform_sentence`) with
  `LONGFORM_FIXTURE_TEXT`: the owner's *The Villainess Malfoy* extract, hard
  line-wraps normalised to single spaces, em-dashes/quotes/apostrophes kept
  exactly. Verified byte-equal to the task file's text.
- **807 characters, 8 sentences, 4 chunks** at the 300-character budget.
  SHA-256 `a5595f416783b1c74439e2b7bdae2d1d37b0e76ed10bf20674ad92ef9632f185`
  (pinned). All six families share the same text; the pinned tests now assert
  the shared literal, SHA, char/sentence counts and chunk count.
- Fixture report gains `fixture: "real_prose"` and the `source` string; the
  markdown block prints `fixture: real prose (…)` instead of a misleading
  "fixture seed", and renders `server default (40 ms/120 ms)` when the report has
  no explicit join timing. `--print-longform-scenarios` now states the fixture
  and the defaults.
- `LONGFORM_SEED` is retained only for report continuity (it no longer
  influences the text).
- **Deviation:** `build_longform_payload` now omits `crossfade_ms` /
  `sentence_pause_ms` when they are `None` and the CLI defaults for
  `--longform-crossfade-ms`/`--longform-sentence-pause-ms` are `None`. Previously
  the harness always sent explicit 0/0, which would have measured butt splices
  and contradicted the shipped defaults. Explicit probe flags (H3 uses 50/50)
  still send their value. The orchestrator's matrix re-run therefore measures
  the shipped 40/120 + conditioning; the existing measured rows are pre-12-5 and
  flagged as such in `docs/benchmarks/throughput.md`.
- `scripts/audio_regression.py` axis literals stay synthetic (numeric
  identity/determinism purpose, never listened to) — owner verdict 3, decision
  recorded here and in the throughput method note.

**Docs**

- `docs/api.md`: field table defaults updated to 40/120 ms and a new "Join
  quality (Phase 12-5)" paragraph covering the gate/bound/guard, the
  `trim_silence` interplay and the unset-vs-explicit rule.
- `CHANGELOG.md`: long-form feature bullet amended; new Fixes bullet for the
  join stutter.
- `docs/benchmarks/throughput.md` §Phase 12: fixture and variants rewritten for
  real prose + defaults, with a dated note that the matrix rows below predate
  12-5 and must be re-measured.

**Measured edge-silence statistics**

- Not captured here: this session is CPU-only with no engine server or model
  render, per the task guardrails. The thresholds/bounds come from the design
  brief; the orchestrator can capture per-chunk edge-silence distributions from
  the re-rendered listening samples. The synthetic tests pin the semantics
  (exact guard/bound/gap behaviour), so a later tuning pass can change the
  constants without changing the contract.

**Full-suite result**

- `1512 passed, 5 skipped` (`tests/unit tests/integration tests/e2e/wire_contract`,
  2026-11-09; baseline 1495 + 17 new tests: 12 edge-conditioning/join-gap unit
  cases (one parametrised ×2), 3 bench-fixture/report pins, 1 wire
  default-resolution pin and 1 end-to-end tuned-default integration test; two
  existing integration tests were rewritten in place, none removed).

---

Dated 2026-11-09 (Phase 12-5 text-fidelity follow-up; per-model default chunk
budget). Host investigation of the owner report *"Chatterbox Turbo output isn't
true to the text — missing a word or two, or saying something slightly
different"*.

**Evidence (Whisper large-v3-turbo, CPU int8, beam 5; real-prose 138-word
fixture; edit distance vs fixture; "Karina/Carina"-style spelling noise ≈ 2)**

| Model | Mode | Edit distance | Notes |
|---|---|---|---|
| chatterbox (base) | unchunked | ~7 | loses the final clause ("loses it towards the end") |
| chatterbox (base) | chunked @300 | **2** | perfect (transcription noise only) |
| chatterbox_turbo | unchunked | **~59** | skips the whole middle + hallucinates ("balfolst", "shvokertat") |
| chatterbox_turbo | chunked @300 | **15–23** (3 renders) | consistently drops ~15–20 words |
| chatterbox_turbo | chunked @150 | **2–3** (2 renders) | faithful |
| chatterbox_turbo | problem sentences SOLO | **0** (2 sentences) | the "dropped" sentences are spoken perfectly alone |
| chatterbox_multilingual | unchunked | **140** (2 words heard) | total breakdown on long English prose |
| chatterbox_multilingual | chunked @300 | **63** → **67** (follow-up 2) | only 73/138 words spoken; partial rescue |
| melo / openvoice_v2 | both modes | **1–2** | perfect |

**Length-artifact resolution.** Turbo's earlier "+35 % audio length when
chunked" was an artifact of comparing against a word-DROPPING unchunked
baseline. Per-word pacing is equal (2.13 MB / 137 words vs 1.48 MB / 88 words),
so **no pacing problem exists**; the length difference was missing words.

**Fix (implemented, `serving_text_to_speech.py`).** A per-model-type default
chunk budget resolved next to `DEFAULT_CHUNK_SIZE`:

- `DEFAULT_CHUNK_SIZE_TURBO = 150`;
  `DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE = {"ChatterboxTurboTTS": 150}` (structured
  and pinned like `NATIVE_SPEED_MODEL_TYPES`).
- `_request_model_types()` resolves the request's configured family from the
  pipeline spec (sentinel ids) **and** a concrete `config.name` lookup (direct
  executor names); `_resolve_default_chunk_size()` returns the override when
  one matches, else the global `DEFAULT_CHUNK_SIZE` (300).
- `_resolve_chunks()` uses that only when `split_text` is on and `chunk_size`
  is unset. **An explicit `chunk_size` always wins, for every model.** Unknown
  models and a bare serving object fall back to 300.
- Deliberately unchanged: base Chatterbox keeps 300 (faithful at 300; fewer
  chunks = better throughput). Multilingual keeps 300; its long-form English is
  weak in both modes (recommend the base model — owner-facing caveat in
  `docs/models.md`/`docs/benchmarks/throughput.md`). Melo/OV2 unaffected.

**Bench fixture (`scripts/bench_throughput.py`).** ON payloads now omit
`chunk_size` unless `--longform-chunk-size` is given, so the harness measures
the server's per-model default. Expected counts are computed per family with
the serving splitter: **4 chunks at 300** for every family and **7 chunks at
150 for `chatterbox_turbo`** (the brief's "≈ 6" was approximate; the splitter's
exact result is 7). `--longform-chunk-size` defaults to `None`; the fixture
report records `default_chunk_size` per family and the markdown budget cell
prints `server default (300; turbo 150)` when unset.

**Tests added (13).** Unit: pinned constants/mapping (2), turbo sentinel and
concrete-name resolution, other-family 300, unknown model 300, bare-serving
300, explicit-wins bypass, `_resolve_chunks` uses 150 (8 in
`tests/unit/wire_contract/test_long_form_fields.py`); bench: per-family chunk
counts, mapping-mirrors-serving, ON omits `chunk_size`, explicit size verbatim,
markdown per-model budget (4 in `tests/unit/benchmarks/test_bench_longform_scenarios.py`);
integration: full serving path turbo default yields 150-char chunks (1 in
`tests/integration/test_tts_chunking.py`). Existing `test_default_chunk_budget_is_300_characters`
still pins the non-turbo/global fallback.

**Docs.** `protocol.py` field description and `docs/api.md` (field row + new
per-model budget paragraph); `docs/models.md` Turbo fidelity note + pointer;
`docs/benchmarks/throughput.md` §Phase 12 dated "Text fidelity (post-12-5)"
subsection (condensed evidence table, recommendations) and the fixture/variants
method bullets; `CHANGELOG.md` Fixes bullet.

**Full-suite result.** `1525 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, CPU-only, 2026-11-09; baseline 1512 + 13 new tests,
none removed).

---

Dated 2026-11-09 (Phase 12-5 text-fidelity follow-up 2; multilingual default
chunk budget + Whisper fidelity gate). Owner-approved 2026-11-09.

**Multilingual budget sweep (owner decision).** The same real-prose fixture and
Whisper large-v3-turbo method as the Turbo investigation (CPU int8, beam 5;
807-char / 138-word text; word edit distance vs the fixture; ~2–6 is
transcription noise: "Karina/Carina", "weaves/leaves"):

| Model | Budget (chars) | Edit distance | Words heard / ref | Notes |
|---|---|---|---|---|
| chatterbox_multilingual | 300 | **67** | 73 / 138 | silently skips half the text |
| chatterbox_multilingual | 150 | **48** | — | partial rescue |
| chatterbox_multilingual | 100 | **26** | — | closer |
| chatterbox_multilingual | **75** | **9** | **139 / 138** | full coverage; residual ≈ noise |
| chatterbox (base) | every budget | 2–6 | — | budget-insensitive (noise) |
| melo | every budget | 2–6 | — | budget-insensitive (noise) |
| chatterbox_turbo | 150 | 2–3 | — | faithful at ≤150 (pinned in follow-up 1) |

Audio duration at 75 is **39 s vs 18.6 s at 300**; that is the whole text
finally being spoken, not slower synthesis (the same per-word pacing argument
as the Turbo length-artifact resolution). The model is budget-sensitive, not
incapable.

**Fix (implemented, `serving_text_to_speech.py`).**

- `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75`;
  `DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE` now
  `{"ChatterboxTurboTTS": 150, "ChatterboxMultilingualTTS": 75}`.
- `_resolve_default_chunk_size()` resolves either override, else the global
  300. Explicit `chunk_size` still wins; unknown/bare-serving falls back to 300.
- `protocol.py` field description and the docs below updated.

**Bench fixture.** The ON payload still omits `chunk_size`, so the server's
per-model default applies. The serving splitter gives **16 chunks at 75** for
the 807-char fixture (the brief's "≈ 11" was approximate; the exact result is
asserted in the pinned tests). Turbo stays 7 at 150; every other family 4 at
300. The fixture report's `default_chunk_size` / `expected_chunks_per_request`
are computed per family by one shared helper used by both bench modes.

**Whisper fidelity gate (implemented, `scripts/bench_throughput.py`).** The gap
that let Turbo's word-dropping ship: benchmarks measured duration, never
fidelity. New `--fidelity` mode (requires `--scenario longform`):

- Renders each family's fixture **once per variant** (default ON at the
  server's per-model budget; `--longform-split both` adds the OFF baseline),
  saving WAVs under `out/bench/fidelity-12/` (`--fidelity-out`).
- Transcribes with faster-whisper `large-v3-turbo` (`--gpu` selects CUDA
  float16; otherwise CPU int8), `language="en"`, beam 5. The import is lazy;
  a missing install fails with an actionable `SystemExit` message.
- Normalises case/punctuation (apostrophes split; the fixture is 138 words),
  computes LCS-based word edit distance (`len(ref) + len(hyp) - 2 * LCS`) and
  words heard vs reference, and writes per-family rows to the JSON report plus
  a markdown table `| family | variant | edit distance | words heard / ref |`.
- Noise floor documented in the report method block and the docs: ~2–6 edit
  distance on this fixture is transcription noise, not model error.
- The report's per-variant `chunks` field reflects the per-family expected
  chunk count, so a fidelity row cannot disagree with the throughput matrix.

**Tests added (existing pins updated in place).** Unit:
- wire-contract: pinned constants/mapping, multilingual sentinel + concrete
  name resolution, multilingual `_resolve_chunks` (3 added; mapping pin
  updated).
- bench fidelity: 138-word normalisation, word-edit-distance known pairs,
  report-row shape, injected stub transcriber, markdown row shape, per-family
  chunk-count reflection, split-default resolution, `--fidelity` requires
  `--scenario longform`, missing faster-whisper message, and an end-to-end
  report run with an injected transcriber and a fake render (10 functions /
  16 parametrised cases).
- bench long-form: per-family chunk-count expectations (16 multilingual),
  mapping-mirrors-serving, markdown budget string, recipe text (updated
  4 tests in place).
Integration: full serving path multilingual default yields 75-char chunks (1).

**Docs.** `docs/api.md` per-model budget paragraph (multilingual sweep);
`docs/models.md` Chatterbox long-form fidelity note; `docs/benchmarks/throughput.md`
§Phase 12 method (fixture counts, variants, new fidelity-gate bullet, noise
floor), text-fidelity table/sweep/recommendation and a Reproduce command;
`CHANGELOG.md` Fixes bullet.

**Full-suite result.** `1545 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, CPU-only, 2026-11-09; baseline 1525 + 20 new test
cases, none removed). The same CPU-only run covers the new multilingual
serving-path integration test and the stub-transcriber fidelity tests.

---

## ⚠️ Post-implementation audit — false assumptions in the 12-5 text-fidelity follow-ups (2026-11-09)

> **CRITICAL FOR IMPLEMENTING AGENTS — DO NOT TRUST THE "LONG-CONTEXT MODEL FAILURE" / "BUDGET-SENSITIVE" EXPLANATIONS ABOVE:**
>
> Code audit on 2026-11-09 proved that both word-dropping symptoms investigated in the two follow-ups above were caused by **bugs in our own engine/wrapper code**, NOT by inherent 150-char or 75-char context limits of `ChatterboxTurboTTS` or `ChatterboxMultilingualTTS`. Furthermore, setting `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75` and `DEFAULT_CHUNK_SIZE_TURBO = 150` forces `_split_oversize` / `_hard_wrap` to chop normal 90–160 char English sentences in half mid-clause, where `punc_norm` appends a fake period `.` and the stitcher inserts a 120 ms pause mid-sentence.
>
> ### What was actually wrong:
>
> 1. **Why `ChatterboxTurboTTS` dropped 15–20 words at `chunk_size=300` while speaking the same sentences perfectly SOLO:**
>    - When tested **SOLO** (`len(inputs) == 1`), `_execute_chatterbox_tts_ar` (`model_runner_base.py:928-929`) **bypassed `ARBatchRunner`** and called `_execute_chatterbox_turbo_tts_sequential` → `T3.inference_turbo` (`max_gen_len = 1000` new speech tokens *after* prefill).
>    - When tested **chunked at 300 chars** (`len(inputs) == 4` concurrent chunks), it ran through **`ARBatchRunner`**, which sets `row.max_new_tokens = min(1001, self.arena.cap_tokens - length)` (`ar_runner.py:727`) with `cap_tokens = 768` (`config.gpu.yml:17`).
>    - Unlike English `ChatterboxTTS` (which compresses its conditioning prompt to 32 tokens via `Perceiver`, `len_cond = 34`), `ChatterboxTurboTTS` has `use_perceiver_resampler = False` and `speech_cond_prompt_len = 375` (`tts_turbo.py:153-154`), so its conditioning prompt is **up to 376 tokens**! Plus ~70 text tokens, Turbo's prefill `length` is ~445 tokens, leaving only **`768 - 445 = 323` speech tokens (~12.9 s)** before `ARBatchRunner` (`ar_runner.py:820`) silently truncated every 300-char chunk mid-sentence (`finish_reason = "length"`).
>    - **Resolution:** Fixed in [12-8-ChatterboxARRunnerAndS3GenBatchFixes.md](12-8-ChatterboxARRunnerAndS3GenBatchFixes.md) (proper arena `cap_tokens` sizing + removal of `DEFAULT_CHUNK_SIZE_TURBO = 150`).
>
> 2. **Why `ChatterboxMultilingualTTS` spoke only 73/138 words at `chunk_size=300` (and 2 words unchunked):**
>    - `prepare_chatterbox_multilingual_tts_inputs` (`harmonyspeech/task_handler/inputs.py:859`) hardcoded `repetition_penalty = 1.2` (copied from English `ChatterboxTTS`) instead of upstream `ChatterboxMultilingualTTS.generate`'s default **`2.0`** (`chatterbox/mtl_tts.py:241`).
>    - Upstream `AlignmentStreamAnalyzer.step()` (`alignment_stream_analyzer.py:154-175`) checks `len(set(self.generated_tokens[-2:])) == 1` (any **2** identical consecutive speech tokens, with `# self.complete and` commented out) and immediately forces EOS (`logits[..., self.eos_idx] = 2**15`). With `repetition_penalty = 1.2` instead of `2.0`, 2-token repeats happen mid-sentence and immediately abort the chunk.
>    - `AlignmentStreamAnalyzer` also leaks 3 GPU→CPU `.cpu()` forward hooks onto `self.tfmr` per call (`alignment_stream_analyzer.py:84`), unmasks end-of-text attention sinks at ~60 % of a chunk (`A_chunk[:, self.curr_frame_pos + 1:] = 0` at 25 Hz speech frames vs ~15 chars/s text tokens), and was the sole reason Multilingual was forced onto `eager` attention and excluded from `ChatterboxT3AR` and `ARBatchRunner`.
>    - **Resolution:** Fixed in [12-7-ChatterboxMultilingualFixesAndBatching.md](12-7-ChatterboxMultilingualFixesAndBatching.md) (restore `repetition_penalty = 2.0`, fix analyzer hook/EOS bugs, enable SDPA + `ChatterboxT3AR` + `ARBatchRunner` for Multilingual, and remove `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75`).
>
> 3. **Why `CrossfadeStitcher` joins still had a missing fade-out when `sentence_pause_ms > 0`:**
>    - In `CrossfadeStitcher.feed()` (`audio_utils.py:610-635`), `bridge = concat(self._carry, _silence(pause_samples))` emits `self._carry` in `bridge[:-fade]` **completely un-faded** (hard step to `0.0` at the pause start) and only crossfades `bridge[-fade:]` (zeros) into the next chunk.
>    - **Resolution:** Fixed in [12-6-TextSplitterAndStitcherFixes.md](12-6-TextSplitterAndStitcherFixes.md).
>
> ### What is KEPT from 12-5:
> - **`LONGFORM_FIXTURE_TEXT`** (the real-prose extract from *The Villainess Malfoy* in `scripts/bench_throughput.py`).
> - **`--fidelity` mode in `scripts/bench_throughput.py`** (Whisper `large-v3-turbo` transcription + LCS word edit distance gate) — kept as the permanent text-coverage regression gate, updated in [12-9-LongFormBenchmarkAndFidelityReverification.md](12-9-LongFormBenchmarkAndFidelityReverification.md) to assert faithful coverage at the unified 300-character default budget.
> - **`condition_edges()` and the 40 ms / 120 ms join defaults**, paired with the corrected two-sided pause fade in [12-6](12-6-TextSplitterAndStitcherFixes.md).


