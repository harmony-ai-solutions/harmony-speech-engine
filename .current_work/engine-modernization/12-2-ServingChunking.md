# Phase 12-2: Serving-Layer Chunking + Stitching (all TTS families)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio
> **Depends on:** 12-1 (utils + fields); reuses 7-2 embedding cache (embed once, synthesize N chunks)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (serving module pattern); March plan 3-1/4-1 sub-task files superseded
> **State reference:** verified 2026-11-08 — synthesis dispatch in `endpoints/openai/serving_text_to_speech.py` (re-verify exact lines before editing); perth watermark is a per-request post-step in `task_handler/model_runner_base.py` (~:1123 per 11-1 spec — re-verify); `speed` consumed natively by some model types only (`NATIVE_SPEED_MODEL_TYPES` concept from the March plan — verify which families read `speed_function` today).

## Objective

When `split_text: true` (or `chunk_size` set): split → synthesize chunks → stitch → return **one** response. Model-agnostic: every TTS family (chatterbox en/turbo/multilingual, hs1, kitten ×4, melo, openvoice v1/v2) gets it from the serving layer, zero model-runner changes.

## Context (self-contained)

- **Chunk submission is concurrent, stitching is ordered.** All chunk sub-requests are submitted at once (asyncio.gather-equivalent on the engine path) so a multi-chunk request can fill an admission batch **by itself** — on `ar_decode` families 4 chunks = 4 rows in one step; on `padded_batch` families the step batches them; this is 12-4's measured hypothesis, do not serialize "for simplicity". Stitch by chunk index, never by completion order.
- **Clone mode embeds ONCE.** `voice_cloning`/`input_audio` requests resolve the reference through the existing embed step **one time** (7-2 cache key unchanged), then every chunk synthesizes against the same conditioning. Assert this in tests — re-embedding per chunk is the silent failure mode.
- **Watermark exactly once, on the stitched buffer.** The per-request post-step would watermark every chunk (muddled watermark, crossfaded over). Design: per-chunk synthesis with watermark suppressed + one application on the final output (reuse the model's watermarker; serving layer needs a small helper or a runner-side flag). Pin with a test that the watermark hook runs exactly once per HTTP request.
- **Per-chunk failure fails the request** — honest all-or-nothing; no partial audio with a 200.
- **Seeding:** a request seed derives per-chunk seeds deterministically (`seed + chunk_index`) so chunked output is reproducible and the 9-3 AR runner rows stay comparable; assert re-run stability.
- **Speed/normalization ordering:** native-speed models keep their native handling; non-native families get `time_stretch` after stitching (whole-output, not per-chunk); `norm_loudness`/peak normalization after stitching. `sentence_pause_ms`/`crossfade_ms` during stitch.
- **Degenerate cases:** short text (0 or 1 chunks) must be **byte-identical** to today's path (no stitcher in the pipeline) — this is the regression gate.
- **Phase 14 seam (from the 2026-11-08 streaming research):** build the stitcher so a streaming consumer can reuse it with the **hold-back-the-crossfade-tail pattern** — each chunk's last `fade_samples` are withheld and emitted crossfaded into the next chunk's head (the upstream `carry` pattern, `server.py:1009-1067`). Concretely: expose a chunk-by-chunk stitch API (feed one chunk → get emittable audio + carried tail), with the whole-buffer stitch a thin fold over it. Do not hard-code whole-buffer-only internals.

## Implementation steps

1. Chunking dispatcher in `serving_text_to_speech.py` (guarded by the 12-1 fields; single-chunk fast path = existing code untouched).
2. Concurrent chunk submission + ordered stitch + the watermark/seed/speed rules above; `NATIVE_SPEED_MODEL_TYPES` frozenset pinned by test (verify current native consumers of `speed_function` first — do not guess the set).
3. Tests: integration — 2-chunk and 5-chunk texts through the real engine loop (stub/tiny models, the 8-x precedent): ordered stitch, clone-embeds-once, watermark-once, chunk-failure ⇒ request failure, seed determinism; e2e per family (the March list: chatterbox, turbo, multilingual, hs1, kitten ×4, melo, openvoice — `single_speaker_tts` + `voice_cloning` where supported) asserting valid WAV + expected duration growth vs chunk count; wire-contract additions for the new fields (default-off stays identical).
4. Docs: `docs/api.md` (fields now effective), `docs/batching.md` note (chunk sub-requests are ordinary admitted requests), `CHANGELOG.md` (with 12-1).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher + stitch + watermark/speed rules |
| `harmonyspeech/task_handler/model_runner_base.py` (or serving-side helper) | watermark-once mechanism; no other runner changes |
| `tests/integration/test_tts_chunking.py` (NEW), `tests/e2e/tts/*` (extend) | gates above |
| `docs/api.md`, `docs/batching.md`, `CHANGELOG.md` | honest user-facing docs |

## Tests / validation

- Byte-identity gate: short text with `split_text: true` ⇒ identical bytes to `split_text: false`.
- Full suites green; no new skips.

## Checklist

- [x] Dispatcher + ordered stitch landed (concurrent submission)
- [x] Clone-embeds-once + watermark-once + seed derivation pinned
- [x] Per-family e2e + integration + wire contracts green
- [ ] Byte-identity gate green; docs/CHANGELOG; `summary.md` ticked

## Deviations / verification log

Dated 2026-11-09 (Phase 12-2 implementation).

**Verified**

- **Watermark seam.** The premise "the per-request post-step watermarks every
  request" was imprecise. There is no universal per-request post-step: the only
  watermark calls are
  * `ModelRunnerBase._artifact_from_waveform` (Chatterbox AR and the padded VC
    vocode path), and
  * upstream `ChatterboxTTS/TurboTTS/MultilingualTTS.generate()`, which applies
    the watermark itself on the sequential path.
  Chosen seam: `_artifact_from_waveform(..., watermark=False)` for the AR/batched
  paths, and a `_without_watermark()` context manager that swaps the model's
  watermarker for an identity `_NullWatermarker` around the one sequential
  `generate()` call. Chunk requests carry an internal, non-wire
  `TextToSpeechRequestInput.suppress_watermark` flag. The serving layer then
  applies the producing model's watermarker exactly once to the stitched
  buffer (`_apply_final_watermark`, under a lock). Pinned by
  `test_watermark_runs_exactly_once_per_request` (chunked = 1, single-shot = 1).
- **`NATIVE_SPEED_MODEL_TYPES` found** (traced from `common/inputs.py` through
  each handler):
  `HarmonySpeechSynthesizer` (`alpha`), `OpenVoiceV1Synthesizer`
  (`length_scale=1/speed`), `MeloTTSSynthesizer` (`length_scale=1/speed`),
  `KittenTTSSynthesizer` (ONNX `speed`). All Chatterbox families ignore
  `speed`, so the chunked path applies `time_stretch` post-stitch only for
  them. Pinned by `test_native_speed_model_types_are_pinned` plus native /
  non-native duration tests.
- **Default chunk budget = 300 characters** when `split_text: true` and
  `chunk_size` is null (rationale: one to three spoken sentences; keeps the AR
  families' per-request token budget and peak memory bounded while leaving
  prosody/crossfade joins non-dominant). Pinned by
  `test_default_chunk_budget_is_300_characters`. `chunk_size` alone also enables
  chunking (pinned).
- **Concurrent submission, ordered stitch.** All chunk sub-requests are added to
  the engine's request tracker before any output stream is consumed (each
  `add_request` completes without yielding to the event loop), so one
  multi-chunk request fills an admission batch by itself; results are then
  consumed concurrently and stitched by chunk index. Pinned by
  `test_chunk_sub_requests_co_batch_in_one_engine_step` (a 5-chunk request
  observed as one 5-row batch) and the ordered-stitch test.
- **Clone embeds once.** A voice-cloning chunk request first runs one
  `SpeechEmbeddingRequestInput` through the existing pipeline embed step, then
  every chunk starts at the synthesize step with the returned
  `SpeakerArtifact`. Pinned for Chatterbox (`prepare_conditionals` calls == 1,
  N `generate` calls) and OpenVoice V1 (`sum(encoder batch sizes) == 1`).
- **Per-chunk failure fails the request.** `asyncio.gather` collects every
  chunk; any error/missing audio returns a 400 `ErrorResponse`, never partial
  audio. Pinned.
- **Seed derivation.** `seed + chunk_index` when the request has a seed; the
  per-chunk seeds reach the model and two identical runs produce identical
  audio. Pinned.
- **Byte-identity gate.** `split_text: true` with 0/1 chunks returns `None`
  from the dispatcher, so the original request goes through the untouched
  single-shot path. Pinned byte-for-byte.
- **CJK pass-through.** `language` is forwarded to `split_text_into_chunks`
  (12-1 handles the enders); pinned with a `zh` request.

**Deviations**

- **`norm_loudness` is not an output normalizer in this codebase.** Verified in
  the installed upstream: `ChatterboxTurboTTS.generate(norm_loudness=...)` only
  passes it to `prepare_conditionals` (reference-clip normalization); it never
  touches the generated waveform. There is therefore no per-chunk output
  normalization to defer, and no post-stitch loudness/peak step was added.
  `audio_utils.peak_normalize` remains available for the Phase 12-3 filters.
  The embed-once sub-request carries the original `generation_options` (new
  internal `SpeechEmbeddingRequestInput.generation_options`), but the existing
  embed handler computes conditionals with its historical defaults; changing
  that would alter pre-existing multi-step clone output and is out of scope.
- **`summary.md` untouched per task scope**, hence the last checklist box stays
  unticked (byte-identity gate, docs and CHANGELOG are green). The 12-1
  precedent is unchanged.
- **Disconnect mid-chunks does not abort outstanding chunk requests.** The
  serving layer checks disconnect before submission and after collection; it
  does not cancel chunks already admitted. Phase 14's streaming carry should
  add per-chunk cancellation/resource release.
- **Serving-side watermark call runs in the event-loop thread** under a lock.
  The runner's own watermark calls remain unlocked, but the watermark forward is
  read-only; no shared mutable state was found. Recorded for review.
- **OpenVoice V2 cloning is not separately stubbed.** It shares the exact
  embed-once → synthesize → tone-convert path pinned for V1, with `MeloTTSSynthesizer`
  (native speed) as the synthesize step. Melo itself is covered as a
  single-speaker family; V1 cloning is covered end-to-end.
- **Real-model e2e tests are gated** by `HSE_E2E_MODEL_TESTS=1`
  (`tests/e2e/tts/test_long_form_chunking.py`, the `tests/e2e/wire_contract`
  convention). The hermetic per-family coverage in
  `tests/integration/test_tts_chunking.py` runs in the default suite for every
  March-list family (chatterbox en/turbo/multilingual, Kitten ×4, Melo,
  OpenVoice V1, HarmonySpeech) and asserts a decodable WAV plus duration growth
  with the chunk count.

**Full-suite result:** `1413 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, 2026-11-09; baseline 1386 + 27 new tests).

Dated 2026-11-09 (Phase 12-2 follow-up — per-chunk deadline override).

**Bug (found by the Phase 12-4 host benchmark).** Long-form chunked requests
failed under concurrent load with HTTP 400 `Long-form synthesis failed: request
exceeded timeout of 120s` (repeated per chunk). Reproduction: 12 concurrent
`split_text: true` requests to `chatterbox_multilingual` — a
`max_batch_size=1` "loop" executor — queue 48 chunk sub-requests; the tail
chunks wait >120 s in the queue and the engine's per-request deadline (stamped
in `add_request`, `request_timeout_s: float | None = 120.0`) expires them before
they are ever processed. Queueing delay is not synthesis failure.

**Fix.** Mirrors the 12-2 `suppress_watermark` precedent: an internal, non-wire
`request_timeout_s: float | None = None` on `TextToSpeechRequestInput` (and on
`SpeechEmbeddingRequestInput` for the one-time reference-embedding sub-request
of a clone request). Engine `add_request` uses the per-request value when it is
not `None`; `None` means **inherit** the engine-wide timeout (never "no
timeout"), and `0` disables the deadline exactly like an engine-wide `0`. The
expiry message now reports the effective per-request timeout. The serving chunk
dispatcher stamps `base_timeout * (chunk_count + 1)` on every chunk sub-request
(worst-case wait for all siblings ahead in the queue, plus its own synthesis);
the original single-shot path is untouched.

**Verification.** New deterministic integration tests
(`tests/integration/test_tts_chunking.py`) use a 0.4 s stub executor with the
engine configured `request_timeout_s=1.0`: two concurrent 4-chunk requests (8
queued chunks) all succeed under the override (`1.0 * 5 = 5.0 s`), and the
companion test that pins the override off reproduces the benchmark failure
(`request exceeded timeout of 1s`). Unit tests pin the stamping semantics
(override wins; `None` inherits; `0` disables; message shows the effective
timeout) in `tests/unit/error_handling/test_request_timeouts.py`, and wire
contract tests confirm the field is internal only
(`tests/unit/wire_contract/test_long_form_fields.py`).

**Full-suite result (follow-up):** `1479 passed, 5 skipped` (`tests/unit
tests/integration tests/e2e/wire_contract`, 2026-11-09; baseline 1471 + 8 new
tests).

Dated 2026-11-09 (Phase 12-2 follow-up #2 — per-pipeline clone conditioning).

**Bug (found by the Phase 12-4 host benchmark, real models).** A chunked
(`generation_options.split_text: true`) **voice_cloning** request to the
OpenVoice V2 pipeline failed instantly with HTTP 400:

```
request ID tts-<id>-embed is not of type TextToSpeechRequestInput or
SynthesisRequestInput
```

The benchmark request named the concrete MeloTTS card
(`model: ov2-synthesizer-en`, the OV2 pipeline's synthesize member) with
`mode: voice_cloning` + `input_audio`, and the failure came from the
one-time reference-embedding sub-request the 12-2 dispatcher builds.

**Root cause.** `_create_chunked_response` treated *every* `input_audio`
request as "there is a standalone embed step": it unconditionally built a
`SpeechEmbeddingRequestInput` with the original `requested_model`/`model` and
ran it as `tts-<id>-embed`. That assumption only holds for a **pipeline id**
(`openvoice_v1`/`openvoice_v2`/`harmonyspeech`/`chatterbox*`), where the Phase
3-4 entry rules route an embedding request to the step that produces
`embedding_artifact`. When the client names a concrete executor instead, no
pipeline routing happens (`get_spec` → `None`), so the embed sub-request went
straight to that executor:

- **OpenVoice (toolchain id).** The reference conditioning is the tone-converter
  encoder step: a non-chunked OV2 clone runs `vad → embed
  (OpenVoiceV2ToneConverterEncoder) → synthesize (MeloTTSSynthesizer) →
  tone_convert (OpenVoiceV2ToneConverter)`. `prepare_inputs` accepts
  `SpeechEmbeddingRequestInput` for the VAD/encoder steps, so the chunked
  embed-once path is correct **as long as the pipeline id is what the request
  names**. It is not the OpenVoice synthesizer that consumes the reference.
- **Concrete synthesizer (`ov2-synthesizer-en`).** The request resolves to
  `MeloTTSSynthesizer`, whose `prepare_inputs` accepts only
  `TextToSpeechRequestInput`/`SynthesisRequestInput` (the same is true for the
  OpenVoice V1 synthesizer, KittenTTS and HarmonySpeech's synthesizer). The
  embed sub-request therefore reached MeloTTS and was rejected — the exact
  benchmark error. The non-chunked flow for that name never reads
  `input_audio` at all, so the chunked flow was doing two things the single-shot
  path does not: inventing conditioning, and sending a request type the
  executor cannot accept.
- **Chatterbox (checked, not broken).** `ChatterboxTTS/TurboTTS/
  MultilingualTTS.prepare_inputs` explicitly accept `SpeechEmbeddingRequestInput`
  (the handler runs `prepare_conditionals`), and the pipeline's `embed` step
  does too, so the dispatcher's embed sub-request matches. It is unchanged; the
  existing pre-existing default-vs-request `exaggeration`/`norm_loudness`
  difference inside the embed handler remains out of scope (see the earlier
  deviation entry).

**Fix shape (per pipeline, no duplication).**
`OpenAIServingTextToSpeech._reference_conditioning_mode(request_data)` now
resolves how the *single-shot* path conditions on the reference:

- `"embed"` when the request names a pipeline whose steps include an
  `embedding_artifact` producer (`_spec_produces_embedding`, so single-step ids
  such as `kitten_tts` are excluded), **or** when it names a Chatterbox
  executor directly (`STANDALONE_EMBED_MODEL_TYPES`, whose `prepare_inputs`
  accepts the standalone embed request and computes the same conditionals).
  Only then is the one-time `SpeechEmbeddingRequestInput` built (still
  `tts-<id>-embed`, still once, still the same 7-2 cache key).
- `"none"` otherwise: the chunks are built with the original request fields
  (`input_audio`/`input_vad_data` forwarded, `clear_clone_fields` stays false),
  i.e. N chunks behave exactly like N single-shot requests — which is what the
  single-shot path does for a concrete synthesizer. There is no "chunk 0 full
  clone artifact reuse" shape here because those executors produce no
  conditioning artifact at all; the honest fallback is parity with the
  single-shot request.

Single-shot dispatch, the byte-identity gate, watermark-once, seed derivation
and the per-chunk timeout override are untouched.

**Which stub hid it.** `tests/integration/test_tts_chunking.py::_prepare_inputs`
produced inputs for *any* request type for every model, and the only OpenVoice
clone test pre-supplied `input_vad_data` so it entered at the embed step and
never exercised the VAD hop either. The 12-2 deviation "OpenVoice V2 cloning is
not separately stubbed" (above) is now superseded. The stub is type-strict:
`_ACCEPTED_REQUEST_TYPES` mirrors `task_handler/inputs.py` and raises the real
"request ID … is not of type …" error before producing inputs. The new meta
test `test_type_strict_stub_would_have_caught_the_embed_regression` restores the
pre-fix behaviour (force `"embed"`) and asserts the request fails with the
benchmark message, proving the gap is closed.

**Verification.** New integration tests (all type-strict, CPU, hermetic):
`test_openvoice_v2_clone_chunking_embeds_once` (full `vad → embed → synthesize
→ tone_convert`, one VAD + one encoder row, one synthesize + one tone-convert
row per chunk), `test_openvoice_v2_chunked_clone_uses_the_same_embedding_cache_key`
(a later non-chunked request with the same reference is a 7-2 cache hit — no
encoder forward), `test_chunked_clone_of_a_concrete_synthesizer_skips_the_embed_step`
(the benchmark repro: succeeds, no `-embed` sub-request is ever admitted),
`test_chunked_clone_of_a_directly_named_chatterbox_model_embeds_once`
(`prepare_conditionals` == 1, N `generate`), and
`test_reference_conditioning_mode_resolves_per_pipeline`. The harness now
patches `BatchedInferencePipeline` with a deterministic fake so the OpenVoice
VAD hop is exercised, and wires the real `EmbeddingCache` into the stub runners
(production `model_identity`), so cache assertions are real rather than stubbed.

**Full-suite result (follow-up #2):** `1485 passed, 5 skipped` (`tests/unit
tests/integration tests/e2e/wire_contract`, 2026-11-09; baseline 1479 + 6 new
tests).

