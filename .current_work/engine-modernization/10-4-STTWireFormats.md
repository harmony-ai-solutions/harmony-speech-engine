# Phase 10-4: STT Wire Formats — srt/vtt + `/v1/audio/translations`

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 10 — Whisper serving
> **Depends on:** nothing (fully parallelizable with 10-1..10-3; pure serving-layer work, Phase 1 patterns)
> **Codebase mapping consulted:** `.planning/codebase/CONVENTIONS.md`; Speaches-harvest from study #13 (`../../soulbits-cloud-backend/.current_work/inference-engine-re-eval-26-11/13-speaches-analysis.md`)
> **State reference:** verified 2026-11-07 — `serving_speech_to_text.py` returns one JSON shape (`SpeechToTextResponse(id, created, model, text, language, timestamps)` built at `:87-94` from `json.loads(final_res.output)` at `:85`; `result_data["segments"]` already carries start/end/text dicts). The six audio endpoints live in `api_server.py`; no `/v1/audio/translations` exists today.

## Objective

Two cheap client-compatibility wins harvested from the Speaches conformance diff: OpenAI-style `srt`/`vtt` transcription response formats, and the `/v1/audio/translations` endpoint (`task=translate` on the same whisper pipeline).

## Context (self-contained)

- **Response formats:** add `response_format` to `SpeechTranscribeRequest` (admission-validated, 400 on unknown — the Phase 1-2 `output_options` pattern). Supported: `json` (today's shape, default, byte-identical), `text` (plain text — verify whether it already exists informally; make it explicit), `srt` and `vtt` rendered from `result_data["segments"]`:
  - `srt`: numbered blocks, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, blank-line separated;
  - `vtt`: `WEBVTT` header, `HH:MM:SS.mmm` arrows, no block numbers;
  - both served as `text/plain` (OpenAI behavior); empty-segments input renders an empty document, not an error.
- **Translations:** `POST /v1/audio/translations` — request model mirrors transcription (model, file/audio, prompt, response_format; `language` is a *source* hint only). Serving module reuses `OpenAIServingSpeechToText` plumbing with a `task="translate"` flag threaded: protocol request type → `SpeechTranscribeRequestInput` (new `task` field) → `prepare_faster_whisper_inputs` → handler passes `task` into `pipeline.transcribe(task=...)` / the 10-2 batcher's generate options (faster-whisper supports `task="translate"` natively). Same response formats as transcriptions.
- **Model cards / registry:** the pipeline specs' `stt` capability covers the same executors; add a `translation` capability flag so `/v1/models` advertises it honestly (Phase 3-3 spec-driven cards — no hand lists).
- **Docs:** `docs/api.md` (+ the endpoint's param semantics), `docs/audio-io-contract.md` format table; wire-contract matrix cases for both endpoints × all formats.

## Implementation steps

1. `protocol.py`: `response_format` field + validation; `SpeechTranslationRequest` (or a task-discriminated request) + response reuse.
2. Renderers (small pure module, unit-tested): segments dict → srt/vtt strings with timestamp edge cases (≥1 h segments, sub-ms rounding, empty list, text with newlines/`-->` sequences escaped per format rules).
3. Serving: transcription response dispatch by format; new translations route in `api_server.py` + serving method sharing the generator path with `task` threaded through inputs/handler.
4. Registry: `translation` capability on the stt specs; card generation picks it up.
5. Wire-contract tests (Phase 1-4 suites) + docs.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/protocol.py` | `response_format`, translation request types |
| `harmonyspeech/endpoints/openai/serving_speech_to_text.py` (+ small renderer module) | format dispatch + translation method |
| `harmonyspeech/endpoints/openai/api_server.py` | `/v1/audio/translations` route |
| `harmonyspeech/common/inputs.py`, `task_handler/inputs.py`, handler | `task` plumbing |
| `harmonyspeech/processing/pipeline_registry.py` | `translation` capability |
| `tests/unit/wire_contract/`, `tests/e2e/wire_contract/` (extend) | renderer units + matrix cases |

## Tests / validation

- Unit: renderer edge cases; admission 400 on unknown format; `task` reaches the handler (stubbed).
- Wire-contract e2e: both endpoints × {json, text, srt, vtt} against the mocked serving layer; real-model spot-check rides the benchmark-host session.

## Checklist

- [x] `response_format` (json/text/srt/vtt) with admission validation; json byte-identical
- [x] `/v1/audio/translations` live with `task=translate` threaded end-to-end
- [x] `translation` capability advertised via specs
- [x] Wire-contract matrix green; docs updated
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-07)

### Upstream facts verified (faster-whisper 1.2.1, no weights downloaded)

- `faster_whisper.tokenizer.Tokenizer.__init__(tokenizer, multilingual, task, language)`
  stores `self.task = token_to_id("<|{task}|>")` (validated against `_TASKS`,
  i.e. `transcribe` / `translate` when multilingual) and `sot_sequence` is
  `[<|startoftranscript|>, <|{language}|>, <|{task}|>]`.
- `WhisperModel.get_prompt()` extends `tokenizer.sot_sequence` into the prompt;
  `TranscriptionOptions` itself has no `task` field. **`task` is encoded only
  through the tokenizer**, so a path that builds its own tokenizer must pass the
  request's task.
- `WhisperModel.transcribe(..., task="transcribe")` and
  `BatchedInferencePipeline.transcribe(..., task="transcribe")` both accept
  `task`; the batched pipeline constructs `Tokenizer(task=task, language=...)`
  (`transcribe.py:495-510`) and `generate_segment_batched` builds one prompt per
  row from that tokenizer. The engine's 10-2 batcher replaces that prompt
  construction, so it now builds its tokenizer with `request_windows.task`.
- `BatchedInferencePipeline.generate_segment_batched` passes
  `previous_tokens=tokenizer.encode(options.initial_prompt)` to `get_prompt`;
  the batcher mirrors that for the new `prompt` request field.
- `language` provided upstream sets `language_probability = 1` and
  `all_language_probs = None` (no detection call); the batcher mirrors this for
  a source hint.

### Implementation

- `harmonyspeech/endpoints/openai/stt_formats.py` (new): pure
  `normalize_response_format` / `validate_response_format` /
  `segments_to_srt` / `segments_to_vtt`. Timestamps are `HH:MM:SS` + 3-digit
  fraction (`Decimal(str(seconds))`, half-up rounding; hours grow past 99).
  Empty segments → `""` (SRT) / `"WEBVTT\n"` (VTT). Cue text normalises CRLF,
  strips outer blank lines and renders a literal `-->` as `- ->` (neither format
  permits the timing arrow in cue text).
- `protocol.py`: `SpeechTranscribeRequest` gains `response_format`, `language`
  (source hint) and `prompt`; `SpeechTranslationRequest` is a subclass.
  `response_format` is validated at admission (Phase 1-2 pattern) → HTTP 400
  before `engine.generate`; unknown values never queue inference.
- `serving_speech_to_text.py`: one shared `_create_stt(..., task=...)` backs
  `create_transcription` (task `transcribe`) and `create_translation` (task
  `translate`). `json` returns the exact pre-10-4 `SpeechToTextResponse`
  construction (pinned by tests); `text`/`srt`/`vtt` return
  `PlainTextResponse` (`text/plain`). Cards are generated from
  `("stt", "translation")`.
- `api_server.py`: `POST /v1/audio/translations` mirrors the transcription route
  (same upload aliases/OpenAPI dual body); both routes pass a
  `PlainTextResponse` through unchanged.
- `common/inputs.py`: `SpeechTranscribeRequestInput` gains `task` (default
  `transcribe`), `language`, `prompt`; `from_openai(..., task=...)`.
- `task_handler/inputs.py`: `prepare_faster_whisper_inputs` resolves/normalises
  `task` (rejecting unknown values as the batch's typed prepare error) and
  normalises empty language/prompt to `None`.
- `task_handler/model_runner_base.py`: the lane pool reads
  `whisper_request_options(initial_request)` and passes `task`, `language`,
  `initial_prompt` to `pipeline.transcribe`.
- `task_handler/whisper_batching.py`: `whisper_request_options()` helper;
  `_RequestWindows` carries `task`/`language_hint`/`initial_prompt`;
  `_tokenizer_for` uses the request task and encodes the prompt as previous
  tokens; a language hint skips detection and reports probability `1`.
- `processing/pipeline_registry.py`: new `TRANSLATION` capability on the
  Faster-Whisper step/spec (`advertises=(STT, TRANSLATION, VAD)`); new
  `capabilities_for_model_type()` so individual cards carry spec-derived
  capability metadata (no hand lists).

### Deviations (2026-11-07)

1. **`prepare_faster_whisper_inputs` keeps returning audio artifacts.** It
   resolves/validates the task on the request input instead of returning a
   per-request `(artifact, options)` tuple. This preserves the 10-1/10-2 input
   contract (a list of 16 kHz `AudioArtifact`s) and every direct-call test; both
   execution paths read the resolved options from the same request input.
2. **`language` and `prompt` are threaded too** (beyond the mandatory `task`):
   `language` is a source hint (skips batcher detection; passed to
   `pipeline.transcribe`), `prompt` becomes `initial_prompt` / previous tokens.
   The spec's request mirror named both fields, and leaving them silently
   ignored would violate the wire-contract principle.
3. **Individual model cards now carry spec-derived `capabilities`** (additive).
   Faster-Whisper cards report `["stt", "translation", "vad"]`, so translation
   is advertised on `/v1/models`; toolchain cards are unchanged. No existing
   field changed.
4. **`-->` escaping** uses `- ->` (both formats) rather than a format-specific
   escape; neither format defines one, and the arrow must not survive in cue
   text.
5. **Hours are not capped at 99** in either format (`100:00:00,000`), matching
   the "≥1 h" requirement.
6. The window batcher's `initial_prompt` is reported in
   `TranscriptionOptions` for payload parity with the sequential path.

### Tests

- `tests/unit/wire_contract/test_stt_formats.py` (33 cases): renderer edge
  cases (≥1 h, sub-ms half-up, invalid timestamps, empty input, multi-cue,
  newlines/`-->`, non-mapping segments), admission 400s on both endpoints with
  engine-not-called assertions, the byte-identical JSON shape pin, and the
  `task`/`language`/`prompt` plumbing.
- `tests/unit/inference_flow/test_whisper_window_batching.py` (+4): translate
  prompt token, mixed-task per-row prompts, language hint skips detection,
  initial prompt as previous tokens.
- `tests/unit/inference_flow/test_whisper_lanes.py` (+1): the sequential path
  forwards `task`/`language`/`initial_prompt`.
- `tests/unit/initialization/test_pipeline_validation_cards.py` (+1): the
  translation capability is declared and advertised on the Whisper card.
- `tests/e2e/wire_contract/test_contract_matrix.py` (+12): both endpoints ×
  {json, text, srt, vtt}, a multipart variant, and admission 400s.
- `tests/integration/test_api_endpoints.py` (+1) + conftest mock.
- Full runs: unit 958 passed, integration 53 passed, wire-contract e2e
  50 passed / 3 skipped (model-gated); `tests/e2e/stt` 2 passed / 2 skipped
  (parity harness needs a local CT2 model).
- Real-model spot-checks remain `_tbd_` for the benchmark host (10-5): no
  weights downloaded, per the task constraint.

