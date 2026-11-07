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

- [ ] `response_format` (json/text/srt/vtt) with admission validation; json byte-identical
- [ ] `/v1/audio/translations` live with `task=translate` threaded end-to-end
- [ ] `translation` capability advertised via specs
- [ ] Wire-contract matrix green; docs updated
- [ ] `summary.md` Implementation Status updated
