# Phase 3-4: VAD Switch & Entry Points

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 3 — Declarative PipelineSpec routing
> **Depends on:** 3-2
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`

## Objective

Deliver the two routing features the old field-presence design could not express cleanly:

1. **`input_vad_mode` becomes functional** — Silero VAD selectable for the OpenVoice pipelines (today the field is decorative; TODO comments at the deleted `:167`/`:234`).
2. **`single_speaker_tts` becomes an entry point** — start at the synthesis step instead of the `FIXME`-flagged field-coercion hack (deleted `:187`/`:254`).

Plus: formalize `EntryRule`s for all request shapes.

## Context (self-contained)

- Silero runs today only as a standalone `/v1/audio/vad` executor (`_execute_silero` → `get_speech_timestamps`, `model_runner_base.py:600-642` region per research §1.3(d)); its output shape (timestamp list) differs from Whisper's segment dicts (`{"start","end","text"}`), which the OpenVoice encoder step consumes (`inputs.py:393-444` — segment iteration + text-length filter).
- Entry decisions currently scattered: request class checks + `mode` field + field presence. With specs, entry = `PipelineState.step_index` set at admission.

## Implementation steps

### 1. Silero segment normalization

In the Silero executor path, produce a `VadArtifact` whose `segments` use the Whisper-compatible schema: map `get_speech_timestamps` output (`[{"start": sample, "end": sample}]`) to `[{"start": s/16000, "end": e/16000, "text": ""}]`. Text-empty segments must survive the encoder's segment filter (`inputs.py:431`: `2 <= len(text) < 200` — **adjust the filter**: for silero-origin segments, skip the text-length condition; make the filter artifact-backend-aware). Document both schemas in `VadArtifact`.

### 2. VAD step preference by `input_vad_mode`

`Step.requires` for the OpenVoice VAD step is `("FasterWhisper", "SileroVAD")`. `next_step()` consults `request.input_vad_mode`: `"silero"` → prefer SileroVAD config if present (else fall back with a warning); `"whisper"`/`None` → FasterWhisper. Make this a first-class field on `Step` (`requires_by_preference: dict[str, tuple[str, ...]]` or a resolver hook — pick the simpler) and cover with unit tests.

### 3. Entry rules

```python
@dataclass(frozen=True)
class EntryRule:
    applies: Callable[[RequestInput], bool]
    start_step: str            # step name
    stop_after: str | None     # for embedding-only requests: stop after 'embed'
```

Rules per family (transcribe from old routing, `harmonyspeech_engine.py:109-335`, now deleted — use the 3-2 parity table):

- `SpeechEmbeddingRequestInput` → start at `vad` (if audio undecoded) **or** `embed` (when `input_vad_data` already present — external VAD data), stop after `embed`.
- `TextToSpeechRequestInput` `mode="voice_cloning"` with audio → start at `vad`.
- `TextToSpeechRequestInput` `mode="single_speaker_tts"` (or embedding already present) → start at `synthesize` — **this replaces the FIXME**: no field mutation, just a different `step_index`.
- `VoiceConversionRequestInput` → start (and stop) at `tone_convert`/`convert`.
- `SpeechTranscribeRequestInput` → single-step whisper/silero spec.
- Chatterbox: `SpeechEmbeddingRequestInput` or TTS with audio+no-embedding → start at `embed`, else `synthesize`.

Admission (`add_request`) resolves `pipeline_state` via the first matching rule; forwarded requests (already carrying `pipeline_state`) never re-enter rule resolution — the state machine owns them from there.

### 4. Docs

`docs/`: VAD mode option (`input_vad_mode: "whisper" | "silero"`), behavior when the requested backend isn't configured (fallback + warning), and the single-speaker entry semantics.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | Silero `VadArtifact` normalization |
| `harmonyspeech/task_handler/inputs.py` | encoder segment filter backend-aware |
| `harmonyspeech/processing/pipeline_registry.py` | `EntryRule` resolution, VAD preference |
| `harmonyspeech/engine/harmonyspeech_engine.py` | admission resolves entry |
| `tests/` | silero-vad-in-pipeline e2e + entry-rule units |

## Tests

- E2E: `input_vad_mode: "silero"` on `/v1/audio/speech` (openvoice_v2, cloning) routes the VAD step to Silero (assert via routing test double or model-call spy) and completes the pipeline.
- Unit: silero segments survive the encoder filter; embedding quality sane (non-zero, normalized).
- Unit: each `EntryRule` maps its request shape to the right start step (incl. `single_speaker_tts` → synth, no field writes).
- Unit: `vad_mode: "silero"` with no Silero config → falls back to Whisper + warning log entry.

## Checklist

- [x] Silero produces Whisper-compatible `VadArtifact` segments; filter adjusted
- [x] `input_vad_mode` honored with graceful fallback
- [x] `EntryRule` set replaces field-coercion entry hacks; `single_speaker_tts` FIXME resolved
- [x] Forwarded requests never re-resolve entry
- [x] Docs + tests green
- [x] `summary.md` Implementation Status updated

## Deviations (recorded after implementation)

- `VadArtifact` gained a `source` field (`"whisper"` / `"silero"`) plus
  `from_silero_timestamps()`; the encoder filter keys off `source`, keeping the
  whisper text-length heuristic exactly as before.
- The Silero executor now always builds normalized segments; only the JSON *response*
  stays dependent on `get_timestamps` (standalone `/v1/audio/vad` unchanged).
- The VAD preference is expressed as `Step.preference_field` +
  `requires_by_preference`, the simpler of the two options in the task.
- `EntryRule.requires_model_type` was added so the Chatterbox embed entry depends on
  whether a dedicated embedding executor is configured (see 3-2 deviations).
- Silero segments still obey the 1.5 s–20 s duration filter (the reference clip must
  contain a usable speech run); only the text-length condition is skipped.
- E2E: `tests/e2e/tts/test_openvoice_vad_mode.py` runs the real OpenVoice V2 cloning
  pipeline with `input_vad_mode="silero"`, `"whisper"`, and with Silero absent
  (fallback) — all three pass on CPU.
