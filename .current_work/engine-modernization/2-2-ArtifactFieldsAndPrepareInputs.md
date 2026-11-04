# Phase 2-2: Artifact Fields & prepare_inputs

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 2 — Artifact bus
> **Depends on:** 2-1
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (dispatch points), `STRUCTURE.md`

## Objective

Wire artifacts into the request/prepare layer: `prepare_*` functions produce artifacts, and a precedence rule makes artifacts win over base64 fields — so a request that has already been decoded once (by an earlier step or by the boundary) is never re-decoded.

## Context (self-contained)

- Request input classes live in `harmonyspeech/common/inputs.py` (plain classes: `TextToSpeechRequestInput`, `SpeechEmbeddingRequestInput`, `VoiceConversionRequestInput`, `AudioConversionRequestInput`, `SynthesisRequestInput`, … with `from_openai` builders). They carry base64 `str` fields: `input_audio`, `input_embedding`, `input_vad_data`, `input_mel_spectrogram`, `source_audio`, `target_audio`, `target_embedding`.
- `prepare_*` functions in `harmonyspeech/task_handler/inputs.py` b64-decode + `librosa.load` (now `decode_audio` from 1-1) on **every step** — e.g. Whisper VAD decodes `input_audio`, then the OpenVoice encoder step decodes the same `input_audio` again (`:393-400`), then the tone converter decodes the synthesized WAV again (`:485-489`).

## Implementation steps

### 1. Artifact-backed getters on the request layer

In `common/inputs.py`, add a mixin (or module-level helpers) used by all request classes:

```python
def get_audio_artifact(self, engine_request, field: str = "input_audio", target_sr: int | None = None) -> AudioArtifact:
    """Precedence: engine_request.artifacts[field] → decode base64 field (once) → cache into artifacts.
    Raises ValueError with the field name if neither is present."""
```

Rules:

- Cache key = field name; the decoded artifact replaces the need to ever re-read the base64 field internally.
- `target_sr` mismatch → resample **when producing** the artifact for a given step, caching per (field, sr) — simplest correct cache: store the raw decode at native sr in `artifacts[field]` and let steps request `artifact.resampled(sr)` transiently (document choice; avoid caching many resampled copies).
- Internal forwarders (Phase 2-3) will set `artifacts[field]` directly; base64 fields stay for the API and for backward compatibility.

### 2. Convert `prepare_*` outputs to artifacts

Each `prepare_*` in `task_handler/inputs.py` returns what its `_execute_*` expects. Convert the audio-carrying ones to return artifact-based input tuples:

| prepare function | Today returns | Now returns |
|---|---|---|
| `prepare_harmonyspeech_encoder_inputs` (`:251-…`) | np array | `AudioArtifact` |
| `prepare_openvoice_tone_converter_encoder_inputs` (`:381-458`) | list of np segment arrays | `list[AudioArtifact]` (segments) + reuses the already-decoded full audio artifact instead of the tempfile double-decode (`:435-444` collapses) |
| `prepare_openvoice_tone_converter_inputs` (`:461-513`) | np waveform + embedding bytes | `AudioArtifact` + `SpeakerArtifact` (embedding tensor decoded via the 0-1-safe path) |
| `prepare_faster_whisper_inputs` (`:516-530`) | np @16k | `AudioArtifact` (16k, CPU tensor) |
| `prepare_silero_vad_inputs` (`:598-602`) | np @16k | `AudioArtifact` |
| `prepare_chatterbox_*` (`:827-876`) | raw bytes | `AudioArtifact` (native sr) / `SpeakerArtifact` for conditionals |

Keep the signature `(model_config, requests_to_batch)` and per-request order (Phase 0-3 asserts still hold).

### 3. Keep the decode-once property at the boundary

The engine's `add_request` (or the serving layer's `from_wire`, formalized in 2-4) decodes inbound audio **once** into `engine_request.artifacts` when the base64 field is present — after this, no step ever touches base64 for internal hops.

### 4. Behavior parity

Existing e2e suites must pass unchanged: for **external** requests nothing observable changes (they still send base64; outputs still base64). The artifact path only changes *internal* representation.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/inputs.py` | artifact getters/mixin, precedence rule |
| `harmonyspeech/task_handler/inputs.py` | all `prepare_*` produce artifacts; tempfile double-decode removed |
| `harmonyspeech/engine/harmonyspeech_engine.py` | boundary decode-once at `add_request` |
| `tests/unit/inference_flow/` | prepare-function artifact tests |

## Tests

- Unit: base64 field + artifact present → artifact wins (mutate base64 after artifact set, result unchanged).
- Unit: same request through two prepares (whisper then encoder) decodes only once — patch/count `decode_audio` calls.
- Unit: OpenVoice TCE prepare returns correct segment artifacts for a fixture with known segments.
- E2E: existing voice-clone suites unchanged (parity).

## Checklist

- [x] Precedence getters implemented on request inputs
- [x] All `prepare_*` functions artifact-based; per-hop re-decodes eliminated
- [x] Boundary decode-once at admission
- [x] Decode-count unit tests prove single decode per request
- [x] E2e parity green
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-04)

- `common/inputs.py` owns the decode helpers and precedence getters
  (`get_audio_artifact`, `get_speaker_artifact`, `get_vad_artifact`,
  `get_mel_artifact`). `RequestInput` exposes thin methods over the same
  functions; prepare code uses the module-level form so test doubles that
  auto-create attributes still work.
- `prepare_inputs` now passes **`EngineRequest`** entries to the prepare
  functions (they carry the artifact store). Bare request inputs are still
  accepted, so direct callers/tests keep working.
- Native-rate decodes are cached on the request; a step asking for another
  sample rate gets a transient `AudioArtifact.resampled(sr)` view (never cached,
  so memory stays flat). Admissions/embeddings/VAD/mel are decoded once.
- Admission decode happens in `HarmonySpeechEngine.add_request`
  (`prefill_audio_artifacts`) and, for the HTTP path, in
  `wire.from_wire_audio`; it is idempotent. Undecodable `input_audio` payloads
  (the HarmonySpeech vocoder's JSON envelope) are left for their step to reject
  with the pre-existing typed error, so error behaviour is unchanged.
- `prepare_harmonyspeech_encoder_inputs` returns the decoded `AudioArtifact`
  (preprocessing moved to the handler) so the Whisper→encoder hop shares one
  decode.
- `prepare_openvoice_tone_converter_encoder_inputs` consumes the
  `VadArtifact` + shared `AudioArtifact`; the tempfile export/import is gone.
  It keeps OpenVoice's int16 quantization for embedding parity and fixes a
  latent `segments[k+1].start` attribute access on a plain dict.
- `prepare_synthesis_inputs` (modeling layer) no longer decodes wire text; it
  accepts the decoded embedding array from the `SpeakerArtifact`.
- `_decode_audio_field` / `_load_conditionals_payload` remain importable from
  `task_handler.inputs` as aliases over the common helpers (test compatibility).

