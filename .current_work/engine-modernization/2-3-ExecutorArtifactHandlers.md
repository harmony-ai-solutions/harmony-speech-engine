# Phase 2-3: Executor Artifact Handlers

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 2 — Artifact bus
> **Depends on:** 2-2
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `CONCERNS.md`

## Objective

Convert every `_execute_*` handler in `model_runner_base.py` to consume artifacts from `prepare_*` and to **produce artifacts** for the next step — eliminating the base64/pickle/JSON-float intermediates on the four worst hops. Serialization to the client stays (via the serving layer) for final outputs only.

## Context (self-contained)

Hop-by-hop conversion targets (research §5.2 table; call sites verified 2026-11-04):

| Hop | Today (`model_runner_base.py`) | After |
|---|---|---|
| Whisper VAD → encoder | JSON text `input_vad_data`; encoder re-decodes `input_audio` (`:371-394` → `inputs.py:393-400`) | `VadArtifact` + shared `AudioArtifact` (already-decoded 16k/44.1k waveform reused) |
| Encoder → synth/TC | base64 pickle embedding (`:302-306`, `:332-333`) | `SpeakerArtifact.embedding` tensor |
| Synth → tone converter | base64 WAV (`:437-442` per report; TC decodes at `inputs.py:485-489`) | `AudioArtifact` waveform (no WAV encode/decode, no lossy intermediate) |
| Chatterbox embed → TTS | base64 pickle conditionals (`:833-838`) | `SpeakerArtifact.conditionals` object passed by reference (0-1's safe-serialization stays for the *external* embed endpoint response) |
| Chatterbox embed → VC | same, `.gen` extracted | `SpeakerArtifact.conditionals` |
| VoiceFixer restorer → vocoder | base64(JSON(list[float])) mel (`:544-549`) | `MelArtifact` tensor — removes ~35–45 MB churn / ~40–80 ms per hop |
| HarmonySpeech synth → vocoder | base64(JSON) mel + breaks (`:212-220`) | `MelArtifact` + `breaks: list[int]` carried alongside |
| Final outputs → client | WAV base64 (multiple sites) | unchanged in this task (serving converts; but the runner may hand the serving layer an `AudioArtifact` via a typed output — see step 3) |

## Implementation steps

### 1. Handler input side

Each handler's inner worker (already request-explicit per Phase 0-3) reads its inputs from the artifact tuples produced by 2-2's `prepare_*`. Torch-native handlers move artifacts to `self.device` explicitly (`artifact.to(self.device)`) and assert `artifact.device` at entry (research risk #8: device-mismatch bugs).

### 2. Handler output side — typed outputs

Outputs flow back as `RequestOutput` subclasses (`common/outputs.py`). Add an optional artifact slot to the relevant output classes (additive field, defaults `None`, ignored by wire serialization):

- `SpeechEmbeddingRequestOutput.embedding_artifact: SpeakerArtifact | None`
- `SpeechTranscriptionRequestOutput.vad_artifact: VadArtifact | None`
- `SpeechSynthesisRequestOutput.audio_artifact: AudioArtifact | None`, `.mel_artifact: MelArtifact | None` (HarmonySpeech)
- `AudioConversionRequestOutput.audio_artifact: AudioArtifact | None`, `.mel_artifact: MelArtifact | None` (VoiceFixer restorer)

Handlers set both the legacy string output (for external callers of that step — e.g. `/v1/embed/speaker` clients) **and** the artifact. Cost: one extra reference; no serialization.

### 3. `check_forward_processing` moves artifacts, not strings

`harmonyspeech_engine.py:363-506`: each forwarding branch (whisper→embed, embed→synth, synth→TC, restorer→vocoder, chatterbox embed→tts) currently base64-encodes the result into the request field. Change to: copy the **artifact** from `result.result_data.*_artifact` into `engine_request.artifacts[field]` (the precedence rule from 2-2 makes the next step read it). Keep setting the base64 field **only** where an external client could observe the intermediate (none today — intermediates are internal), so: do not set base64 on forward; leave the old field `None`.

Edge case — Chatterbox conditionals via `SpeakerArtifact.conditionals`: the object holds device tensors; it must be moved to the target executor's device by the consuming handler (TTS/VC), not by the bus (explicit moves only).

### 4. Final-output artifact (optional but recommended)

For TTS/VC final outputs, set `audio_artifact` on the output and let the serving layer encode from the artifact **or** from the base64 (whichever is present). This removes one WAV encode for client responses when 1-2's `apply_output_options` needs to re-encode anyway (decode step disappears). Keep base64 emission as fallback so nothing breaks.

### 5. VoiceFixer breaks & HarmonySpeech vocoder pairing

Preserve the exact mel/breaks pairing semantics (re-read `:212-220`, `:528-598` before editing; the vocoder inserts silence via `breaks` — keep the logic, change the container).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | all handlers artifact-in/artifact-out; device asserts |
| `harmonyspeech/common/outputs.py` | optional artifact fields on output classes |
| `harmonyspeech/engine/harmonyspeech_engine.py` | `check_forward_processing` forwards artifacts |
| `tests/unit/inference_flow/`, `tests/e2e/` | artifact-hop tests + parity |

## Tests

- Unit per converted hop: artifact produced by step A is the identical object (or equal tensor) consumed by step B; **no** base64 field set after forwarding.
- Unit: VoiceFixer restorer→vocoder — assert `json.dumps`/`base64` not called on the mel path (monkeypatch counters).
- Unit: Chatterbox embed→tts passes conditionals object by reference (identity check with a stub).
- E2E parity: all families' outputs byte-comparable to pre-Phase-2 (fixed text, fixed seed where models are deterministic; for stochastic ones compare duration + non-silence + spectral sanity).
- Device test (if GPU available): CPU artifact into GPU executor moves exactly once.

## Checklist

- [ ] All 8 hops from the table converted; base64 intermediates gone from internal flow
- [ ] Output classes carry optional artifacts; external endpoints unchanged
- [ ] `check_forward_processing` forwards artifacts (no re-encode)
- [ ] Device asserts + explicit moves in handlers
- [ ] Hop unit tests + e2e parity green
- [ ] `summary.md` Implementation Status updated
