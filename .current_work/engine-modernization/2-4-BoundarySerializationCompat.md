# Phase 2-4: Boundary Serialization & Compatibility

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 2 — Artifact bus
> **Depends on:** 2-3
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `INTEGRATIONS.md`

## Objective

Formalize the rule "**serialization happens only in `serving_*.py`**" as code: `from_wire()` at admission (bytes/base64/multipart → request + artifacts) and `to_wire()` at response (artifacts → requested output format). Verify full backward compatibility: every existing client (including Soulbits' `cmd/worker` `HSEAdapter`, which treats HSE's base64 JSON as opaque) keeps working without changes.

## Context (self-contained)

- Inbound: serving modules build `RequestInput.from_openai(...)` from protocol models; Phase 1-3 added multipart; Phase 2-2 added decode-once at admission. These now live in slightly different places — consolidate into one `from_wire` helper per serving module (or one shared function parameterized by endpoint type).
- Outbound: Phase 1-2's `apply_output_options` decodes base64 WAV → converts. With 2-3's `audio_artifact` on outputs, `to_wire` should encode directly from the artifact, skipping the decode.
- The engine core (`harmonyspeech_engine.py`, runners, scheduler) must contain **zero** `base64` calls for audio after this task (grep-able invariant; embeddings served externally keep their 0-1-safe serialization at the serving layer).

## Implementation steps

### 1. Shared wire helpers

`harmonyspeech/endpoints/openai/wire.py` (new):

```python
def from_wire_audio(protocol_request, engine_request, fields: list[str]) -> None:
    """For each named base64/file field: decode via audio_codec (1-1) → store AudioArtifact
    in engine_request.artifacts; leave base64 fields as sent (compat/observability)."""

def to_wire_audio(result_output, output_options) -> str:
    """Prefer audio_artifact (encode directly, honoring output_options via 1-2 helpers);
    fall back to legacy base64 WAV string; returns response `data` payload."""
```

### 2. Apply to all audio serving modules

`serving_text_to_speech.py`, `serving_voice_conversion.py`, `serving_audio_conversion.py`, `serving_speech_to_text.py`, `serving_voice_embed.py`, `serving_voice_activity_detection.py`: admission calls `from_wire_audio`; response calls `to_wire_audio`. External embedding responses (speak/embed endpoints) serialize `SpeakerArtifact` via the 0-1-safe format at this seam only.

### 3. Engine-core serialization purge

- Remove now-dead base64/JSON encode paths from `model_runner_base.py` handlers **for internal hops** (kept only where an output is directly client-visible and no artifact exists — ideally none after 2-3 step 4).
- Add the grep invariant to CI or a unit test: `grep -rn "base64" harmonyspeech/` may only match `endpoints/`, `common/inputs.py` (field definitions), `common/outputs.py` (legacy fields), and `common/audio_codec.py`. Encode the check as a small pytest that scans files and asserts the allowed set (keeps future contributors honest).

### 4. Compatibility verification matrix

- JSON base64 in → base64 WAV out (no options): **byte-identical** to pre-Phase-2 for deterministic models.
- Multipart in → format-converted out (Phase 1 paths).
- Embedding endpoints: response format unchanged (0-1-safe base64 payload).
- `cmd/worker` HSEAdapter contract (soulbits-cloud-backend): opaque JSON envelope unchanged — verify field names/types of response models did not change.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/wire.py` (new) | `from_wire_audio` / `to_wire_audio` |
| All six `serving_*.py` | use wire helpers |
| `harmonyspeech/task_handler/model_runner_base.py` | purge internal base64 paths |
| `tests/unit/` (new) | `test_serialization_boundary.py` (grep invariant + wire helpers) |

## Tests

- Unit: grep-invariant test passes.
- Unit: `to_wire_audio` from artifact == from legacy base64 for the same audio (byte-equal WAV).
- E2E: full compat matrix (step 4) recorded in this file when done.

## Checklist

- [x] `wire.py` helpers exist; all serving modules use them
- [x] Engine core free of audio base64 (invariant test)
- [x] Compat matrix verified (JSON path byte-identical; HSEAdapter envelope unchanged)
- [x] `summary.md` Implementation Status updated

## Compatibility matrix (verified 2026-11-04)

| Case | Result |
|---|---|
| JSON base64 in → base64 WAV out, no options | **byte-identical** — `encode_artifact` produces the same PCM-16 WAV bytes the executor wrote (`test_to_wire_audio_from_artifact_matches_legacy_payload`); real-model e2e defaults pass unchanged |
| Multipart in → format-converted out | green (`tests/e2e/wire_contract`, 38 passed / 3 model-only skips) |
| Output format/sample-rate options | green; `encode_artifact` reports the same `sample_rate` echo as `apply_output_options` |
| Embedding endpoints | response format unchanged: `to_wire_embedding` reproduces the legacy payload per backend (raw float32 / `torch.save` tensor / 0-1-safe conditionals) — byte-equality unit tests |
| `cmd/worker` HSEAdapter envelope | unchanged: response models (`data`, `sample_rate`, `text`, `language`, `timestamps`, `speech_activity`) and field names/types are untouched; only internal containers changed |
| Engine core wire encoding | none: `test_engine_core_contains_no_wire_encoding` scans `harmonyspeech/` and allows only `common/audio_codec.py`, `common/inputs.py`, `common/outputs.py`, `endpoints/` |

## Implementation notes (2026-11-04)

- `harmonyspeech/endpoints/openai/wire.py` provides `from_wire_audio`,
  `to_wire_audio`, `to_wire_embedding`; every serving module calls
  `from_wire_audio` at admission, and the audio/embedding responses go through
  the outbound helpers.
- `output_encoding.encode_artifact` is the artifact-side counterpart of
  `apply_output_options` (no base64 WAV decode; one encode honoring
  `output_format`/`sample_rate`).
- `serving_voice_embed` is the only place a `SpeakerArtifact` is serialized for
  a client; the engine core never serializes embeddings.
- The multipart envelope still performs its documented in-process base64
  round-trip for exact JSON parity; it is now immediately decoded into artifacts
  at admission, so no pipeline step ever touches it.

