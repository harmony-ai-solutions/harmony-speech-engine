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

- [ ] `wire.py` helpers exist; all serving modules use them
- [ ] Engine core free of audio base64 (invariant test)
- [ ] Compat matrix verified (JSON path byte-identical; HSEAdapter envelope unchanged)
- [ ] `summary.md` Implementation Status updated
