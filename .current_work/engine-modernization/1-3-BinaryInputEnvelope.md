# Phase 1-3: Binary Input Envelope (Multipart)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 1 — Wire contract
> **Depends on:** 1-1 (decoder is format-agnostic already)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `INTEGRATIONS.md`

## Objective

Let clients upload audio as **binary multipart/form-data files** instead of base64 inside JSON, on every endpoint that accepts audio input. This is a *transport envelope change only* — the payload is still complete-then-process, with identical request semantics, so there is **no streaming and no cutoff/stutter risk**. Benefits: ~25% smaller uploads (no base64 inflation), no giant-JSON parse, and standard OpenAI-SDK / `curl -F` compatibility.

## Context (self-contained)

Endpoints accepting base64 audio today (all in `harmonyspeech/endpoints/openai/`):

| Endpoint | Serving module | Audio fields |
|---|---|---|
| `/v1/audio/speech` (voice cloning) | `serving_text_to_speech.py` | `input_audio` |
| `/v1/audio/transcriptions` | `serving_speech_to_text.py` | `input_audio` |
| `/v1/audio/vad` | `serving_voice_activity_detection.py` | `input_audio` |
| `/v1/embed/speaker` | `serving_voice_embed.py` | `input_audio` |
| `/v1/voice/convert` | `serving_voice_conversion.py` | `source_audio`, `target_audio` |
| `/v1/audio/convert` | `serving_audio_conversion.py` | `source_audio` |

All consume JSON models from `protocol.py` with `str` base64 fields.

## Implementation steps

### 1. Add multipart request variants

For each endpoint above, add a sibling route (e.g. same path, `POST`, declared with `File(...)`/`Form(...)` parameters) that accepts:

- `audio` (or `source_audio` + `target_audio` for VC/AC): `UploadFile`
- remaining scalar options as `Form(...)` fields (model, mode, language, voice, …)

Reconstruct the **existing** `Request*` model internally: read file bytes → `base64.b64encode(...).decode()` into the same field the JSON path uses → hand over to the unchanged handler. This guarantees behavior parity and keeps the diff small; the base64 round-trip inside the process is acceptable for now and disappears in Phase 2 (artifacts) if worth it.

```python
@router.post("/v1/audio/transcriptions", ...)  # existing JSON route unchanged

@router.post("/v1/audio/transcriptions", ...)  # content-type negotiated variant
async def create_transcription_multipart(
    audio: Annotated[UploadFile, File(description="Audio file (wav/flac/ogg/mp3/aac...)")],
    model: Annotated[str, Form()],
    ...
):
    request = SpeechTranscribeRequest(model=model, input_audio=b64(await audio.read()), ...)
    # delegate to the same internal path as the JSON route
```

FastAPI supports two request bodies per path via explicit `openapi_extra` or separate paths — decide per endpoint whether to register a distinct path (e.g. `/v1/audio/transcriptions` stays JSON; multipart accepted on the same path via content-type detection with a single route declaring both a JSON body and File params is *not* clean in FastAPI — prefer **two routes, one path is not possible; use the same path with `request.stream()` sniffing OR a `/multipart` suffix**). Recommended: accept multipart on the **same path** by declaring the route parameter as the JSON model *or* files is messy — instead follow the OpenAI convention: these endpoints are *already* multipart in the OpenAI API (`-F file=@x.mp3`). Therefore: **make multipart the primary body for the STT/VAD/embed/VC/AC endpoints** (matching OpenAI clients), keep the JSON body as the documented legacy alternative on the same path via a manual content-type dispatch in the route handler (`request.headers.get("content-type")` → parse as Form or JSON). For `/v1/audio/speech`, keep JSON primary (OpenAI speech API is JSON) and add optional multipart for the cloning `input_audio` field only.

### 2. Size guards

Multipart bypasses Pydantic length intuition — add an explicit max upload size (e.g. 50 MB configurable via args/env, consistent with gateway limits) returning 413 with a clear message.

### 3. Docs & OpenAPI

Ensure the OpenAPI schema shows both content types (FastAPI generates this automatically for the multipart variant; for the manual dispatch route, document in the docstring + `docs/` page with `curl -F` examples).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_speech_to_text.py` | multipart primary body + JSON legacy |
| `harmonyspeech/endpoints/openai/serving_voice_activity_detection.py` | same |
| `harmonyspeech/endpoints/openai/serving_voice_embed.py` | same |
| `harmonyspeech/endpoints/openai/serving_voice_conversion.py` | same (two files) |
| `harmonyspeech/endpoints/openai/serving_audio_conversion.py` | same |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | optional multipart `input_audio` for cloning |
| `harmonyspeech/endpoints/openai/api_server.py` | upload-size limit middleware/param |
| `docs/` | transport guide with curl examples |

## Tests

- E2E per endpoint: multipart upload of a wav fixture → identical response to the JSON/base64 path (compare decoded audio hashes for TTS-input-dependent endpoints, response fields for STT/VAD/embed).
- E2E: mp3 + aac uploads work end-to-end (exercising 1-1 decoder through the new envelope).
- E2E: oversized upload → 413.
- E2E: JSON path still works unchanged (all existing e2e tests already cover this).

## Checklist

- [ ] Multipart accepted on all six audio-input endpoints (OpenAI-style `file` fields)
- [ ] JSON/base64 paths unchanged and still primary for `/v1/audio/speech`
- [ ] Upload size limit with 413
- [ ] Parity tests (multipart vs JSON same result) added
- [ ] Docs with curl/SDK examples
- [ ] `CHANGELOG.md` entry
- [ ] `summary.md` Implementation Status updated
