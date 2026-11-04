# Phase 3-2: Family Migration

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 3 — Declarative PipelineSpec routing
> **Depends on:** 3-1
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`

## Objective

Express every model family as a `PipelineSpec`, switch the engine's routing and forwarding to `next_step()`/`advance()`, and **delete** the `if/elif` ladders (`reroute_request_*` and the forwarding branches).

## Pipeline definitions (authoritative table)

| Spec id | Steps (in order) | final |
|---|---|---|
| `harmonyspeech` | `encoder` (HarmonySpeechEncoder) → `synthesizer` (HarmonySpeechSynthesizer) → `vocoder` (HarmonySpeechVocoder) | vocoder |
| `openvoice_v1` | `vad` (FasterWhisper | SileroVAD*) → `embed` (OpenVoiceV1ToneConverterEncoder) → `synthesize` (OpenVoiceV1Synthesizer, language-matched) → `tone_convert` (OpenVoiceV1ToneConverter) | tone_convert |
| `openvoice_v2` | `vad` (FasterWhisper | SileroVAD*) → `embed` (OpenVoiceV2ToneConverterEncoder) → `synthesize` (MeloTTSSynthesizer, language-matched) → `tone_convert` (OpenVoiceV2ToneConverter) | tone_convert |
| `voicefixer` | `restore` (VoiceFixerRestorer) → `vocode` (VoiceFixerVocoder) | vocode |
| `chatterbox` / `chatterbox_turbo` / `chatterbox_multilingual` | `embed` (ChatterboxEmbedding | variant TTS as fallback, optional) → `synthesize` (variant TTS) | synthesize |
| `chatterbox_vc` | `convert` (ChatterboxVC) | convert |
| `kitten_tts`† | `synthesize` (KittenTTSSynthesizer) | synthesize |
| `fasterwhisper`† / `silero_vad`† | single step | — |

\* Silero selection via `input_vad_mode` is wired in 3-4 — the `requires` tuple already declares the preference list.
† Single-step specs keep uniform machinery (their "routing" today is the explicit-model check at `harmonyspeech_engine.py:384`).

Step `needs`/`produces` per family follow the 2-3 hop table (e.g. `openvoice_v2.embed`: needs `{"input_audio","input_vad_data"}`, produces `{"input_embedding"}`).

## Implementation steps

### 1. Populate the registry

Write the table above into `PIPELINE_SPECS` (3-1). Chatterbox embed fallback: `requires=("ChatterboxEmbedding", "ChatterboxTTS")` with the embedding-preference rule preserved (`harmonyspeech_engine.py:300-317` logic, now data).

### 2. Replace routing

`check_reroute_request_to_model` (`:337-361`) becomes:

```python
spec = get_spec(request.requested_model)
if spec is None:
    return  # unknown model → 0-4 error path
if request.model in self.model_executors and not request_is_forwarded(request):
    return  # explicit single-model request (existing semantics, :384)
cfg = next_step(spec, request.pipeline_state, request, self.model_configs)
request.model = cfg.name if cfg else None   # None → 0-4 typed error
```

Delete `reroute_request_harmonyspeech/_openvoice_v1/_openvoice_v2/_voicefixer/_chatterbox/_chatterbox_vc` (`:109-335`).

### 3. Replace forwarding

`check_forward_processing` (`:363-506`): keep the error-result guard (`:379-381`) and the explicit-model short-circuit (`:384`). Replace every family branch with:

```python
done = advance(spec, engine_request, result.result_data)
if done:
    new_status = RequestStatus.FINISHED_STOPPED
    # final-step output → endpoint response type mapping (existing conversion for
    # TextToSpeechRequestOutput stays as one small, spec-driven mapping)
else:
    new_status = RequestStatus.FINISHED_FORWARDED
self.scheduler.update_request_status(result.request_id, new_status)
if not done:
    self.add_request(result.request_id, input_data)
```

The `TODO: Post-Generation Processing` at `:504` remains as the Phase 6 seam — `advance()` gains a documented extension point (append extra steps to the spec instance per request).

### 4. Mixed single-step semantics

Direct requests naming a concrete configured model (e.g. `model: "kitten-tts-mini"`), and the STT/VAD/embed endpoints driving single executors, must keep working: treat them as single-step specs with `pipeline_state` present but final (no forwarding). Verify against `serving_speech_to_text.py` / `serving_voice_embed.py` request paths (they set `requested_model`/`model` directly — re-read before editing).

### 5. Behavioral parity sweep

Every routing decision the old ladders made must be reproducible: build the routing test table in 3-5 *from the old code* before deleting it (transcribe the field-presence matrix per family as test cases), then run it against the new machinery.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/pipeline_registry.py` | full registry per the table |
| `harmonyspeech/engine/harmonyspeech_engine.py` | spec-driven routing/forwarding; delete `:109-335` reroute functions and forwarding branches |
| `tests/unit/inference_flow/` | parity table tests (see 3-5) |

## Tests

- Transcribed-parity tests (field-presence matrix per family → expected model sequence) — all green on new machinery.
- Forwarded requests: artifacts move per `produces`; no base64 fields written (2-3 invariant).
- Explicit-model and single-step requests unaffected.

## Checklist

- [ ] All 8+ specs registered exactly per the table
- [ ] Both ladders deleted; routing/forwarding spec-driven
- [ ] Error guard + explicit-model short-circuit preserved
- [ ] Phase-6 extension point documented at `advance()`
- [ ] Parity table tests transcribed from old code and passing
- [ ] `summary.md` Implementation Status updated
