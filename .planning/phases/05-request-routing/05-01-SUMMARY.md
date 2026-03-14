---
phase: request-routing
plan: 1
subsystem: engine-routing
tags:
  - chatterbox
  - request-routing
  - engine
  - serving-layer
dependency_graph:
  requires:
    - 04-01
  provides:
    - chatterbox-routing
  affects:
    - harmonyspeech_engine
    - serving_text_to_speech
    - serving_voice_embed
    - serving_voice_conversion
tech_stack:
  added:
    - ChatterboxTTS
    - ChatterboxTurboTTS
    - ChatterboxMultilingualTTS
    - ChatterboxEmbedding
    - ChatterboxVC
  patterns:
    - reroute_request_* pattern
    - check_forward_processing multi-step chain
key_files:
  created: []
  modified:
    - harmonyspeech/engine/harmonyspeech_engine.py
    - harmonyspeech/endpoints/openai/serving_text_to_speech.py
    - harmonyspeech/endpoints/openai/serving_voice_embed.py
    - harmonyspeech/endpoints/openai/serving_voice_conversion.py
decisions:
  - "Chatterbox uses embed→synthesize multi-step chain similar to OpenVoice"
  - "All 3 TTS variants (chatterbox, chatterbox_turbo, chatterbox_multilingual) share reroute_request_chatterbox()"
  - "Group key for embedding is 'chatterbox' (not 'chatterbox_embedding') to match sentinel"
metrics:
  duration: null
  completed_date: "2026-03-14"
---

# Phase 05 Plan 01: Request Routing — Engine and Serving Layer Summary

## Objective

Wired up Chatterbox request routing across three layers:
1. **Engine rerouting** — two new `reroute_request_chatterbox*()` methods + extensions to `check_reroute_request_to_model()` in `harmonyspeech_engine.py`
2. **Forward processing** — new Chatterbox block in `check_forward_processing()` to handle the embed→synthesize multi-step chain
3. **Serving layer** — registered Chatterbox model types/groups in `serving_text_to_speech.py`, `serving_voice_embed.py`, and `serving_voice_conversion.py`

## One-Liner

Chatterbox request routing with embed→synthesize multi-step chain support

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add reroute_request_chatterbox() | 0f68364 | harmonyspeech_engine.py |
| 2 | Add reroute_request_chatterbox_vc() | 0f68364 | harmonyspeech_engine.py |
| 3 | Extend check_reroute_request_to_model() | 0f68364 | harmonyspeech_engine.py |
| 4 | Add Chatterbox block to check_forward_processing() | 0f68364 | harmonyspeech_engine.py |
| 5 | Add Chatterbox types/groups to serving_text_to_speech.py | 0f68364 | serving_text_to_speech.py |
| 6 | Add ChatterboxEmbedding to serving_voice_embed.py | 0f68364 | serving_voice_embed.py |
| 7 | Add ChatterboxVC to serving_voice_conversion.py | 0f68364 | serving_voice_conversion.py |

## Verification Results

- [x] `check_reroute_request_to_model()` handles all 4 Chatterbox sentinels
- [x] `reroute_request_chatterbox()` routes `input_audio` present → `ChatterboxEmbedding`
- [x] `reroute_request_chatterbox()` routes `input_audio` absent → correct TTS variant
- [x] `reroute_request_chatterbox()` routes `SpeechEmbeddingRequestInput` → `ChatterboxEmbedding`
- [x] `reroute_request_chatterbox_vc()` routes `VoiceConversionRequestInput` → `ChatterboxVC`
- [x] `check_forward_processing()` Chatterbox block re-submits with `input_audio=None` + `input_embedding` set
- [x] `_TTS_MODEL_TYPES` has 6 entries (3 existing + 3 new Chatterbox)
- [x] `_TTS_MODEL_GROUPS` has 6 entries (3 existing + 3 new Chatterbox)
- [x] `_EMBEDDING_MODEL_TYPES` has 2 entries (1 existing + ChatterboxEmbedding)
- [x] `_EMBEDDING_MODEL_GROUPS` has 4 entries (3 existing + chatterbox)
- [x] `_VOICE_CONVERSION_MODEL_TYPES` has 3 entries (2 existing + ChatterboxVC)
- [x] `_VOICE_CONVERSION_MODEL_GROUPS` has 3 entries (2 existing + chatterbox_vc)

## Success Criteria (from ROADMAP Phase 5)

1. ✅ TTS requests without voice cloning route directly to TTS model — `reroute_request_chatterbox()` `input_audio is None` + `input_embedding is None` → ChatterboxTTS*
2. ✅ TTS requests with pre-computed embedding route directly to TTS model — `reroute_request_chatterbox()` `input_audio is None` + `input_embedding is not None` → ChatterboxTTS*
3. ✅ TTS requests with `input_audio` route to Embedding first, then forward to TTS — routes to `ChatterboxEmbedding`; `check_forward_processing()` then forwards with `input_embedding` set
4. ✅ Embedding requests route to ChatterboxEmbedding model — `reroute_request_chatterbox()` `SpeechEmbeddingRequestInput` → `ChatterboxEmbedding`
5. ✅ VoiceConversion requests route to ChatterboxVC model — `reroute_request_chatterbox_vc()` → `ChatterboxVC`
6. ✅ Forward processing transfers embedding from embed step to synthesize step — `check_forward_processing()` Chatterbox block sets `input_audio=None`, `input_embedding=result`, re-submits

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] All modified files exist
- [x] Commit hash 0f68364 exists
- [x] All verification checks passed