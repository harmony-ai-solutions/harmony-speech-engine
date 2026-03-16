# Chatterbox Audio Enhancements — Implementation Plan (Phase A + Phase B)

## Overview

This plan implements **Phase A and Phase B audio quality improvements** for the Harmony Speech Engine (HSE), inspired by the feature set of the [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server).

The features are designed to be **model-agnostic** where possible — particularly text chunking+stitching, speed factor, and all Phase B post-processing filters, which benefit all HSE TTS models.

### Phase A Features

| Feature | Scope | Priority |
|---|---|---|
| **Text chunking + smart crossfade stitching** | All TTS models (model-agnostic) | 🔴 Highest |
| **Speed factor post-processing** (librosa pitch-preserving) | Chatterbox models only (others use native speed) | 🔴 High |
| **Peak normalization / clip prevention** | All TTS models (model-agnostic) | 🟡 Medium |

### Phase B Features

| Feature | Scope | Priority |
|---|---|---|
| **DC offset removal** (`remove_dc_offset`) | All TTS models; applied per-chunk before stitching | 🟡 Medium |
| **Leading/trailing silence trimming** (`trim_silence`) | All TTS models; applied to final audio | 🟡 Medium |
| **Internal silence reduction** (`fix_internal_silence`) | All TTS models; applied to final audio | 🟡 Medium |

### Design Principles

1. **Model-agnostic** — Chunking and stitching live in the serving layer (`serving_text_to_speech.py`), not in any model runner, so all TTS model types benefit automatically.
2. **Opt-in per request** — Clients can control chunking via new request fields (`split_text`, `chunk_size`). Disabled by default to preserve backward compatibility.
3. **Non-breaking** — All new fields are optional with sensible defaults. Existing clients continue to work unchanged.
4. **Reuse existing infrastructure** — Hooks into `GenerationOptions` and `AudioOutputOptions` in `protocol.py` and the `TextToSpeechRequest` Pydantic model.
5. **Native speed guard** — `NATIVE_SPEED_MODEL_TYPES` frozenset in `audio_utils.py` identifies model types that handle speed internally; post-synthesis time-stretch is skipped for these.

### Codebase Mapping Consulted

- [`.planning/codebase/ARCHITECTURE.md`](../../.planning/codebase/ARCHITECTURE.md) — Layer overview, data flow, serving module pattern
- [`.planning/codebase/CONVENTIONS.md`](../../.planning/codebase/CONVENTIONS.md) — Coding conventions, naming, imports

---

## Files Modified / Created

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | **NEW** — Shared audio post-processing utilities |
| `harmonyspeech/endpoints/openai/protocol.py` | **MODIFY** — Add `split_text`, `chunk_size`; replace `norm_loudness` with `normalize_audio`; add `crossfade_ms`, `sentence_pause_ms` |
| `harmonyspeech/common/inputs.py` | **MODIFY** — Rename `norm_loudness` → `normalize_audio` in `TextToSpeechGenerationOptions` and `TextToSpeechRequestInput` |
| `harmonyspeech/task_handler/inputs.py` | **MODIFY** — Update `prepare_chatterbox_turbo_tts_inputs()` to use `normalize_audio`; update `NATIVE_SPEED_MODEL_TYPES` guard |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | **MODIFY** — Add chunking dispatcher, stitching, speed factor, normalization |
| `tests/unit/inference_flow/test_chatterbox_inputs.py` | **MODIFY** — Update `CHATTERBOX_FIELD_NAMES`, remove `test_tts_rejects_norm_loudness`, update Turbo result tuple |
| `tests/unit/test_audio_utils.py` | **NEW** — Unit tests for audio utility functions |
| `tests/e2e/tts/test_chatterbox.py` | **MODIFY** — Add chunking, speed, normalize_audio tests |
| `tests/e2e/tts/test_chatterbox_turbo.py` | **MODIFY** — Rename `norm_loudness` → `normalize_audio`; add chunking, speed, normalize tests |
| `tests/e2e/tts/test_chatterbox_multilingual.py` | **MODIFY** — Add chunking, speed, normalize_audio tests |
| `tests/e2e/tts/test_harmonyspeech.py` | **MODIFY** — Add chunking and normalize_audio tests |
| `tests/e2e/tts/test_kittentts.py` | **MODIFY** — Add chunking and normalize_audio tests for all 4 variants |
| `tests/e2e/tts/test_melotts.py` | **MODIFY** — Add chunking and normalize_audio tests |
| `tests/e2e/tts/test_openvoice_v1.py` | **MODIFY** — Add chunking and normalize_audio tests |
| `docs/api.md` | **MODIFY** — Document new request fields and updated `GenerationOptions` |
| `docs/models.md` | **MODIFY** — Note native vs post-synthesis speed per model type |
| `CHANGELOG.md` | **MODIFY** — Add changelog entry |
| `README.md` | **MODIFY** — Replace any `norm_loudness` occurrences |

**Phase B additions:**

| File | Change |
|---|---|
| `harmonyspeech/common/audio_utils.py` | **MODIFY** — Add `remove_dc_offset()`, `trim_silence()`, `fix_internal_silence()` |
| `harmonyspeech/endpoints/openai/protocol.py` | **MODIFY** — Add `remove_dc_offset`, `trim_silence`, `fix_internal_silence` to `GenerationOptions` |
| `harmonyspeech/common/inputs.py` | **MODIFY** — Add same three fields to `TextToSpeechGenerationOptions` |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | **MODIFY** — Wire Phase B flags in single and chunked synthesis paths |
| `tests/unit/test_audio_utils.py` | **MODIFY** — Add `TestRemoveDcOffset`, `TestTrimSilence`, `TestFixInternalSilence` test classes |
| `tests/e2e/tts/test_chatterbox.py` | **MODIFY** — Add Phase B tests (single_speaker + voice_cloning) |
| `tests/e2e/tts/test_chatterbox_turbo.py` | **MODIFY** — Add Phase B tests (single_speaker + voice_cloning with embed step) |
| `tests/e2e/tts/test_chatterbox_multilingual.py` | **MODIFY** — Add Phase B tests (single_speaker + voice_cloning) |
| `tests/e2e/tts/test_harmonyspeech.py` | **MODIFY** — Add Phase B tests (voice_cloning) |
| `tests/e2e/tts/test_kittentts.py` | **MODIFY** — Add Phase B tests for all 4 variants (single_speaker only) |
| `tests/e2e/tts/test_melotts.py` | **MODIFY** — Add Phase B tests (single_speaker + voice_cloning) |
| `tests/e2e/tts/test_openvoice_v1.py` | **MODIFY** — Add Phase B tests (single_speaker + voice_cloning) |

---

## Implementation Status

Track the completion of each phase as implementation progresses:

- [ ] **Phase 1: Audio Utilities Module**
  - [ ] Core audio utilities ([1-1-AudioUtilitiesModule.md](1-1-AudioUtilitiesModule.md))
- [ ] **Phase 2: Protocol Additions**
  - [ ] TTS Request Protocol Fields ([2-1-ProtocolAdditions.md](2-1-ProtocolAdditions.md))
- [ ] **Phase 3: Serving Layer — Text Chunking and Stitching**
  - [ ] Serving Text-to-Speech Chunking Logic ([3-1-ServingLayerChunking.md](3-1-ServingLayerChunking.md))
- [ ] **Phase 4: Speed Factor and Peak Normalization Wiring**
  - [ ] Speed Factor and Normalization in Serving ([4-1-SpeedFactorNormalization.md](4-1-SpeedFactorNormalization.md))
- [ ] **Phase 5: Tests and Documentation (Phase A)**
  - [ ] Unit Tests for Audio Utils and Existing Test Updates ([5-1-UnitTests.md](5-1-UnitTests.md))
  - [ ] E2E Tests — Chatterbox Suites (single_speaker + voice_cloning) ([5-2-E2ETests-Chatterbox.md](5-2-E2ETests-Chatterbox.md))
  - [ ] E2E Tests — Other Model Suites (single_speaker + voice_cloning where supported) ([5-3-E2ETests-Other-models.md](5-3-E2ETests-Other-models.md))
  - [ ] API and Documentation Update ([5-4-DocumentationUpdate.md](5-4-DocumentationUpdate.md))
- [ ] **Phase 6: Phase B — Audio Utilities**
  - [ ] DC Offset Removal, Silence Trimming, Internal Silence Reduction ([6-1-AudioUtilitiesPhaseB.md](6-1-AudioUtilitiesPhaseB.md))
- [ ] **Phase 7: Phase B — Protocol Additions**
  - [ ] Add `remove_dc_offset`, `trim_silence`, `fix_internal_silence` fields ([7-1-ProtocolPhaseB.md](7-1-ProtocolPhaseB.md))
- [ ] **Phase 8: Phase B — Serving Layer Wiring**
  - [ ] Wire Phase B flags into single and chunked synthesis paths ([8-1-ServingLayerPhaseB.md](8-1-ServingLayerPhaseB.md))
- [ ] **Phase 9: Phase B — Tests**
  - [ ] Unit Tests for Phase B Audio Utilities ([9-1-UnitTests-PhaseB.md](9-1-UnitTests-PhaseB.md))
  - [ ] E2E Tests — Chatterbox Suites Phase B (single_speaker + voice_cloning) ([9-2-E2ETests-Chatterbox-PhaseB.md](9-2-E2ETests-Chatterbox-PhaseB.md))
  - [ ] E2E Tests — Other Model Suites Phase B (single_speaker + voice_cloning where supported) ([9-3-E2ETests-Other-models-PhaseB.md](9-3-E2ETests-Other-models-PhaseB.md))
