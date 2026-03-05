---
phase: 04-e2e-tts-models
plan: 3
subsystem: testing
tags: [e2e, harmonyspeech, tts, voice-cloning]
dependency_graph:
  requires: []
  provides:
    - tests/e2e/tts/test_harmonyspeech.py
    - harmonyspeech_engine fixture
  affects:
    - tests/e2e/conftest.py
tech_stack:
  added:
    - harmonyspeech_engine session-scoped pytest fixture
    - 4 HarmonySpeech E2E test functions
  patterns:
    - Full pipeline E2E testing via serving handlers
    - Direct stage access via engine.generate()
    - Session-scoped engine fixtures for model weight caching
key_files:
  created:
    - tests/e2e/tts/test_harmonyspeech.py
  modified:
    - tests/e2e/conftest.py
decisions:
  - "HarmonySpeech V1 is a pure voice cloner with no single-speaker component - only 4 tests included"
  - "Used actual sample audio (wanda4.wav) instead of silent audio to avoid division-by-zero warnings"
  - "Direct synthesis/vocode stages accessed via engine.generate() with SynthesisRequestInput/VocodeRequestInput"
metrics:
  duration: ~13 seconds (test execution)
  completed: 2026-03-05
---

# Phase 04 Plan 03: HarmonySpeech E2E Tests Summary

## Objective

Create true end-to-end tests for HarmonySpeech covering:
1. Voice cloning (full pipeline: embed reference → synthesize → vocode)
2. Direct stage access: embed-only, synthesize-only, vocode-only

## What Was Built

### Fixture: `harmonyspeech_engine`
- Session-scoped pytest fixture in `tests/e2e/conftest.py`
- Loads 3 HarmonySpeech component models:
  - `hs1-encoder` (HarmonySpeechEncoder)
  - `hs1-synthesizer` (HarmonySpeechSynthesizer)
  - `hs1-vocoder` (HarmonySpeechVocoder)
- Returns tuple: `(engine, serving_tts, serving_embed)`

### Test Functions (4 total)

| Test | Description | Path |
|------|-------------|------|
| `test_harmonyspeech_voice_cloning` | Full embed→synthesize→vocode pipeline via TTS endpoint | serving_tts.create_text_to_speech() |
| `test_harmonyspeech_embed_stage` | Direct encoder stage via EmbedSpeaker endpoint | serving_embed.create_voice_embedding() |
| `test_harmonyspeech_synthesize_stage` | Direct synthesizer stage via engine.generate() | SynthesisRequestInput |
| `test_harmonyspeech_vocode_stage` | Direct vocoder stage via engine.generate() | VocodeRequestInput |

All tests:
- Marked `@pytest.mark.e2e` and `@pytest.mark.slow`
- Use actual sample audio (`wanda4.wav`) from `tests/test-data/samples/`
- Run with `--device=cpu --dtype=float32`

## Verification

```bash
# Collect tests
python -m pytest tests/e2e/tts/test_harmonyspeech.py --collect-only -q
# Output: 4 tests collected

# Run tests
python -m pytest tests/e2e/tts/test_harmonyspeech.py -v
# Output: 4 passed in ~13s
```

## Deviations from Plan

### Single-Speaker Test Not Included
- **Original plan:** 5 tests including single-speaker TTS
- **Actual:** 4 tests (voice cloning + 3 direct stages)
- **Reason:** HarmonySpeech V1 is a pure voice cloner with no single-speaker component. The routing logic in `reroute_request_harmonyspeech()` requires `input_audio` for all TTS requests.

### Test Audio Source
- **Original plan:** Use `make_silent_wav_b64()` for reference audio
- **Actual:** Use `load_sample_audio_b64("wanda4")` 
- **Reason:** Silent audio causes division-by-zero warnings in `preprocess_wav()` due to `np.abs(wav).max()` returning 0

## Commits

- `6faed08` - feat(04-e2e-tts-models): add HarmonySpeech E2E tests

## Requirements Coverage (E2E-03)

| Requirement | Status | Notes |
|-------------|--------|-------|
| HarmonySpeech voice cloning flows through full toolchain | ✅ | test_harmonyspeech_voice_cloning |
| Direct embed-stage produces non-empty EmbedSpeakerResponse | ✅ | test_harmonyspeech_embed_stage |
| Direct synthesize-stage produces non-empty output | ✅ | test_harmonyspeech_synthesize_stage |
| Direct vocode-stage produces non-empty VocodeAudioResponse | ✅ | test_harmonyspeech_vocode_stage |
| All tests marked @pytest.mark.e2e | ✅ | Auto-added by pytest_collection_modifyitems |
| Tests run with --device=cpu --dtype=float32 | ✅ | Default pytest options |

## Self-Check

- [x] `tests/e2e/tts/test_harmonyspeech.py` exists
- [x] `harmonyspeech_engine` fixture is importable
- [x] pytest --collect-only discovers 4 tests
- [x] pytest -m e2e discovers 4 tests
- [x] All 4 tests pass
- [x] Commit 6faed08 exists
