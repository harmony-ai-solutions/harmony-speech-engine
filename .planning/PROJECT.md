# Harmony Speech Engine Testing Framework

## What This Is

A comprehensive testing framework for Harmony Speech Engine that enables unit, integration, and end-to-end testing of all models and toolchains (TTS, STT, Voice Conversion, VAD, Audio Restoration) individually. Built on pytest with fixtures, mocking support, and CI/CD integration. All tests execute in CPU mode for CI compatibility.

## Core Value

Enable reliable, automated verification of all model inference pipelines through comprehensive test coverage that runs in CI environments without GPU dependencies.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implement pytest-based test framework with fixtures and configuration
- [ ] Create unit tests for core components (config, engine, model loaders)
- [ ] Create integration tests for API endpoints
- [ ] Create end-to-end tests for all TTS models (KittenTTS, MeloTTS, HarmonySpeech)
- [ ] Create end-to-end tests for STT model (Whisper)
- [ ] Create end-to-end tests for Voice Conversion (OpenVoice)
- [ ] Create end-to-end tests for VAD
- [ ] Create end-to-end tests for Audio Restoration (Voicefixer)
- [ ] Add CI/CD configuration for automated test execution
- [ ] Add test documentation and reporting

### Out of Scope

- GPU-based tests — CI environment is CPU-only; GPU testing done manually
- Load/stress testing — defer to future phase
- Security penetration testing — defer to future phase

## Context

Harmony Speech Engine is a unified, high-performance inference platform consolidating multiple open-source speech AI technologies. Currently, no tests exist in the codebase. The existing `.planning/codebase/TESTING.md` provides recommendations but has not been implemented.

The engine supports:
- **TTS**: KittenTTS, MeloTTS, HarmonySpeech
- **STT**: Whisper (via Faster-Whisper)
- **Voice Conversion**: OpenVoice V1/V2
- **VAD**: Voice Activity Detection
- **Audio Restoration**: Voicefixer

## Constraints

- **CPU-Only Execution**: All tests must run in CPU mode — GPU tests are out of scope for CI compatibility
- **Python 3.12+**: Target Python version matches engine requirements
- **pytest Framework**: Must use pytest as the testing framework per existing recommendations
- **CI/CD Integration**: Tests must integrate with GitHub Actions workflow

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| pytest over unittest | Industry standard, rich plugin ecosystem, better fixture support | — Pending |
| CPU-only for CI | Ensures reproducibility, no GPU dependency, faster execution | — Pending |
| Model-by-model test structure | Enables individual model testing and debugging | — Pending |

---
*Last updated: 2026-02-28 after project initialization*
