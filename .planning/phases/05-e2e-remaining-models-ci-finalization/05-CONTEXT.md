# Phase 5: E2E Testing - Remaining Models & CI Finalization - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete E2E test coverage for STT (Whisper), VAD (Silero), and Audio Restoration (Voicefixer) models, and finalize the CI pipeline with comprehensive coverage reporting and documentation. Voice Conversion (OpenVoice) is excluded as it was already covered in Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Test Fixtures
- **Dedicated Fixtures:** Create separate pytest fixtures for each model type: `whisper_engine`, `vad_engine`, and `voicefixer_engine`.
- **Fixture Content:** Each fixture will return a tuple containing the `engine` and its corresponding `serving handler` (consistent with Phase 4 patterns like `melotts_en_engine`).
- **Model Variants:**
    - **Whisper (STT):** Test only with the `whisper-tiny` variant.
    - **Others (VAD, Voicefixer):** Test all available variants if they exist.

### Coverage & CI
- **Scope:** Coverage reporting will include everything under `harmonyspeech/`, **excluding** `harmonyspeech/modeling/models/` (which contains proprietary/adapted code not requiring unit tests).
- **E2E in CI:** E2E tests will be executed with coverage enabled to provide a full picture of the codebase's tested paths.

### Documentation (DOC-02)
- **Enhancements:** Update `docs/testing.md` to include:
    - Instructions on how to run specific test categories (unit, integration, e2e).
    - Guidance on how to interpret coverage reports, specifically explaining line vs. branch coverage.

</decisions>

<deferred>
## Deferred Ideas
*None captured during discussion.*
</deferred>
