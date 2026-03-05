---
phase: 05-e2e-remaining-models-ci-finalization
plan: 3
subsystem: testing
tags: [pytest, coverage, ci, github-actions, e2e]

# Dependency graph
requires:
  - phase: 05-e2e-remaining-models-ci-finalization
    provides: E2E tests for Whisper, SileroVAD, VoiceFixer models
provides:
  - pytest-cov configuration for coverage reporting
  - 3 new parallel e2e CI jobs (whisper, vad, voicefixer)
  - Coverage artifact upload in main test job
  - Updated testing documentation with coverage interpretation
affects: [testing, ci]

# Tech tracking
tech-stack:
  added: [pytest-cov]
  patterns: [CI parallelization, coverage reporting]

key-files:
  created: []
  modified:
    - pyproject.toml (added pytest-cov addopts)
    - .github/workflows/test.yml (added coverage upload + 3 e2e jobs)
    - docs/testing.md (added coverage interpretation guide)

key-decisions:
  - "Coverage excludes harmonyspeech/modeling/models (third-party code)"
  - "E2E jobs use --no-cov to avoid misleading partial coverage"
  - "Branch coverage (BrCov) explained as stricter than line coverage"

patterns-established:
  - "CI parallel e2e jobs with needs: [test] dependency"

requirements-completed: [CI-03, DOC-02]

# Metrics
duration: 5min
completed: 2026-03-05
---

# Phase 5 Plan 3: CI Coverage and Documentation Summary

**Added pytest-cov coverage reporting to CI, three new parallel e2e jobs for Whisper/SileroVAD/VoiceFixer, and comprehensive testing documentation.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-05T23:33:31Z
- **Completed:** 2026-03-05T23:38:31Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added pytest-cov configuration to pyproject.toml for automatic coverage on all test runs
- Added coverage.xml artifact upload to main test job in GitHub Actions
- Created 3 new parallel e2e CI jobs: e2e-whisper, e2e-vad, e2e-voicefixer
- Updated docs/testing.md with guidance on running specific test categories
- Added comprehensive coverage report interpretation (line vs branch coverage)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add coverage to pyproject.toml and update CI with coverage + 3 new e2e jobs** - `f1a6036` (feat)
2. **Task 2: Update docs/testing.md with category run instructions and coverage interpretation** - `f1a6036` (feat)

**Plan metadata:** `f1a6036` (docs: complete plan)

## Files Created/Modified
- `pyproject.toml` - Added pytest-cov addopts for coverage reporting
- `.github/workflows/test.yml` - Added coverage upload step and 3 new e2e jobs
- `docs/testing.md` - Added test category run instructions and coverage interpretation guide

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- [x] pyproject.toml addopts includes --cov flags
- [x] test.yml has e2e-whisper, e2e-vad, e2e-voicefixer jobs with needs: [test]
- [x] Main test job uploads coverage.xml as artifact
- [x] docs/testing.md explains line vs branch coverage (BrCov column)
- [x] docs/testing.md explains how to run each test category individually

## Self-Check: PASSED

All files exist and commit verified.
