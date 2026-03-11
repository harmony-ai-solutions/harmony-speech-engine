---
phase: 04-e2e-tts-models
plan: 4
subsystem: testing
tags: [ci, github-actions, pytest, e2e]

# Dependency graph
requires:
  - phase: 04-e2e-tts-models
    provides: E2E test files for each TTS model group
provides:
  - CI workflow with 4 parallel e2e jobs
  - Main test job excludes e2e tests
affects: [testing, ci]

# Tech tracking
tech-stack:
  added: [github-actions, pytest markers]
  patterns: [parallel ci jobs, test filtering]

key-files:
  created: []
  modified: [.github/workflows/test.yml]

key-decisions:
  - "Used pytest -m 'not e2e' to exclude e2e tests from main job"
  - "Used pytest -k flag for per-model-group test filtering in e2e jobs"

patterns-established:
  - "Parallel CI jobs: Each TTS model group runs in isolated job"

requirements-completed: [E2E-01, E2E-02, E2E-03]

# Metrics
duration: 2min
completed: 2026-03-05
---

# Phase 4 Plan 4: CI Workflow Update Summary

**Updated CI workflow with 4 parallel e2e jobs, each running tests for a specific TTS model group.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-05T01:43:00Z
- **Completed:** 2026-03-05T01:45:49Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Modified main test job to exclude e2e tests via `-m "not e2e"` marker
- Added 4 new parallel e2e CI jobs: e2e-kittentts, e2e-melotts-openvoice-v2, e2e-openvoice-v1, e2e-harmonyspeech
- Each e2e job runs only its model group's tests via pytest -k flag
- All e2e jobs have `needs: [test]` dependency for sequential execution after unit/integration tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Update CI workflow with parallel e2e jobs** - `6660bc8` (ci)

## Files Created/Modified

- `.github/workflows/test.yml` - Updated CI workflow with 6 total jobs (test, lint, 4 e2e jobs)

## Decisions Made

- Used pytest `-m "not e2e"` to exclude e2e tests from main job (as specified in plan)
- Used pytest `-k` flag for per-model-group test filtering in e2e jobs (as specified in plan)

## Deviations from Plan

None - plan executed exactly as written.

## Verification

All verification checks passed:

1. ✅ YAML valid and has correct 6 jobs: `['e2e-harmonyspeech', 'e2e-kittentts', 'e2e-melotts-openvoice-v2', 'e2e-openvoice-v1', 'lint', 'test']`
2. ✅ Main test job excludes e2e via `-m "not e2e"`
3. ✅ All 4 e2e jobs have `needs: [test]` dependency
4. ✅ All e2e jobs use correct `-k` selectors: kittentts, melotts or openvoice_v2, openvoice_v1, harmonyspeech

## Self-Check

- [x] File `.github/workflows/test.yml` exists
- [x] Commit `6660bc8` exists in git history
- [x] All 4 e2e jobs present with correct -k selectors
- [x] Main test job has `-m "not e2e"` flag
- [x] All e2e jobs have `needs: [test]`

## Self-Check: PASSED
