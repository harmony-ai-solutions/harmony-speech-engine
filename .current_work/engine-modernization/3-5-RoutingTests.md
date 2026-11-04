# Phase 3-5: Routing Tests & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 3 — Declarative PipelineSpec routing
> **Depends on:** 3-1…3-4
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`

## Objective

Lock the new routing in with a dedicated, exhaustive test suite, verify the phase's exit criteria, and update knowledge docs.

## Exit criteria (from the phase plan)

1. **Adding a toolchain touches one file** (the registry).
2. **Deleting a model from config produces a startup error naming the pipeline** (warning by default, `--strict-pipelines` error) — never a hang.
3. Routing behavior for all existing request shapes is provably identical to the pre-migration engine (parity table).

## Steps

### 1. Routing test suite `tests/unit/inference_flow/test_pipeline_routing.py`

- **Parity table** (transcribed in 3-2 before the old code was deleted): for every family × every entry shape (cloning w/ audio, cloning w/ VAD data, embedding-only, single-speaker, VC, conversion w/ mel, explicit model) → assert the exact model sequence and final output type.
- **Artifact flow**: each forwarding writes exactly the declared `produces` keys; base64 fields untouched.
- **Degenerate cases**: mid-pipeline error at each step → request finishes with error (no requeue loop); client disconnect mid-pipeline; step output missing its artifact → typed error naming the step.
- **New-toolchain test** (exit criterion #1): register a two-step toy spec + two fake configs in the test, request it end-to-end with stub executors — only the registry file gained content.

### 2. Full-suite run + manual verification

Per workspace rules tests run manually — coordinate; record results here.

### 3. Knowledge updates

- `memory-bank/systemPatterns.md`: routing is spec-driven now (update the pattern description); `progress.md`, `activeContext.md`.
- `.planning/codebase/CONCERNS.md`: retire routing-related findings (field-presence fragility, hang-forever class, decorative `input_vad_mode`, FIXME entry hack).
- `CHANGELOG.md`, `docs/` (architecture note: pipeline registry is the extension point).
- Tick Phase 3 in [summary.md](summary.md).

## Checklist

- [x] Parity table complete (every old-ladder decision covered) and passing
- [x] Degenerate + artifact-flow cases covered
- [x] New-toolchain test proves one-file extension
- [x] Strict-pipelines boot error names pipeline + step
- [x] Full suite recorded; docs/memory-bank/CONCERNS/CHANGELOG updated
- [x] Phase 3 ticked in summary.md

## Exit criteria verification

1. **Adding a toolchain touches one file** — `test_new_two_step_toolchain_is_a_registry_only_extension`
   registers a two-step toy spec via `monkeypatch.setitem(PIPELINE_SPECS, ...)` and runs
   it end to end (routing, forwarding, `/v1/models` card) with stub executors.
2. **Deleting a model produces a startup error naming the pipeline** — warnings by
   default, `RuntimeError` under `--strict-pipelines`
   (`test_strict_pipelines_boot_error_names_pipeline_and_step`), and the same
   diagnosis is returned per request instead of hanging.
3. **Parity** — 40 transcribed cases in `test_pipeline_routing.py` cover every family
   × every entry shape (cloning with audio, with VAD data, with embedding, embedding-only,
   single-speaker, VC, transcription, conversion with mel, explicit model, sentinel aliases).

## Test results (recorded 2026-11-04, CPU)

| Suite | Result |
|---|---|
| `tests/unit` | 469 passed (346 baseline + 123 net new/adapted) |
| `tests/integration` | 23 passed |
| `tests/e2e/wire_contract` | 38 passed, 3 model-only skips |
| `tests/e2e/tts` (real weights) | 30 passed + `test_harmonyspeech.py` 4 passed with `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1` (run separately) |
| `tests/e2e/stt`, `tests/e2e/vad` | 7 passed |
| `tests/e2e/tts/test_openvoice_vad_mode.py` (new) | 3 passed |

## Deviations (recorded after implementation)

- The parity table has 40 cases (not the illustrative "8+"), including explicit
  single-model and sentinel-alias shapes.
- Adapted existing tests (behaviour preserved, mechanics updated):
  `test_artifact_hops.py` (legacy-string fallback became a typed missing-artifact
  error; VAD hop asserts the artifact instead of the JSON routing signal; vocoder
  completion returns `None` instead of the request object),
  `test_engine_error_resilience.py` (embedding forward carries the artifact),
  `test_chatterbox_routing.py` (rewritten against `resolve_entry`/`next_step`/`advance`),
  `test_startup_validation.py` (card generator name; undeclared model types produce no cards).
- `tests/unit` total is 469 after Phase 3 (the brief's 346 baseline was Phase 2).
