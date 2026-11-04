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

- [ ] Parity table complete (every old-ladder decision covered) and passing
- [ ] Degenerate + artifact-flow cases covered
- [ ] New-toolchain test proves one-file extension
- [ ] Strict-pipelines boot error names pipeline + step
- [ ] Full suite recorded; docs/memory-bank/CONCERNS/CHANGELOG updated
- [ ] Phase 3 ticked in summary.md
