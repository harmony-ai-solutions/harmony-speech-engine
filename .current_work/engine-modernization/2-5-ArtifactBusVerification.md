# Phase 2-5: Artifact Bus Verification & Benchmark

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 2 — Artifact bus
> **Depends on:** 2-1…2-4
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`

## Objective

Prove the phase's exit criteria with numbers, and leave a repeatable benchmark script in the repo.

## Exit criteria (from the phase plan)

1. **Serialization time removed from inner hops** — measured before/after on the OpenVoice V2 voice-cloning path and the VoiceFixer path.
2. **Fixed-seed output byte-identical** (or statistically equivalent for stochastic models) vs pre-Phase-2.
3. All e2e suites green.

## Steps

### 1. Benchmark script `scripts/bench_pipeline_hops.py`

- Boots the engine from a given config (CPU config acceptable for relative serialization measurements; GPU for absolute latency).
- Runs the OpenVoice V2 clone path (Whisper→TCE→Melo→TC) and the VoiceFixer path N times; records per-step wall time, `tracemalloc` peak, and — if GPU — `torch.cuda.max_memory_allocated`.
- Accepts a `--label` and appends results to `docs/benchmarks/artifact-bus.md` (create folder) so before/after numbers live in the repo.
- To get "before" numbers on the same codebase, support `--legacy-serialization` env toggle **if** cheap to keep during 2-3 development (a temporary branch switch is also acceptable — record whichever approach was used).

### 2. Expected wins to validate (research §5.2 estimates)

| Hop | Expected saving |
|---|---|
| Restorer → vocoder | ~40–80 ms + ~35–45 MB host churn removed |
| Whisper → encoder | 1 base64 decode + librosa load (~5–15 ms) removed |
| Embedding → TTS | ~1–5 ms + host allocation |
| TTS → tone converter | ~10–25 ms + avoids lossy intermediate WAV |

If measured savings materially diverge (e.g. < 25% of estimate), investigate before closing the phase — the estimate arithmetic came from the research document and a miss may indicate a remaining hidden encode path.

### 3. Determinism check

For each family, generate output pre-Phase-2 (recorded fixture hashes kept in `tests/test-data/expected/`) vs post-Phase-2 with identical inputs and seeds: WAV-level byte comparison; for stochastic models (Chatterbox sampling) compare duration (±1%) + RMS/spectral sanity. Document per-family result in the phase file.

### 4. Knowledge updates

- `docs/benchmarks/artifact-bus.md` with the table.
- `memory-bank/progress.md`, `CONCERNS.md` (retire the base64-intermediate findings), `CHANGELOG.md` (performance entry, no internals).
- Tick Phase 2 in [summary.md](summary.md).

## Checklist

- [ ] Benchmark script committed; before/after numbers recorded in `docs/benchmarks/artifact-bus.md`
- [ ] VoiceFixer + OpenVoice-clone hop savings within (or explained against) expectations
- [ ] Determinism fixtures verified per family
- [ ] Full suite green; memory-bank/CONCERNS/CHANGELOG updated
- [ ] Phase 2 ticked in summary.md
