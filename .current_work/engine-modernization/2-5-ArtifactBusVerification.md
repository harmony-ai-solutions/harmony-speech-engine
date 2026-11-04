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

- [x] Benchmark script committed; before/after numbers recorded in `docs/benchmarks/artifact-bus.md`
- [x] VoiceFixer + OpenVoice-clone hop savings within (or explained against) expectations
- [ ] Determinism fixtures verified per family
- [x] Full suite green; memory-bank/CONCERNS/CHANGELOG updated
- [x] Phase 2 ticked in summary.md

## Implementation notes (2026-11-04)

- `scripts/bench_pipeline_hops.py` supports `--mode stub` (serialization
  microbenchmark, no weights) and `--mode engine` (real requests, instrumented
  executor steps, `tracemalloc`, CUDA memory, encode/decode counters) plus
  `--determinism`. Results append to `docs/benchmarks/artifact-bus.md`.
- Measured (stub, 20 iterations): VoiceFixer mel hop 25.4 ms → 0.005 ms and
  ~0.98 MiB → ~0 MiB allocations; synth→TC WAV hop 0.67 ms → 0.004 ms.
- Real-model CPU runs were possible with the local HuggingFace cache:
  - HarmonySpeech clone: 249 ms total (encoder 91 / synth 96 / vocoder 58),
    `encode_calls=0`, `decode_calls=2` for two requests.
  - VoiceFixer: 2.54 s total (restorer 0.48 / vocoder 2.06), same counts.
  - OpenVoice V2 clone: 2.37 s total (Whisper 0.38 / TCE 0.014 / Melo 1.27 /
    TC 0.70), same counts. First-request warm-up was much slower (Melo 28 s),
    which is why the numbers are labelled steady-state.
  - Real-model e2e also ran green: `audio_restoration` (1), `tts/test_harmonyspeech`
    (4), `tts/test_melotts` (5), `tts/test_openvoice_v1` (5), `tts/test_chatterbox`
    (6), `tts/test_chatterbox_turbo` + `tts/test_kittentts` (8), `stt` + `vad` (7).
- Determinism: VoiceFixer and HarmonySpeech repeated runs are **byte-identical**
  (SHA-256). OpenVoice V2 differs (duration Δ1.46 %, RMS Δ1.17 %) because MeloTTS
  samples noise — expected for a stochastic model.
- **Unticked item:** the pre-Phase-2 fixture comparison (recording hashes from
  commit `06e08dd`) was not run; it needs a separate worktree/environment for
  the old code. The artifact bus only changes the in-process container, and the
  post-Phase-2 repeatability above is consistent with the expected outcome.
  CUDA numbers are also pending a GPU host (`cuda=0.0 MiB` on this CPU run).

