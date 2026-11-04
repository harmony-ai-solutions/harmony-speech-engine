# Phase 4-2: Audio Regression Gate

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 4 — VRAM: fp16 rollout + Chatterbox weight dedup
> **Depends on:** Phase 2 (artifacts make output capture trivial); developed in parallel with 4-1, and re-gates 4-4
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md` (CPU-only CI today — this harness is the GPU quality signal the repo lacks; research risk #7, #10)

## Objective

A repeatable, threshold-based audio regression harness comparing model outputs across a *change axis* (dtype, weight sharing, later compile/capture): fixed inputs → outputs → objective spectral-distance metrics + pass/fail thresholds per family. This is the quality gate that allows the 4-1 config flip and later the 4-4 sharing, without a human listening to every clip (a spot-check listening pass is still part of sign-off).

## Context (self-contained)

- CI is CPU-only (10 e2e jobs; no GPU regression signal — research risk #10). The harness therefore runs **on demand on a GPU host** and stores signed results in the repo; CPU families (KittenTTS) run in CI.
- Stochastic models (Chatterbox sampling, OpenVoice `noise_scale`) need seed pinning where the model API exposes it, plus stochastic-robust metrics (duration, RMS, spectral envelope) instead of sample-exact comparison.

## Implementation steps

### 1. Fixtures `tests/audio_fixtures/`

- Inputs: fixed texts (EN + one non-EN) per family; fixed reference-audio clips (2 speakers × ~8 s, wav) from `tests/test-data/samples` or newly generated (documented provenance; no third-party copyrighted audio).
- Expected baselines are **not** committed as golden audio (bit-exactness across torch versions is unrealistic); instead the harness compares *two configurations of the current build* (e.g. fp32 vs fp16) plus stores metric snapshots in `docs/benchmarks/audio-regression.md` for trend tracking.

### 2. Harness `scripts/audio_regression.py`

```
--families openvoice_v1,openvoice_v2,chatterbox,voicefixer,melotts,harmonyspeech
--axis dtype:float32-vs-float16          # or sharing:dedup-vs-dup, or baseline:commit-A-vs-commit-B
--gpu                                    # requires CUDA host
```

For each family: boot the engine twice (both sides of the axis) with identical configs except the axis knob, run the fixed fixture set, capture `AudioArtifact`s, compute per-clip metrics:

| Metric | Tool |
|---|---|
| Log-mel spectral distance (MSE over mel frames) | `librosa.feature.melspectrogram` |
| Duration delta | samples/sr |
| RMS loudness delta | `librosa.feature.rms` |
| Sample correlation (deterministic families only) | `np.corrcoef` |

Thresholds (initial, tune on first run): spectral distance < 1e-2 (log-mel MSE), |duration delta| < 1% , |RMS delta| < 5%; correlation > 0.98 for deterministic families. Per-family overrides in a small config block in the script.

### 3. Verdict output

Machine-readable JSON + human table appended to `docs/benchmarks/audio-regression.md` with axis, commit hashes, GPU model, driver/torch versions, per-family pass/fail. Exit code non-zero on any family failure → usable as a gate.

### 4. Spot-check listening pack

On failure-or-marginal cases the script writes a small folder of A/B wav pairs (`out/regression-pack/`) for a human listening pass — documented as part of the sign-off procedure for dtype/sharing changes.

## Files to create/modify

| File | Change |
|---|---|
| `scripts/audio_regression.py` (new) | the harness |
| `tests/audio_fixtures/` (new) | texts + reference clips + config |
| `docs/benchmarks/audio-regression.md` (new) | recorded runs & thresholds |

## Tests / validation

- Self-test: axis `dtype:float32-vs-float32` must pass trivially on all families (harness sanity, CI-runnable for CPU families).
- First real run: `dtype:float32-vs-float16` on GPU host → results recorded; failures become per-family fp32 opt-outs in `config.gpu.yml` (per 4-1 step 3).

## Checklist

- [x] Fixtures committed (documented provenance)
- [x] Harness with metrics, thresholds, JSON+md output, exit-code gate
- [x] Self-test (identity axis) passes; CPU families run in CI
- [x] First fp32-vs-fp16 run recorded; opt-out list derived
- [x] Listening-pack procedure documented
- [x] `summary.md` Implementation Status updated

## Outcome & deviations (2026-11-04)

* `scripts/audio_regression.py` runs each side of the axis in its own worker
  process (real VRAM release between sides) and computes metrics in the parent.
  Deterministic families: log-mel MSE + duration + RMS + waveform correlation.
  Stochastic families: duration + RMS + LTAS RMSE. Sharing axis: identical
  length + `max|Δsample|` ≤ 1e-3 + log-mel MSE ≤ 1e-2.
* Identity self-test (`dtype:float32-vs-float32`) passes on KittenTTS (CPU) with
  bit-identical output.
* dtype axis recorded for Chatterbox, Turbo and Multilingual — all pass
  (details in `docs/benchmarks/audio-regression.md`).
* OpenVoice V1/V2 and MeloTTS fail structurally under fp16 (float32 tensors in
  the adapted VITS code; MeloTTS spline-bin assertion) → per-entry `float32`
  opt-outs. VoiceFixer could not be measured: its GPU path raises a
  pre-existing "input cuda vs weight cpu" error; kept `float32`.
* **Deviation:** across two processes the CUDA autoregressive path is not
  bit-identical (kernel/workspace selection varies with free VRAM; measured
  `max|Δsample|` ≈ 1.5e-4). Exact equality is therefore asserted in-process by
  `tests/e2e/tts/test_chatterbox_sharing_identity.py`, and the cross-process
  sharing gate uses the −60 dBFS tolerance above. Watermarking is disabled for
  the sharing axis because PerthNet's conv kernels are not reproducible either.
* Synthetic fixture clips (`tests/audio_fixtures/ref_voice_*.wav`) are used
  where no VAD is involved; the Whisper-VAD-driven toolchains use the
  pre-existing repository speech samples because synthetic vowels are not
  detected as speech (documented in the fixture module and the benchmark doc).
