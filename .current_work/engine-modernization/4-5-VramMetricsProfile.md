# Phase 4-5: VRAM Metrics & 16 GiB Profile

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 4 — VRAM: fp16 rollout + Chatterbox weight dedup
> **Depends on:** 4-1, 4-4
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, research §1.7, risk #10

## Objective

Make VRAM observable and validate the phase's headline claim on the real card: the full shipped catalogue boots with **large headroom on 16 GiB** (research arithmetic: ~5.9 GiB weights+context+activations after fp16+dedup, vs ~16.4 GiB shipped today). What gets measured gets trusted — and Phase 6's governor needs these gauges.

## Implementation steps

### 1. Per-model VRAM accounting

In each executor/worker after model load (and after sharing, via `WeightRegistry.stats()`):

- `torch.cuda.memory_allocated()` delta per model load on CUDA devices → cumulative per-model weight bytes (shared components attributed to the group, listed once).
- Expose on the engine: `HarmonySpeechEngine.get_vram_stats() -> dict` with per-model entries, registry-shared entries, totals, plus `torch.cuda.memory_reserved()`.

### 2. Health & metrics surface

- Extend `check_health()` (fixed in 0-5) payload with the VRAM stats — `/health` shows per-model bytes and totals.
- `/metrics` (if a metrics endpoint exists — re-read `api_server.py`; if none exists, add a minimal Prometheus-text endpoint): `hse_model_vram_bytes{model}`, `hse_vram_reserved_bytes`, `hse_resident_models`, `hse_shared_backbone_bytes{group}`.

### 3. Measurement script `scripts/measure_vram.py`

Boots the engine from a config (default `config.gpu.yml`), waits for readiness, samples `torch.cuda.memory_allocated/reserved` + `nvidia-smi --query-gpu=memory.used` at steady state, runs one warmup request per family (captures activation peak via `reset_peak_memory_stats`), prints a table, and appends to `docs/benchmarks/vram-profiles.md`.

### 4. The 16 GiB validation run

On the 4060 Ti host: run the script before Phase 4 (fp32, duplicated — from git history if needed) and after (fp16 + dedup). Record both in `docs/benchmarks/vram-profiles.md`. Exit target: full `config.gpu.yml` catalogue resident with ≥ 8 GiB headroom (research estimate ~10 GiB headroom; ≥8 is the acceptance floor). If below the floor, diagnose per-model actuals vs `docs/models.md` estimates and correct the docs — the estimates are the research's order-of-magnitude figures, now to be replaced by measurements.

### 5. Docs & knowledge

- `docs/models.md`: replace estimate table with measured values (fp16 + dedup).
- `memory-bank/`, `CONCERNS.md` (retire "at or beyond a 16 GB card" finding), `CHANGELOG.md`.
- Tick Phase 4 in [summary.md](summary.md).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/engine/harmonyspeech_engine.py` | `get_vram_stats()`; health payload extension |
| `harmonyspeech/endpoints/openai/api_server.py` | `/metrics` (minimal prometheus text) |
| `scripts/measure_vram.py` (new) | boot + steady-state + peak sampler |
| `docs/benchmarks/vram-profiles.md` (new) | recorded runs |
| `docs/models.md` | measured numbers |

## Tests

- Unit: `get_vram_stats` shape/mocking (CPU: zeros/empty for CUDA stats — no crash on CPU-only builds).
- Validation (GPU host, manual): before/after tables recorded; acceptance floor met.

## Checklist

- [x] Per-model + shared-group VRAM accounting implemented
- [x] `/health` payload and `/metrics` endpoint live
- [x] `measure_vram.py` committed; before/after runs recorded
- [x] 16 GiB acceptance floor (≥8 GiB headroom, full catalogue) met or diagnosed
- [x] `docs/models.md` now carries measured figures
- [x] Phase 4 ticked in summary.md

## Outcome (2026-11-04)

* `HarmonySpeechEngine.get_vram_stats()` — CUDA allocated/reserved/peak/total/
  headroom, per-model `vram_bytes` (load-time allocation delta), per-model
  `weights_bytes`, per-model dtype/device/group, registry `shared` detail,
  per-group `groups`, and `dedup_saved_bytes`. Safe on CPU-only builds.
* `/health` gains a `vram` block and per-model `vram_bytes`;
  `/metrics` is now a hand-rendered endpoint combining the default Prometheus
  registry with `hse_model_vram_bytes{model,model_type}`,
  `hse_model_weights_bytes{model}`, `hse_vram_allocated_bytes`,
  `hse_vram_reserved_bytes`, `hse_vram_peak_allocated_bytes`,
  `hse_vram_total_bytes`, `hse_vram_headroom_bytes`, `hse_resident_models`,
  `hse_shared_backbone_bytes{group}`, `hse_shared_total_bytes`,
  `hse_shared_components`.
* `scripts/measure_vram.py`: boots a config, samples torch + `nvidia-smi`,
  optional per-family warmup with `reset_peak_memory_stats`, table + optional
  JSON/markdown; pure helpers are unit-tested.

### Measured result (full `config.gpu.yml`, RTX 5060 Ti 16 GiB)

| | Before (fp32, duplicated) | After (fp16 T3 + dedup) |
|---|---:|---:|
| torch allocated | 16.45 GiB | 8.52 GiB |
| nvidia-smi used | 15.67 GiB | 8.83 GiB |
| headroom (reserved) | 0.00 GiB | 7.30 GiB |
| Chatterbox block | 14.29 GiB | 6.35 GiB |

**Acceptance floor: missed by ≈0.7–0.9 GiB** (7.10–7.30 GiB headroom vs 8 GiB).
Diagnosed in `docs/benchmarks/vram-profiles.md`: the remaining fp32 families
(OpenVoice/MeloTTS ≈2.0 GiB, VoiceFixer ≈0.4 GiB) are upstream half-precision
incompatibilities, and Chatterbox Turbo carries ≈1.5 GiB of GPT2 causal-mask
buffers in its checkpoint module tree. The card is exactly 16 GiB, so the floor
was applied as written rather than scaled.
