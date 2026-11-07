# Phase 10-3: Whisper Memory Governance & Device-Level VRAM Truth

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 10 — Whisper serving
> **Depends on:** nothing (land **first** in Phase 10 if possible — 10-1's worker counts, 10-2's batch caps and 9-2's pool budget all want its numbers)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; precedent: Phase 4-5 VRAM metrics
> **State reference:** verified 2026-11-07 — `HarmonySpeechEngine.get_vram_stats` (`harmonyspeech_engine.py:759-849`) reads **torch-allocator counters only** (`torch.cuda.memory_allocated/reserved/max_memory_allocated`); `vram_metrics.py:12-16` renders `hse_vram_headroom_bytes` as "Free device memory (total minus reserved)". CTranslate2 allocates **outside PyTorch's caching allocator**, so whisper's device memory is invisible to every current VRAM series — and `headroom` **overstates** free VRAM whenever a CT2 model holds memory.

## Objective

Make the engine's VRAM story true at device level, then bound Whisper's footprint by construction + measurement: CT2 has **no memory-cap knob** (`gpu_memory_utilization` does not exist there), so the budget is enforced on our side — through the knobs we own (workers M, windows/batch, beam_size, admitted audio duration) and a measured peak.

## Context (self-contained)

- **The observability fix:** add device-level truth via `torch.cuda.mem_get_info(device)` → `device_used = total − free` (captures torch + CT2 + CUDA contexts — everything). New series `hse_vram_device_used_bytes`, and **`hse_vram_headroom_bytes` is redefined in place** as the true device-level free memory (owner decision 2026-11-07) — one clean breaking change with a changelog + `_HELP` note, no lookalike series. Keep the torch-scoped series too (they still answer "what does torch hold").
- **Per-model attribution honesty:** CT2 memory cannot be attributed per model via torch. Approximate honestly: known weights bytes (checkpoint sizes, already reported via `hse_model_weights_bytes`) + a **startup probe**: after loading each whisper config at its configured M, run a max-shape synthetic forward (M workers busy, full window batch, longest admitted audio) and record the `device_used` delta as that config's `peak_activation_bytes`. Report per-model in `/health` with a "measured probe" annotation — never guessed.
- **Enforcement:** boot-time warning (or error, config-key choice: `whisper_vram_budget_bytes` optional hard bound) when the projected whisper peak (sum of configured whisper configs' weights + probes) exceeds the measured free headroom; at runtime, if `device_used` crosses the budget, admission shrinks concurrent windows (10-2's cap is a live bound, not a boot constant). This is the engine-side budget vLLM gets from `gpu_memory_utilization` — implemented as admission scaling, not allocator magic.
- **Side effect for Phase 9:** the same device-level numbers re-baseline the "7.1 GiB headroom" claim that 9-2's pool budget is carved from — record the corrected figure in `docs/benchmarks/vram-profiles.md`.
- **Request-size coupling:** admitted-audio duration caps are part of the bound (the open "request size limits" concern in CONCERNS); this task defines the whisper-side cap, the general endpoint limit stays its own concern.

## Implementation steps

1. `get_vram_stats`: add `mem_get_info`-based `device_used_bytes`/`free_bytes`; renderer + `_HELP` text; unit tests with mocked CUDA calls.
2. Startup probe for CT2 whisper configs (synthetic max-shape run at configured M; CPU-safe skip); results into `get_vram_stats` model entries + boot log table.
3. Budget check at boot (warn/error per config key) + runtime admission-scaling hook consumed by 10-1/10-2.
4. `docs/benchmarks/vram-profiles.md`: corrected headroom table + whisper peaks; note the torch-vs-device delta (that delta *is* CT2 + context overhead).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/engine/harmonyspeech_engine.py`, `engine/vram_metrics.py` | device-level stats + series |
| `harmonyspeech/common/config.py` | `whisper_vram_budget_bytes` (optional) |
| `harmonyspeech/modeling/loader.py` or runner init | max-shape probe per whisper config |
| `tests/unit/engine/test_vram_stats.py` (extend) | new series, probe math, budget check |

## Tests / validation

- Unit: series rendering; probe delta math; boot warning/error path; admission-scaling decision function.
- Benchmark host: real probe numbers into `vram-profiles.md`; corrected headroom recorded.

## Checklist

- [ ] Device-level `device_used`/true headroom series landed (torch-scoped kept)
- [ ] Whisper startup probe + per-model peak reporting
- [ ] Boot budget check + runtime admission-scaling hook for 10-1/10-2
- [ ] `vram-profiles.md` corrected (incl. the Phase 9 pool-sizing input)
- [ ] `summary.md` Implementation Status updated
