# Phase 9-2: Generic AR KV-Cache Framework (shared pool)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** nothing in Phase 9 (foundational; 9-3 consumes it). Do **after or alongside 10-3** if that lands first — 10-3's device-level headroom truth is the right input for sizing the default budget.
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`; design input: study #14 D2/D5 (`../../soulbits-cloud-backend/.current_work/inference-engine-re-eval-26-11/14-vllm-whisper-kvcache-and-hse-ar-batching.md`)
> **State reference:** config surface verified against `harmonyspeech/common/config.py` 2026-11-07 (`EngineConfig` frozen dataclass at `:148-168`, YAML loader at `:178-197`); metrics precedent `harmonyspeech/engine/embedding_cache_metrics.py` + `vram_metrics.py`.

## Objective

A model-agnostic, size-capped AR KV-cache **pool** owned by the engine: one device-wide budget, a free-list slot allocator, per-family layout descriptors, slot lifecycle + metrics. No model-specific logic here — 9-3 registers a Chatterbox T3 arena; a future torch-owned AR decoder (see study #14 §"what if we ever fork Whisper decode") registers another. This is the "generic and configurable, shared across models" layer: **shared = allocator + machinery, never cache content** (owner direction 2026-11-07; see summary.md Deferred for the rejected alternatives).

## Context (self-contained)

- **Layout (study #14 D2):** static per-slot slabs, not paged blocks — PyTorch has no paged-attention kernel, and gather-based page reads cost more than the attention at these sizes. One arena tensor per family: `[L, 2, B, cap, H, D]` (layers × {K,V} × slots × cap × kv-heads × head_dim), fp16 on CUDA. `cap = 768` tokens ≈ 30 s of speech at 25 Hz (2 mel frames/token; derived in study #14 — re-derive from the vendored/upstream token rate when 9-3 lands). Fixed strides give a zero-copy `[B, cap, H, D]` view per layer for one `scaled_dot_product_attention` call.
- **Cross-request isolation (Phase 0-2 discipline):** a slot belongs to exactly one in-flight request. Stale KV beyond a new sequence's write position is never *read* (decode writes position k before any attention reads ≤ k, and attention is masked to the current length) — document this invariant and pin it with a poisoned-slot test (pre-fill a slot with NaN/garbage, decode, assert outputs equal a clean-slot run).
- **Slots ≠ admission width** (the Phase 10 lesson, applied from day one): slot count comes from pool availability + per-model descriptor; the scheduler's `max_batch_size` stays a separate capacity signal. An arena that cannot grant a slot returns `None` and the runner takes the honest sequential fallback — never an OOM.
- **Config surface:**

```yaml
ar_inference:
  kv_cache_budget_bytes: 1073741824   # device-wide pool — initial 1 GiB, owner-approved 2026-11-07; tune on data like 8-6's "initial 8"
  default_cap_tokens: 768             # ≈30 s; per-descriptor override allowed
```

  EngineConfig keys + CLI overrides, mirroring the Phase 7 embedding-cache pattern (`--ar-cache-budget-bytes`, `--no-ar-cache`).
- **Boot validation:** refuse descriptors whose slab (B × cap × …) exceeds the remaining budget; log the arena table (family, slots, bytes) at startup like the weight registry does.
- **Metrics:** `hse_ar_cache_{pool_bytes,allocated_bytes,slots,slots_active,slots_free,grants_total,refusals_total}` on `/metrics` + an `ar_cache` block on `/health` (zeros when disabled, never absent — the 7-3 convention).

## Implementation steps

1. `harmonyspeech/engine/ar_cache.py` (new): `ARCacheDescriptor` (model_key, layers, kv_heads, head_dim, dtype, cap_tokens, slots), `ARCachePool` (register/acquire/release/stats), free-list allocator, CPU fallback mode (pool on CPU device → still correct, slower; lets unit tests run without CUDA).
2. EngineConfig keys + YAML/CLI plumbing (`common/config.py`, `engine/args_tools.py`); boot-time arena table log + validation errors.
3. `/metrics` + `/health` wiring (`engine/harmonyspeech_engine.py`, following the embedding-cache metrics precedent).
4. Threading: pool owned by the engine, handed down executor → worker → runner exactly like `WeightRegistry`/`EmbeddingCache` (Phase 4-3/7-1 threading precedent).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/engine/ar_cache.py` (new) | pool + descriptor + allocator |
| `harmonyspeech/common/config.py`, `engine/args_tools.py` | `ar_inference.*` keys + CLI |
| `harmonyspeech/engine/harmonyspeech_engine.py` | own pool, thread down, boot log/validation |
| `tests/unit/engine/test_ar_cache.py` (new) | allocator lifecycle, budget refusal, poisoning, metrics |

## Tests / validation

- Unit: acquire/release/grant/refuse; FIFO-of-free-list slot reuse; poisoned-slot decode equality; budget-below-slab boot error; metrics rendering with and without CUDA; disabled pool (`--no-ar-cache`) fast path.
- No model integration yet — that is 9-3.

## Checklist

- [ ] Pool + descriptor + allocator landed with unit tests
- [ ] Config/CLI keys + boot validation + startup arena table
- [ ] `/metrics` + `/health` series (zero-valued when disabled)
- [ ] Slot-lifecycle isolation invariant documented + poisoned-slot test green
- [ ] `summary.md` Implementation Status updated
