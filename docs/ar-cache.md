# Shared AR KV-cache pool

Autoregressive speech models (the Chatterbox T3 decoder today, any future
token-level AR decoder) need a key/value cache that grows one position per
decode step. Before Phase 9 every request carried its own cache, which is why
the family could not batch: there was nowhere to put a second sequence's KV
state. The engine now owns **one device-wide, size-capped pool** of static KV
slabs. The pool is shared machinery — an allocator and a set of layout
descriptors — never cache *content*: a slot belongs to exactly one in-flight
request.

> **Status:** the framework (pool, allocator, descriptors, config, metrics) is
> implemented, and the first consumer landed in Phase 9-3: the batched Chatterbox
> AR runner registers one arena per English/Turbo model at load and acquires one
> slot per decode row. Phase 9-6 moved both roles into the `ar_decode` execution
> class with an initial admission width of 4, justified from this pool's arena
> table. Multilingual stays the sequential skip. The startup arena table lists
> every registered family.

## Layout

One arena tensor per model family, shape `[L, 2, B, cap, H, D]`:

| Axis | Meaning |
|---|---|
| `L` | decoder layers that hold KV state |
| `2` | keys and values |
| `B` | slots (one per concurrent sequence) |
| `cap` | token capacity per slot |
| `H` | key/value heads (GQA/MQA reduce this below the attention-head count) |
| `D` | head dimension |

PyTorch has no paged-attention kernel, and gather-based page reads cost more
than the attention itself at these sizes, so the layout is static slabs rather
than paged blocks. Fixed strides give a **zero-copy** `[B, cap, H, D]` view per
layer for one `scaled_dot_product_attention` call, and a per-slot `[1, cap, H, D]`
view for one sequence.

The slab size is `L × 2 × cap × H × D × dtype_size` bytes; an arena is
`slab × slots`. Slabs are fp16 on CUDA (the serving choice) and may be fp32 on
CPU, where the pool is correct but slower — that fallback is what lets the unit
suite run without a GPU.

## Configuration

The knobs live in one nested `ar_inference` block in `config.yml` /
`config.gpu.yml`, or on the command line:

| Config key | CLI flag | Default | Meaning |
|---|---|---|---|
| `ar_inference.kv_cache_budget_bytes` | `--ar-cache-budget-bytes` | `1073741824` (1 GiB) | Device-wide byte budget for every registered arena. |
| `ar_inference.default_cap_tokens` | — | `768` | Token capacity per slot when a descriptor does not set its own (≈30 s of speech at 25 Hz). |
| `ar_inference.enabled` | `--no-ar-cache` (disable) | `true` | Master switch. |

```yaml
# config.yml
ar_inference:
  kv_cache_budget_bytes: 1073741824   # 1 GiB, initial value pending measurement
  default_cap_tokens: 768             # ≈30 s per sequence
```

```bash
# shrink the budget for a shared GPU
python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \
    --ar-cache-budget-bytes 536870912

# turn the pool off entirely
python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \
    --no-ar-cache
```

The 1 GiB budget is the owner-approved initial value (2026-11-07) and, like the
Phase 8-6 batching widths, ships pending measurement on the benchmark host.
Flat `ar_cache_budget_bytes` / `ar_cache_default_cap_tokens` keys are accepted
as aliases for deployments that prefer the embedding-cache style.

### Boot validation

Each family registers a descriptor at model load. The engine refuses to start if
a slab does not fit in the remaining budget, or if an explicitly requested slot
count does not — that is a configuration error, not a runtime condition. With no
explicit slot count, the pool takes **as many whole slabs as the remaining
budget fits** (at least one). The registered arena table (family, slots, cap,
slab MiB, arena MiB, dtype) is logged at startup next to the model load table.

### Slots are not admission width

The slot count comes from pool availability plus the descriptor. The scheduler's
`max_batch_size` stays a separate capacity signal and is never derived from the
pool. When every slot of an arena is in use, `acquire` returns `None`; the
request takes its sequential fallback. The pool therefore cannot cause an
out-of-memory failure at runtime.

## Isolation invariant

A slot belongs to exactly one in-flight request between acquire and release.
A released slot keeps the bytes the previous request wrote; the next request
reuses it **without a clearing pass**. This is safe because every consumer must
honour the decode contract:

* a decode step writes position `k` (the K/V of the token being generated)
  **before** any attention reads positions `<= k`, and
* attention is masked or sliced to the sequence's current length, so stale KV
  **beyond the write position is never read**.

The invariant is pinned by a unit test that pre-fills a reused slot with NaN,
writes a real prefix, and asserts a `scaled_dot_product_attention` decode over
that prefix is bit-identical to the clean-slot run — while the same decode over
the full capacity still sees the poison, proving the prefix slice is the guard.

## Metrics

`/metrics` always emits every series, with zeros when the pool is disabled, so a
dashboard can tell "idle" from "absent":

| Series | Type | Meaning |
|---|---|---|
| `hse_ar_cache_enabled` | gauge | `1` enabled, `0` disabled. |
| `hse_ar_cache_pool_bytes` | gauge | Resident slab bytes of every registered arena. |
| `hse_ar_cache_allocated_bytes` | gauge | Bytes of currently granted (in-flight) slots. |
| `hse_ar_cache_slots` | gauge | Total slots across every arena. |
| `hse_ar_cache_slots_active` | gauge | Slots granted to an in-flight request. |
| `hse_ar_cache_slots_free` | gauge | Slots available for a new request. |
| `hse_ar_cache_grants_total` | counter | Successful slot grants since boot. |
| `hse_ar_cache_refusals_total` | counter | Grants refused because the arena was full. |

`/health` carries the same accounting in an `ar_cache` block, including a
per-family `arenas` map with `active_rows` (the value a capacity-based admission
policy can consume), `slab_bytes` and `arena_bytes`. The block also appears
inside the `vram` summary: the slabs are ordinary PyTorch allocations, so they
are already included in `hse_vram_allocated_bytes` / `hse_vram_device_used_bytes`
— the explicit `ar_cache` entry makes the footprint visible rather than hidden
inside "torch allocations".

A rising `hse_ar_cache_refusals_total` means requests are falling back to the
sequential path: either raise the budget / lower `default_cap_tokens`, or accept
the smaller concurrency. A `hse_ar_cache_pool_bytes` larger than the configured
budget is impossible by construction; if it approaches the budget, the pool is
fully sized and further families will be refused at boot.

### Decode progress (Phase 9-4)

Multi-step AR requests stay `RUNNING` across engine steps, so the per-model
decode view is separate from the device-wide pool accounting. `/metrics` emits a
labelled `hse_ar_decode_*` family — `active_rows`, `slots`, `slots_active`,
`slots_free`, `tokens_total`, `steps_total`, `tokens_per_step` — and `/health`
carries the same rows in an `ar_decode` block. Every series is present (as
`{model="none"}` zeros) when no AR model is configured. A row's `active_rows`
tracks the pool's granted slots; the token counters advance once per partial
executor result, so a decode that survives many steps is visible while it runs.

## What this is not

* **Not a content cache.** Slots are never shared between requests, and nothing
  survives a request's release as meaningful state.
* **Not prefix caching.** Reusing an identical conditioning + text KV prefix is
  deferred (Phase 9 note in the plan); decode dominates AR-TTS latency, so
  prefill reuse has no big win.
* **Not a scheduler.** Admission width stays in the scheduler; the pool only
  reports availability.
