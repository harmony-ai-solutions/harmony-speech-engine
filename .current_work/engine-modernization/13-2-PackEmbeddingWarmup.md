# Phase 13-2: Pack Embedding Warm-Up + Persistent Disk Cache

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 13 — Standard voice pack & voices endpoint
> **Depends on:** 13-1 (catalog + routing); 7-2 embedding cache (in-memory layer)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; boot observations from the 11-3 flip smoke (2026-11-08)
> **State reference:** verified 2026-11-08 — executors fill `/health` as they come up (background init; observed in the flip smoke runs); the 7-2 cache is **in-memory only** today (TTL/FIFO/byte-ceiling, `/health embedding_cache` accounting); no engine-level disk-cache convention exists yet (weights go through the HF cache) — the cache-root choice below is new surface, keep it small.

## Objective

Standard-voice embeddings survive restarts: after first use (or an explicit warm-up), a pack voice's per-family conditioning artifact is on disk, and every later boot hydrates the 7-2 in-memory cache from disk with **zero embedding compute**. Owner decision 2026-11-08: "store the embeddings in a cache folder so we don't have to do this on every startup."

## Context (self-contained)

- **Warm what a first request would compute — nothing more.** Per family the 7-2 handlers already define the cacheable artifact (chatterbox: VE embedding/conditioning from the embed step; openvoice: its speaker-encoder artifact; hs1: encoder output — verify each handler's cache key/value shape first and reuse **exactly** those). Warm-up = replaying 13-1's routing path for every (cloning-capable model × pack voice) with a synthetic request, then persisting what the handler cached.
- **Disk layout:** one artifact file per `(model_type, model repo, revision, voice, wav-content-hash)` under a configurable `voice_pack_cache_dir` (default a stable user-cache location, e.g. `~/.cache/harmony-speech/voice-pack/` — verify platformdirs/cachedir availability in deps before adding anything). The wav hash in the key is the invalidation story: pack update ⇒ new hash ⇒ old entries ignored (and lazily prunable).
- **Boot discipline:** warm-up runs **after** the model is resident (post-executor-init hook — verify the exact hook; the flip smoke showed executors land asynchronously), async, never blocking or failing boot; per-voice failures are logged and skipped (on-demand path from 13-1 remains the fallback). A config gate `voice_pack_warmup: true` (global) with the measured boot cost recorded here once known (N voices × E embedding models).
- **Races:** a real request for a pack voice during warm-up must not duplicate work or corrupt state — on-demand compute wins, warm-up dedupes (the 7-2 cache already dedupes compute per key within a step; verify cross-step).
- **Hydration order:** disk → 7-2 memory at model-init time (cheap I/O), so first real request after boot is a cache hit.

## Implementation steps

1. Verify-first: each cloning family's 7-2 cache key/value shapes (what exactly to persist); the post-load hook point; concurrent-warmup vs on-demand semantics.
2. Persistent store: write/read/prune helpers (hash-keyed, atomic write, corrupt-entry ⇒ ignore + log); `voice_pack_cache_dir` + `voice_pack_warmup` config knobs.
3. Warm-up task per cloning-capable model: iterate catalog standard voices → hydrate from disk or compute via the 13-1 routing path → persist; bounded concurrency (one model at a time).
4. Tests: warm-up populates disk + memory; second boot with warm disk performs **zero** embed forwards (spy/gate on the embed step); hash invalidation re-embeds; corrupt file tolerated; warm-up failure doesn't block boot; on-demand-during-warmup dedupe.
5. Docs: `docs/models.md` (cache behavior + how to clear), CHANGELOG with 13-3.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/voice_pack_cache.py` (or extend catalog module) | NEW — disk store + warm-up |
| `harmonyspeech/common/config.py` (+ both configs) | `voice_pack_cache_dir`, `voice_pack_warmup` |
| engine/loader hook point (verify) | post-init warm-up trigger |
| tests (unit/integration), docs, CHANGELOG | gates above |

## Tests / validation

- Full suites green; boot-time cost of warm-up recorded here after first host run (dated).

## Checklist

- [ ] Per-family artifact shapes verified + recorded here
- [ ] Disk store + hydration + invalidation landed
- [ ] Zero-compute second boot pinned by test
- [ ] Failure/race gates green; docs/CHANGELOG; `summary.md` ticked
