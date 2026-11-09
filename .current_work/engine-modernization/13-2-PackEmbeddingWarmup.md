# Phase 13-2: Pack Embedding Warm-Up + Persistent Disk Cache

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 13 — Standard voice pack & voices endpoint
> **Depends on:** 13-1 (catalog + routing); 7-2 embedding cache (in-memory layer)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; boot observations from the 11-3 flip smoke (2026-11-08)
> **State reference:** verified 2026-11-08 — executors fill `/health` as they come up (background init; observed in the flip smoke runs); the 7-2 cache is **in-memory only** today (TTL/FIFO/byte-ceiling, `/health embedding_cache` accounting); no engine-level disk-cache convention exists yet (weights go through the HF cache) — the cache-root choice below is new surface, keep it small.

## Objective

Standard-voice embeddings survive restarts: after first use (or an explicit warm-up), a pack voice's per-family conditioning artifact is on disk, and every later boot hydrates the 7-2 in-memory cache from disk with **zero embedding compute**. Owner decision 2026-11-08: "store the embeddings in a cache folder so we don't have to do this on every startup."

## Context (self-contained)

- **Warm what a first request would compute — nothing more.** Per family the 7-2 handlers already define the cacheable artifact (chatterbox: VE embedding/conditioning from the embed step; openvoice: its speaker-encoder artifact; hs1: encoder output — verify each handler's cache key/value shape first and reuse **exactly** those). Warm-up = replaying 13-1's routing path for every (cloning-capable model × pack voice) with a synthetic request, then persisting what the handler cached. **Concrete seam (post-12, 2026-11-09):** the synthetic request is the *embed-step* request the serving layer already builds — reuse `_reference_conditioning_mode()` + `SpeechEmbeddingRequestInput` (`_resolve_reference_embedding` / `STANDALONE_EMBED_MODEL_TYPES`, `serving_text_to_speech.py:117-124, 598-665`) rather than inventing a new request shape. That seam exists precisely to route an embedding request correctly for both pipeline ids and directly named Chatterbox executors — hand-rolling a warm-up request risks exactly the request-type-rejection bug class the 12-2 follow-up #2 fixed (`ov2-synthesizer-en` + `-embed` ⇒ 400). Pack voices apply only to cloning-capable models, where the mode resolves to `"embed"` — but route through the helper anyway so the invariant lives in one place.
- **Disk layout:** one artifact file per `(model_type, model repo, revision, voice, wav-content-hash)` under a configurable `voice_pack_cache_dir` (default a stable user-cache location, e.g. `~/.cache/harmony-speech/voice-pack/` — verify platformdirs/cachedir availability in deps before adding anything). The wav hash in the key is the invalidation story: pack update ⇒ new hash ⇒ old entries ignored (and lazily prunable).
- **Boot discipline:** warm-up runs **after** the model is resident (post-executor-init hook — verify the exact hook; the flip smoke showed executors land asynchronously), async, never blocking or failing boot; per-voice failures are logged and skipped (on-demand path from 13-1 remains the fallback). A config gate `voice_pack_warmup: true` (global) with the measured boot cost recorded here once known (N voices × E embedding models).
- **Races:** a real request for a pack voice during warm-up must not duplicate work or corrupt state — on-demand compute wins, warm-up dedupes (the 7-2 cache already dedupes compute per key within a step; verify cross-step).
- **Mutable-artifact precondition (verified 2026-11-09, commit `8b7d222`):** persisting cached conditionals is safe only because Phase 12-8 stopped mutating them. `_ar_conditionals` → `cast_conditionals(..., exaggeration=…)` now **clones** the T3 half before the dtype/device move and before applying an exaggeration override, so a cached or shared `Conditionals` object is never written to per request. Re-verify this still holds when you add the disk layer (it is the single thing that would silently corrupt a persisted artifact across restarts), and add a regression test that a second request with a different `exaggeration` leaves the cached object byte-identical.
- **Boot cost changed since the 2026-11-08 estimate:** `ChatterboxMultilingualTTS` is no longer the slow eager-attention loop — it runs native SDPA with the per-row analyzer and full AR batching (12-7). Warm-up is therefore cheaper than the earlier estimate for that family; record the measured N-voices × E-embedders cost as the docstring requires rather than reasoning from the old numbers.
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

- [x] Per-family artifact shapes verified + recorded here
- [x] Disk store + hydration + invalidation landed
- [x] Zero-compute second boot pinned by test
- [x] Failure/race gates green; docs/CHANGELOG; `summary.md` ticked

## Verification log (2026-11-09)

**Verify-first findings.**

- **Artifact shapes (7-2 key/value), verified from `task_handler/model_runner_base.py`:**
  - Chatterbox (`_execute_chatterbox_embedding`): value `SpeakerArtifact(conditionals=<Conditionals>, backend="chatterbox")`; key params `{"exaggeration": <float, default 0.5>}`; audio hash = decoded pack PCM.
  - OpenVoice (`_execute_openvoice_tone_converter_encoder`): value `SpeakerArtifact(embedding=<tensor>, backend="openvoice")`; key params `{"vad_source": "whisper"|"silero"}`; audio hash = decoded pack PCM.
  - HarmonySpeech (`_execute_harmonyspeech_encoder`): value `SpeakerArtifact(embedding=<L2-normalised tensor>, backend="harmonyspeech")`; params `{}`.
  - The 7-2 key is `make_key(model_identity, audio_hash, params)`; `model_identity` embeds the per-boot `v{load_version}`, so the disk key is **not** the 7-2 key: it is the stable `(model_type, repo, revision, voice, wav_hash, params)` tuple, and hydration recomputes the 7-2 key with the current identity.
- **Post-executor-init hook, verified:** models load **synchronously** in `HarmonySpeechEngine.__init__` → `init_custom_executors()` → `_init_worker()` → `load_model()` (the flip-smoke "asynchronous executors" observation does not hold in this checkout). The background loop starts lazily on the first `add_request`. Warm-up is therefore scheduled from `AsyncHarmonySpeech.start_background_loop()` (after the executors are resident) as a task; hydration runs synchronously at the end of `HarmonySpeechEngine.__init__` (disk → 7-2 memory).
- **Cross-step vs intra-step dedupe, verified:** the 7-2 cache dedupes across steps (second request is a hit) but does **not** dedupe duplicate keys inside one batch (slots are resolved before compute). Warm-up therefore submits one voice at a time and checks `EmbeddingCache.contains()` first, so an on-demand compute wins; a same-step collision can cost at most one redundant embed forward and cannot corrupt state (handlers keep per-request copies).
- **Mutable-artifact precondition, re-verified:** `ar_runner.cast_conditionals` deep-copies the T3 half before `.to(device, dtype)` and before writing `emotion_adv`; `gen` is shared read-only. A new regression test pins that two calls with different `exaggeration` leave the cached object byte-identical.

**Deviations.**

1. `voice_pack_warmup` **code default is `false`**; both shipped configs set `true`. A `true` code default would have made every test engine hydrate/warm up against the user cache (and broke exact cache-stats assertions); the shipped-config gate is the product switch.
2. The disk key folds in the family **params** (`exaggeration`, `vad_source`) in addition to the task file's `(model_type, repo, revision, voice, wav_hash)`: params change the artifact, so omitting them would let a non-default request hydrate a default artifact.
3. Warm-up covers **TTS cloning embed steps only** (Chatterbox en/turbo/multilingual, OpenVoice v1/v2, HarmonySpeech), matching the task file's "pack voices apply only to cloning-capable models, where the mode resolves to `embed`". ChatterboxVC `target_voice` (`set_target_voice` → `ref_dict`) is a different artifact and is not warmed; it stays on-demand.
4. `EmbeddingCache.contains()` was added (presence check without bumping hit/miss counters) so hydration/warm-up dedupe does not distort the 7-3 hit ratio.
5. The disk store is internal and uses a `torch.save` payload of tensors/primitives loaded with `weights_only=True` (documented in the module docstring); `reference_audio` is never persisted.
6. Platform cache root is zero-dependency (`%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME`/`~/.cache` elsewhere) plus `HSE_VOICE_PACK_CACHE_DIR`; `platformdirs` is installed transitively but is **not** a declared dependency, so it was not used.
7. GitNexus MCP was available but its `harmony-speech-engine` index is **12 commits behind HEAD** (and cannot reflect the uncommitted 13-1/13-2 tree), so per the `12-6` precedent impact analysis was done by grep instead: `HarmonySpeechEngine.__init__` (kwargs-only callers), `EmbeddingCache` (additive method), `AsyncHarmonySpeech.start_background_loop` (additive task), `EngineConfig`/both YAMLs (additive fields). No existing symbol body was edited.

**Measured boot cost:** `_tbd_` (host session). Structure: `N` pack voices × `E` cloning-capable embed executors = `N × E` embed forwards on a cold cache; with the shipped GPU config `E = 4` (ChatterboxEmbedding, OV1 encoder, OV2 encoder, HarmonySpeech encoder) and `N = 28` ⇒ 112 forwards, but the three Chatterbox TTS families share one `ChatterboxEmbedding` executor (13-1 cross-family identity), so the per-family count is 28. On a warm disk it is zero forwards; the number is recorded once measured on the host (Multilingual warm-up is cheaper post-12-7 SDPA, so measure, do not extrapolate).

**Test counts:** full suite (`tests/unit tests/integration tests/e2e/wire_contract`) **1631 passed, 5 skipped** (baseline after 13-1: 1612 passed, 5 skipped; +19 new tests: 13 disk/planning/hydration/invalidation unit gates, 5 warm-up/hydration integration gates, 1 `cast_conditionals` mutation regression).

