# Phase 7-1: Embedding Cache Core

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 7 — Cross-request embedding cache
> **Depends on:** Phase 2 (artifacts — cached values are `SpeakerArtifact`s); independent of Phase 8 (composes with it: hits shrink embed batches)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (engine/executor ownership pattern of `WeightRegistry`, 4-3), `CONVENTIONS.md`
> **State reference:** verified 2026-11-05 at `b81352d`+`d28b81d`

## Objective

A process-wide, ref-count-free **cross-request embedding cache**: identical reference audio no longer recomputes speaker embeddings/conditionals on every request. Policy per the product decision (2026-11-05): **TTL eviction, default 300 s, configurable**; **FIFO eviction when more than 1000 embeddings are cached in total**. This task delivers the cache module + config plumbing; 7-2 wires it into the executors.

## Context (self-contained)

- Today `_execute_openvoice_tone_converter_encoder` (serves OpenVoice V1 **and** V2 encoder steps) and `_execute_harmonyspeech_encoder` compute speaker embeddings, and `_execute_chatterbox_embedding` computes `Conditionals`, **per request** from `input_audio`. The Soulbits worker pattern (repeated cloning with the same handful of reference voices) recomputes the same tensors hundreds of times.
- Phase 2 artifacts: embeddings/conditionals flow as `SpeakerArtifact` (`.embedding` tensor for OpenVoice/HarmonySpeech, `.conditionals` object for Chatterbox) and are serialized only at the boundary (`to_wire_embedding`). The cache stores **artifacts, never wire bytes**.
- Phase 0-2 invariant to preserve: cached values must never share mutable state between requests — `Chatterbox Conditionals` get `.to(device)`-moved by consumers, so a cached entry **must be deep-copied on serve** (entries are a few MB at most; copy cost ~ms).
- Engine-owned singleton precedent: `WeightRegistry` (4-3) is created by `HarmonySpeechEngine.__init__` and threaded executor → worker → runner → loader. The cache follows the same ownership.

## Implementation steps

### 1. New module `harmonyspeech/processing/embedding_cache.py`

```python
@dataclass
class CacheEntry:
    value: SpeakerArtifact          # stored on CPU (tensors moved to cpu on store)
    key: str
    inserted_at: float              # time.monotonic()
    expires_at: float
    approx_bytes: int

class EmbeddingCache:
    """Cross-request embedding/conditionals cache.

    Eviction: TTL (default 300 s) checked lazily on get + swept on insert;
    FIFO beyond `max_entries` (default 1000 — oldest insertion evicted first,
    per product decision 2026-11-05, NOT LRU); optional byte ceiling as a
    safety bound (default 1 GiB) since Chatterbox conditionals are MB-sized.
    """
    def __init__(self, ttl_s: float, max_entries: int, max_bytes: int | None): ...
    def make_key(self, *, model_identity: str, audio_hash: str, params: Mapping) -> str: ...
    def get(self, key: str) -> SpeakerArtifact | None:      # deep copy on serve; None on miss/expired
    def put(self, key: str, value: SpeakerArtifact) -> None: # stores CPU deep copy; evicts per policy
    def invalidate_model(self, model_identity: str) -> int:  # called on executor (re)load
    def stats(self) -> dict:                                # counters + entries/bytes, feeds /metrics + /health
```

Rules:

- **Key** = sha256 over `(model_identity, audio_hash, params_hash)`:
  - `model_identity` = `ModelConfig.name + model_type + dtype + repo/revision + load_version`. `load_version` is a per-model-name counter the executor bumps on every (re)load — stale entries from a previous weight load become unreachable and age out via TTL/FIFO (no unbounded growth from reloads).
  - `audio_hash` = the request's `AudioArtifact.content_hash` (added in 7-2): sha256 over the **decoded** waveform bytes + sample rate, so the same audio in different containers (wav/flac/mp3 of one clip) is one key.
  - `params_hash` = generation options that influence the embedding — Chatterbox `exaggeration` (it bakes into `emotion_adv`), OpenVoice TCE: none today but include the VAD source (`"whisper"` vs `"silero"` — segment sets differ, see 3-4) so the key can't conflate them.
- **FIFO, not LRU** — per the product decision. `collections.OrderedDict` in insertion order; on `put` over capacity, pop oldest until within `max_entries`; also enforce `max_bytes` the same way. A `get` must NOT reorder (that would silently turn it into LRU).
- **TTL**: expired entries are treated as misses on `get`; a cheap sweep (iterate until first non-expired, given insertion order approximates expiry order only loosely — just sweep all keys) runs on every `put` (or every N puts, N=16, to bound cost at 1000 entries).
- **Copy discipline**: `put` deep-copies the artifact and moves tensors to CPU; `get` deep-copies again and leaves device placement to the consumer (explicit moves only, per Phase 2 device discipline). Both copies are counted in `approx_bytes` bookkeeping.
- **Thread safety**: `threading.Lock` around map mutations (prepare functions use a ThreadPoolExecutor; cheap insurance).
- **Counters** (monotonic): `hits`, `misses`, `stores`, `ttl_expirations`, `fifo_evictions`, `byte_evictions`.

### 2. Config plumbing

Follow the `request_timeout_s` precedent (`common/config.py:149`, `EngineConfig` + YAML key + CLI override):

| Key (YAML, engine section) | Default | Meaning |
|---|---|---|
| `embedding_cache_enabled` | `true` | master switch (`false` = no cache at all; also `ttl_s: 0` disables) |
| `embedding_cache_ttl_s` | `300` | TTL seconds (product decision default) |
| `embedding_cache_max_entries` | `1000` | FIFO cap (product decision default) |
| `embedding_cache_max_bytes` | `1073741824` | safety ceiling (1 GiB), documented as an addition on top of the 1000-entry FIFO cap |

CLI: `--embedding-cache-ttl-s` / `--embedding-cache-max-entries` / `--no-embedding-cache` (mirror how `--request-timeout-s` is wired in `endpoints/args.py`). Engine creates `self.embedding_cache` in `__init__` when enabled and threads it to executors alongside the weight registry.

### 3. Metrics surface (implementation; exposure wired in 7-3)

`stats()` returns the dict; series naming finalized in 7-3 (`hse_embedding_cache_*`), so implement `stats()` now with stable field names: `entries`, `bytes`, `hits`, `misses`, `stores`, `ttl_expirations`, `fifo_evictions`, `byte_evictions`.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/embedding_cache.py` (new) | `EmbeddingCache` + `CacheEntry` |
| `harmonyspeech/common/config.py` | 4 engine-level keys on `EngineConfig` (+ YAML load) |
| `harmonyspeech/endpoints/args.py` | CLI overrides |
| `harmonyspeech/engine/harmonyspeech_engine.py` | own + thread the cache (like the weight registry) |
| `tests/unit/processing/test_embedding_cache.py` (new) | policy tests |

## Tests

- Unit: TTL expiry (monkeypatched `time.monotonic`) — expired `get` = miss + counter; FIFO order — insert 1001, oldest evicted first, and `get` does **not** reorder (verify by eviction order after a hit); byte ceiling; disabled cache (`enabled=false`, `ttl_s=0`) never stores.
- Unit: copy isolation — mutate the artifact returned by `get` (move to device, overwrite tensor) → subsequent `get` returns the pristine stored value; stored entry unaffected by caller mutations.
- Unit: `make_key` stability — same inputs → same key; differs on audio hash / params / model identity / load_version.
- Unit: `invalidate_model` bumps reachability (new version → old key misses).
- Unit: counters monotonic and consistent after a scripted hit/miss/evict sequence.

## Checklist

- [ ] `EmbeddingCache` implemented (TTL + FIFO 1000 + byte ceiling, copy-on-serve, lock, counters)
- [ ] Config keys + CLI overrides wired; engine owns the cache
- [ ] Unit tests for every policy rule green
- [ ] `summary.md` Implementation Status updated
