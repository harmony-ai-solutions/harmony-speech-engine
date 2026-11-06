# Cross-request embedding cache

Voice cloning starts by deriving a speaker representation from the caller's
reference audio: a speaker embedding (OpenVoice, Harmony Speech) or the
Chatterbox conditioning tensors. That work depends only on the reference clip,
not on the text or the request id — but before Phase 7 every request paid for it
again. The engine now keeps the derived values in one bounded in-process cache
per engine, so a flood of clone requests against the **same** voice skips the
embedding model entirely after the first request.

This page documents what is cached, how it is bounded, how to tune it, and what
it means for data retention.

## What is cached

| Pipeline step | Cached value | Cache key |
|---|---|---|
| `chatterbox_embedding` / Chatterbox TTS inline embed | Chatterbox `Conditionals` (speaker embedding + S3Gen reference dict) | model identity, reference-audio content hash, `exaggeration` |
| Chatterbox voice conversion, target-audio path | Target-voice `ref_dict` | model identity, target-audio content hash |
| OpenVoice tone-converter encoder (V1 + V2) | Speaker embedding tensor | model identity, reference-audio content hash, VAD source |
| Harmony Speech encoder | Speaker embedding tensor | model identity, reference-audio content hash |

Notes:

* The key is content-addressed over the **decoded** audio (sample rate + PCM), so
  the same clip sent as WAV, FLAC or MP3 hits the same entry, while a 16 kHz and
  a 24 kHz decode never collide.
* The model identity includes the configured model, its revision, its dtype and
  a per-name load counter: reloading weights invalidates every entry written by
  the previous load (they age out via the TTL/FIFO policy below).
* Values are copied on the way in and on the way out, so two concurrent requests
  can never share mutable conditioning state.

Nothing is written to disk, and the cache never stores request text, output
audio or client identifiers — only values derived from reference audio.

## Configuration

Set the knobs in `config.yml` / `config.gpu.yml` (top level, next to
`request_timeout_s`), or override them on the command line.

| Config key | CLI flag | Default | Meaning |
|---|---|---|---|
| `embedding_cache_enabled` | `--no-embedding-cache` (disable) | `true` | Master switch. |
| `embedding_cache_ttl_s` | `--embedding-cache-ttl-s` | `300` | Entry lifetime in seconds. `0` disables the cache. |
| `embedding_cache_max_entries` | `--embedding-cache-max-entries` | `1000` | Entry cap. |
| `embedding_cache_max_bytes` | — | `1073741824` (1 GiB) | Resident-memory ceiling. |

```yaml
# config.yml
embedding_cache_enabled: true
embedding_cache_ttl_s: 300          # 5 minutes
embedding_cache_max_entries: 1000
embedding_cache_max_bytes: 1073741824
```

```bash
# lower the retention window for a privacy-sensitive deployment
python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \
    --embedding-cache-ttl-s 30

# turn it off entirely
python -m harmonyspeech.endpoints.openai.api_server --config-file-path config.gpu.yml \
    --no-embedding-cache
```

The cache is an optimisation only: with it disabled, behaviour and output are
identical to a build without it (every request recomputes the embedding).

## Eviction policy

Three independent bounds, all enforced per insert:

1. **TTL (default 300 s).** Entries expire lazily on lookup and are swept every
   16 inserts. A request that arrives after the TTL recomputes and re-stores.
2. **Entry cap (default 1000).** Beyond the cap the **oldest insertion** is
   evicted — FIFO, not LRU: a cache hit never reorders the store. Re-inserting
   an existing key counts as a new insertion.
3. **Byte ceiling (default 1 GiB).** A hard memory bound on top of the entry cap,
   because Chatterbox conditionals are megabytes each. The oldest insertion is
   evicted first; an entry larger than the whole ceiling is stored and
   immediately evicted.

FIFO (rather than LRU) is a deliberate product decision: it makes eviction
timing predictable from the request stream and keeps the hit path free of the
bookkeeping a true LRU needs.

## Observability

### Prometheus (`GET /metrics`)

Every series is present on every scrape — with zeros when the cache is disabled —
so dashboards and alerts never see a missing metric family.

| Series | Type | Meaning |
|---|---|---|
| `hse_embedding_cache_enabled` | gauge | `1` enabled, `0` disabled. |
| `hse_embedding_cache_entries` | gauge | Resident entries. |
| `hse_embedding_cache_bytes` | gauge | Approximate resident bytes. |
| `hse_embedding_cache_hits_total` | counter | Lookups served from the cache. |
| `hse_embedding_cache_misses_total` | counter | Lookups that recomputed. |
| `hse_embedding_cache_stores_total` | counter | Values written. |
| `hse_embedding_cache_ttl_expirations_total` | counter | Entries dropped by the TTL. |
| `hse_embedding_cache_fifo_evictions_total` | counter | Entries dropped by the entry cap. |
| `hse_embedding_cache_byte_evictions_total` | counter | Entries dropped by the byte ceiling. |
| `hse_embedding_cache_hit_ratio` | gauge | `hits / (hits + misses)`; `0` when there was no traffic. |

Useful alerts:

* `hse_embedding_cache_hit_ratio` near `0` with a steady `hse_embedding_cache_misses_total`
  rate on a cloning-heavy deployment means callers are sending distinct audio
  (the cache cannot help — that is the `distinct` flood case in
  [throughput benchmarks](benchmarks/throughput.md)).
* `hse_embedding_cache_fifo_evictions_total` or
  `hse_embedding_cache_byte_evictions_total` climbing steadily means the working
  set is larger than the caps: raise `embedding_cache_max_entries` /
  `embedding_cache_max_bytes` only if the host has memory headroom.
* `hse_embedding_cache_hit_ratio` dropping to `0` while traffic continues can be
  a sign of a TTL that is shorter than the caller's request interval.

### Health (`GET /health`)

`/health` carries the same numbers in an `embedding_cache` block, alongside the
per-model `vram` block:

```json
{
  "text_to_speech": "healthy",
  "voice_embedding": "healthy",
  "embedding_cache": {
    "enabled": true,
    "entries": 12,
    "bytes": 48234496,
    "hits": 340,
    "misses": 18,
    "stores": 12,
    "ttl_expirations": 0,
    "fifo_evictions": 0,
    "byte_evictions": 0,
    "hit_ratio": 0.9497,
    "ttl_s": 300.0,
    "max_entries": 1000,
    "max_bytes": 1073741824
  }
}
```

With the cache disabled the block reads `"enabled": false` and every counter `0`.

## Privacy note

The cached values are **derived voice biometrics**: a speaker embedding or the
Chatterbox conditioning tensors are enough to reproduce a person's voice timbre.
They are held in process memory (never on disk) for up to the configured TTL.

Deployments with strict data-retention requirements should therefore:

* disable the cache (`--no-embedding-cache` / `embedding_cache_enabled: false`),
  or
* lower `embedding_cache_ttl_s` to the shortest window that still helps them
  (e.g. `30`), and
* size `embedding_cache_max_entries` / `embedding_cache_max_bytes` to the
  smallest working set that sustains their traffic, and
* treat a process restart as the only guaranteed way to clear the cache
  immediately (there is no explicit flush endpoint).

The cache never logs reference audio or embeddings, and `/health` only reports
counts and sizes — never the values themselves.

## Sizing guidance

* Chatterbox conditionals are the large entries (single-digit MB each), so the
  1 GiB byte ceiling — not the 1000-entry cap — is usually what binds there.
  Roughly 200–400 resident Chatterbox voices fit in the default ceiling.
* Speaker embeddings (OpenVoice, Harmony Speech) are kilobytes each: the entry
  cap binds first and the full 1000 entries cost well under a megabyte.
* A TTL of 300 s matches interactive/agent workloads (a conversation or a batch
  job that keeps reusing one voice). Raise it only if the same voices are
  reused across long-lived sessions; every extra second is extra retention of
  biometric data.

## Related

* [Batching & concurrency](batching.md) — why cloning is bound by the
  per-request tone-converter hops, which is what the cache leaves untouched.
* [Throughput benchmarks](benchmarks/throughput.md) — the Phase 7 clone-flood
  measurements (`--clone-reference same|distinct` × cache on/off).
* [API reference](api.md) — `/health`, `/metrics` and the model endpoints.
