# Phase 14-5: Stream Contract Tests, Metrics, Docs (phase exit)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-1..14-4 (the whole surface under test)
> **State reference:** verified 2026-11-08 (research) — Kokoro's disconnect test is the ecosystem's reference correctness gate; httpx `TestClient.stream` supports streaming-body assertions; our wire-contract suite (`tests/e2e/wire_contract/`) is the natural home; metrics house style: `hse_*` Prometheus series with zero-values-never-absent discipline (9-2 precedent).

## Objective

The streaming surface is pinned by contract, observable in production, and honestly documented — Phase 14's exit gate.

## Implementation steps

1. **Wire-contract tests** (`tests/e2e/wire_contract/`): binary mode — `Transfer-Encoding: chunked`, no `content-length`, the `X-Audio-*` header set (incl. target rate under resampling), WAV header field values, payload = buffered-path PCM (single-chunk text); SSE mode — content type, event ordering, delta shape (`audio` field), exactly one terminal event, `[DONE]` last line, keepalive tolerated by the parser; admission — mp3+stream 400, alias equivalence (`stream:true` ≡ `stream_format:"audio"`), plus the 2026-11-09 owner matrix: `trim_silence`/`fix_internal_silence`/non-native-`speed` + stream ⇒ 400 naming the whole-buffer restriction, `remove_dc_offset` + stream ⇒ 200.
2. **Real-server disconnect test** (httpx against live uvicorn, the server-lifecycle guardrails): client aborts mid-stream ⇒ engine.abort fired, in-flight chunk released (AR row + cache slot freed — assert via the 9-4/9-2 stats), no further chunk synthesis (log discipline).
3. **Audio regression**: streamed (multi-chunk) vs buffered chunked synthesis — identical **pre-watermark** PCM given the same seed and chunk plan (determinism across envelopes; extends the 4-2 harness axes). Scope notes (2026-11-09): (a) the watermark diverges **by design** — per-segment perth in streaming vs one whole-clip call buffered — so the axis compares stitch/DC output, never final wire bytes; (b) admission already excludes `trim_silence`/`fix_internal_silence`/non-native `speed` from streaming, so both paths share stitch + DC only and no filter asymmetry can leak in. Phase 12 already pins the primitive: `crossfade_stitch(...)` ≡ streaming `feed()+flush()` sample-for-sample in `tests/unit/common/test_audio_utils.py`. This axis therefore validates the **stream generator's** chunk ordering and frame slicing, not the stitcher — if it diverges, look at `_chunked_stream` first.
4. **Metrics**: `hse_tts_stream_first_frame_seconds` (histogram — the TTFA number the docs will quote), `hse_tts_stream_chunks_total{outcome=completed|aborted|error}`, `hse_tts_stream_bytes_total`; zero-state present when unused (house rule). Record TTFA split by **batch composition** (chunk 0 alone vs co-batched), because Phase 12-8's deferred vocode makes TTFA depend on the slowest co-batched row, not only on text length (see 14-3's Phase 12 delta note).
5. **Docs**: `docs/api.md` streaming section finalized (curl `--no-buffer | ffplay -` example, browser `fetch`+AudioWorklet snippet for SSE+pcm, proxy notes: `X-Accel-Buffering`, `proxy_read_timeout`, no-gzip); README feature line; CHANGELOG (user-visible streaming block for the whole phase); `summary.md` tick.
6. Optional measured follow-up (separate, NOT in this phase): pipelined chunk submission with a bounded reorder window (k=2) — only with a proven TTFA/wall-clock win.

## Files to create/modify

| File | Change |
|---|---|
| `tests/e2e/wire_contract/`, `tests/integration/` | contract + disconnect gates |
| `scripts/audio_regression.py` | streamed-vs-buffered axis |
| metrics module(s), `docs/api.md`, `README.md`, `CHANGELOG.md` | observability + docs |

## Checklist

- [ ] Wire-contract suites green (both envelopes + admission matrix)
- [ ] Live disconnect test green (abort + resource release asserted)
- [ ] Metrics + regression axis landed; docs/CHANGELOG final; `summary.md` ticked
