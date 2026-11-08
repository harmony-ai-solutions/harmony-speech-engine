# Phase 14-3: Chunked Streaming Serving Path (`stream_format: "audio"`)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-1 (admission), 14-2 (primitives), **12-2** (chunk loop + chunk-by-chunk carry stitcher)
> **State reference:** verified 2026-11-08 (research) — `AsyncHarmonySpeech.generate` yields from a per-request `AsyncStream` (`engine/async_harmonyspeech.py:49-79, 493-527`); disconnect handling already aborts (`serving_text_to_speech.py:80-85` → `engine.abort` → `_release_request_resources` frees AR row + 9-2 slot, `engine/harmonyspeech_engine.py:701-722, 461-484`); `UnfinishedRequestOutput` carries progress only — **audio exists only per completed chunk**, so frame = chunk (sub-utterance streaming is out of scope, promise "first audio after the first sentence").

## Objective

The binary envelope: `create_text_to_speech` becomes a dispatcher (`_single` / `_chunked` / `_chunked_stream`); the streaming generator emits header → crossfaded PCM frames → end, with **correct disconnect abort** as the single most important correctness item.

## Implementation steps

1. Dispatcher refactor: today's path becomes `_single`/`_chunked` untouched; `_chunked_stream` is the new generator (12-2's loop + carry stitcher, sequential chunk submission — ordering by construction, memory bounded to one chunk; pipelining is a measured later optimization, not MVP).
2. Frame loop: per chunk — engine result → carry-crossfade (hold back `fade_samples`, emit emittable audio; research: 480 samples @20 ms/24 k, a one-chunk 20 ms delay) → resample via the 14-2 stream if `sample_rate` set → PCM frame via `to_thread` (never encode on the event loop) → yield. First yield = WAV header (`0xFFFFFFFF` sizes) with **target** rate; `pcm` format omits it.
3. **Disconnect:** `try/except asyncio.CancelledError: await engine.abort(current_chunk_request_id); raise` + `finally` aborting whatever chunk is in flight (Starlette cancels the generator on disconnect — the only reliable hook). A closed tab must stop synthesizing chunk 7..20.
4. Errors: pre-first-byte failures ⇒ normal HTTP error (status not yet committed — the header must not be sent before the first chunk is *in hand*: send headers lazily with the first frame); post-first-byte ⇒ truncated body (binary mode's known weakness; SSE mode fixes it in 14-4 — document).
5. Keepalive: none in binary mode (cannot inject bytes); document the long-first-chunk proxy-timeout caveat (`proxy_read_timeout` guidance in `docs/api.md`; SSE mode has `: keepalive`).
6. `api_server.create_speech`: replace the commented scaffolding (`:291-292`) with the real `StreamingResponse` branch + the 14-1 header set.
7. Tests: streamed-vs-buffered **PCM payload byte-equality** when the text is one chunk (header sizes differ by design — assert payload only); carry continuity across frames (no discontinuity at boundaries beyond the crossfade); disconnect aborts the in-flight chunk (engine.abort asserted, pending chunks never submitted); resampled stream header carries the target rate; `speed` composes per chunk.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher + `_chunked_stream` |
| `harmonyspeech/endpoints/openai/api_server.py` | real `StreamingResponse` branch |
| tests (integration + unit) | gates above |

## Checklist

- [ ] Dispatcher + frame loop + lazy headers landed
- [ ] Disconnect-abort gate green (the correctness item)
- [ ] Payload byte-equality + carry + resample gates green; `summary.md` ticked
