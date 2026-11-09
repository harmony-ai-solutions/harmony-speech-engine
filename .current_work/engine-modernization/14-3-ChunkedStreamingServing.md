# Phase 14-3: Chunked Streaming Serving Path (`stream_format: "audio"`)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-1 (admission), 14-2 (primitives), **12-2** (chunk loop + chunk-by-chunk carry stitcher)
> **State reference:** verified 2026-11-08 (research) — `AsyncHarmonySpeech.generate` yields from a per-request `AsyncStream` (`engine/async_harmonyspeech.py:49-79, 493-527`); disconnect handling already aborts (`serving_text_to_speech.py:80-85` → `engine.abort` → `_release_request_resources` frees AR row + 9-2 slot, `engine/harmonyspeech_engine.py:701-722, 461-484`); `UnfinishedRequestOutput` carries progress only — **audio exists only per completed chunk**, so frame = chunk (sub-utterance streaming is out of scope, promise "first audio after the first sentence").

## Objective

The binary envelope: `create_text_to_speech` becomes a dispatcher (`_single` / `_chunked` / `_chunked_stream`); the streaming generator emits header → crossfaded PCM frames → end, with **correct disconnect abort** as the single most important correctness item.

## Phase 12 final state — deltas this task must build on (verified 2026-11-09, commit `8b7d222`)

The research date was 2026-11-08, before Phase 12 landed. Four numbers/behaviours changed; do **not** re-derive them from the older notes:

1. **The carry hold-back is 40 ms, not 20 ms.** The shipped chunked defaults are `crossfade_ms = 40` / `sentence_pause_ms = 120` (`DEFAULT_CHUNK_CROSSFADE_MS` / `DEFAULT_CHUNK_SENTENCE_PAUSE_MS` in `common/audio_utils.py`). At 24 kHz that is **960** samples held back, so the first frame of an N-chunk stream trails by 960 samples. The audible inter-chunk gap is `sentence_pause_ms - crossfade_ms + 2 * guard_ms` = `120 - 40 + 2*10` = **100 ms**, not `120 ms`.
2. **`feed()` conditions and fades internally — never pre- or double-condition.** `CrossfadeStitcher.feed` runs `condition_edges` on every fed chunk (near-silence trimmed to a 10 ms guard, gate −45 dBFS relative to that chunk's own peak, ≤150 ms per edge) and, when `pause_samples > 0`, applies the **two-sided** pause fade: the outgoing tail fades to silence with `cos²` *before* the pause, the incoming head fades up from silence with `sin²`. The carry therefore holds **conditioned** samples. Feeding an already-conditioned chunk would trim twice and break the byte-equality gate below.
3. **Byte-equality is already proven at the unit level — make that the first line of defence.** `crossfade_stitch(chunks, …)` and streaming `feed()+flush()` are pinned sample-for-sample identical in `tests/unit/common/test_audio_utils.py` (including the paused path). The 14-5 end-to-end streamed-vs-buffered comparison is therefore a *confirmation*, not a discovery tool: if it fails, the bug is in the stream generator's chunk ordering or its frame slicing, not in the stitcher.
4. **Deferred vocode can delay the first frame — this is the one genuine conflict with 14-3's promise.** Phase 12-8 holds a finished AR row's S3Gen vocode until the other rows co-batched with it finish T3 decoding (so staggered-EOS chunks vocode in one padded `inference_batched` call). With 14-3's **sequential** chunk submission the first chunk runs alone (`len(inputs) == 1` → the 9-1 sequential path), so the "first audio after the first sentence" promise holds. But if the engine co-batches chunk 0 with **another client's** rows, chunk 0's audio waits for the slowest row in that batch.
   - Consequence: TTFA is a function of batch composition, not just of text length. Bounded by the slowest co-batched request's decode, not by 1/4 of the caller's own render.
   - Mitigation if 14-5's measured TTFA is unacceptable: add a runner-side "flush pending vocode when this request is the only unfinished row in the batch" rule (a `_pending_vocode` drain on the last-row condition), not a stitcher/serving hack. Record the decision and its measurement in 14-5.
   - Measured ceiling from 12-9 for a 4-chunk 807-char request at `c=1` (`split=on`): `chatterbox` p50 10.3 s, `chatterbox_multilingual` 10.5 s, `chatterbox_turbo` 7.0 s, `melo` 0.9 s, `openvoice_v2` 1.5 s. One chunk alone is roughly a quarter of the AR figures (~2–3 s) and near-instant for the padded-batch families; treat those as the TTFA budget the docs may quote until 14-5 measures the real histogram.

## Implementation steps

1. Dispatcher refactor: today's path becomes `_single`/`_chunked` untouched; `_chunked_stream` is the new generator (12-2's loop + carry stitcher, sequential chunk submission — ordering by construction, memory bounded to one chunk; pipelining is a measured later optimization, not MVP).
2. Frame loop: per chunk — engine result → carry-crossfade (hold back `fade_samples`; the shipped default is **40 ms = 960 samples @ 24 kHz**, and the stitcher conditions + two-sided-pause-fades internally — see the Phase 12 delta section) → resample via the 14-2 stream if `sample_rate` set → PCM frame via `to_thread` (never encode on the event loop) → yield. First yield = WAV header (`0xFFFFFFFF` sizes) with **target** rate; `pcm` format omits it. Emit the final `flush()` tail before the terminating frame — the stitcher never materialises a trailing pause, so the last chunk's conditioned tail is only in `flush()`.
3. **Disconnect:** `try/except asyncio.CancelledError: await engine.abort(current_chunk_request_id); raise` + `finally` aborting whatever chunk is in flight (Starlette cancels the generator on disconnect — the only reliable hook). A closed tab must stop synthesizing chunk 7..20.
4. Errors: pre-first-byte failures ⇒ normal HTTP error (status not yet committed — the header must not be sent before the first chunk is *in hand*: send headers lazily with the first frame); post-first-byte ⇒ truncated body (binary mode's known weakness; SSE mode fixes it in 14-4 — document).
5. Keepalive: none in binary mode (cannot inject bytes); document the long-first-chunk proxy-timeout caveat (`proxy_read_timeout` guidance in `docs/api.md`; SSE mode has `: keepalive`).
6. `api_server.create_speech`: replace the commented scaffolding (`:291-292`) with the real `StreamingResponse` branch + the 14-1 header set.
7. Tests: streamed-vs-buffered **PCM payload byte-equality** when the text is one chunk (header sizes differ by design — assert payload only); carry continuity across frames (no discontinuity at boundaries beyond the crossfade); multi-chunk streamed output equals `crossfade_stitch` of the same chunk set sample-for-sample (the unit gate already proves the primitive — this proves the generator's frame slicing); disconnect aborts the in-flight chunk (engine.abort asserted, pending chunks never submitted); resampled stream header carries the target rate; `speed` composes per chunk; **TTFA under co-batching** recorded (chunk 0 alone vs chunk 0 co-batched with another request's rows) so the deferred-vocode caveat has a number attached.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher + `_chunked_stream` |
| `harmonyspeech/endpoints/openai/api_server.py` | real `StreamingResponse` branch |
| `harmonyspeech/task_handler/{ar_runner,model_runner_base}.py` | only if the measured TTFA forces the last-row `_pending_vocode` drain |
| tests (integration + unit) | gates above |

## Checklist

- [ ] Dispatcher + frame loop + lazy headers landed (final `flush()` tail emitted)
- [ ] Disconnect-abort gate green (the correctness item)
- [ ] Payload byte-equality + carry + resample gates green
- [ ] Multi-chunk stream ≡ whole-buffer stitch sample-for-sample
- [ ] TTFA under co-batching measured and recorded; deferred-vocode mitigation decided
- [ ] `summary.md` ticked
