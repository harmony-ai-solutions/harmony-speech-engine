# Phase 14-4: SSE Envelope (`stream_format: "sse"`)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-3 (the frame generator), 14-2 (event primitives)
> **State reference:** verified 2026-11-08 (research) — OpenAI contract: `speech.audio.delta {type, audio}` → `speech.audio.done {type, usage}` (vLLM-Omni's first impl used the wrong field name `delta` and was corrected — `audio` is authoritative); `speech.audio.error` exists in vLLM-Omni/speaches; `[DONE]` emission is inconsistent ecosystem-wide (AssemblyAI documents none) — we emit it, clients must not require it; `EventSource` is GET-only and cannot POST — browsers use `fetch` + `ReadableStream` (document this).

## Objective

A thin wrapper turning 14-3's frame generator into the event stream: the "contract-bearing" mode — boundaries, end-of-stream metadata (`sample_rate`, `duration_s`, `usage`), heartbeats, and **mid-stream error events** (binary mode's truncation weakness fixed here).

## Implementation steps

1. Envelope generator: same frame source as 14-3; each frame → `speech.audio.delta` with `audio` (base64 PCM, resampled if requested) + additive `chunk_index`; terminal `speech.audio.done` with `id/model/created/sample_rate/format/chunks/duration_s/usage` (duration from the **frame generator's own emitted-sample counter** — sum of `feed()`/`flush()` return lengths ÷ sample rate. Corrected 2026-11-09: the earlier note said "consume 12-2's exposure", but `CrossfadeStitcher` exposes no cumulative counter (`audio_utils.py:605-613`); count in the generator, or add a `samples_emitted` property if 14-2 touches the class — don't recompute from wire bytes, the resampled stream length ≠ source length); then `data: [DONE]`.
2. Error semantics: pre-first-byte failure ⇒ normal HTTP JSON error; post-first-byte ⇒ `speech.audio.error {error:{message,type,code}}` then close (exactly one terminal event, `done` xor `error` — pinned by test).
3. Keepalive: `: keepalive` comment line when a chunk takes >15 s (before the first delta too — browsers/proxies keep the connection warm during first-chunk synthesis).
4. Content-Type `text/event-stream; charset=utf-8` + the 14-1 headers; no gzip.
5. Tests (unit on the envelope + integration through the route): event ordering, delta `audio` b64 round-trip, exactly-one-terminal-event, error-after-N-deltas path, keepalive emission, done-payload fields.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | SSE branch over the shared generator |
| `tests/` | envelope gates |

## Checklist

- [x] Envelope + error/keepalive semantics landed
- [x] Terminal-event + ordering + round-trip gates green
- [x] `summary.md` ticked

## Verification log (2026-11-10)

**Status:** implementation + gates complete, suite green. No engine/model/runner
change, no new dependency, CPU-only tests. The 501 stub is gone; both envelopes
are served from the one frame source.

### What changed

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | `_encode_sse_frames` (the envelope generator), `_pcm16_base64` (one quantizer + base64 in a worker thread), `SSE_KEEPALIVE_TIMEOUT_S = 15.0` (module-level, resolved at call time), `_open_sse_stream` (priming + headers), `_prepend_frames` generalized to any frame type (PEP 695; binary behaviour identical), `StreamingAudioResponse.body` widened to `AsyncIterator[bytes \| str]`, dispatcher `"sse"` branch replaces `_sse_stream_unavailable` (the 501 stub is deleted), module docstring |
| `harmonyspeech/endpoints/openai/streaming.py` | additive `SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"` (+ `__all__`); no existing symbol changed |
| `harmonyspeech/endpoints/openai/api_server.py` | docstring only — the existing `StreamingResponse` branch is already envelope-agnostic (media type + header dict + body iterator) |
| `tests/unit/wire_contract/test_streaming_sse.py` | **new** — 19 unit gates on the envelope |
| `tests/integration/test_tts_streaming_sse.py` | **new** — 9 integration gates through the real serving object + engine loop |
| `tests/integration/test_tts_streaming_admission.py` | 14-1's interim-501 pin rewritten in place (`sse` now streams: body consumed, headers asserted) |
| `tests/unit/wire_contract/test_streaming_protocol.py` | 14-1's wire 501 pin rewritten in place (route-level SSE contract: content type, header set, one `done`, `[DONE]` last, PCM deltas) |
| `docs/api.md` | streaming section: SSE event semantics, the always-PCM rule, error/keepalive behaviour, `EventSource`-cannot-POST note |

Envelope (`_encode_sse_frames`), per frame of the **shared** source
`_stream_audio_segments` (identical ordering, DC-before-feed, per-segment
watermark, stateful resampling):

* one `speech.audio.delta` per non-empty segment — `audio` = base64 of
  `pcm16_frame(segment, target_rate)` (the one 14-2 quantizer, in
  `run_in_threadpool` with the base64) + additive `chunk_index` (0-based);
* one `speech.audio.done` from `done_payload`, reading `duration_s` from
  `StreamState`'s **emitted-sample counter** (source-rate samples ÷ source rate
  — the corrected 2026-11-09 fact; never the wire length, which differs once
  resampled), `sample_rate` = target rate, `format` = `"pcm"`, `chunks` = delta
  frames sent, `usage` = `{input_characters: len(input_text), audio_seconds}`
  (the primitive copies and extends; the caller's mapping is never mutated);
* then `SSE_DONE_FRAME` (`data: [DONE]`) as the very last line.

`format`/`X-Audio-Format` are `pcm` even when `format: "wav"` was requested:
SSE delta audio is headerless s16le by contract, so a WAV header would have to
ride inside the first delta (wrong for every consumer). The requested container
only frames the binary envelope. Of the 14-1 header set, **none** is
WAV-specific — `X-Audio-Bits`/`Channels` describe the PCM samples in the deltas,
`X-Audio-Format` names their encoding — so the full set is emitted unchanged
(with `X-Audio-Format: pcm`). `Content-Type` is `SSE_MEDIA_TYPE` (the charset is
stated in the constant, not left to Starlette's default). No compression is
added: the app has no compression middleware (only CORS) and
`Cache-Control: no-cache, no-transform` + `X-Accel-Buffering: no` are the
hygiene contract; gzip would defeat incremental delivery.

Error semantics: the first frame is **primed** exactly like 14-3
(`await anext(body)` inside `_open_sse_stream`), so a failure before anything is
on the wire is re-raised and becomes an ordinary JSON error (normal status, no
committed header). A failure after at least one frame becomes **one**
`speech.audio.error` `{error:{message,type,code}}` and the stream closes — no
`done`, no `[DONE]` (done XOR error, pinned both ways). A keepalive counts as
"on the wire": once it is sent the response is committed, so a later
first-delta failure is an error event (tested). `CancelledError` is re-raised
untouched — never converted into an error event — and the generator's `finally`
cancels the pending `__anext__` task and closes the frame source, so the shared
generator's abort-in-flight-chunk logic runs exactly as in 14-3 (tested through
the SSE branch, plus the polled-disconnect path).

Keepalive: a timeout race around the **frame wait only** — `asyncio.wait` on the
pending `frames.__anext__()` task. On timeout the same task is awaited again
(the synthesis task is never cancelled or re-issued, so a slow chunk costs no
CPU and no frame is lost or duplicated), and a `: keepalive` comment is yielded.
This covers the first delta too: the priming `anext` can return a keepalive, so
a cold first sentence keeps the connection warm. `SSE_KEEPALIVE_TIMEOUT_S`
(15 s) is module-level and resolved at call time, so tests
`monkeypatch.setattr` a tiny value (0.01 s) — no 15 s sleeps. A non-positive
override disables keepalives.

### Blast radius

`gitnexus` is available but its index is at `1243ddb` (one commit behind `HEAD`
7357358) and predates the untracked 14-1..14-4 files, so it cannot see
`streaming.py` or the new serving methods. Advisory only:
`gitnexus.detect_changes(scope="unstaged")` → 91 changed symbols / 12 files,
1 affected, **risk medium** (dominated by docs sections and the uncommitted
14-1..14-3 tree); `gitnexus.api_impact(route="/v1/audio/speech")` → **LOW**,
0 direct consumers, 0 affected flows.

Grep blast radius (authoritative; the graph is stale):

* `_sse_stream_unavailable` — **removed**; the only remaining mentions are the
  historical 14-3 task log. No code or test referenced it.
* `_open_sse_stream` / `_encode_sse_frames` / `_pcm16_base64` /
  `SSE_KEEPALIVE_TIMEOUT_S` / `SSE_MEDIA_TYPE` — new symbols, referenced only by
  the dispatcher/`_open_sse_stream` and the two new test files.
* `_prepend_frames` — two production call sites (`_open_audio_stream`,
  `_open_sse_stream`) and 14-3's unit test; the PEP 695 generalization is
  type-only, behaviour unchanged (14-3 tests untouched and green).
* `StreamingAudioResponse` — `api_server.create_speech` (branch unchanged),
  14-3's tests (untouched), the new SSE tests; the widened `body` annotation is
  the only structural change.
* `create_text_to_speech` — unchanged signature; the envelope selection moved
  below `from_wire_audio`/`_resolve_chunks` (both envelopes now share one
  dispatch point). Callers are the route and tests, all unaffected.
* `api_server.create_speech` — route handler; docstring only, no logic change.

### Tests

Command:
`& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header`

* before (14-3 baseline): **1877 passed, 5 skipped**
* after: **1905 passed, 5 skipped** (+28: 19 unit + 9 integration; the two
  14-1 interim-501 pins were rewritten in place, net +0)
* `ruff check` clean on every touched file except `api_server.py`'s
  pre-existing `SIM108` in `run_server` (not this change).

Gates — unit (`tests/unit/wire_contract/test_streaming_sse.py`):

1. `test_delta_audio_round_trips_to_the_exact_pcm_bytes` — b64 decodes to
   exactly `pcm16_frame(segment, rate)`, `chunk_index` 0..n-1, delta payload has
   only `{type, audio, chunk_index}`.
2. `test_event_ordering_is_deltas_then_one_done_then_done_frame` — kinds list,
   one `done`, no `error`, last frame is `SSE_DONE_FRAME`, no keepalive on a
   fast stream.
3. `test_empty_segments_are_skipped_and_do_not_consume_a_chunk_index` — no
   empty deltas; `chunks` counts real frames.
4. `test_done_payload_reports_target_rate_and_emitted_sample_duration` — target
   rate 16 k while `duration_s` = 24 000 source samples ÷ 24 kHz = 1.0 s,
   `format` `pcm`, `chunks` = deltas, `usage` extended.
5. `test_done_usage_is_extended_with_zero_input_characters_when_absent` /
   `test_done_never_mutates_the_caller_usage_mapping`.
6. `test_done_duration_is_zero_without_a_source_rate`.
7. `test_error_after_a_delta_is_one_terminal_error_event_and_no_done` —
   exact `{error:{message,type,code}}`, no `done`.
8. `test_error_event_is_not_followed_by_the_done_frame`.
9. `test_failure_before_the_first_frame_is_raised_not_an_event`.
10. `test_a_keepalive_counts_as_the_first_byte_for_error_semantics`.
11. `test_cancellation_propagates_and_is_never_an_error_event` — `CancelledError`
    out of `anext`, no error event.
12. `test_keepalive_threshold_default_is_15_seconds` (constant pin).
13. `test_keepalive_is_emitted_before_the_first_delta` /
    `test_keepalive_is_emitted_mid_stream_and_never_drops_a_frame`.
14. `test_module_level_threshold_is_resolved_at_call_time` (the monkeypatch
    seam) / `test_a_non_positive_threshold_disables_keepalives`.
15. `test_pcm16_base64_is_the_one_canonical_quantizer` /
    `test_quantization_runs_on_a_worker_thread`.

Gates — integration (`tests/integration/test_tts_streaming_sse.py`):

1. `test_sse_stream_orders_deltas_done_then_done_frame` — content type, full
   14-1 header set (`X-Audio-Format: pcm`, `X-Audio-Sample-Rate: 8000`),
   header-priming (`kinds[0]` is a delta), one `done` immediately before
   `[DONE]`, `X-Request-Id` == `done.id`, `chunks` == deltas,
   `usage.input_characters`/`audio_seconds`.
2. `test_delta_audio_round_trips_to_the_exact_pcm_bytes` — concatenated deltas
   == `pcm16_frame` of the stub's own chunk waveform (pre-watermark equality; the
   stub watermarker is the identity).
3. `test_multi_chunk_sse_stream_matches_the_whole_buffer_stitch` — deltas ==
   `crossfade_stitch` sample-for-sample, `state.chunks == 3`,
   `done.chunks == 4` (three chunk frames + the stitcher's flush tail),
   `duration_s` == stitched length ÷ source rate.
4. `test_done_reports_target_rate_and_source_sample_duration_under_resampling` —
   requested `format: "wav"` still yields `X-Audio-Format`/`done.format` `pcm`,
   `sample_rate` 16000, frames bit-identical to one-shot `soxr` (1 LSB), and
   `duration_s` == source samples ÷ source rate while
   `len(pcm)//2 != state.emitted_samples` (the resampled wire length can never
   be the source of the number).
5. `test_error_after_deltas_emits_exactly_one_error_and_no_done` — chunk 0's
   delta on the wire, chunk 1 fails: one `error`, no `done`, no `[DONE]`,
   `type` `StreamSynthesisError`, chunk 2 never submitted.
6. `test_first_chunk_failure_is_a_plain_json_error` — 400, no stream.
7. `test_keepalive_before_the_first_delta_and_between_deltas` — injected 0.01 s
   threshold + 0.05 s executor delay: primed first frame is `: keepalive`,
   keepalives before the first delta and between two deltas, all four frames
   intact, stream completes.
8. `test_disconnect_aborts_in_flight_chunk_through_the_sse_branch` — chunk 0
   primed alone, chunk 1 cancelled mid-synthesis: `engine.abort` fired for
   `-chunk-1`, exactly two submissions, chunk 2 never generated.
9. `test_polled_disconnect_stops_before_the_next_chunk_is_submitted` — the
   per-chunk poll stops before chunk 1; the frame source ends normally (it
   polls, it does not raise), so the envelope still writes its one terminal
   frame to a connection that is gone.

Route-level (rewritten 14-1 pins):
`test_wire_sse_envelope_streams_events` (status 200, exact content type, the
header set, one `done`, `[DONE]` last, non-RIFF PCM deltas) and
`test_sse_envelope_streams_after_admission` (serving level).

### Deviations / decisions recorded

1. **`done.chunks` counts delta frames, not planned synthesis chunks.** The
   client can verify it against `chunk_index` (0..chunks-1). For N ≥ 2 chunks
   it is `state.chunks + 1` because the stitcher's `flush()` tail is its own
   frame (14-3's `state.chunks` stays the planned count).
2. **A keepalive is the first byte.** "Headers commit with the first event"
   therefore commits on the first keepalive when the first chunk is slow. From
   that point failures are error events; before it they are JSON errors. This is
   the only coherent reading of "keepalive before the first delta" + lazy
   headers, and it is pinned by test.
3. **SSE always names `pcm`** in `X-Audio-Format`/`done.format`, whatever
   `output_options.format` said (see above).
4. **The polled-disconnect path still emits `done`** when the frame source ends
   normally (the shared generator has no "aborted" signal and 14-3's body ends
   the same way). The frame goes to a connection the poll just declared gone, so
   nothing is delivered; changing that would need an additive `StreamState` flag
   and is left to 14-5 if the metrics want to distinguish `completed` from
   `aborted` at the envelope level.
5. **`_prepend_frames` uses PEP 695 generics** (ruff `UP047`; the project
   targets 3.12). No runtime behaviour change; 14-3's tests untouched.
6. **`usage.input_characters` is the request's input text length** (the full
   text for a chunked request), passed in by `_open_sse_stream`; the 14-2
   primitive still defaults it to 0 when a caller supplies nothing.

### Reproduce

```powershell
# the whole gate set
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header

# just the 14-4 gates
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/wire_contract/test_streaming_sse.py tests/integration/test_tts_streaming_sse.py -q --no-header

# the rewritten 14-1 pins + the untouched 14-3 gates
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/wire_contract/test_streaming_protocol.py tests/integration/test_tts_streaming_admission.py tests/integration/test_tts_streaming.py tests/unit/wire_contract/test_streaming_serving.py -q --no-header
```

**Blocker:** none.

