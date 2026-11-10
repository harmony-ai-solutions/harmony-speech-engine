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
5. **The post-stitch chain is whole-buffer — admission resolves it, not the frame loop.** Grounded at `8b7d222`: the buffered chunked path (`serving_text_to_speech.py:466-506`) gathers **all** chunks, then runs `crossfade_stitch → trim_silence → fix_internal_silence → time_stretch (non-native speed) → one whole-clip perth watermark` on the final buffer; the silence gate is relative to the **global** RMS peak (`audio_utils.py:853-859`), so trim/internal/stretch cannot be applied sample-identically to incrementally-emitted frames. Owner decisions 2026-11-09: `trim_silence`/`fix_internal_silence`/non-native `speed` + streaming ⇒ 400 at admission (14-1 — this path never sees them); `remove_dc_offset` runs per chunk **before** `stitcher.feed()` (matching the buffered ordering, `serving_text_to_speech.py:471-474`); the watermark runs **per emitted segment** — one perth `apply_watermark` call on each emittable frame before PCM encoding (reuse `_apply_final_watermark`'s model-resolution helper, `serving_text_to_speech.py:714-723`; divergence from the buffered whole-clip call is documented, and the 14-5 equality axis is scoped to pre-watermark PCM).
6. **Deadline stamping differs for sequential submission.** The buffered dispatcher stamps `request_timeout_s = base × (chunk_count + 1)` on every chunk sub-request because the whole burst queues at once (`_chunk_request_timeout_s`, `serving_text_to_speech.py:564-577`, internal non-wire field on `TextToSpeechRequestInput`). Sequential streaming submission must derive its own rule — a chunk waits only for itself plus whatever other requests are ahead of it in the queue, so stamping `base` per chunk is the MVP rule; record it next to the 14-5 TTFA numbers for review.

## Implementation steps

1. Dispatcher refactor: today's path becomes `_single`/`_chunked` untouched; `_chunked_stream` is the new generator (12-2's loop + carry stitcher, sequential chunk submission — ordering by construction, memory bounded to one chunk; pipelining is a measured later optimization, not MVP).
2. Frame loop: per chunk — engine result → `remove_dc_offset` per chunk **before** `feed()` (the one post-option that composes) → carry-crossfade (hold back `fade_samples`; the shipped default is **40 ms = 960 samples @ 24 kHz**, and the stitcher conditions + two-sided-pause-fades internally — see the Phase 12 delta section) → watermark the emittable segment (per-segment perth, owner decision 2026-11-09 — delta item 5) → resample via the 14-2 stream if `sample_rate` set → PCM frame via `to_thread` (never encode on the event loop) → yield. Nothing else from the buffered post-stitch chain applies here: trim/internal/non-native `speed` were rejected at admission. First yield = WAV header (`0xFFFFFFFF` sizes) with **target** rate; `pcm` format omits it. Emit the final `flush()` tail before the terminating frame — the stitcher never materialises a trailing pause, so the last chunk's conditioned tail is only in `flush()`.
3. **Disconnect:** `try/except asyncio.CancelledError: await engine.abort(current_chunk_request_id); raise` + `finally` aborting whatever chunk is in flight (Starlette cancels the generator on disconnect — the only reliable hook). A closed tab must stop synthesizing chunk 7..20.
4. Errors: pre-first-byte failures ⇒ normal HTTP error (status not yet committed — the header must not be sent before the first chunk is *in hand*: send headers lazily with the first frame); post-first-byte ⇒ truncated body (binary mode's known weakness; SSE mode fixes it in 14-4 — document).
5. Keepalive: none in binary mode (cannot inject bytes); document the long-first-chunk proxy-timeout caveat (`proxy_read_timeout` guidance in `docs/api.md`; SSE mode has `: keepalive`).
6. `api_server.create_speech`: replace the commented scaffolding (`:291-292`) with the real `StreamingResponse` branch + the 14-1 header set.
7. Tests: streamed-vs-buffered **pre-watermark PCM payload byte-equality** when the text is one chunk (header sizes differ by design; the watermark diverges by design — per-segment vs whole-clip, owner decision 2026-11-09 — so assert the stitch/DC output, not final wire bytes); carry continuity across frames (no discontinuity at boundaries beyond the crossfade); multi-chunk streamed output equals `crossfade_stitch` of the same chunk set sample-for-sample (the unit gate already proves the primitive — this proves the generator's frame slicing); disconnect aborts the in-flight chunk (engine.abort asserted, pending chunks never submitted); resampled stream header carries the target rate; per-chunk deadline stamping pinned (delta item 6); **TTFA under co-batching** recorded (chunk 0 alone vs chunk 0 co-batched with another request's rows) so the deferred-vocode caveat has a number attached. Admission matrix itself (trim/internal/non-native-speed 400s, DC allowed) is 14-1's gate — one cross-check test here that the generator never receives such a request.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher + `_chunked_stream` |
| `harmonyspeech/endpoints/openai/api_server.py` | real `StreamingResponse` branch |
| `harmonyspeech/task_handler/{ar_runner,model_runner_base}.py` | only if the measured TTFA forces the last-row `_pending_vocode` drain |
| tests (integration + unit) | gates above |

## Checklist

- [x] Dispatcher + frame loop + lazy headers landed (final `flush()` tail emitted)
- [x] Disconnect-abort gate green (the correctness item)
- [x] Payload byte-equality + carry + resample gates green
- [x] Multi-chunk stream ≡ whole-buffer stitch sample-for-sample
- [x] TTFA under co-batching measured and recorded; deferred-vocode mitigation decided (host pass 2026-11-10 — see the verification log below)
      *Mechanism recorded and pinned structurally in `tests/integration/test_tts_streaming.py` (chunk 0 alone ⇒
      first frame after chunk 0; chunk 0 co-batched behind a slower row ⇒ first frame waits for the slowest row).
      The real-GPU TTFA histogram (and the last-row `_pending_vocode` drain decision) is the orchestrator's 14-5
      benchmark pass — 14-3 deliberately made no runner change. See the verification log below.*
- [x] `summary.md` ticked

## Verification log (2026-11-10)

**Status:** implementation + gates complete, suite green. The only open item is the
real-GPU TTFA histogram, which is the orchestrator's 14-5 benchmark pass (see the
"TTFA" section below); no runner/model change was made.

### What changed

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher split (`_single` / `_chunked` / `_chunked_stream` seam), the shared frame source `_stream_audio_segments`, `_open_audio_stream` (lazy headers), `StreamSynthesisError` / `StreamState` / `StreamingAudioResponse`, `_encode_stream_frames` / `_prepend_frames`, `_stream_chunk_request_timeout_s`, `_stream_clone_artifacts`, `_result_waveform` / `_emit_stream_segment` / `_finish_stream_segment`, `_abort_stream_request`, `_sse_stream_unavailable` stub |
| `harmonyspeech/endpoints/openai/api_server.py` | `create_speech` returns the real `StreamingResponse` over the primed body + the 14-1 header set (replaces the commented scaffolding at the old `:291-292`) |
| `tests/integration/test_tts_streaming.py` | **new** — 13 gates (see below) |
| `tests/unit/wire_contract/test_streaming_serving.py` | **new** — 9 unit gates (frame encoder, thread offload, primed-first-frame, deadline rule, duration math) |
| `tests/integration/test_tts_streaming_admission.py` | 14-1's interim-501 pins updated: `audio` now streams (body consumed, headers asserted), `sse` still 501 |
| `tests/unit/wire_contract/test_streaming_protocol.py` | 14-1's wire pins updated the same way; added a `pcm`-omits-header test |
| `tests/unit/wire_contract/test_multipart.py` | the `stream: true` alias test now asserts a streamed `200 audio/* RIFF` body instead of the 501 |
| `tests/wire_contract_support.py` | `FakeEngine.add_request` (+ `_SingleOutputStream`) so the streaming path can drive the wire fake exactly like `AsyncHarmonySpeech` (additive; buffered tests untouched) |

Dispatcher details:

* `create_text_to_speech` keeps the 14-1 admission gate where it was and now
  selects the envelope *before* the buffered dispatch. `"sse"` → `_sse_stream_unavailable()`
  (501, never a silent buffered 200); `"audio"` → `_open_audio_stream`;
  otherwise `_chunked` when the splitter produced 2+ chunks, else `_single`.
* `_single` is the pre-12 tail moved verbatim (engine `generate` +
  `text_to_speech_full_generator`); `_chunked` is the 12-2 method renamed only.
  Both bodies are unchanged — the byte-identity tests in
  `tests/integration/test_tts_chunking.py` (and the wire matrix) still pin them.
* `_chunked_stream` is the binary **encoder** half (`_encode_stream_frames`);
  the engine-facing half is the one shared frame source
  `_stream_audio_segments`, which 14-4's SSE wrapper will consume unchanged
  (it reads `state.chunks` / `state.duration_s` for `speech.audio.done`).

Frame loop (per chunk, sequential): engine result → `_result_waveform`
(`audio_artifact.numpy`, the accessor the buffered path uses; the legacy
base64 payload goes through `audio_codec.decode_wav_b64`) → `remove_dc_offset`
**before** `CrossfadeStitcher.feed` (which conditions edges and applies the
two-sided pause fade internally — never pre-/double-conditioned) → one perth
`apply_watermark` per **emitted segment** (via the existing
`_apply_final_watermark` model-resolution helper) → `StreamResampler.feed`
when `sample_rate` was requested → `pcm16_frame` in `run_in_threadpool` →
yield. DC removal, the stitcher feed, the watermark and the resampler all run
in the same worker-thread call (`_emit_stream_segment`), so no CPU-heavy work
touches the event loop. After the loop: `stitcher.flush()` tail (the last
chunk's conditioned tail exists only there) through the same watermark →
resample chain, then `resampler.flush()` (already-marked audio, not re-marked).
The 0/1-chunk case bypasses the stitcher, mirroring `crossfade_stitch`'s
single-chunk byte-identity rule.

First yield = `wav_stream_header(target_rate) + chunk-0 PCM` (`pcm` omits the
header). `_open_audio_stream` **primes** that first frame before returning, so
`StreamingResponse` never commits headers for a stream that failed before the
first byte: a pre-first-byte `StreamSynthesisError` becomes an ordinary JSON
error (normal HTTP status). Post-first-byte failures truncate the body — the
documented weakness of binary streaming, fixed by the 14-4 SSE envelope.

Disconnect: `except asyncio.CancelledError: await engine.abort(in_flight); raise`
plus a `finally` that aborts whatever chunk is in flight (the clone `-embed`
sub-request is tracked the same way). Sequential submission means the remaining
chunks are never even submitted. uvicorn 0.37 sends ASGI `spec_version 2.3`, so
Starlette's anyio task group cancels the response task — the CancelledError path
is the live one; the `finally` is the belt-and-braces hook for servers that end
the response by exception (spec >= 2.4). Also checked `await raw_request.is_disconnected()`
before each submission: pre-first-byte it is an honest error, later it just stops
submitting.

Deadline: `_stream_chunk_request_timeout_s()` returns the engine-wide base
(`None`/0 stay disabled), stamped on every chunk sub-request; the buffered
`_chunk_request_timeout_s(n) = base * (n + 1)` burst rule is untouched.

### Blast radius

`gitnexus` is available but its index is at `1243ddb` (one commit behind `HEAD`
7357358) and predates the untracked 14-1/14-2/14-3 files, so it cannot see
`streaming.py` or the new serving methods. Advisory results:
`gitnexus.detect_changes(scope="unstaged")` → 61 changed symbols, 1 affected,
**risk medium** (almost entirely docs sections + touched symbols);
`gitnexus.api_impact(route="/v1/audio/speech")` → **LOW**, 0 direct consumers,
0 affected flows.

Grep blast radius for the symbols touched (graph is stale, grep is authoritative):

* `create_text_to_speech(` — 44 call sites, all *callers*: `api_server.create_speech`
  (the only production caller, updated), `scripts/{measure_vram,audio_regression}.py`,
  and tests (`tests/integration/*`, `tests/e2e/**`). All send buffered requests
  except the new streaming tests; return type is additive
  (`StreamingAudioResponse` only when an envelope is selected).
* `api_server.create_speech` — route handler for `POST /v1/audio/speech`; the only
  in-repo reference is the router decorator + wire tests. LOW risk.
* `_create_chunked_response` — referenced only by historical task logs
  (12-2/12-5/13-1 notes); no code or test calls it. Renamed to `_chunked`.
* `text_to_speech_full_generator` — internal only (called by `_single`).
* `_chunk_request_timeout_s` — unchanged; pinned by
  `tests/unit/wire_contract/test_long_form_fields.py:92` and monkeypatched by
  `tests/integration/test_tts_chunking.py:1674`.
* `_apply_final_watermark` / `_resolve_serving_model` — unchanged, now also called
  per emitted segment from the frame source (threadpool, same lock).
* `tests/wire_contract_support.FakeEngine` — additive `add_request`; every existing
  buffered wire test still passes untouched.

### Tests

Command:
`& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header`

* before (14-2 baseline): **1854 passed, 5 skipped**
* after: **1877 passed, 5 skipped** (+23: 14 in `tests/integration/test_tts_streaming.py`,
  9 in `tests/unit/wire_contract/test_streaming_serving.py`; the 14-1 501 pins were
  rewritten in place, net +0)
* `ruff check` clean on every touched file except `api_server.py`'s pre-existing
  `SIM108` in `run_server` (not this change; the repo does not enforce format).

Gates in `tests/integration/test_tts_streaming.py`:

1. `test_one_chunk_stream_matches_buffered_pre_watermark_pcm[False|True]` — the
   streamed PCM equals the buffered path's pre-watermark samples (stub watermarker
   is the identity, so the buffered WAV body *is* the stitch/DC reference); the
   `remove_dc_offset` variant pins DC-before-feed ordering. A separate 1-LSB
   tolerance cross-check against the buffered WAV documents that the **WAV
   container** (libsndfile, x32768) and headerless PCM (`_to_s16le`, x32767) differ
   by one LSB — a container difference, not a streaming one.
2. `test_multi_chunk_stream_matches_whole_buffer_stitch_sample_for_sample` — the
   streamed bytes equal `pcm16_frame(crossfade_stitch(same chunks))` exactly
   (proves generator ordering + frame slicing; the primitive is already pinned in
   `tests/unit/common/test_audio_utils.py`), plus `response.state.chunks/duration_s`.
3. `test_carry_continuity_across_frame_boundaries[0|120]` — max per-sample step over
   the whole stream < 0.05 (a naive concat of the 0.16/0.41 blocks would step 0.25),
   asserted at every frame seam; both the direct-crossfade and the pause-fade path.
4. `test_disconnect_aborts_in_flight_chunk_and_never_submits_the_rest` — chunk 0 is
   primed alone, chunk 1 is cancelled mid-synthesis: `engine.abort` fired exactly
   for `-chunk-1`, exactly two submissions, `generate_calls == [chunk 0 text]`
   (chunks 2.. never submitted).
5. `test_polled_disconnect_stops_before_the_next_chunk_is_submitted` — the
   complementary hook: `raw_request.is_disconnected()` turning true after chunk 0
   stops before chunk 1 is submitted and skips the tail flushes.
6. `test_resampled_stream_header_carries_the_target_rate` — `X-Audio-Sample-Rate`
   and the RIFF header say 16000, frames equal one-shot `soxr.resample` of the same
   stitch (1 LSB).
7. `test_sequential_stream_stamps_the_base_deadline_per_chunk` — every chunk carries
   `request_timeout_s == 60.0` with `base=60`, while `_chunk_request_timeout_s(3)
   == 240.0` (control: the buffered burst rule is unchanged).
8. `test_generator_never_receives_a_request_the_admission_gate_rejects` —
   `trim_silence` / `fix_internal_silence` / non-native `speed` / `mp3` all 400 with
   zero submissions and zero model calls.
9. `test_stream_state_counts_emitted_source_samples` — the counter equals the
   whole-buffer stitch length and `duration_s == samples / source_rate`.

### TTFA under co-batching (structural) + host deferral

`test_ttfa_structural_chunk_zero_alone_emits_before_the_slow_row` and
`test_ttfa_structural_cobatched_chunk_zero_waits_for_the_slowest_row` model 12-8's
deferred-vocode rule directly in the harness: a `_BatchModelExecutor` releases
**every** row of a batch only when the slowest row is done (a text starting with
`Z` is the slow row), and records batch composition + per-row generation order.

* chunk 0 submitted alone ⇒ its batch is `[(chunk-0, SENTENCE_A)]` (size 1) and the
  first frame is produced before the other client's slow row is even queued;
* chunk 0 submitted in the same engine iteration as the slow row (both requests are
  `add_request`ed without yielding, so the scheduler sees them together) ⇒ the batch
  is `[(slow-row, Z…), (chunk-0, A…)]` and the first frame is produced **after** the
  slow row's audio, i.e. TTFA is bounded by the slowest co-batched row.

This pins the mechanism, not a number. The real-GPU TTFA histogram (and the
last-row `_pending_vocode` drain mitigation decision) is deferred to the
orchestrator's 14-5 benchmark pass — 14-3 intentionally made **no** runner/model
change and did not touch the GPU. The 12-9 ceiling quoted in delta item 4 remains
the only figure the docs may quote until 14-5 measures.

### Deviations / decisions recorded

1. **Priming instead of true lazy headers.** Starlette's `StreamingResponse` sends
   `http.response.start` before iterating, so "headers with the first frame" is
   implemented by awaiting the first encoded frame inside `_open_audio_stream`
   (one chunk of audio held briefly) and handing the response a generator that
   re-yields it. Same observable contract; no header is ever committed for a stream
   with no audio.
2. **A 0/1-chunk stream bypasses the stitcher** (no edge conditioning, no 40 ms
   hold-back) so it matches the buffered single-shot samples exactly, mirroring
   `crossfade_stitch`'s documented single-chunk rule. The hold-back note therefore
   applies to N >= 2 chunks.
3. **Single-chunk streaming still uses a chunk sub-request** (`-chunk-0`,
   `suppress_watermark=True`) so the watermark is applied exactly once by the
   serving layer, per emitted segment, rather than twice (model + serving).
4. **The resampler's `flush()` tail is not re-watermarked** — it is the filter's
   continuation of already-marked samples, not new audio.
5. **`StreamingAudioResponse.state` is exposed** so 14-4's `done` payload (and any
   future stream metrics) can read `duration_s`; binary mode has no trailer.
6. **Chunks shorter than the carry window**: the first frame can then span more than
   one chunk (the stitcher returns empty for the first feed and the encoder skips
   empty frames). Not a bug — it is the hold-back — but it is why the gates use
   sentences longer than two 40 ms windows at 8 kHz.
7. **No runner change** for the deferred-vocode TTFA: the mitigation stays a 14-5
   decision with real numbers (delta item 4).

### Reproduce

```powershell
# the whole gate set
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header

# just the 14-3 gates
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/integration/test_tts_streaming.py tests/unit/wire_contract/test_streaming_serving.py -q --no-header
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/wire_contract/test_streaming_protocol.py tests/integration/test_tts_streaming_admission.py tests/unit/wire_contract/test_multipart.py -q --no-header
```

**Blocker:** none.


## Verification log (2026-11-10, host) — TTFA measured; no runner change

`out/bench/measure-14-stream.py` against the shipped `config.gpu.yml` (INFO
boot, GPU 1, 807-char fixture = 4 chunks, 5 runs, p50):

| Family | TTFA alone | TTFA +1 co-batched | TTFA +3 co-batched | Stream wall | Buffered wall |
|---|---|---|---|---|---|
| chatterbox | **3.62 s** | 13.87 s | 33.61 s | 16.19 s | 9.74 s |
| melo | **0.22 s** | 0.82 s | 2.44 s | 1.07 s | 0.91 s |

* The "first audio after the first sentence" promise holds when chunk 0 runs
  alone (3.6 s for one AR chunk ≈ a quarter of the 12-9 whole-paragraph p50,
  as predicted; 0.22 s for the padded family).
* The co-batching cost is real and large: behind one/tres other clients' rows,
  chunk 0's first frame waits for the slowest row. **Mitigation decision: no
  runner change in Phase 14** — the last-row `_pending_vocode` drain only
  helps when the stream's own row is the last unfinished one; it cannot skip
  waiting on *other* rows' T3 decode, which is what the +1/+3 rows measure.
  Priority admission or pipelined submission is the honest future fix
  (14-5 optional follow-up, unchanged).
* Sequential streaming forgoes the burst's co-batched decode: whole-paragraph
  wall 16.2 s streamed vs 9.7 s buffered — the expected, documented cost of
  MVP sequential submission, not a regression.
