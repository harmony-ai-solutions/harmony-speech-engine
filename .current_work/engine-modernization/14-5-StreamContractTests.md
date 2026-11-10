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

- [x] Wire-contract suites green (both envelopes + admission matrix)
- [x] Live disconnect test green (abort + resource release asserted via the abort seam; 9-4/9-2 stats not reachable with the stub executor)
- [x] Metrics + regression axis landed; docs/CHANGELOG final; `summary.md` ticked
- [ ] **DEFERRED TO HOST** — real-GPU TTFA histogram + throughput verification (orchestrator's pass; the metrics are live on `/metrics`)

## Verification log (2026-11-10)

### What changed

**Observability (new `harmonyspeech/endpoints/openai/streaming_metrics.py`, wired in `serving_text_to_speech.py` + `api_server.py`)**

- `hse_tts_stream_first_frame_seconds` — histogram, one observation per
  streaming response at the **first audio frame** (binary envelope: the first
  non-empty payload, header included; SSE: the first `speech.audio.delta`).
  Keepalive comments are not audio and never observe. The observation is taken
  **pre-header-priming**, immediately before the first audio frame is yielded
  to Starlette (headers commit only after the primed frame returns). Both
  envelopes therefore share one clock and the number excludes keepalive time;
  the client-visible difference is only the header write.
- `hse_tts_stream_chunks_total{outcome="completed"|"aborted"|"error"}` — one
  increment per streaming **response**, not per audio frame (decision below).
- `hse_tts_stream_bytes_total` — audio bytes handed to the wire (streaming WAV
  header included; SSE counts raw s16le, never the larger base64 text).
- All three series are rendered on every `/metrics` scrape in both the
  engine-present and engine-absent branches; idle values are zeros (the 7-3 /
  9-2 / 10-1 house rule).
- `StreamState` gained three **type-additive** fields: `started_at`
  (`default_factory=time.monotonic`, the histogram clock), `aborted` (set in
  the `CancelledError` handler, the `finally` in-flight abort, and the
  polled-disconnect branch) and `first_frame_observed`. No envelope bytes,
  ordering or terminal-event semantics changed; the 14-3/14-4 pins stay green.

**Wire-contract e2e (`tests/e2e/wire_contract/test_streaming_wire_contract.py`, 16 tests)** — boots the real FastAPI app under **uvicorn on a loopback socket** with the boundary-decoding `FakeEngine` (hermetic, CPU-only, no weights). Pins, through a real HTTP client:

- binary `wav`: `Transfer-Encoding: chunked`, no `Content-Length`, the full
  `X-Audio-*`/`X-Request-Id`/`X-Accel-Buffering`/`Cache-Control` set, the 44-byte
  RIFF header field by field (`0xFFFFFFFF` in both size fields, target rate,
  byte rate, block align), and a single-chunk payload byte-equal to the
  buffered path's canonical `pcm16_frame` output;
- binary `pcm`: headerless body at the requested target rate, equal to the
  buffered resampled path (≤1 LSB, the documented container difference);
- WAV header under resampling: rate/byte-rate fields carry the target rate;
- SSE: `text/event-stream; charset=utf-8`, deltas → exactly one
  `speech.audio.done` → `data: [DONE]` last line, delta shape
  `{type, audio, chunk_index}` with base64 s16le `audio`, `done` metadata, a
  keepalive-comment-tolerant parser, and a **real** keepalive emitted before the
  first delta (injected threshold + slow fake engine);
- admission over HTTP: `mp3`/`flac` + stream 400, alias ≡ `stream_format:"audio"`,
  `trim_silence`/`fix_internal_silence`/non-native `speed` 400 naming the
  whole-buffer restriction, `remove_dc_offset` and `sample_rate` 200;
- `/metrics`: the three series are present with zeros and move after one
  completed stream.

**Real-server disconnect (`tests/integration/test_tts_streaming_disconnect_live.py`, 1 test)** — the 12-2 `EngineHarness` (real `AsyncHarmonySpeech` loop, stub Chatterbox, deterministic per-chunk delay) served by uvicorn **on the test's event loop**; an `httpx.AsyncClient` closes the TCP connection while chunk 1 is in flight. Asserts `engine.abort` fires for the in-flight `-chunk-1`, chunk 2 is never submitted (request-log discipline; chunk 1 may finish synthesis before the abort is processed, but its audio is never delivered), and the metrics classify the stream as `aborted`.

**Audio regression (`scripts/audio_regression.py`, + `tests/unit/benchmarks/test_audio_regression_streaming.py` (7), + `tests/integration/test_audio_regression_streaming_axis.py` (1))** — new `streaming:off-vs-on` axis. Side A is the real buffered `_chunked` path; side B drives the real `_stream_audio_segments` **in-process** (the harness has no HTTP server and does not need one — documented choice). Both sides use `LONGFORM_CHUNKED_TEXT`, `LONGFORM_AXIS_CHUNK_SIZE=120`, `LONGFORM_AXIS_SEED=1234`, `crossfade_ms=40`, `sentence_pause_ms=120`, `max_batch_size=1` and **watermark off** (it diverges by design: per-segment vs whole-clip). The worker writes both sides through `sf.write(PCM_16)`, so the parent's `compute_metrics` compares the same quantized container. Verdict: duration is a hard invariant; deterministic families must be bit-identical, stochastic families get the documented cross-process CUDA noise floor (`SHARING_MAX_ABS_DIFF`/`SHARING_LOG_MEL_MSE`), and a model that samples every run (`MODEL_SAMPLES_EVERY_RUN`) gets the stochastic distributional bounds. The hermetic integration test proves the synthetic stub's streamed pre-watermark PCM is sample-identical to the buffered path.

**Docs** — `docs/api.md` streaming section finalized with a `curl --no-buffer | ffplay -` binary example (WAV and PCM), a browser `fetch` + `ReadableStream` + `AudioWorklet` SSE/PCM snippet, and reverse-proxy notes (`X-Accel-Buffering: no`, `proxy_read_timeout` sized for the first chunk, no gzip/`no-transform`). `README.md` feature line added; `CHANGELOG.md` carries one user-visible streaming block for the whole phase (both envelopes, alias, admission rules, per-segment watermark divergence, headers).

### Blast radius (grep, 2026-11-10)

- `StreamState` — defined `serving_text_to_speech.py:303`; constructed only in
  the serving layer (2 streaming openers), the regression script and tests.
  No production caller outside the streaming module.
- `_encode_stream_frames` / `_encode_sse_frames` — called only by
  `_open_audio_stream` / `_open_sse_stream`; the unit suites call them
  directly. The only new behaviour is metrics bookkeeping and the
  `StreamState.aborted` read; no bytes changed.
- `_stream_audio_segments` — called only by the two openers and the
  regression/integration seams; the polled-disconnect branch now also sets
  `state.aborted` (additive).
- `streaming_metrics` functions — imported by the serving layer, the
  `/metrics` route and tests; no engine or task-handler imports.
- `audio_regression.py` CLI — `parse_axis`, `side_settings`, `_cases_for_axis`
  gained an additive `streaming` branch; `run_worker` gained an `axis_kind`
  branch and `_run_streaming_worker`; existing axes are untouched (the 59
  pre-existing benchmark unit tests plus the 162-test benchmark directory stay
  green).
- `api_server.metrics` — one extra renderer call in each branch; no existing
  series or ordering changed.

### Test counts

Command (the task's gate):

```powershell
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header
```

- Baseline after 14-4: **1905 passed / 5 skipped**.
- After 14-5: **1943 passed / 5 skipped** (+38: 16 e2e wire-contract, 1 live
  disconnect, 13 streaming-metrics unit, 7 audio-regression-axis unit, 1
  audio-regression-axis integration).
- `tests/unit/benchmarks` explicitly (the script is touched): **162 passed**.

The 5 skips are the pre-existing model-gated/placement skips; none is new.

### Decisions / deviations

1. **First-frame observation point** — pre-header-priming, at the first audio
   frame (keepalives excluded). Chosen so both envelopes share one clock and
   the histogram measures synthesis TTFA, not header-write time; documented in
   `streaming_metrics.py` and here.
2. **`hse_tts_stream_chunks_total` counts streams, not audio frames.** The
   plan's name is kept, but `completed`/`aborted`/`error` are terminal
   stream states, and a per-frame counter could not represent the
   polled-disconnect case 14-4 flagged. One sample per response matches the
   one-per-request `first_frame` histogram.
3. **`StreamState.aborted`** — added (type-additive, no envelope change) so a
   polled disconnect (frame source ends normally) is classified `aborted`
   instead of `completed`. It is set in the `CancelledError` handler, the
   `finally` in-flight abort, and the polled-disconnect branch. The 14-4
   envelope pins are unchanged.
4. **Disconnect resource release** — the stub executor has no AR row or 9-2
   cache slot, so the 9-4/9-2 stats surfaces are not reachable hermetically;
   the test asserts the abort-call seam (`engine.abort("-chunk-1")`), the
   request-log discipline (chunk 2 never submitted) and the `aborted` metric.
   A real-model host pass can additionally read `hse_ar_decode_*` /
   `hse_ar_cache_*`.
5. **Audio-regression "streamed" without a server** — side B drives
   `_stream_audio_segments` directly (the integration-test seam), documented in
   the script. This is the explicit fallback the task allows.
6. **WAV-container quantization** — the axis writes both sides through
   `sf.write(PCM_16)`; this avoids a false failure from the known 1-LSB
   `pcm16_frame` (×32767) vs libsndfile (×32768) difference. Deterministic
   families still require the quantized PCM to be identical.
7. **No new dependencies, no GPU/CUDA touched.** All tests are hermetic and
   CPU-only; the metrics are ready for the host's real histogram.
8. **TTFA batch-composition split is by host scenario, not a metric label.**
   The serving layer submits chunks sequentially and cannot observe whether
   chunk 0 was co-batched behind another request's row (that is a scheduler
   fact). The host pass runs the same histogram under "chunk 0 alone" and
   "chunk 0 co-batched" workloads and compares the two; 14-3's structural test
   already pins the mechanism. Adding a label would require an engine→serving
   batch-composition signal that does not exist.

### Reproduce

```powershell
# the 14-5 gates
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest `
  tests/unit/wire_contract/test_streaming_metrics.py `
  tests/unit/benchmarks/test_audio_regression_streaming.py `
  tests/integration/test_tts_streaming_disconnect_live.py `
  tests/integration/test_audio_regression_streaming_axis.py `
  tests/e2e/wire_contract/test_streaming_wire_contract.py -q --no-header

# the whole gate set (task command)
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header

# the regression script's own unit suite
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/benchmarks -q --no-header

# the host axis (real models, CPU or --gpu)
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" scripts/audio_regression.py `
  --families chatterbox,chatterbox_turbo,chatterbox_multilingual,openvoice_v1,openvoice_v2,melotts,harmonyspeech,kittentts `
  --axis streaming:off-vs-on
```

### Deferred to host (orchestrator's pass)

- **Real-GPU TTFA histogram** — `hse_tts_stream_first_frame_seconds` split by
  batch composition (chunk 0 alone vs co-batched behind a slower row), as
  14-3's deferred-vocode note requires. The metric is live; the runner is the
  orchestrator's.
- **Streaming throughput suite** — streamed vs buffered wall-clock/throughput
  at the benchmark host (the `streaming:off-vs-on` audio axis is correctness,
  not throughput).
- **Real-model streaming axis** — the script's `streaming:off-vs-on` at family
  breadth on the host (the synthetic stub gate is already green in-process).
- **Real-model AR row / cache-slot release on disconnect** — read
  `hse_ar_decode_*` / `hse_ar_cache_*` during a host disconnect run (the
  hermetic test asserts the abort seam instead).

## Recalibration of the `streaming:off-vs-on` axis (2026-11-10)

> **Status:** owner-approved recalibration landed and hermetic-green.
> **GPU re-run pending (orchestrator).** This entry does **not** claim the host
> gate is green; the 2026-11-10 table in `docs/benchmarks/throughput.md` is the
> *pre-recalibration* run.

### Owner decision

The streamed-vs-buffered axis was run at family breadth on the GPU host
(`out/bench/gate-14-verify-streaming-axis.json`): **6 of 8 families pass,
exactly the two OpenVoice clone pipelines fail.** Decision (binding):

1. The clone-pipeline families (`openvoice_v1`, `openvoice_v2`) are compared
   with the **distributional** thresholds — the same class the stochastic cases
   already use (duration/RMS/LTAS, plus the spectral-correlation floor) — not
   the sample-exact duration/max-|Δ|/log-mel-≤0.01 strictness.
2. A streamed **run1-vs-run2 determinism sub-check** is added: two streamed
   renders of the same request must be identical given the same seed. It
   mirrors the existing `determinism:run1-vs-run2` equality class and runs for
   every family on the axis (cheap: side B's manifest is run 1, so only one
   extra worker process per family).
3. The axis documents *why*: the case labels and the script constants record
   the B=1-vs-B=4 tone-converter numerics and the 2026-11-10 measurements.
4. **Nothing is weakened for the other six families** — they keep the strict
   gate that now passes.

### Measured evidence (RTX 5060 Ti, 2026-11-10)

| Family | Class | dur Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | LTAS corr | waveform corr | max \|Δsample\| | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| chatterbox | strict (stochastic kind) | 0.0 | 0.0 | 1e-06 | 0.0 | 1.0 | 1.0 | 3.052e-05 | PASS |
| chatterbox_turbo | strict (stochastic kind) | 0.0 | 0.0 | 2e-06 | 0.0 | 1.0 | 1.0 | 3.052e-05 | PASS |
| chatterbox_multilingual | strict (stochastic kind) | 0.0 | 0.0 | 1e-06 | 0.0 | 1.0 | 1.0 | 3.052e-05 | PASS |
| melotts | strict (stochastic kind) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.637e-05 | PASS |
| harmonyspeech | strict (deterministic) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 (bit-identical) | PASS |
| kittentts | strict (samples every run) | 0.0 | 0.083 | 3.145006 | 0.1369 | 1.0 | 0.9922 | 0.23858643 | PASS |
| **openvoice_v1** | **distributional** | **0.054** | **2.081** | **43.81633** | **0.5927** | **0.9995** | **0.0206** | **0.45618698** | **FAIL (strict)** |
| **openvoice_v2** | **distributional** | **0.026** | **0.831** | **20.941809** | **0.3676** | **0.9999** | **0.1045** | **0.66954184** | **FAIL (strict)** |

The failure is inaudible: length moves ±0.03-0.05 %, the waveform is
decorrelated (corr 0.02-0.10) but the long-term average spectrum is identical
(LTAS corr ≥ 0.9995, RMS ≤ 2.1 %).

### Attribution (by elimination, plus recorded engine facts)

* Sequential streaming submits chunk sub-requests **one at a time**, so each
  chunk's tone-conversion is a **B=1** forward. The buffered path submits the
  whole 4-chunk burst at once and the engine **co-batches** it into one padded
  tone-converter forward (Phase 8-3). The padded-batch TC equivalence has always
  been **distributional** (log-mel ≤ 1.48e-4), never sample-exact.
* The TC edge-condition trim decision flips under that numeric noise: the
  stitched length moves by a few samples and the waveform shifts phase, so a
  sample-wise comparison is meaningless by construction.
* MeloTTS (the same synthesizer, standalone) passes sample-exact → the
  synthesizer is exonerated. The AR families pass → seeded decode is
  exonerated. HarmonySpeech is bit-identical → the frame generator + stitcher
  are exonerated. Only the TC batch shape differs.
* The mirrored determinism class is known-satisfiable: the buffered
  `determinism:run1-vs-run2` run of the same day
  (`out/bench/gate-14-verify-determinism.json`) measured openvoice_v1
  **bit-identical** and openvoice_v2 **max |Δ| 4.458e-05** across processes. A
  real streaming defect (wrong chunk order, duplicated/dropped chunk, unseeded
  noise) moves samples by O(1) — orders of magnitude above that floor.

### What changed in the axis (`scripts/audio_regression.py`)

1. **`STREAMING_AXIS_DISTRIBUTIONAL = {openvoice_v1, openvoice_v2}`** (and the
   derived `STREAMING_AXIS_STRICT` for the six). `verdict_for_axis` scores
   those two families with `THRESHOLDS["stochastic"]` — duration ≤ 12 %,
   RMS ≤ 15 %, LTAS RMSE ≤ 4 dB — plus a spectral-correlation floor
   `STREAMING_DISTRIBUTIONAL_LTAS_CORR_MIN = 0.99` (the *waveform* corr and
   max |Δsample| are reported, never gated: they are decorrelated by
   construction). The `longform` and `determinism` axes are untouched, and the
   six strict streaming families keep the exact gate they shipped with.
2. **Streamed determinism sub-check** (`streaming_determinism_entry`, new
   `STREAMING_DETERMINISM_FAMILIES` = the whole axis). `run_harness` renders
   side B a second time (`run_tag="on_run2"`, identical settings, own work
   dir/manifest so run 1 cannot be overwritten) and compares run1 vs run2 with
   the `determinism:run1-vs-run2` equality class: duration is a hard invariant;
   deterministic families must be bit-identical; seeded stochastic families get
   the documented cross-process CUDA floor (`SHARING_MAX_ABS_DIFF` /
   `SHARING_LOG_MEL_MSE`); a model that samples every run (`kittentts`) is
   scored distributionally. The result is recorded per case
   (`streaming_determinism`), folded into the case verdict, and shown in a new
   **stream det** column; the run-2 manifest is recorded as `side_b_run2`.
3. **Self-documenting labels/constants.** The clone cases' label now reads
   `...; distributional: B=1 vs B=4 tone-converter numerics (Phase 8-3
   log-mel <= 1.48e-4), run1-vs-run2 determinism gated`, and the constants carry
   the rationale + the 2026-11-10 numbers.
4. **Tests** (`tests/unit/benchmarks/test_audio_regression_streaming.py`, +15,
   hermetic/CPU): the class split; the exact measured OV metrics pass for the
   clone pair and still fail for every strict non-sampling family; a
   decorrelated-waveform/identical-LTAS audio pair passes only for OV; the
   ±0.05 % trim shift is tolerated only for OV; a duration/LTAS/LTAS-corr/RMS
   regression still fails; labels; the determinism entry class (bit-identical,
   cross-process floor, O(1) breakage, KittenTTS distributional + hard
   duration); and the `run_harness` wiring (side B rendered twice, sub-check
   folded into the verdict and the report).

### Test counts

```powershell
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header
```

- Before: **1943 passed / 5 skipped**.
- After: **1958 passed / 5 skipped** (+15, the streaming-axis unit suite;
  `tests/unit/benchmarks` 178 passed including the axis integration test).
- The 5 skips are the pre-existing model-gated/placement skips; none is new.

### Cost / deviations

* The sub-check adds **one extra engine boot per family** to the host gate
  (side B is rendered twice). That is the price of mirroring the existing
  cross-process determinism topology; it was chosen over an in-process second
  render because it reuses the proven `determinism:run1-vs-run2` equality class
  and its measured bounds.
* No production code, no engine change, no GPU/CUDA touched; everything above
  is harness + tests + this log.


## Verification log (2026-11-10, host pass) — gates, benchmarks, TTFA

Full record: `docs/benchmarks/throughput.md` "Phase 14 pre-commit verification
vs commit 8b7d222" (per-run blocks appended verbatim; JSON under
`out/bench/phase14-verify-*.json` / `out/bench/gate-14-verify-*.json`).

**Gates (GPU 1, real weights).** `longform:off-vs-on` 8/8 PASS;
`determinism:run1-vs-run2` PASS; `batching:1-vs-4` PASS;
`dtype:fp32-vs-autocast` fp16 PASS (7/7, matches record); bf16 FAIL 5/7 —
reproduces the recorded 2026-11-08 verdict verbatim (corr 0.8475 / mse 2.9995
vs recorded 0.8475 / 3.000; the documented reason fp16 ships). AR parity e2e
**2 passed**; whisper window parity e2e **2 passed** (requires the cuDNN shim
— without it CT2 hard-aborts the process, 0xC0000409). Recalibrated
`streaming:off-vs-on` (this file's previous entry): **overall PASS** — six
families sample-exact (harmonyspeech bit-identical), OV clone cases
distributional + streamed-determinism sub-check PASS per case.

**Benchmarks vs the 8b7d222 record.** Every measured table matches within
noise or improves (long-form matrix 5/6 families, short-text post rows, Phase
8 clone/OV1/HS/VF/kitten curves, Phase 11-1 VC post rows 2.52/2.62/3.53/3.95
vs recorded 2.59/2.65/3.58/4.03 req/s, Phase 10 whisper windows on GPU
25.77/32.10/35.16 vs recorded 26.92/32.14/32.03, fidelity at the noise floor,
Phase 9-6 runner rows and cache composition better than their pre-12
baselines). One attribution finding recorded for the owner: the 12-9 OV2
long-form rows were measured on the 8-entry `config-phase12.yml` fixture;
the shipped 28-model catalogue + 2.75 GiB AR pool does not leave that
headroom on a 16 GiB card, so long-row TC padded batches re-trip the WDDM
paging cliff on the shipped config (the same tree reproduces the recorded
numbers on the fixture — no code regression; envelope follow-up in
throughput.md).

**TTFA + metrics (live server).** TTFA p50: chatterbox 3.62 s alone / 13.87 s
behind +1 / 33.61 s behind +3 co-batched rows; melo 0.22 s alone. Sequential
stream wall 16.19 s vs 9.74 s buffered for the 807-char paragraph (burst
co-batching forgone by design). SSE end-to-end verified (5 deltas, one
`speech.audio.done` — duration_s 39.48, usage input_characters 807 /
audio_seconds 39.48 — then `data: [DONE]`); `hse_tts_stream_*` series
observed moving (completed outcomes, bytes, first-frame histogram) with
zero-state present. Deferred-vocode mitigation decision: no runner change
(see 14-3's verification log — the last-row drain cannot skip other rows'
decode).
