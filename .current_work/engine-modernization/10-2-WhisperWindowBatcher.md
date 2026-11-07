# Phase 10-2: Whisper Window Batcher (cross-request 30-s windows)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 10 — Whisper serving
> **Depends on:** 10-1 (in-flight accounting, lane admission); benefits from 10-3's memory bounds
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; design input: study #14 D7/D10 (`../../soulbits-cloud-backend/.current_work/inference-engine-re-eval-26-11/14-vllm-whisper-kvcache-and-hse-ar-batching.md`)
> **State reference:** verified 2026-11-07 — the handler's Phase 8-5 docstring (`model_runner_base.py:1234-1256`) names the exact skip being reversed here: cross-request batching "would mean re-implementing `transcribe()`… a drifting copy". This task consciously accepts a **thin** re-implementation on the public CT2 surface, with the fallback keeping the stable path.

## Objective

Form cross-request window batches: collect the ≤30 s clips of **all** admitted whisper requests and run them through **one** CT2 `encode(batch)` + **one** `generate(encoder_output, prompts, beam_size=…)` call per step (the study's D7). The KV cache and beam search stay inside CT2 — we own batch *formation*, never kernels. Expected: deep-queue throughput rises sharply (today a <30 s request decodes `batch=1` while up to 16 sit admitted).

## Context (self-contained)

- **What the public API gives (verified by study #14 against installed ctranslate2 4.4.0 / faster-whisper 1.2.1 — re-verify before building):** `WhisperModel.model.encode(features_batch)` → stacked encoder outputs; `generate(encoder_output, prompts=[…], beam_size=…)` handles decode + native beam search for the whole batch. faster-whisper's `feature_extractor`, tokenizer, and `generate_segments`-style result handling are the surrounding pieces to reuse rather than reinvent.
- **Language detection:** `detect_language` re-runs an encoder pass today; reuse the batched `encode()` output instead (one encoder pass per window, not two — record the saving).
- **Per-request semantics preserved:** per-request language → per-request prompt/tokenizer (CT2 takes a `prompts` list — rows may differ); timestamp restoration + segment stitching per window with its time offset; `condition_on_previous_text` and initial-prompt semantics must match the `pipeline.transcribe` path exactly — transcribe upstream's `_batched_segments_generator` behavior into a parity table first.
- **Do NOT port (study D10):** vLLM's client-side beam search (re-prefill per beam per token). CT2's native beams are strictly better here.
- **Fallbacks:** `len == 1` request → today's `pipeline.transcribe` verbatim; `batched_inference: false` → always; window-batcher failure (any exception) → per-request `pipeline.transcribe` for that step's requests (log + metric, never a user-visible error).
- **Bounded memory:** windows per batch is capped (initial 8, owner-approved 2026-11-07); with 10-1, concurrent CT2 calls = M workers × ≤1 formed batch each; 10-3's probe validates the peak.

## Implementation steps

1. Verify-first: introspect the installed CT2/faster-whisper surface (encode/generate signatures, feature extractor reuse, segment-result decoding); record the API table in this file.

### Step 1 result — API verification table (2026-11-07, installed env `hse`)

Introspected directly from `ctranslate2==4.4.0` and `faster-whisper==1.2.1` in
`C:\Users\sge20\miniconda3\envs\hse`. **The batched encode/generate capability the
design assumes exists in the installed versions.** No hard blocker.

| Surface | Verified signature / behaviour | Use in 10-2 |
|---|---|---|
| `ctranslate2.models.Whisper.encode` | `encode(features: StorageView, to_cpu: bool = False) -> StorageView`; `features` is `[batch_size, n_mels, chunk_length]`; output `[batch_size, chunk_length // 2, d_model]` | one batched encoder pass per formed window batch |
| `ctranslate2.models.Whisper.generate` | `generate(features, prompts: list[list[str]] \| list[list[int]], *, asynchronous=False, beam_size=5, patience=1, num_hypotheses=1, length_penalty=1, repetition_penalty=1, no_repeat_ngram_size=0, max_length=448, return_scores=False, return_logits_vocab=False, return_no_speech_prob=False, max_initial_timestamp_index=50, suppress_blank=True, suppress_tokens=[-1], sampling_topk=1, sampling_temperature=1) -> list[WhisperGenerationResult]`. The docstring states it **also accepts the encoded features returned by `encode()`** and takes a **batch of prompts** (rows may differ). Native beam search stays inside CT2. | one batched decode per formed window batch, per-row prompts |
| `WhisperGenerationResult` | fields `sequences`, `sequences_ids`, `scores`, `no_speech_prob` | per-window tokens + avg logprob + no-speech probability |
| `ctranslate2.models.Whisper.detect_language` | `detect_language(features) -> list[list[(token, probability)]]`; also **accepts the encoded features returned by `encode()`** | language detection reuses the batched encoder output (no second encoder pass) |
| `WhisperModel.encode` | `encode(features: np.ndarray) -> StorageView`; adds a batch dim for 2-D input; `to_cpu` only when multi-GPU | numpy `[B, n_mels, 3000]` → encoder output |
| `WhisperModel.get_prompt` | `get_prompt(tokenizer, previous_tokens, without_timestamps=False, prefix=None, hotwords=None) -> list[int]` | per-request prompt; `BatchedInferencePipeline` forces `condition_on_previous_text=False`, so no cross-window prompt state |
| `WhisperModel._split_segments_by_timestamps` | `(tokenizer, tokens, time_offset, segment_size, segment_duration, seek) -> (segments, seek, single_timestamp_ending)` | upstream timestamp restoration per window (reused verbatim) |
| `faster_whisper.transcribe.restore_speech_timestamps` | `(segments, speech_chunks, sampling_rate) -> Iterable[Segment]` | maps concatenated-speech time back to original audio time |
| `faster_whisper.transcribe.collect_chunks` | `(audio, chunks, sampling_rate=16000, max_duration=inf) -> (audio_chunks, chunks_metadata)`; merges VAD clips into ≤30 s windows; metadata carries `offset` (speech-only timeline), `duration`, `segments` | window collector reuses it, so window formation matches the pipeline |
| `faster_whisper.transcribe.pad_or_trim` | `pad_or_trim(array, length=3000, axis=-1)` | every window feature row is padded/trimmed to 3000 frames, exactly like `forward()` |
| `faster_whisper.transcribe.get_compression_ratio` | `get_compression_ratio(text) -> float` | per-subsegment compression ratio |
| `faster_whisper.transcribe.get_suppressed_tokens` | `(tokenizer, [-1]) -> tuple[int, ...]` | generation suppression set (language-independent) |
| `faster_whisper.vad.get_speech_timestamps` | `(audio, vad_options=None, sampling_rate=16000, **kwargs) -> list[dict]`; sample offsets by default | same VAD call as `BatchedInferencePipeline.transcribe` (`max_speech_duration_s=chunk_length`, `min_silence_duration_ms=160`) |
| `faster_whisper.feature_extractor.FeatureExtractor` | `__call__(waveform) -> np.ndarray [n_mels, frames]`; attributes `sampling_rate`, `chunk_length`, `nb_max_frames`, `time_per_frame` | per-window mel features (`[..., :-1]` trim, same as upstream) |
| `faster_whisper.tokenizer.Tokenizer` | `Tokenizer(hf_tokenizer, multilingual, task, language)`; `sot_sequence`, `timestamp_begin`, `decode` | one tokenizer per request (per detected language), shared for its windows |

**Upstream parity facts transcribed from `BatchedInferencePipeline.transcribe` / `forward` / `generate_segment_batched` (1.2.1):**

| Behaviour | Pipeline value | Consequence for the batcher |
|---|---|---|
| `condition_on_previous_text` | forced **False** | windows are independent; cross-request batching is semantically valid |
| `initial_prompt` | applied to **every** window (no history) | handler never passes one; prompt is constant per request |
| `temperature` | only `temperature[0]` (`0.0`); **no fallback chain** | batched generation pins `sampling_temperature=0.0` |
| `max_initial_timestamp_index` | **not passed** → CT2 default `50` | batcher must not pass it either (the non-batched `generate_with_fallback` uses `0`) |
| language detection input | first ≤30 s of **concatenated VAD-chunk features** (padded/trimmed to 3000) | batcher reuses the first window's encoder row; divergence only when the first VAD window is <30 s and a second window exists (documented parity edge) |
| `multilingual` | `False` by default (no per-window re-detection) | one tokenizer/language per request |
| `word_timestamps` | `False` (handler passes no override) | `Segment.words = None` |
| `vad_filter` | `True` by default; `VadOptions(max_speech_duration_s=30, min_silence_duration_ms=160)` | collector uses the same VAD options |
| `info` shape | `asdict(TranscriptionInfo)` with `TranscriptionOptions` + `VadOptions` | batcher rebuilds the same dataclasses, so the payload is byte-shape identical |
| segment ids | per request, starting at `1` | batcher enumerates stitched segments per request |
| segment `seek` | `int(chunk_metadata["offset"] * frames_per_second)` | carried per window |
| segment start/end | `round(..., 3)` after `restore_speech_timestamps` | batcher rounds identically |
2. Window collector: per request → VAD/clip timestamps (as today's pipeline does) → features per window; rows tagged `(request, window_index, offset)`.
3. Batched step: one `encode` + one `generate` per formed batch → per-window results → per-request segment stitching → the same `{"text", "segments", "info"}` payload the handler emits today (`:1281`) so downstream (VadArtifact, wire) is untouched.
4. Parity gate: batched path B=1 == `pipeline.transcribe` output for the same audio (pin temperature fallback; document any stochastic edge like temperature-fallback divergence honestly).
5. Pairing/invariant tests (5-1 discipline): one result per request id; a poisoned window errors alone (degrades its request, not the batch).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/whisper_batching.py` (new) | window collector + batched step + stitching |
| `harmonyspeech/task_handler/model_runner_base.py` | `_execute_faster_whisper` gains the batched path + fallbacks |
| `tests/unit/inference_flow/test_whisper_window_batching.py`, integration (new/extend) | gates below |

## Tests / validation

- Unit (stubbed CT2): window formation from clip timestamps; per-request prompts; offset stitching; poisoned-window isolation; fallback on failure.
- Integration: 4 mixed-length requests (2 short, 1 >60 s, 1 VAD-step request via the same handler) — correct pairing, correct segment offsets.
- GPU parity gate + throughput: benchmark host (10-5 records).

## Checklist

- [x] CT2/faster-whisper API table recorded here (verified signatures)
- [x] Window collector + one-encode/one-generate step + stitching landed
- [ ] Parity gate B=1 vs `pipeline.transcribe` green — **harness ready**
      (`tests/e2e/stt/test_whisper_window_parity.py`, skipped unless
      `HSE_WHISPER_PARITY_MODEL` points at a local model); the benchmark-host
      run and the `_tbd_` cells in `docs/benchmarks/throughput.md` are the
      Phase 10-5 gate
- [x] Fallbacks wired (`len==1`, flag, failure)
- [x] `summary.md` Implementation Status updated

## Implementation notes / deviations (2026-11-07)

1. **API verified, no blocker.** CT2 4.4.0's `generate` accepts the encoder
   output from `encode()` and a per-row prompt list, and `detect_language`
   accepts the same encoder output. The design's one-encode/one-generate shape
   is exactly the installed surface (table above).
2. **Cross-request batching is opt-in for Whisper.** The phase spec says the
   batched path applies "when >1 request and `batched_inference: true`" and
   "flag off → always sequential". `None` (unset) therefore keeps the
   pre-10-2 per-request path; only an explicit `true` enables the window
   batcher. This also keeps every earlier whisper test/config honest until the
   operator opts in. The shipped configs intentionally do not set the flag
   (10-5 decides after measurement).
3. **Window batch cap is a module constant, not a config key.** The spec's
   files table did not include `config.py`; the owner-approved initial `8`
   lives in `whisper_batching.WHISPER_WINDOW_BATCH_SIZE`. If 10-5 wants it
   operator-tunable, adding the key is a small follow-up (the batcher already
   accepts `window_batch_size`).
4. **`WhisperRequestAbortedError` moved to `whisper_batching.py`.** Both the
   Phase 10-1 lane pool and the Phase 10-2 batcher signal it, and
   `model_runner_base` re-exports the name, so existing imports/tests are
   unchanged.
5. **Language detection parity edge.** The batcher reuses the first window's
   encoder output (one encoder pass). Upstream `detect_language` runs on the
   concatenation of the first VAD windows (padded/trimmed). For a single-window
   request (≤30 s of speech) they are identical; for a multi-window request
   whose first window is shorter than 30 s the inputs differ. Documented in
   `docs/batching.md` and reported (not asserted) by the e2e parity harness.
6. **Batch-level vs request-level failure.** A request whose collection or
   decode fails errors alone. A batch-level CT2 failure propagates to the
   handler, which falls back to per-request `pipeline.transcribe` for the whole
   step (the spec's fallback), where the poisoned request errors alone. This is
   the closest honest reading of "poisoned window/request errors alone, never
   de-batches neighbours" on an API whose `generate` is one atomic batch call.
7. **Metrics addition.** `hse_whisper_batch_fallbacks_total` (per model)
   records fallbacks; it is threaded through `whisper_lane_stats()` and the
   pure renderer, so the existing series contract is unchanged.
8. **Admission liveness floor.** The governor's window allowance is consumed in
   true window units. When nothing is in flight and a request needs more
   windows than the allowance, the step admits that request anyway (Phase 10-1
   liveness preserved); `allowed_windows == 0` remains a hard stop.

**Verification (local, no weights):** `tests/unit` 919 passed (26 new in
`tests/unit/inference_flow/test_whisper_window_batching.py`), `tests/integration`
52 passed (3 new in `tests/integration/test_whisper_window_batching.py`).
Targeted Phase 10/8 suites (183 cases) green. Ruff clean on every changed file
(the pre-existing engine/scheduler lint findings are untouched).
