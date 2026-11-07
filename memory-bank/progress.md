# Progress: Harmony Speech Engine

## Implementation Status Overview

**Current Version**: 0.1.0-rc9  
**Overall Completion**: ~80%  
**Release Target**: Stable v0.1.0 within 1-2 months  

## Engine Modernization — Phase 10 (COMPLETE, 2026-11-07; benchmark-host measurements pending)

**Plan folder**: `.current_work/engine-modernization/`; task: `10-5-WhisperGatesBenchmarksExit.md`

- ✅ **Exhaustion scenarios** — `scripts/bench_throughput.py` gained `--scenario {whisper-flood,whisper-fairness,whisper-budget-trip}`: a 100-request heterogeneous flood (durations 2 s–8 min, mixed languages, transcription + translation, `json`/`text`/`srt`/`vtt`) against tight caps (M=2, window batch=4, queue 16), per-request wall-time traces, `/metrics` lane-depth sampling, timeout/loss/pairing accounting; a long-first fairness proof; a memory-budget trip run; and the lanes-vs-windows / M-sweep / window-sweep cells (`--print-whisper-scenarios`). The default `throughput` path is unchanged.
- ✅ **Local degradation proof** — `tests/integration/test_whisper_exhaustion.py` drives the real engine loop, scheduler, lane admission and window batcher with stubbed CT2/VAD/features: the 100-request flood completes with **zero loss and zero pairing errors** (each transcript is checked against its own window markers), a timeout storm resolves every request as a typed error, a waiting+in-flight abort storm loses no unrelated request, a batch-level batcher failure falls back and counts once, and a mocked-`mem_get_info` budget trip shrinks admission 6→2→1→0 without a crash.
- ✅ **Five-class execution-model table** — `tests/unit/initialization/test_config.py` now maps every registered type to `padded_batch` / `loop` / `queue_depth` / `lanes_windows` / `ar_decode` with per-class semantic validators; `ar_decode` is asserted pending until 9-6. Whisper is `lanes_windows` with its real knobs. `docs/batching.md` carries the class table and the final "lanes M × windows B" Whisper row; the docs-matrix lint gained a Phase 10 assertion.
- ✅ **Dated config decision (2026-11-07)** — `batched_inference` stays **opt-in** until the benchmark-host B=1 real-model parity gate (the batcher re-implements window formation/stitching; fallbacks protect availability, not output correctness). `num_workers` (turbo 2 / tiny 4) and the window cap 8 ship unmeasured; width is behaviour-safe and tested.
- ✅ **Docs + knowledge** — `docs/benchmarks/throughput.md` §Phase 10 (lanes vs windows, fairness, M/window sweeps, 200-connection, exhaustion, budget trip, parity, reproduce commands; every cell `_tbd_`), `vram-profiles.md` budget-trip validation, `docs/models.md` capacity honesty, CONCERNS retired, CHANGELOG already carried the user-visible entries.

**Verification**: unit 974 / integration 59 / wire-contract e2e 50 passed + 3 model-gated skips (4 pre-existing TTS e2e weight/checkpoint errors unchanged). No weights downloaded; every measurement table is `_tbd_` for the benchmark-host session (same idle RTX 5060 Ti run as Phase 7/8).

**Deviations**: benchmark scenarios are a new `--scenario` axis (default unchanged); queue-wait is the engine histogram delta (true per-request pairing is proven locally); local tests include the 8-minute request but run in ~14 s; the class table lives before the model matrix so the docs lint still parses the matrix; no config values changed.

**Next**: benchmark-host run fills the Phase 10 `_tbd_` cells, including the B=1 parity gate that flips `batched_inference`.

## Engine Modernization — Phase 10-4 (CODE COMPLETE, 2026-11-07; real-model spot-check pending 10-5)

**Plan folder**: `.current_work/engine-modernization/`; task: `10-4-STTWireFormats.md`

- ✅ **`response_format`** — `SpeechTranscribeRequest` accepts `json` (default, byte-identical shape pinned by a test), `text`, `srt`, `vtt`; validated at admission (HTTP 400 before inference, Phase 1-2 pattern). `text`/`srt`/`vtt` return `text/plain`; JSON stays the unchanged `SpeechToTextResponse`.
- ✅ **Pure renderer** — `harmonyspeech/endpoints/openai/stt_formats.py`: SRT (`HH:MM:SS,mmm`, numbered blocks) and WebVTT (`WEBVTT`, `HH:MM:SS.mmm`), `Decimal`-based half-up millisecond rounding, ≥1 h (hours grow past 99), empty input → empty document, CRLF normalisation and `-->` → `- ->` in cue text.
- ✅ **`/v1/audio/translations`** — mirrors the transcription route/model; `create_translation` shares `_create_stt(..., task="translate")`. `SpeechTranscribeRequestInput` gained `task`/`language`/`prompt`; `prepare_faster_whisper_inputs` normalises/validates `task`; the lane pool passes `task`/`language`/`initial_prompt` to `pipeline.transcribe`, and 10-2's batcher builds its `Tokenizer` with the request task and encodes `prompt` as previous tokens. Verified faster-whisper 1.2.1 encodes `task` through `Tokenizer.sot_sequence` (`<|transcribe|>`/`<|translate|>`), not `TranscriptionOptions`.
- ✅ **Registry** — `translation` capability on the Faster-Whisper step/spec; new `capabilities_for_model_type()` gives individual cards spec-derived capabilities (`["stt","translation","vad"]`), so `/v1/models` advertises translation without hand lists.
- ✅ **Docs** — `docs/api.md` (response formats + translation endpoint), `docs/audio-io-contract.md` (endpoint/upload tables + STT format table), `docs/models.md`, `docs/routing.md`, CHANGELOG.

**Verification**: `tests/unit` 958 passed (33 new in `tests/unit/wire_contract/test_stt_formats.py`, +4 batcher, +1 lane, +1 cards), `tests/integration` 53 passed (+1 translations route), wire-contract e2e 50 passed / 3 model-only skips (+12 matrix/admission cases). No weights downloaded; real-model translation/subtitle spot-checks remain `_tbd_` for the benchmark host (10-5).

**Deviations**: `prepare_faster_whisper_inputs` keeps returning audio artifacts (options resolved on the request input) to preserve the 10-1/10-2 input contract; `language`/`prompt` are threaded too (source hint / initial prompt); individual cards now carry additive spec-derived capabilities.

**Next**: 10-5 benchmark-host gates (including real translation + subtitle spot-checks).

## Engine Modernization — Phase 10-2 (CODE COMPLETE, 2026-11-07; benchmark-host parity/throughput pending)

**Plan folder**: `.current_work/engine-modernization/`; task: `10-2-WhisperWindowBatcher.md`

- ✅ **API verified** — CT2 4.4.0 `Whisper.encode(features[B,n_mels,3000])` → stacked encoder output; `generate(encoder_output, prompts: list[list[int]], beam_size=…)` takes the encoded output + one prompt per row (native beam search inside CT2); `detect_language(encoder_output)` reuses the same encoder pass. No hard blocker; the API table is in the task file.
- ✅ **`harmonyspeech/task_handler/whisper_batching.py`** — collects every admitted request's ≤30 s windows through faster-whisper's own VAD/`collect_chunks`/`pad_or_trim`, tags rows `(request, window_index, offset)`, forms ≤8-window batches (owner-approved initial 8), issues one `encode` + one `generate` per batch with per-request prompts (detected language), and stitches with upstream `_split_segments_by_timestamps` + `restore_speech_timestamps`. Payload is the same `{"text","segments","info"}` shape, so VadArtifact/wire are untouched.
- ✅ **Handler wiring** — opt-in `batched_inference: true`; `len==1`, flag off, and any batch-level batcher failure use the per-request `pipeline.transcribe` path. Per-request collection/decode failures error alone; abort is cooperative at formed-batch boundaries.
- ✅ **Admission counts windows** — `whisper_window_count(request) = ceil(seconds/30)` feeds the 10-3 governor composition (`allowed_windows - running_windows`), with a liveness floor when a single long request would otherwise starve; `plan_lane_admission` gained an optional window budget.
- ✅ **Metrics** — `hse_whisper_batch_fallbacks_total{model}`.
- ✅ **Docs** — `docs/batching.md` whisper row + window-batching section; `docs/benchmarks/throughput.md` Phase 10-2 with `_tbd_` parity/throughput cells; CHANGELOG.

**Verification**: `tests/unit` 919 passed (26 new in `tests/unit/inference_flow/test_whisper_window_batching.py`), `tests/integration` 52 passed (3 new in `tests/integration/test_whisper_window_batching.py`). Real-model B=1 parity harness `tests/e2e/stt/test_whisper_window_parity.py` skips unless `HSE_WHISPER_PARITY_MODEL` is set; no weights downloaded. Benchmark-host parity/throughput + `batching:1-vs-N`-style gates outstanding (10-5).

**Deviations**: Whisper batching is opt-in (unset = previous path) per the phase wording; window cap is a module constant (no config key in the spec's files table); language detection reuses the first window's encoder output (documented edge on multi-window requests whose first window <30 s); `WhisperRequestAbortedError` moved to `whisper_batching` and re-exported.

**Next**: 10-5 fills the `_tbd_` parity/throughput cells and decides whether to set `batched_inference: true` in the shipped configs.

## Engine Modernization — Phase 10-1 (CODE COMPLETE, 2026-11-07; benchmark-host width sanity pending)

**Plan folder**: `.current_work/engine-modernization/`; task: `10-1-WhisperLanes.md`

- ✅ **`num_workers` plumbing** — per-`FasterWhisper` `ModelConfig` key (default `1` = unchanged behaviour) forwarded to `WhisperModel(..., num_workers=...)`, which maps it to CT2 `inter_threads` (verified: faster-whisper 1.2.1 `transcribe.py:628/695`). Shipped initial values, owner-approved 2026-11-07: turbo `2`, tiny `4`.
- ✅ **M-worker handler** — `_execute_faster_whisper` dispatches a step to the runner-owned `WhisperLanePool` (`ThreadPoolExecutor(max_workers=M)` + `BoundedSemaphore(M)`), one per-thread pipeline each, generators consumed inside the worker, `len == 1` direct path kept, segment-bound cooperative abort (`WhisperRequestAborted`, error isolated to that request).
- ✅ **Two-lane fairness admission** — Whisper models are capacity-based; `<=30 s` vs `>30 s` lanes drained at 3 short : 1 long with persistent per-model credits; width composes capacity × Phase 10-3 `get_whisper_admission()` minus running × lane credits. Non-Whisper models keep the boolean rule unchanged.
- ✅ **Metrics** — `hse_whisper_{workers,in_flight,lane_short_depth,lane_long_depth}` per model, `hse_whisper_queue_wait_seconds` histogram, plus `hse_whisper_ct2_{queued,active}_batches` (CT2 4.4.0 exposes `num_queued_batches` / `num_active_batches`).
- ✅ **Docs** — `docs/batching.md` whisper row + lanes section, `docs/models.md` knob, `CHANGELOG.md` entry.

**Verification**: `tests/unit` 893 passed (36 new in `tests/unit/inference_flow/test_whisper_lanes.py`), `tests/integration` 49 passed (3 new in `test_parallel_lanes.py`). No model weights downloaded; the handler is exercised with stubbed pipelines. Benchmark-host width sanity (c=4, M=1 vs 2 vs 4) outstanding (10-5).

**Next**: 10-2 window batcher consumes the lane in-flight accounting; 10-5 measures the real width.

## Engine Modernization — Phase 10-3 (CODE COMPLETE, 2026-11-07; benchmark-host run pending)

**Plan folder**: `.current_work/engine-modernization/`; task: `10-3-WhisperMemoryGovernance.md`

- ✅ **Device-level VRAM truth** — `get_vram_stats()` now reads `torch.cuda.mem_get_info()`: `device_used_bytes` / `free_bytes` / `hse_vram_device_used_bytes` include CTranslate2 + CUDA contexts; `hse_vram_headroom_bytes` is redefined in place as true free device memory (breaking change, documented in `_HELP`, CHANGELOG and `docs/benchmarks/vram-profiles.md`). Torch-scoped counters kept.
- ✅ **Whisper max-shape probe** — `harmonyspeech/modeling/whisper_probe.py` runs a synthetic max-shape transcription (workers × window batch × `whisper_max_audio_seconds`) after a CUDA Whisper load and records the device-level delta; CPU/missing-import/CPU-placed-model paths skip explicitly. Reported per model on `/health` (`probe_measured`, `probe_reason`) and as `hse_model_peak_activation_bytes{model}`.
- ✅ **Budget + admission governance** — optional `whisper_vram_budget_bytes` turns a projected-peak overshoot into a boot error (otherwise it warns against the measured free headroom); `whisper_max_audio_seconds` is the whisper request-size cap; the runtime hook `get_whisper_admission(active_windows)` is driven by the pure `allowed_windows()` in `harmonyspeech/engine/vram_budget.py` (ready for 10-1/10-2 and 10-5's mocked-`mem_get_info` floods).
- ✅ **Docs** — `docs/benchmarks/vram-profiles.md` §Phase 10-3 carries the corrected semantics, the `_tbd_` device-level/probe cells and the Phase 9 pool-sizing input; `docs/models.md` documents the knobs.

**Verification**: `tests/unit` 857 passed (45 new in `tests/unit/engine/test_vram_stats.py`; existing `test_vram.py`, `test_measure_vram.py`, config/loader/weight-registry suites green) and `tests/integration` 46 passed. Benchmark-host device-level figures and real Whisper probe peaks outstanding (no weights downloaded).

**Next**: benchmark host fills the `_tbd_` cells, then 10-1/10-2 consume `get_whisper_admission()`.

## Engine Modernization — Phase 8 (CODE COMPLETE, 2026-11-06; benchmark-host run pending)

**Plan folder**: `.current_work/engine-modernization/` (Phase 8 of the modernization; `summary.md` tracks all phases)

- ✅ **8-1 Batching core extensions** — `pad_feature_batch(return_lengths=True)`, `trim_feature_output`, `pad_waveform_batch`, axis-unit docs; `audio_regression.py` gained a pending-family activation map so 8-2…8-4 could gate immediately.
- ✅ **8-2 Embedder batching** — `ReferenceEncoder.forward_batched` (per-layer padding masks + packed GRU) and `SpeakerEncoder.forward_batched` (packed LSTM); OpenVoice V1/V2 tone-converter encoders and the HarmonySpeech encoder batch `(request, segment/partial)` rows on the mel-frame axis; cache hits resolve before the batch; B=1 bit-identical.
- ✅ **8-3 Tone-converter batching** — `SynthesizerTrn.voice_conversion_batched` with an opt-in per-stage decoder padding mask (a padded row drifted 0.30 peak before it); handler pads per-request mel frames, plans length-similar sub-batches, trims by `y_lengths × prod(upsample_rates)`.
- ✅ **8-4 HarmonySpeech + VoiceFixer batching (+ GPU placement fix)** — `MelGANGenerator.inference_batched` and `VoiceFixerVocoder.forward_batched` mask every stage, restore reflection tails and replay the length-dependent skip per row; `VoiceFixerRestorer.forward_batched` packs the GRUs and masks the UNet; VoiceFixer's STFT conv weights are now placed on the configured device (the pre-existing GPU device-mismatch fix). HarmonySpeech synthesizer is a documented skip (0.16 peak padding drift).
- ✅ **8-5 VAD/STT audit, lanes and skips** — whisper is batched within one audio (one pipeline per step, admission width 16); Silero keeps its sequential loop (single-stream API, 0.95× at 8 lanes, shared wrapper not thread-safe); KittenTTS runs `min(requests, max_batch_size)` lanes over one shared ONNX session (1.71× at 4 lanes, 1.91× at 8, `intra_op` = ORT default, phonemizer lock, load warmup); 19-type matrix in `docs/batching.md` + docs-lint test.
- ✅ **8-6 Honest config, flood benchmark tooling & exit** — both configs now declare `max_batch_size: 8` (+ `batched_inference: true`) for the newly batched families (initial value; tuning pending the benchmark-host run) while hs1-synthesizer and Chatterbox stay 1, whisper 16, Silero 16, Kitten 4. The config-honesty test now encodes the three-way policy (batched family / per-request loop / admission-lane-queue depth) and a docs test cross-checks every shipped entry against the `docs/batching.md` matrix. `scripts/bench_throughput.py` gained `--concurrency` list parsing, `--embedding-cache {auto,on,off}` verification against `/health`, `--engine-log` per-step sub-batch composition, client-pool sizing for c=200, `--print-scenarios`, and the OV1/HarmonySpeech/VoiceFixer/Kitten families. `docs/benchmarks/throughput.md` §Phase 8 carries the scenario matrix, per-step composition slot, 200-connection observation and the sequential-steps honesty note (all measurements `_tbd_`).

**Verification**: 838-test suite was green at Phase 8 hand-off; 8-6 added 19 test cases (46 green across `tests/unit/benchmarks/test_bench_throughput_references.py`, `tests/unit/initialization/test_config.py`, `tests/unit/test_batching_docs_matrix.py`). Remaining: the phase owner's full-suite run and the benchmark-host measurement (local HF cache holds only BERT text encoders, so real-model runs and the GPU `batching:1-vs-N` gates run on the RTX 5060 Ti host).

**Deviations**: `8` is an initial capacity, not a measured tuning; flood/family tables are `_tbd_` until the benchmark host run; real-model e2e for the touched families is deferred for the same weights reason; Chatterbox remains excluded by product decision.

**Next**: run the Phase 8 benchmark host measurements, then fill the `_tbd_` cells and tick the Phase 8 exit.

## Engine Modernization — Phase 7 (COMPLETE, 2026-11-06)

**Plan folder**: `.current_work/engine-modernization/` (Phase 7 of the modernization; `summary.md` tracks all phases)

- ☑ **7-1 Cache core** — `harmonyspeech/processing/embedding_cache.py`: `EmbeddingCache` (one per engine) keyed by `(model_identity, audio_hash, params)`; values are `SpeakerArtifact`s deep-copied on put and serve (Phase 0-2 isolation). TTL 300 s default (lazy + sweep every 16 puts), FIFO beyond 1000 entries, 1 GiB byte ceiling; per-model load version invalidates old generations.
- ☑ **7-2 Handler integration** — cache seams in Chatterbox embedding, Chatterbox VC target audio, OpenVoice V1/V2 tone-converter encoder and Harmony Speech encoder; `AudioArtifact.content_hash` (sha256 over decoded PCM + rate) makes the key container-independent. Config keys + CLI (`--embedding-cache-ttl-s`, `--embedding-cache-max-entries`, `--no-embedding-cache`).
- ☑ **7-3 Metrics, benchmark tooling & exit** — `/metrics` series (`hse_embedding_cache_{enabled,entries,bytes,hits_total,misses_total,stores_total,ttl_expirations_total,fifo_evictions_total,byte_evictions_total,hit_ratio}`) rendered zero-valued when disabled; `get_embedding_cache_stats()`; `embedding_cache` block on `/health`; `--clone-reference {same,distinct}` in `scripts/bench_throughput.py`; `docs/embedding-cache.md` (incl. privacy/retention note); Phase 7 section in `docs/benchmarks/throughput.md`; CHANGELOG.

**Verification**: unit + integration suites green (663 before 7-3; new metrics/benchmark/engine-flow tests added in 7-3). Engine-level proof `tests/integration/test_embedding_cache_flow.py`: two clone requests with the same reference → one embedding pass (cache) vs two (cache disabled), identical synthesized audio. Remaining: the GPU clone-flood numbers (`same`/`distinct` × cache on/off, c={1,4,20}) must be filled into `docs/benchmarks/throughput.md` (structure + reproduce commands already there).

**Deviations**: the cache is FIFO, not LRU (product decision) — a hit never reorders entries. Two extra Prometheus series beyond the plan's list (`_enabled`, `byte_evictions_total`) so "disabled" is distinguishable from "cold". 7-3 also fixed a pre-existing telemetry gap: `AsyncHarmonySpeech` now proxies `get_vram_stats()`/`get_embedding_cache_stats()` and returns the `check_health` payload, so `/health`'s `vram` block and `/metrics`' VRAM series reach the server again.

**Next**: run the Phase 7 GPU benchmark, then Phase 8 (batching across all steps, except Chatterbox).

## Engine Modernization — Phase 5 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 5 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **5-1 Type-checked batches & pairing invariants** — the type partition runs before `prepare_inputs` for every handler; `assert_batch_aligned` raises `EngineBatchError` which the runner converts into per-request errors; `_build_result` refuses a second result for the same request id inside one step. New engine-level suite `tests/integration/test_batching_invariants.py` (KittenTTS/Chatterbox/Silero/Whisper model types, batch sizes 1→5, delayed/failing items, forwarded+fresh scheduling). **Found and fixed a scheduler bug**: requests above a model's `max_batch_size` were dropped from the waiting queue (never ran, never errored, caller hung).
- ✅ **5-2 Padded batch on `SynthesizerTrn`** — vendored `infer_batched` on OpenVoice V1 and MeloTTS (per-sample `length_scale`/noise, returns decoder frame counts), `harmonyspeech/task_handler/batching.py` (padding, length-similar sub-batching with `T_max/T_min ≤ 4`, token budget, unpadding), both handlers batched with the per-item loop as `len==1` fast path and failure fallback, per-model `batched_inference` flag (default on for these two families). Equivalence: `batching:1-vs-4` gate passes (OpenVoice V1 byte-identical, MeloTTS max |Δ| 1.6e-5); B=1 unit tests bit-identical.
- ✅ **5-3 Honest batch config & benchmarks** — truthful `max_batch_size` in both configs (synth 8 + `batched_inference`, every loop 1, Chatterbox 1, Whisper 16, Kitten 4, Silero 16) asserted by a config test; `scripts/bench_throughput.py` (HTTP, warmup, p50/p95 + req/s at c={1,2,4,8}); `docs/benchmarks/throughput.md` pre/post tables; `docs/batching.md` ops guide (semantics, per-family concurrency, Chatterbox replicas-not-batch).

**Verification**: unit 530 → 578, integration 23 → 34, wire-contract e2e 38 + 3 skipped; real-model e2e green for OpenVoice V1, MeloTTS and the OpenVoice VAD-mode suite; HTTP pre/post benchmark on an idle RTX 5060 Ti (MeloTTS 12.62 → 14.43 req/s at c=8, p50 629 → 544 ms; Chatterbox flat at 0.6 req/s and Whisper at 2.8 req/s in both).

**Deviations**: full OpenVoice V2 clone throughput is unchanged (bound by per-item tone-converter hops, left at `max_batch_size: 1` by design); long-text batching shows no gain (compute-bound) — both documented in `docs/benchmarks/throughput.md`.

**Next**: Phase 6 is an opt-in capabilities backlog (not implemented).

## Engine Modernization — Phase 4 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 4 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **4-1 dtype plumbing & config flip** — `ChatterboxTTS/TurboTTS/MultilingualTTS/VC` wrappers accept `dtype` and apply a documented per-submodule policy (T3 → configured dtype, S3Gen/VE/Perth watermarker pinned float32); the loader forwards `model_config.dtype` in every Chatterbox branch and logs effective dtypes. `config.gpu.yml` Chatterbox entries flipped to `float16`; other torch families stay `float32` (measured opt-outs: OpenVoice/MeloTTS build float32 tensors internally and MeloTTS splines assert under fp16; VoiceFixer has a pre-existing GPU placement bug).
- ✅ **4-2 Audio regression gate** — `scripts/audio_regression.py` (+ `tests/audio_fixtures/` with synthetic reference clips and documented provenance): two-process A/B per family, log-mel MSE / duration / RMS / correlation for deterministic families, LTAS/duration/RMS for stochastic ones, JSON + markdown output, non-zero exit, listening pack. Identity axis passes; fp16 Chatterbox/Turbo/Multilingual pass; sharing axis passes.
- ✅ **4-3 Weight registry** — `harmonyspeech/modeling/weight_registry.py`: ref-counted, lock-protected `(repo, revision, component, dtype)` store with `stats()`, read-only enforcement and `release()`; engine owns one and threads it through executor → worker → model runner → loader; `ModelConfig.backbone_group` added.
- ✅ **4-4 Chatterbox backbone sharing** — `modeling/models/chatterbox/factory.py` builds one reference S3Gen/VE per checkpoint through the registry and assembles distinct wrapper instances per role (own `conds`/`ref_dict`); embedding shares T3 too. `backbone_group: "chatterbox-base"` on chatterbox/multilingual/vc/embedding; turbo ungrouped. Multilingual grouped on hash evidence (`s3gen.pt` ≡ `s3gen.safetensors`, `ve.pt` ≡ `ve.safetensors`, all tensors bit-identical). In-process A/B is bit-identical; cross-process gate within −60 dBFS kernel noise.
- ✅ **4-5 VRAM metrics & profile** — `get_vram_stats()` per model + shared-group accounting, `/health` payload extension, Prometheus `/metrics` (`hse_model_vram_bytes`, `hse_vram_reserved_bytes`, `hse_resident_models`, `hse_shared_backbone_bytes{group}`, …), `scripts/measure_vram.py`, `docs/benchmarks/vram-profiles.md`, measured figures in `docs/models.md`.

**Verification**: unit 469 → 530, integration 23, wire-contract e2e 38 passed + 3 skipped. Real-model e2e: Chatterbox family (routing/behavior unchanged after sharing) + OpenVoice V1 on GPU/CPU. Full-catalogue VRAM: 16.45 GiB (fp32, duplicated — overflowed the card) → **8.52 GiB (8.83 GiB nvidia-smi), 7.1–7.3 GiB headroom**. Chatterbox block 14.29 → 6.35 GiB. Acceptance floor of ≥8 GiB headroom missed by ~0.7 GiB; causes diagnosed in `docs/benchmarks/vram-profiles.md` (remaining float32 families + turbo's GPT2 mask buffers).

**Deviations**: S3Gen stays float32 (fp16 audibly degrades synthesis — measured); turbo ungrouped; VoiceFixer/precision opt-outs documented.

**Next**: Phase 5 — honest batching (HSE only).

## Engine Modernization — Phase 3 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 3 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **3-1 PipelineSpec model & registry** — `Step`/`PipelineSpec`/`EntryRule`/`PipelineState` plus `next_step()`/`advance()` in `harmonyspeech/processing/pipeline_registry.py`; artifact-first `needs`, declared `produces`/`clears`; `EngineRequest.pipeline_state`.
- ✅ **3-2 Family migration** — registry holds 11 specs (8 client-facing toolchains + 3 single-step executors); both `if/elif` ladders deleted (~250 lines); routing/forwarding spec-driven; error guard, explicit-model short-circuit and Phase 6 extension point preserved.
- ✅ **3-3 Startup validation & model cards** — `validate_registry()` names each missing step; `--strict-pipelines`/`HSE_STRICT_PIPELINES`; cards generated from specs (capabilities/languages/members) and every per-serving-module hand list deleted; `/v1/models` ids unchanged.
- ✅ **3-4 VAD switch & entry points** — Silero emits whisper-schema `VadArtifact` segments (backend-aware encoder filter); `input_vad_mode` honoured with fallback+warning; `EntryRule` set replaces field-presence entry hacks (`single_speaker_tts` FIXME resolved); forwarded requests never re-resolve entry.
- ✅ **3-5 Routing tests & exit** — `tests/unit/inference_flow/test_pipeline_routing.py` parity table (40 cases over every family × entry shape), artifact-flow, per-step error, missing-artifact, disconnect, registry-only extension and strict-boot tests.

**Verification**: 469 unit + 23 integration green; wire-contract e2e 38 passed / 3 model-only skips; real-model e2e green on CPU for all TTS families (HarmonySpeech with `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1`), Whisper STT, Silero/Whisper VAD, VoiceFixer, plus a new Silero-VAD-in-pipeline cloning test.

**Next**: Phase 4 — VRAM (fp16 rollout + Chatterbox weight dedup).

## Engine Modernization — Phase 2 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 2 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **2-1 Artifact type system** — `harmonyspeech/common/artifacts.py` (`AudioArtifact`, `MelArtifact`, `SpeakerArtifact`, `VadArtifact`), torch.Tensor waveforms (`[1, samples]` float32, `from_numpy`/`.numpy`), explicit `to(device)`, `EngineRequest.artifacts` + deterministic `clear_artifacts()` release on finish.
- ✅ **2-2 Artifact fields & prepare_inputs** — precedence getters decode each inbound field once and cache it; every `prepare_*` returns artifacts; admission decode-once (`add_request` / `wire.from_wire_audio`); transient resampled views instead of per-rate caches.
- ✅ **2-3 Executor artifact handlers** — all 8 hops converted (VAD→encoder, encoder→synth/TC, synth→TC, Chatterbox embed→TTS/VC, VoiceFixer restorer→vocoder, HarmonySpeech synth→vocoder, final outputs); output classes gained optional artifact slots; forwarding copies artifacts, not strings; device asserts + explicit moves; Chatterbox conditionals by reference. Removed `ensure_librosa_readable_bytes` from the pipeline; fixed Chatterbox's bundled default voice path.
- ✅ **2-4 Boundary serialization & compatibility** — `endpoints/openai/wire.py` (`from_wire_audio`/`to_wire_audio`/`to_wire_embedding`), all six serving modules migrated, engine core free of wire encoding (pytest grep invariant), default WAV responses byte-identical, embed response format unchanged.
- ✅ **2-5 Verification & benchmark** — `scripts/bench_pipeline_hops.py` + `docs/benchmarks/artifact-bus.md`; stub microbenchmark (VoiceFixer mel hop 25.4 ms → 0.005 ms/s of audio) and real-weight CPU runs; determinism: VoiceFixer/HarmonySpeech byte-identical, MeloTTS stochastic (Δduration 1.46 %).

**Verification**: 346 unit + 23 integration green; wire-contract e2e 38 passed / 3 model-only skips. Real-model e2e ran green on CPU for HarmonySpeech, OpenVoice V1/V2, VoiceFixer, Chatterbox/Turbo, KittenTTS, Whisper STT, Silero/Whisper VAD. Outstanding: pre-Phase-2 hash-fixture comparison and CUDA numbers.

**Next**: Phase 3 — declarative `PipelineSpec` routing.

## Engine Modernization — Phase 1 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 1 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **1-1 Boundary audio decoder** — `harmonyspeech/common/audio_codec.py` sniffs → decodes with libsndfile → falls back to an optional ffmpeg subprocess → raises typed `AudioDecodeError`. Every `librosa.load` boundary decode in `task_handler/inputs.py` now goes through it, and Chatterbox reference audio accepts mp3/aac too. SoXR resampling (`soxr` pinned).
- ✅ **1-2 Output format & sample rate** — `output_options` is validated at admission (400 before synthesis) and applied once at the response boundary; default path is byte-identical WAV passthrough. `pcm` (raw s16le) and `aac` (ffmpeg-only) supported; responses echo the effective `sample_rate`.
- ✅ **1-3 Binary input envelope** — every audio-input endpoint accepts OpenAI-style `multipart/form-data` uploads on the same path via content-type dispatch, JSON stays primary for `/v1/audio/speech`; configurable upload limit returns 413.
- ✅ **1-4 Wire-contract tests & docs** — parametrized contract matrix (input × transport × output), `docs/audio-io-contract.md`, README/CHANGELOG/CONCERNS updates.

**Verification**: 281 unit + 23 integration green; wire-contract e2e matrix (38 passed / 3 skipped without model weights). Real-model e2e remains manual (`HSE_E2E_MODEL_TESTS=1`).

**Next**: Phase 2 — artifact bus (typed intermediates, decode once per request).

## Engine Modernization — Phase 0 (COMPLETE, 2026-11-04)

**Plan folder**: `.current_work/engine-modernization/` (Phase 0 of a 6-phase modernization; `summary.md` tracks all phases)

- ✅ **0-1 Safe deserialization** — request embeddings and model checkpoints load without executing pickle code; unsupported/hostile payloads fail closed with a typed error.
- ✅ **0-2 Per-request state isolation** — Chatterbox conditioning is restored to the post-load baseline after every request; requests without conditioning fail loudly instead of inheriting the previous voice.
- ✅ **0-3 Batch pairing correctness** — mixed-type batches are partitioned before preparation, worker helpers receive their request explicitly, and alignment is asserted.
- ✅ **0-4 Startup validation & no-hang** — declarative family registry, warnings for incomplete toolchains, immediate error on unresolvable routing, scheduler unknown-model safety net, only complete toolchains advertised.
- ✅ **0-5 Timeouts & health** — per-request deadlines (`request_timeout_s`), recoverable loop watchdog with consecutive-trip escalation, structured `check_health` payload.
- ✅ **0-6 Exit review** — unit (162) and integration (23) suites green; E2E (39) collects but requires model downloads (manual verification outstanding).

**Next**: Phase 2 — artifact bus (typed intermediates between pipeline steps).

## Core System Components

### ✅ Engine Architecture (COMPLETE)

**HarmonySpeechEngine Core**
- ✅ Multi-executor initialization and management
- ✅ Request lifecycle management
- ✅ Model configuration loading and validation
- ✅ Health check and monitoring infrastructure
- ✅ Statistics collection and logging

**Request Scheduler**
- ✅ Request queue management
- ✅ Resource allocation and optimization
- ✅ Status lifecycle tracking (WAITING → RUNNING → FINISHED)
- ✅ Request forwarding for multi-step workflows
- ✅ Batch processing coordination

**Executor Framework**
- ✅ CPU executor implementation
- ✅ GPU executor with async support
- ✅ Parallel execution using ThreadPoolExecutor
- ✅ Resource management and optimization
- ✅ Error handling and recovery

### ✅ Request Routing System (COMPLETE)

**Intelligent Model Selection**
- ✅ Harmony Speech routing (Encoder → Synthesizer → Vocoder)
- ✅ OpenVoice V1 routing (VAD → Embedding → Synthesis → Voice Conversion)
- ✅ OpenVoice V2 routing (VAD → Embedding → MeloTTS → Voice Conversion)
- ✅ Language-based model selection
- ✅ Automatic workflow orchestration

**Multi-Step Processing**
- ✅ Request forwarding between models
- ✅ Status management for complex workflows
- ✅ Data transformation between processing steps
- ✅ Error propagation and handling

### ✅ API Layer (COMPLETE)

**FastAPI Implementation**
- ✅ OpenAI-compatible endpoints
- ✅ Extended functionality for voice cloning
- ✅ Automatic OpenAPI/Swagger documentation
- ✅ Request validation with Pydantic
- ✅ Async request handling

**Endpoint Coverage**
- ✅ `/v1/audio/speech` - Text-to-speech synthesis
- ✅ `/v1/audio/transcriptions` - Speech-to-text
- ✅ `/v1/audio/embeddings` - Speaker embeddings
- ✅ `/v1/audio/voice-conversion` - Voice conversion
- ✅ `/v1/audio/convert` - Audio restoration and filtering
- ✅ `/docs` - Interactive API documentation
- ✅ `/redoc` - Alternative documentation interface

## Model Integration Status

### ✅ Harmony Speech V1 (COMPLETE)
- ✅ HarmonySpeechEncoder - Speaker embedding generation
- ✅ HarmonySpeechSynthesizer - Text-to-speech synthesis
- ✅ HarmonySpeechVocoder - Audio vocoding
- ✅ Multi-step pipeline integration
- ✅ Configuration and device management

### ✅ OpenVoice V1 (COMPLETE)
- ✅ OpenVoiceV1Synthesizer - English and Chinese TTS
- ✅ OpenVoiceV1ToneConverter - Voice conversion
- ✅ OpenVoiceV1ToneConverterEncoder - Speaker embedding
- ✅ Emotion support (whispering, shouting, excited, etc.)
- ✅ Multi-step workflow integration

### ✅ OpenVoice V2 / MeloTTS (COMPLETE)
- ✅ MeloTTSSynthesizer - Multilingual synthesis
- ✅ Language support: English, Chinese, Spanish, French, Japanese, Korean
- ✅ OpenVoiceV2ToneConverter - Advanced voice conversion
- ✅ OpenVoiceV2ToneConverterEncoder - Improved embeddings
- ✅ Cross-lingual voice cloning capabilities

### ✅ Faster-Whisper (COMPLETE)
- ✅ Speech recognition and transcription
- ✅ Voice Activity Detection (VAD)
- ✅ Multiple model sizes (tiny, medium, large-v3-turbo)
- ✅ Language detection and multilingual support
- ✅ Integration with voice cloning workflows

### ✅ VoiceFixer (COMPLETE)
- ✅ VoiceFixerRestorer - Audio denoising and enhancement
- ✅ VoiceFixerVocoder - Mel-spectrogram to audio conversion
- ✅ Audio restoration for degraded speech
- ✅ Support for noise, reverberation, and clipping removal
- ✅ Integration with HSE model loading system

### ✅ Silero VAD (COMPLETE)
- ✅ High-performance voice activity detection
- ✅ CPU-optimized ONNX implementation
- ✅ Dynamic parameter support for fine-tuning
- ✅ Enhanced `/v1/audio/vad` endpoint with configurable parameters
- ✅ Native model integration following HSE patterns

### ✅ KittenTTS (COMPLETE)
- ✅ KittenTTSSynthesizer — ONNX-based English TTS
- ✅ 4 model variants: mini (80MB), micro (41MB), nano-fp32 (56MB), nano-int8 (25MB)
- ✅ 8 voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo
- ✅ Phonemizer: misaki[en]
- ✅ 4 e2e tests validated and passing on CPU

### ✅ Chatterbox TTS (COMPLETE — All 7 Phases)
- ✅ Phase 1: Dependencies installed — chatterbox-tts==0.1.6 (--no-deps), resemble-perth, pyloudnorm, einops, omegaconf, conformer, s3tokenizer, spacy-pkuseg; all 7 import tests pass
- ✅ Phase 2: Model Registration — ChatterboxTTS, ChatterboxVC, ChatterboxMultilingualTTS, ChatterboxTurboTTS in ModelRegistry
- ✅ Phase 3: Input Preparation — speed, normalize_audio (renamed from norm_loudness), split_text parameters
- ✅ Phase 4: Model Execution — execute methods and dispatch
- ✅ Phase 5: Request Routing — Chatterbox request routing to engine and serving layer, embed-dispatch support
- ✅ Phase 6: Configuration & Performance — Chatterbox entries in config.yml/config.gpu.yml, watermarker swap
- ✅ Phase 7: Testing & Documentation — Comprehensive unit and e2e tests for all 4 variants

### 🔄 Planned Model Integrations (IN PROGRESS)

**StyleTTS 2** (Priority: High)
- ❌ Text-to-speech synthesis
- ❌ Voice conversion capabilities
- ❌ Few-shot voice cloning
- ❌ Integration with existing workflows

**XTTS V2** (Priority: High)
- ❌ Multilingual zero-shot voice cloning
- ❌ Real-time synthesis capabilities
- ❌ Cross-lingual voice transfer
- ❌ Streaming support

**Vall-E-X** (Priority: Medium)
- ❌ Advanced zero-shot voice cloning
- ❌ Multilingual support
- ❌ High-quality synthesis
- ❌ Integration framework

**Additional Models** (Priority: Low)
- ❌ CosyVoice - Multilingual TTS
- ❌ EmotiVoice - Emotional TTS with multiple speakers
- ❌ SenseVoice - Advanced speech recognition

## Frontend Development

### ✅ React Application (COMPLETE)
- ✅ Modern React 18 + Vite setup
- ✅ TypeScript integration
- ✅ TailwindCSS styling
- ✅ Responsive design implementation

### ✅ Core Components (COMPLETE)
- ✅ TTS module with voice selection
- ✅ Audio player with waveform visualization
- ✅ Settings management interface
- ✅ Performance metrics display
- ✅ Interactive API testing interface

### 🔄 Frontend Enhancements (PLANNED)
- ❌ Real-time processing status updates
- ❌ Advanced audio editing capabilities
- ❌ Batch processing interface
- ❌ Model performance comparison tools

## Infrastructure and Deployment

### ✅ Docker Support (COMPLETE)
- ✅ Multi-stage Dockerfile for different targets
- ✅ Docker Compose configurations
- ✅ NVIDIA GPU support (docker-compose.nvidia.yml)
- ✅ AMD GPU support (docker-compose.amd.yml)
- ✅ CPU-only deployment option

### ✅ Configuration Management (COMPLETE)
- ✅ YAML-based model configuration
- ✅ Environment variable support
- ✅ Device assignment and resource allocation
- ✅ Flexible deployment options

### 🔄 Production Features (IN PROGRESS)
- 🔄 Comprehensive monitoring and logging
- ❌ Health check endpoints
- ❌ Horizontal scaling support
- ❌ Load balancing configuration
- ❌ Performance optimization guides

## Testing and Quality Assurance

### ✅ Testing Infrastructure (ESTABLISHED)
- ✅ 162 unit tests — config parsing, dtype resolution, engine init, model loader registry, safe deserialization, state isolation, batch pairing, startup validation, routing/timeout resilience
- ✅ 23 integration tests — all 7 HTTP endpoints via mocked serving layer
- ✅ 39 e2e tests — full model pipelines for KittenTTS, MeloTTS, OpenVoice V1/V2, HarmonySpeech, FasterWhisper, SileroVAD, VoiceFixer (require model downloads)
- ✅ 80% overall coverage
- ❌ GPU executor path untested (gpu_model_runner.py, gpu_worker.py — 0% coverage)
- ❌ Performance benchmarking suite
- ❌ Load testing framework

### 🔄 Documentation (PARTIAL)
- ✅ API documentation (auto-generated via Swagger/ReDoc)
- ✅ Setup and installation guides
- 🔄 Model configuration examples
- ❌ Advanced usage tutorials
- ❌ Troubleshooting guides
- ❌ Performance optimization documentation

## Performance and Optimization

### ✅ Core Optimizations (COMPLETE)
- ✅ Parallel model execution
- ✅ Resource-aware scheduling
- ✅ Memory management for GPU/CPU
- ✅ Request batching capabilities

### 🔄 Advanced Optimizations (IN PROGRESS)
- 🔄 Model caching and sharing
- ❌ Dynamic model loading/unloading
- ❌ Request prioritization
- ❌ Adaptive quality settings
- ❌ Streaming capabilities

### ❌ Performance Features (PLANNED)
- ❌ TTS streaming for real-time applications
- ❌ Voice conversion post-processing pipelines
- ❌ Batch processing optimization
- ❌ Memory pressure handling

## Known Issues and Limitations

### Current Limitations

**Model Memory Usage**
- Large models require significant GPU/CPU memory
- Limited concurrent model loading capacity
- No automatic model unloading based on usage

**Windows Support**
- Limited Windows compatibility
- CUDA support tested but not fully optimized
- Recommendation to use WSL for Windows users

**Error Handling**
- Complex workflow debugging can be challenging
- Limited error recovery for multi-step processes
- Need for enhanced debugging interfaces

**Performance**
- No streaming support for real-time applications
- Batch processing not fully optimized
- Memory management could be improved

### Known Bugs

**Minor Issues**
- Occasional GPU memory leaks with repeated requests
- Configuration validation could be more comprehensive
- Frontend error handling needs improvement

**Documentation Gaps**
- Missing advanced configuration examples
- Limited troubleshooting documentation
- Need for more integration examples
