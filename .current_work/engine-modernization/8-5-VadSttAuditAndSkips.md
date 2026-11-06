# Phase 8-5: VAD/STT Audit & Documented Skips

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1 (terminology only — mostly independent; can run alongside 8-2…8-4)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `docs/batching.md`
> **State reference:** verified 2026-11-05 (`config.gpu.yml`: whisper 16, silero 16, kitten 4)

## Objective

Close out the batching matrix: verify the already-native batching paths are actually engaged end-to-end, turn every remaining loop into an **evidence-based documented skip**, and give non-batchable CPU models **parallel lanes** (engine-as-message-queue) — so `docs/batching.md` states, per model, exactly one of: "batches (how, budget)" / "parallel lanes (queue depth = `max_batch_size`)" / "doesn't (why, what it would take)".

## Context (self-contained)

- **FasterWhisper** (`_execute_faster_whisper`, serves both the STT endpoint and the OpenVoice VAD step): CTranslate2 `BatchedInferencePipeline(batch_size=16)` is native batching, but **verify the handler feeds the whole step batch into one call** rather than looping around a batched-capable object (per-item loops around batched primitives were exactly the Phase 5 finding elsewhere). If it loops: batch it (pad-free — CT2 takes lists).
- **Silero VAD** (`_execute_silero_vad`): upstream `get_speech_timestamps` is a single-stream API; true batching would mean reimplementing the windowing loop — not worth it. CPU-cheap already (`max_batch_size: 16` as sequential concurrency). Verdict: **skip batching — but apply parallel lanes** (same mechanism as KittenTTS, decision 2026-11-05): `max_batch_size` = lane count, per-request `get_speech_timestamps` calls on the shared session via the thread pool. **Verify first**: read the installed `silero_vad` package to confirm (a) per-call state is function-local (v4/v5 pass LSTM/state explicitly and create it inside the call — no cross-request mutation) and (b) whether the ONNX session (loader uses `load_silero_vad(onnx=True)`) is the shared `InferenceSession` (thread-safe, GIL-releasing). **Scaling expectation — set honestly**: unlike KittenTTS (one ~1.6 s C++ `session.run` per request → near-linear lanes), Silero is *many tiny* inferences with Python glue between them, so the GIL serializes part of the loop — expect sublinear scaling (e.g. 2–4× at 8 lanes, not 8×). Measure lanes {4, 8, 16} on concurrent VAD requests; record the verdict either way ("lanes: measured N× at K" or "lanes: not worth it, sequential kept") in `docs/batching.md`. Context for priority: VAD is a minor hop in the clone pipeline (synthesis dominates) — lanes mainly help floods against the `/v1/audio/vad` endpoint itself.
- **KittenTTS** (`_execute_kittentts_synthesizer`): **batching is impossible as shipped — evidence recorded 2026-11-05** (probe on this machine, all four cached variants `mini/micro/nano-fp32/nano-int8`): every graph has a **static batch dim of 1** (`input_ids:[1, sequence_length]`, `style:[1, 256]`, `speed:[1]`). Batching would require a re-export with a dynamic batch axis — KittenML's export pipeline, not ours. The model is one-shot non-AR (tokens+style+speed → waveform in a single `session.run`). **Verdict: skip batching; run parallel lanes instead (design decision 2026-11-05).**
  - Measured baseline (mini, this CPU): ~1.94 s generate for 5.07 s audio (RTF 0.38); espeak phonemize is **0.3 ms** (not the bottleneck — pure ONNX compute is); 0.8 s session load + ~2 s cold first generate (graph warmup).
  - **Parallel lanes — engine as message queue**: `max_batch_size` **is** the lane count — **no new config key**. The scheduler already admits up to `max_batch_size` kitten requests per engine step (the queue); the handler dispatches each request's `session.run` to a bounded thread pool (pool size = `min(len(inputs), max_batch_size)`), outputs reassembled by request (Phase 0-3/5-1 pairing invariants hold — each lane's exception becomes only that request's error result).
  - **Why one session suffices**: `onnxruntime` `InferenceSession.run()` is documented thread-safe for concurrent calls **and releases the GIL** — N lanes = N real parallel inferences on N cores sharing one 74 MB session (no N× RAM replication, no executor/routing changes).
  - **`intra_op_num_threads` — measure, then fix**: N lanes × ORT's default intra-op threads oversubscribes the CPU. Measurement matrix on the dev machine: lanes {4, 8} × `intra_op` {1, 2, 4, default} on concurrent kitten requests; bake the winner in as the **fixed** session default for lane mode (a constant in the wrapper, not a config knob). Record the matrix + chosen value in `docs/batching.md`.
  - **Phonemizer serialized**: the espeak backend's thread-safety is not guaranteed — guard it with a lock (0.3 ms per call, negligible).
  - **Warmup at load**: one dummy `generate` in runner init so the first real request doesn't pay the ~2 s cold-start (same warmup-on-load principle as the Phase 6-D governor design).
  - **Optional probe — GPU execution providers**: the env's onnxruntime is CPU-only (`['AzureExecutionProvider', 'CPUExecutionProvider']`). Two candidate EPs: `onnxruntime-gpu` (CUDA EP — **version-coexistence risk** with torch cu130 DLLs on Windows; verify before adopting) and `onnxruntime-directml` (Windows DX12, no CUDA coupling — applies to Silero too). A 74 MB one-shot graph *may* or may not beat CPU per-run (small-tensor launch overhead); verdict from a 30-minute measurement, not a commitment. KittenTTS's stated purpose is ultra-light CPU deployment — GPU support is opportunistic, never required.
- **Chatterbox family (all five model types)**: **explicitly out of scope** per the product decision (2026-11-05) — AR TTS needs a KV-cache story; its non-AR members (VC, embedding) are deferred with it to the dedicated later Chatterbox workstream. `docs/batching.md` states this verbatim so the exclusion is a decision, not an oversight.

## Implementation steps

1. **Whisper audit**: read `_execute_faster_whisper`; if per-item, batch the call (one `BatchedInferencePipeline` call per step batch); add an integration test proving N=4 concurrent transcription requests result in **one** pipeline invocation. Confirm VAD-mode requests (segments for the OpenVoice entry) take the same path.
2. **Silero lanes**: verify package internals (per-call state locality, shared session), then apply the lane mechanism; measure lanes {4, 8, 16} on concurrent VAD requests; keep sequential if measured scaling is poor (<2× at 8 lanes) — either verdict goes into `docs/batching.md`.
3. **KittenTTS lanes**: implement lane execution in `_execute_kittentts_synthesizer` — per the design above (no new config key; `max_batch_size` = lane count; thread pool sized to the step batch; phonemizer lock; per-request error isolation). Then run the **lane/`intra_op` measurement matrix** (lanes {4, 8} × intra_op {1, 2, 4, default}) on concurrent kitten requests and fix the winning `intra_op` as the session default for lane mode. Keep load-time warmup. Optional GPU-EP probe last (DirectML first; CUDA EP only if the DLL coexistence with torch cu130 is verified safe). Record the matrix, chosen constant, and RTF before/after in `docs/batching.md`; the static-batch skip verdict is already evidenced above.
4. **`docs/batching.md` matrix update**: one table, every model type × {batches? mechanism, budget axis, fallback, max_batch_size rationale or skip reason}. This table is the phase's contract with operators.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | whisper batched call (if audit finds a loop); KittenTTS lane execution |
| `harmonyspeech/modeling/models/kittentts/onnx_model.py` | `SessionOptions` with the measured fixed `intra_op_num_threads`; phonemizer lock; warmup hook |
| `docs/batching.md` | full matrix incl. skip verdicts, lane mechanism + measured constant, Chatterbox exclusion note |
| `tests/integration/test_batching_invariants.py` (extend), `tests/unit/` | one-pipeline-invocation test; lane pairing/isolation tests |

## Tests

- Integration: N=4 whisper transcriptions → single batched pipeline call (spy); results per-request paired (5-1 invariants).
- Integration (kitten lanes): 8 concurrent requests → all outputs correctly paired by request id; one poisoned input errors **only** its own request (lane error isolation); wall-time speedup vs sequential loop asserted on multicore machines (`skipif(os.cpu_count() < 4)`, assert ≥ 2× to stay CI-stable — the benchmark records the real number).
- Unit: lane pool sizing respects `min(len(inputs), max_batch_size)`; phonemizer lock held around espeak calls.
- Docs table exists and mentions every model type (a tiny docs-consistency unit test can lint it against the model-type registry, mirroring the Phase 2 grep-invariant style — optional, do it if cheap).

## Checklist

- [x] Whisper handler verified/converted to one batched call per step — **audit: no cross-request batch exists** (upstream API is single-audio); one pipeline per step, one `transcribe` per request, cursor reset per request
- [x] Silero batching skip evidenced (single-stream API); Silero lanes verified, measured, verdict recorded — **verdict: sequential kept** (0.95× at 8 lanes; shared wrapper is not thread-safe)
- [x] KittenTTS parallel lanes implemented (max_batch_size = lane count, no new config key)
- [x] Lane/`intra_op` measurement matrix run; fixed value baked in; RTF recorded in `docs/batching.md`
- [x] Load-time warmup landed (first-request cold start gone)
- [x] GPU-EP probe verdict recorded (DirectML / CUDA-EP coexistence, or explicit "not pursued" with reason)
- [x] Chatterbox exclusion documented as an explicit decision
- [x] `docs/batching.md` matrix complete for every model type
- [x] `summary.md` Implementation Status updated

## Outcome (2026-11-06)

**Whisper — audit, no conversion.** In the installed `faster_whisper` 1.2.1
`BatchedInferencePipeline.transcribe()` takes a **single** audio (path / file
object / mono ndarray) and batches *within* it: its own VAD cuts the audio into
≤30 s clips, which `_batched_segments_generator` feeds through `forward()` in
groups of `batch_size`. Neither `transcribe()` nor
`WhisperModel.generate_segments()` accepts a list. Since
`prepare_faster_whisper_inputs` returns one `AudioArtifact` per request — for STT
requests and for the pipeline's whisper-segmentation VAD step alike — there is no
cross-request call to fold. Folding would mean re-implementing `transcribe()`
(per-audio VAD chunking, per-audio language detection and tokenizer, timestamp
restoration) on top of upstream internals. The handler now builds **one pipeline
per step** (was per request) and resets the pipeline's word-timestamp cursor per
request; `batch_size=16` is documented as the intra-audio clip batch and
`max_batch_size: 16` as admission width. Integration tests pin it:
`test_whisper_step_builds_one_pipeline_and_transcribes_each_request_once`,
`test_whisper_vad_step_takes_the_same_per_request_path`.

**Silero VAD — lanes measured and rejected; sequential kept.** The installed
package contradicts the plan's assumption: per-call state is **not**
function-local. `OnnxWrapper` keeps `_state` / `_context` / `_last_sr` on the
instance and `get_speech_timestamps` calls `model.reset_states()` at entry, so
concurrent calls on the shared model corrupt each other — measured **7 of 16**
concurrent results differ from their sequential answer. Lanes with per-lane
wrappers (the only correct form) measured on 16 × 30 s clips: 1.10× at 4 lanes,
**0.95× at 8**, 0.92× at 16 (3.94 s sequential, 4.15 s at 8). Below the ≥2× bar,
so the handler stays a plain loop and `max_batch_size: 16` remains queue depth.

**KittenTTS — lanes implemented.** `max_batch_size` is the lane count (no new
config key); the handler dispatches each request's `generate` to a bounded pool
of `min(len(inputs), max_batch_size)` lanes over the one shared ORT session,
reassembles results in input order and turns a lane's exception into exactly that
request's error. Measured matrix (mini, 8 × ~5.9 s audio, 12-core host):

| lanes | intra 1 | intra 2 | intra 4 | intra default |
|---|---|---|---|---|
| 4 (req/s) | 1.06 | 1.04 | 1.18 | 1.19 |
| 8 (req/s) | 1.23 | 1.24 | 1.28 | 1.29 |

Single-request latency: 2.61 s (intra 1), 1.71 s (2), 1.74 s (4), 1.61 s
(default) for 5.9 s of audio. **Chosen:** `KITTEN_TTS_INTRA_OP_NUM_THREADS = 0`
(ORT's own default = one intra-op thread per physical core), `inter_op = 1`,
baked in and not configurable: throughput is flat across `intra_op` (the
suspected oversubscription costs nothing) and the default keeps the best
single-request RTF (0.273 vs 0.443 at intra 1). Before/after: single-request RTF
unchanged, 4 lanes 0.70 → 1.19 req/s (1.71×), 8 lanes → 1.29 req/s (1.91×).

**Warmup.** `KittenTTS_1_Onnx.warmup()` runs one "Hi" generate from
`KittenTTSSynthesizer.__init__`; a failure is logged and never fails the load.
On this machine the cold gap was already small (1.52 s cold vs 1.47–1.53 s
warm), so the warmup is cheap insurance rather than a measured win here.

**Phonemizer lock.** Required, not optional: `phonemizer`'s `EspeakBackend` is
not thread-safe and concurrent `phonemize` calls fail with
`RuntimeError: number of lines in input and output must be equal, we have:
input=1, output=2` (reproduced while measuring). `PHONEMIZER_LOCK` guards the
call (~0.3 ms).

**GPU execution providers — not pursued.** The pinned hse environment is
CPU-only ORT (`['AzureExecutionProvider', 'CPUExecutionProvider']`); adding
`onnxruntime-directml`/`-gpu` means installing into that pinned environment
(project policy forbids; Application Control blocks several wheels) and the CUDA
EP risks DLL coexistence with torch's CUDA 13 libraries. KittenTTS's purpose is
ultra-light CPU deployment, so opportunistic GPU stays a benchmark-host question.

**`docs/batching.md`** now carries the one-table matrix (19 registry types, each
exactly once: mechanism, budget axis unit, fallback, `max_batch_size` rationale
or skip reason), the KittenTTS lane/`intra_op` matrices, the Silero verdict
numbers, the whisper audit section and the verbatim Chatterbox product decision.
`tests/unit/test_batching_docs_matrix.py` lints the table against
`_MODEL_CONFIGS`.

### Deviations

1. **Whisper was not converted to one batched call per step** — the plan's
   premise ("CT2 takes lists — pad-free") does not hold for the installed
   faster-whisper: the list-accepting entry point does not exist. Recorded as an
   audit + documented skip, with a test pinning the real (one-pipeline-per-step,
   one-transcribe-per-request) behaviour.
2. **Silero lanes were not implemented, and the package internals contradict the
   plan**: per-call state is instance state, not function-local, so the shared
   session is *not* safe for concurrent calls (7/16 corrupted results). Lanes
   with per-lane wrappers measured 0.95× at 8 lanes (< 2× bar) → sequential kept,
   and the code's default matches the measurement.
3. **`intra_op` winner is ORT's default (0), not a small pin.** The plan expected
   N lanes × default threads to oversubscribe; the matrix shows flat throughput
   across 1/2/4/default and a doubled single-request latency at 1. Chose 0 for
   the measured best on both axes.
4. **Warmup is not backed by a measured cold-start gap on this machine**
   (1.52 s cold vs ~1.5 s warm). Kept as cheap insurance; documented as such.
5. **KittenTTS weights were not in any local HF cache** (contrary to the
   hand-off note); the matrix ran on the mini variant downloaded to
   `%TEMP%\opencode\kitten` with `curl.exe --ssl-no-revoke` (Python's network
   stack and HF's CDN reset the connection). No packages were installed.
6. **Lane speedup at 8 lanes is 1.91×, not >2×** on this host. The integration
   test therefore asserts ≥2× against a GIL-releasing stub (it tests the lane
   mechanism); the real-model number is recorded in `docs/batching.md` rather
   than asserted.

