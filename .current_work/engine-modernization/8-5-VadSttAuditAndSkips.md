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

- [ ] Whisper handler verified/converted to one batched call per step
- [ ] Silero batching skip evidenced (single-stream API); Silero lanes verified, measured, verdict recorded
- [ ] KittenTTS parallel lanes implemented (max_batch_size = lane count, no new config key)
- [ ] Lane/`intra_op` measurement matrix run; fixed value baked in; RTF recorded in `docs/batching.md`
- [ ] Load-time warmup landed (first-request cold start gone)
- [ ] GPU-EP probe verdict recorded (DirectML / CUDA-EP coexistence, or explicit "not pursued" with reason)
- [ ] Chatterbox exclusion documented as an explicit decision
- [ ] `docs/batching.md` matrix complete for every model type
- [ ] `summary.md` Implementation Status updated
