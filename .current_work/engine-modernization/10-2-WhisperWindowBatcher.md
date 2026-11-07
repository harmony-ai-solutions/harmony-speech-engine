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

- [ ] CT2/faster-whisper API table recorded here (verified signatures)
- [ ] Window collector + one-encode/one-generate step + stitching landed
- [ ] Parity gate B=1 vs `pipeline.transcribe` green
- [ ] Fallbacks wired (`len==1`, flag, failure)
- [ ] `summary.md` Implementation Status updated
