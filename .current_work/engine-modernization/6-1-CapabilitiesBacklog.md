# Phase 6-1: Capabilities Backlog (opt-in)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 6 — Capabilities on the new foundation
> **Status:** overview only — each item is planned in detail **when picked** for a release. Do not implement from this file alone.

After Phases 0–5, each of these becomes a small, well-seamed addition. Prerequisites and design anchors from the research document (§ references) are recorded here so a future planning session starts warm.

## 6-A `post_generation_filters` as pipeline steps

- **What:** execute the parsed-but-dead `post_generation_filters` field (`common/inputs.py:142`, engine `:504` TODO) as declarative post-steps: "TTS → VoiceFixer" in one request, no extra HTTP call, no intermediate WAV.
- **Prereqs:** Phase 3 (specs; the extension point is already documented at `advance()`).
- **Anchors:** research §5.3(e); spec step-append per request at admission.
- **Rough size:** ~days once specs are in.

## 6-B `RemoteStepExecutor`

- **What:** an executor type that runs one pipeline step against an external OpenAI-compatible engine (vllm-omni / sglang-omni container) over HTTP — never an in-process import (§4.3: the 823-line patch/dependency clash makes imports a non-starter).
- **Prereqs:** Phase 0-5 per-request timeouts (**hard** — a remote step must never live under the loop watchdog), Phase 3 (step types).
- **Anchors:** research §6.2 Phase 6 config sketch + guardrails (health probe with cached result, one-retry policy, documented "not supported on single 16 GiB card").
- **Rough size:** ~1 week.

## 6-C `torch.compile` / CUDA-graph capture on fixed-shape modules

- **What:** compile/capture the vocoder stack (S3Gen decoder, HiFiGAN upsample, VoiceFixer UNet, `ref_enc`) with T-bucketing + pad; expected 10–30 % on vocoder-bound steps [research estimate].
- **Prereqs:** Phase 4-2 harness (the quality gate), Phase 2 (artifacts make shapes explicit). **Never** capture Chatterbox T3 (autoregressive, no KV — research §5.3(b), §6.5).
- **Anchors:** research §5.3(b) capture table; `gpu_model_runner.py` already reserves the `capture_model` seam.
- **Rough size:** ~2–3 weeks, per model, gated.

## 6-D Lazy-load memory governor

- **What:** `resident: true|false|auto` per model, lazy load on first routed request, LRU eviction under a `vram_budget_bytes` ceiling, loud 503 when it cannot fit, load-miss metrics.
- **Prereqs:** Phase 4-5 metrics (the gauges), 4-3 registry (`release()` is already wired for this).
- **Anchors:** research §6.2 Phase 3 items 1–4, risk #9 — **resident-by-default, warmup-on-load (dummy forward at load time, not first-request time), pin the hot set, evict only cold**. After Phase 4 the full catalogue fits (~6 GiB), so the governor is an edge-case tool (growing catalogues), not a necessity.
- **Rough size:** ~2 weeks.

## Explicitly deferred (separate workstreams — recorded for completeness)

- **Streaming output** (`stream=true`, today a 400): blocked on the sentence-chunking QOL workstream; needs chunked synthesis + SSE/PCM transport design (research §6.2 Phase 4).
- **Chatterbox AR batched execution**: needs per-sequence KV/continuous-batching machinery (mini-vLLM). Revisit only on demonstrated throughput need; scale by replicas meanwhile.
- **Worker concurrency (soulbits-cloud-backend)**: handled at deployment level, out of HSE scope.

## Checklist

- [ ] (when an item is picked) create its detailed phase file(s) `6-x-*` following the same structure, update this file's status and [summary.md](summary.md)
