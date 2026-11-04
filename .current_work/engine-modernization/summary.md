# HSE Engine Modernization — Implementation Plan

**Created:** 2026-11-04
**Source research:** `../soulbits-cloud-backend/.current_work/inference-engine-re-eval-26-11/11-hse-gateway-vs-executor-analysis.md` (HSE gateway-vs-executor analysis, 2026-11-02) and `00-SYNTHESIS.md` in the same folder.
**Codebase mapping consulted:** `.planning/codebase/{ARCHITECTURE,STRUCTURE,CONVENTIONS,TESTING,CONCERNS}.md`

## Overview

This plan modernizes the Harmony Speech Engine's internals in a foundation-first order. It deliberately defers **new capabilities** (streaming, post-generation filters, remote executors) until the engine is *safe to build on*: free of deserialization RCE, request-state leaks, request-mispairing bugs, and hang-forever scheduling, with an honest wire contract, tensor-native pipeline steps, declarative routing, a truthful VRAM footprint, and honest batching signals.

The architectural targets are taken verbatim from the research document (§6 roadmap): keep HSE as the single-process multi-model host, replace base64 intermediates with in-process artifacts, replace `if/elif` routing with a declarative `PipelineSpec`, and make the 16 GiB residency story a measured fact (fp16 + Chatterbox backbone dedup: ~16.4 GiB → ~6 GiB) rather than an accident.

## Ordering rationale (gates)

- **Phase 0 blocks Phases 4 & 5**: sharing backbones or batching requests amplifies the existing `conds`/`ref_dict` state leaks and the request-mispairing closure bug unless they are fixed first.
- **Phase 1 before Phase 2**: the boundary decoder/encoder becomes the `from_wire()`/`to_wire()` seams the artifact bus keeps — writing them in the other order means doing them twice.
- **Phase 2 before Phase 3**: `spec.advance()` should write typed artifacts, not base64 fields; otherwise the routing rewrite touches the forwarding code twice.
- Phases 4 and 5 are independent of each other once 0–3 are done.

## Phases

| Phase | Title | Effort | Addresses |
|---|---|---|---|
| **0** | Correctness & security foundation | ~days | RCE via `torch.load(weights_only=False)`, cross-request voice leaks (`model.conds`/`ref_dict`), request-mispairing closures, hang-forever scheduling, the 120 s loop-killing timeout, `check_health` bug |
| **1** | Wire contract — input decode & output formats | ~days | Output `format`/`sample_rate` silently ignored; undocumented input codec support; centralized decode with ffmpeg fallback; binary (multipart) upload envelope |
| **2** | Artifact bus | 1–2 wks | base64/WAV/pickle intermediates between pipeline steps; per-hop re-decode & resample; ~35–45 MB host churn on the VoiceFixer hop |
| **3** | Declarative `PipelineSpec` routing | ~1 wk | 250 lines of field-presence routing if/elif; output-class-sniffing forwarding; decorative `input_vad_mode`; `single_speaker_tts` FIXME; advertised-but-incomplete toolchains |
| **4** | VRAM — fp16 rollout + Chatterbox weight dedup | 1–2 wks | All-fp32 config (~16.4 GiB); 4× duplicated Chatterbox backbones; no VRAM observability |
| **5** | Honest batching — HSE only | ~1 wk | Fake `max_batch_size` values; per-item loops; type-unsafe mixed batches; padded batched inference for `SynthesizerTrn` |
| **6** | Capabilities backlog (opt-in) | per item | `post_generation_filters`, `RemoteStepExecutor`, `torch.compile`/CUDA graphs, lazy-load governor — planned in detail only when picked |

## Implementation Status

Track the completion of each phase as implementation progresses:

- [x] **Phase 0: Correctness & security foundation** — **COMPLETE 2026-11-04** (unit 162 / integration 23 green; E2E 39 collected, model-download run outstanding). **Gate for Phases 1–5 is open.**
  - [x] Safe deserialization — [0-1-SafeDeserialization.md](0-1-SafeDeserialization.md)
  - [x] Per-request state isolation — [0-2-PerRequestStateIsolation.md](0-2-PerRequestStateIsolation.md)
  - [x] Batch pairing correctness — [0-3-BatchPairingCorrectness.md](0-3-BatchPairingCorrectness.md)
  - [x] Startup validation & no-hang guarantees — [0-4-StartupValidationNoHang.md](0-4-StartupValidationNoHang.md)
  - [x] Timeouts & health — [0-5-TimeoutsAndHealth.md](0-5-TimeoutsAndHealth.md)
  - [x] Phase 0 exit review — [0-6-Phase0ExitReview.md](0-6-Phase0ExitReview.md)
- [x] **Phase 1: Wire contract — input decode & output formats** — **COMPLETE 2026-11-04** (unit 281 / integration 23 green; wire-contract e2e 38 passed / 3 skipped without model weights; real-model e2e run outstanding).
  - [x] Boundary audio decoder — [1-1-BoundaryAudioDecoder.md](1-1-BoundaryAudioDecoder.md)
  - [x] Output format & sample-rate honoring — [1-2-OutputFormatSampleRate.md](1-2-OutputFormatSampleRate.md)
  - [x] Binary input envelope (multipart) — [1-3-BinaryInputEnvelope.md](1-3-BinaryInputEnvelope.md)
  - [x] Wire-contract tests & docs — [1-4-WireContractTestsDocs.md](1-4-WireContractTestsDocs.md)
- [ ] **Phase 2: Artifact bus**
  - [ ] Artifact type system — [2-1-ArtifactTypeSystem.md](2-1-ArtifactTypeSystem.md)
  - [ ] Artifact fields & prepare_inputs — [2-2-ArtifactFieldsAndPrepareInputs.md](2-2-ArtifactFieldsAndPrepareInputs.md)
  - [ ] Executor artifact handlers — [2-3-ExecutorArtifactHandlers.md](2-3-ExecutorArtifactHandlers.md)
  - [ ] Boundary serialization & compatibility — [2-4-BoundarySerializationCompat.md](2-4-BoundarySerializationCompat.md)
  - [ ] Verification & benchmark — [2-5-ArtifactBusVerification.md](2-5-ArtifactBusVerification.md)
- [ ] **Phase 3: Declarative PipelineSpec routing**
  - [ ] PipelineSpec model & registry — [3-1-PipelineSpecModel.md](3-1-PipelineSpecModel.md)
  - [ ] Family migration — [3-2-FamilyMigration.md](3-2-FamilyMigration.md)
  - [ ] Startup validation & model cards — [3-3-StartupValidationModelCards.md](3-3-StartupValidationModelCards.md)
  - [ ] VAD switch & entry points — [3-4-VadSwitchEntryPoints.md](3-4-VadSwitchEntryPoints.md)
  - [ ] Routing tests — [3-5-RoutingTests.md](3-5-RoutingTests.md)
- [ ] **Phase 4: VRAM — fp16 + Chatterbox weight dedup**
  - [ ] dtype plumbing & config flip — [4-1-DtypePlumbingConfigFlip.md](4-1-DtypePlumbingConfigFlip.md)
  - [ ] Audio regression gate — [4-2-AudioRegressionGate.md](4-2-AudioRegressionGate.md)
  - [ ] Weight registry — [4-3-WeightRegistry.md](4-3-WeightRegistry.md)
  - [ ] Chatterbox backbone sharing — [4-4-ChatterboxBackboneSharing.md](4-4-ChatterboxBackboneSharing.md)
  - [ ] VRAM metrics & profile — [4-5-VramMetricsProfile.md](4-5-VramMetricsProfile.md)
- [ ] **Phase 5: Honest batching (HSE only)**
  - [ ] Type-checked batches & pairing invariants — [5-1-TypeCheckedBatches.md](5-1-TypeCheckedBatches.md)
  - [ ] Padded batch on SynthesizerTrn — [5-2-SynthesizerTrnPaddedBatch.md](5-2-SynthesizerTrnPaddedBatch.md)
  - [ ] Honest batch config & benchmarks — [5-3-HonestBatchConfigBenchmarks.md](5-3-HonestBatchConfigBenchmarks.md)
- [ ] **Phase 6: Capabilities backlog (opt-in)**
  - [ ] Overview & prerequisites — [6-1-CapabilitiesBacklog.md](6-1-CapabilitiesBacklog.md)

## Deferred to separate workstreams (not in this plan)

- **Streaming output** (`stream=true`) — after the sentence-chunking QOL workstream lands.
- **Chatterbox AR batched execution** — needs a per-sequence KV-cache story (mini-vLLM); revisit only if throughput demands it. Interim honesty fix is Phase 5-3.
- **Worker concurrency on workload machines** — already being raised at deployment level (soulbits-cloud-backend side).

## Conventions for executors

- Follow `.roo/rules/workspace-rules.md`: re-read files before editing; do not build/test mid-task (manual verification after task completion); update `docs/`, `README.md`, `CHANGELOG.md` (no implementation details in changelog entries).
- After completing a subtask: tick its checklist, note deviations in the phase file, and update this summary's Implementation Status.
- Line numbers referenced in phase files were verified on 2026-11-04 and may drift; always re-read the target file before editing.
