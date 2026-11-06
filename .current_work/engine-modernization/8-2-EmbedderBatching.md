# Phase 8-2: Embedder Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1; composes with 7-2 (cache hits leave the batch)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, vendored model trees under `harmonyspeech/modeling/models/{openvoice,harmonyspeech}/`
> **State reference:** handler names verified 2026-11-05

## Objective

Batch the speaker-embedding steps so a flood of clone requests with **distinct** voices (the cache-miss path) also moves through the embed step in one forward: `_execute_openvoice_tone_converter_encoder` (V1+V2) and `_execute_harmonyspeech_encoder` move from per-item loops to padded batched calls.

## Context (self-contained)

- Both handlers consume audio (the OpenVoice TCE consumes **VAD-segmented** audio — per request there may be multiple segments, mirroring the `(request, sentence)` rows of 5-2's synthesizers; HarmonySpeech encoder consumes the full clip) and emit `SpeakerArtifact.embedding`.
- **Verify-first (mandatory):** re-read the vendored encoder forwards before planning surgery —
  - OpenVoice `SpeakerEncoder`/TCE encoder (`modeling/models/openvoice/`): a `resnet`/`lstm`-style ref encoder over mel frames. Determine whether `forward` accepts `[B, T]` (batch-first mel) or is written single-sample; whether segment batching is supported upstream or must be added (our code — same pattern as 5-2's `_infer_impl` extraction).
  - HarmonySpeech encoder (`modeling/models/harmonyspeech/`): same questions. If its checkpoint graph is single-sample-only and surgery is disproportionate, record the verdict and leave it a documented loop (honest value stays 1) — an explicit, evidence-based skip is an acceptable outcome.
- Embedders are small; the win is **step latency under flood** (20 requests × per-item forward → 1–2 padded forwards), not single-request latency.

## Vendored forward audit

> Audited 2026-11-06 against the working tree (line numbers are post-change; the
> pre-change `ReferenceEncoder.forward` was ``openvoice.py:317``). Both families
> **do** support batching — neither is a documented skip — but neither could be
> reused untouched for padded rows, for the same reason: both consume every
> frame they are given and derive the embedding from the *final recurrent
> hidden state*, so trailing padding is not "harmless extra context", it changes
> the answer.

### OpenVoice V1/V2 tone-converter encoder

- **Class.** `OpenVoiceV1ToneConverterEncoder` / `OpenVoiceV2ToneConverterEncoder`
  are registered as `("openvoice", "SynthesizerTrn")`
  (`harmonyspeech/modeling/models/__init__.py:14`, `:18`). The embed step uses
  exactly one submodule of it: `SynthesizerTrn.ref_enc`, built as
  `ReferenceEncoder(spec_channels, gin_channels)` when `n_speakers == 0`
  (`harmonyspeech/modeling/models/openvoice/openvoice.py:428`; the converter
  config has no speaker table). `spec_channels` comes from
  `filter_length // 2 + 1` (`harmonyspeech/modeling/loader.py:378`).
- **Forward signature (before this task).**
  `ReferenceEncoder.forward(self, inputs, mask=None)` (`openvoice.py:389`):
  - accepts `[N, Ty, n_mels]` — mel frames, batch-first
    (`inputs.view(N, 1, -1, self.spec_channels)`) — so `N > 1` was already
    accepted by the graph;
  - the `mask` argument is **never read** (a dead upstream parameter), and the
    GRU runs over every frame present;
  - returns `self.proj(gru_out.squeeze(0))` → `[N, gin_channels]`: the embedding
    is the GRU's **final hidden state**, so padding is part of the sequence.
- **Verdict.** Batch-capable per row, **not** usable with right-padded rows:
  measured on random weights, a short row batched with longer ones through
  `forward` differs from its solo run by ~1.2e-2 (a padded 64-frame row next to
  a 100-frame row). A batched variant was required.
- **Chosen path.**
  `ReferenceEncoder.forward_batched(inputs, frame_lengths=None)`
  (`openvoice.py:397`), sharing a `_encode` extraction (`openvoice.py:344`) with
  the untouched `forward`:
  - rows padded on the **mel-frame** axis (`frame_lengths` = pre-conv frame
    counts);
  - each row's padding region is **zeroed after every conv**
    (`_zero_padding`, `openvoice.py:34`). This is the non-obvious part: a solo
    conv only ever sees its own frames plus implicit zero padding, while a padded
    row would otherwise hand the next layer's edge frames the conv **bias**
    (`relu(conv(0)) ≠ 0`), which drifted the embedding by up to 6.5e-4 before the
    masking was added;
  - the GRU runs over the valid frames only (`pack_padded_sequence`, valid count
    from `_downsample_step`, which mirrors `calculate_channels`);
  - a batch whose rows are all full length (including `B == 1`) routes to
    `forward`, so the historical path is bit-identical.
- **Frontend stays per row.** The handler's STFT is `spectrogram_torch`
  (`openvoice/mel_processing.py:40`), which `F.pad(..., mode="reflect")`s each
  clip before `torch.stft` (`mel_processing.py:52`); a padded waveform batch
  would reflect off the pad and change the shorter rows' last frames, so the mel
  frontend is not batched and the batch is built on its output.
- **Row unit.** `(request, VAD segment)`: `prepare_openvoice_tone_converter_encoder_inputs`
  returns *a list of* `AudioArtifact`s per request
  (`harmonyspeech/task_handler/inputs.py:422`), i.e. the same `(request,
  sentence)` shape as 5-2.
- **Aggregation to reproduce.** Today's loop stores
  `ref_enc(...).unsqueeze(-1)` → `[1, gin, 1]` per segment and folds them with
  `torch.stack(gs).mean(0)`. The batched path stores each row as
  `g[row].detach().unsqueeze(-1).unsqueeze(0)` in segment order and calls the
  **same** `torch.stack(...).mean(0)`, so the arithmetic is bit-identical.

### HarmonySpeech encoder

- **Class.** `HarmonySpeechEncoder` → `("harmonyspeech", "SpeakerEncoder")`
  (`harmonyspeech/modeling/models/__init__.py:20`).
- **Forward signature (before this task).**
  `SpeakerEncoder.forward(self, utterances, hidden_init=None)`
  (`harmonyspeech/harmonyspeech.py:166`): `utterances` is
  `[B, n_frames, n_channels]`, LSTM `batch_first=True`, then `hidden[-1]` → ReLU
  → `Linear` → L2-normalise. It is **already batch-capable** and the handler has
  always called it with one row per partial utterance (all partials same length).
- **Verdict.** Reusable for equal-length rows, **not** for padded rows: the LSTM
  consumes every frame, and the embedding is the final hidden state, so a
  right-padded row differs from its solo run by ~1.2e-2. `hidden_init` exists
  but cannot mask. So: reuse the graph, add a length-aware entry point — no
  surgery on the math, and **no evidenced skip needed**.
- **Chosen path.**
  `SpeakerEncoder.forward_batched(utterances, frame_lengths=None,
  hidden_init=None)` (`harmonyspeech.py:180`). Equal-length rows (the only shape
  this frontend produces) route straight to `forward` (bit-identical, `B == 1`
  included); mixed lengths pack the LSTM and take `hidden[-1]` from the packed
  run.
- **Frontend stays per row, and no waveform padding is needed.**
  `preprocess_wav` + `get_input_frames`
  (`harmonyspeech/encoder/inputs.py:66`) run per clip, and
  `compute_partial_slices(..., partial_utterance_n_frames=160)`
  (`encoder/inputs.py:6`) yields partials of **fixed** length, so `get_input_frames`
  returns `[P, 160, n_mels]`. The row unit is therefore `(request, partial)` on
  the frame axis; `pad_feature_batch(..., return_lengths=True)` degenerates to a
  plain stack. This is a deliberate deviation from the plan sketch ("full clip
  per request … on the waveform axis"): padding waveforms would buy nothing (the
  mel frontend is numpy and per clip) and would reintroduce exactly the
  padded-row hazard the model has. Recorded under Deviations.
- **Aggregation to reproduce.** Today: `partial_embeds = model(...)` →
  `[P, emb]`, `np.mean(partial_embeds, axis=0)`, then
  `raw / np.linalg.norm(raw, 2)`. The batched path reassembles each request's
  rows into `np.stack([...])` in partial order and calls the same
  `np.mean(axis=0)` / norm through one shared helper
  (`_finalize_encoder_embedding`, `task_handler/model_runner_base.py:218`).

### Evidence summary

| Check | Result |
|---|---|
| OV `forward_batched` `B=1` vs `forward` | bit-identical (`torch.equal`) |
| OV mixed-length rows in one padded batch vs solo | ≤1e-8 (float32 accumulation) |
| HS `forward_batched` `B=1` / equal-length batch vs `forward` | bit-identical |
| HS mixed-length rows (packed) vs solo | ≤6e-8 |
| Padded row through the *plain* forward (why masking/packing is needed) | ~1.2e-2 |
| OV conv padding region **not** zeroed between layers | up to 6.5e-4 |

Tests: `tests/unit/modeling/test_batched_embedders.py` (B=1 bit-equality,
mixed-length correctness, padding-independence, aggregation equivalence).

## Implementation steps

1. **Vendored forward audit** (both families): document the exact signature, whether lengths are accepted, and the batch-capable path. Add a batched variant only if missing, mirroring the single-sample implementation exactly (no math changes), with B=1 bit-equality unit tests (5-2 precedent).
2. **Handler batching**: build `(request, segment)` rows → `pad_feature_batch`/`pad_waveform_batch` (per 8-1, on the axis the model wants) → `plan_batches` with the appropriate lengths → single forward per sub-batch → split outputs by lengths → one `SpeakerArtifact.embedding` per request (OpenVoice: aggregate per-request segment embeddings exactly the way the loop does today — re-read the aggregation (mean/stack) and reproduce it bit-for-bit).
3. **Config**: extend `supports_batched_inference` for the two model types; `batched_inference` default **on** only after the gate passes; fallback loop for `len==1` and on exceptions.
4. **Cache composition** (7-2): resolve cache hits first; only misses enter the batch; outputs reassembled in request order.
5. **Gate**: `batching:1-vs-4` axis for the full clone pipelines whose embed step changed (openvoice_v1, openvoice_v2 — clone cases; the embedding output feeds synthesis, so the end-to-end audio comparison covers it; add an embedding-tensor equality assert for the embed-only path).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/openvoice/openvoice.py` | `ReferenceEncoder.forward_batched` + `_encode`/`_zero_padding` extraction |
| `harmonyspeech/modeling/models/harmonyspeech/harmonyspeech.py` | `SpeakerEncoder.forward_batched` + `_encode`/`_embed` extraction |
| `harmonyspeech/task_handler/model_runner_base.py` | batched `_execute_openvoice_tone_converter_encoder`, `_execute_harmonyspeech_encoder`, shared `_finalize_encoder_embedding` |
| `harmonyspeech/task_handler/batching.py` | `BATCHED_INFERENCE_FAMILIES` += the three embedder model types |
| `scripts/audio_regression.py` | `openvoice_v2` moved from `BATCHING_AXIS_PENDING` to `BATCHING_AXIS_ENABLED` |
| `tests/unit/modeling/test_batched_embedders.py` (new), `tests/unit/inference_flow/test_batched_embedder_handlers.py` (new) | vendored-forward and handler tests |
| `tests/integration/test_batched_synthesis.py`, `tests/unit/benchmarks/test_audio_regression.py`, `tests/unit/initialization/test_config.py`, `tests/unit/modeling/test_batched_synthesis.py` | extended |
| `docs/batching.md`, `docs/models.md` | batch-capability table + embedder section |

## Tests

- Unit: vendored batched forward B=1 == loop path (bit-identical); segment aggregation equivalence (N segments sequentially vs padded batch → same embedding within tolerance / bitwise if the op order allows).
- Integration: 4 clone requests with distinct references → correct per-request embeddings (pairing invariants), embeddings equal to loop-computed reference values.
- Harness: `batching:1-vs-4` green for openvoice_v1/v2 clone cases.

## Checklist

- [x] Vendored forward audit documented (both families; skip verdicts recorded if any) — both batched, no skip; see "Vendored forward audit"
- [x] Both handlers batched (or evidence-based skip documented) with loop fallback — `_execute_openvoice_tone_converter_encoder` + `_execute_harmonyspeech_encoder`, per-request loop kept for `len == 1` and as the exception fallback
- [x] Segment aggregation bit-faithful; B=1 gates green — same `torch.stack(...).mean(0)` / `np.mean(axis=0)` calls on reassembled rows; `tests/unit/modeling/test_batched_embedders.py` + `tests/unit/inference_flow/test_batched_embedder_handlers.py`
- [x] Cache-hit composition verified (hits never enter the batch) — `test_*_batch_of_four_with_two_hits_only_forwards_the_misses`; hits resolved before rows are built
- [x] `summary.md` Implementation Status updated

## Deviations

1. **HarmonySpeech row unit is the mel-frame axis, not the waveform axis.**
   The plan sketch said "full clip per request → pad on the waveform axis".
   The audit showed the vendored `SpeakerEncoder` consumes
   `[B, frames, mels]`, its frontend (`preprocess_wav` + `get_input_frames`) is
   numpy and per clip, and `compute_partial_slices` emits **fixed-length**
   partials. Rows are therefore `(request, partial)` padded on the frame axis,
   which needs no waveform padding at all and keeps the frontend unbatched (it
   cannot be batched bit-faithfully: the frontend's reflection/partial split is
   per clip). Reassembly and averaging are unchanged.
2. **The OpenVoice mel frontend stays per segment.** `spectrogram_torch`
   reflect-pads each clip (`mel_processing.py:52`), so a padded waveform batch
   would change the shorter rows' last frames. The batch is built on the mel
   frames the frontend already produces; `pad_waveform_batch` is therefore not
   used by this task (`pad_feature_batch(..., return_lengths=True)` is).
3. **Batch-capability is declared before the batch *capacity*.** The three model
   types were added to `BATCHED_INFERENCE_FAMILIES` (so `batched_inference`
   defaults on), but the shipped `config.gpu.yml`/`config.yml` keep
   `max_batch_size: 1` for the encoders — 8-6 owns the truthful values, and the
   task brief forbids touching the configs here. Consequence: a multi-request
   step batches the embed rows, while normal operation still admits one request
   per step. `tests/unit/initialization/test_config.py` was updated to assert
   `max_batch_size > 1` only for the Phase 5-2 synthesis families and to state
   the 8-6 deferral for the embedders (it previously asserted it for *every*
   batched family, which is exactly the transitional state this task creates).
4. **No real-model family-gate run on GPU in this task.** The
   `batching:1-vs-N` harness axis is now allow-listed for `openvoice_v2`
   (`BATCHING_AXIS_ENABLED`), and the harness tests are green, but the actual
   GPU gate run against downloaded weights is deferred to the benchmark host
   exactly like Phase 7-3. The unit tier backs it: B=1 bit-equality for both
   vendored forwards, mixed-length row equality ≤1e-8, plus handler-level
   loop-vs-batch equality for both embed steps and the 4-clone-request engine
   test.
5. **Embedding-tensor equality assert (plan step 5).** Covered at the engine
   level instead of inside the harness worker: the harness already compares the
   rendered audio of the full clone pipeline (which cannot pass if the embedding
   differs), and per-request embedding equality (batched vs loop) is asserted in
   `tests/integration/test_batched_synthesis.py`
   (`test_four_clone_requests_with_distinct_references_pair_their_embeddings`).
   Adding a torch-level assert to the subprocess harness would require shipping
   tensors across the worker boundary for no extra coverage.
6. **`ReferenceEncoder.forward`'s `mask` parameter is dead upstream code** and
   was deliberately left untouched (documented in the audit); the batched path
   masks explicitly instead.

