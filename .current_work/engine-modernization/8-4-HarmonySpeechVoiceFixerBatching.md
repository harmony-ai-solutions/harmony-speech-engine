# Phase 8-4: HarmonySpeech & VoiceFixer Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-1
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; vendored trees `harmonyspeech/modeling/models/{harmonyspeech,voicefixer}/`; Phase 4 leftovers (VoiceFixer GPU placement bug)
> **State reference:** handler names verified 2026-11-05; audit + implementation 2026-11-06

## Objective

Batch the remaining one-shot, fixed-graph steps: the HarmonySpeech pipeline (synthesizer + vocoder; its encoder is handled in 8-2) and the VoiceFixer pipeline (restorer + vocoder). Also fix the pre-existing **VoiceFixer GPU device-placement bug** (`input cuda vs weight cpu`, flagged since Phase 4) — it blocks this family's GPU gate entirely.

## Context (self-contained)

- `_execute_harmonyspeech_synthesizer` produces a `MelArtifact` (+ `breaks`) per request; `_execute_harmonyspeech_vocoder` turns mels into audio. The synth is expected to be VITS/`SynthesizerTrn`-family (the 5-2 pattern should transfer almost verbatim); **verify-first** the vendored signature (`modeling/models/harmonyspeech/`) — if it is already the same `SynthesizerTrn` used by OpenVoice V1, this may be a near-copy of 5-2's handler change.
- `_execute_voicefixer_restorer` runs a UNet over mel (input via `MelArtifact`/audio per Phase 2 — no JSON path remains); `_execute_voicefixer_vocoder` is a conv vocoder over mel. UNets and conv vocoders are batch-friendly with frame padding; **verify-first** both vendored forwards for `[B, bins, T]` acceptance and length semantics.
- VoiceFixer GPU bug: the restorer's weights stay on CPU while its input lands on CUDA (loader moves only part of the model). Fix in the loader/runner init (consistent with how the other GPU runners place their models) — it's a prerequisite for running this family's gate on GPU; CPU runs are unaffected.
- Both families currently run honest `max_batch_size: 1` loops.

## Vendored forward audit

> Audited 2026-11-06 against the working tree (line numbers are post-change). Three
> of the four forwards were batch-able with the Phase 8-2/8-3 padding isolation;
> the HarmonySpeech **synthesizer** is a documented skip — it is not the
> `SynthesizerTrn` this plan expected, and its generative path runs five coupled
> bidirectional modules over the padded axis.

### 1. HarmonySpeech synthesizer — `ForwardTacotron` (**documented skip**)

- **Class/registration.** `HarmonySpeechSynthesizer` → `("harmonyspeech",
  "ForwardTacotron")` (`harmonyspeech/modeling/models/__init__.py:21`). Config:
  `synthesizer/config.json`; the shipped GPU config pins it to CPU.
- **Exact entry point.** `ForwardTacotron.generate(self, x, spk_emb, alpha=1.0,
  pitch_function=..., energy_function=...)` (`harmonyspeech.py:386`) →
  `_generate_mel(...)` (`harmonyspeech.py:420`). `x` is `[B, T_tokens]`, `spk_emb`
  `[B, 768]`; the handler calls it with `B == 1` per request and consumes
  `mels = x_post` (`[B, n_mels, frames]`).
- **Acceptance of `[B, T]`.** The graph *does* accept `B > 1`: `dur_pred` /
  `pitch_pred` / `energy_pred` (`SeriesPredictor.forward`, `harmonyspeech.py:211`)
  are `Embedding` → 3× `BatchNormConv` → bidirectional GRU → `Linear`, and
  `LengthRegulator.forward` (`harmonyspeech.py:21`) expands each row by its own
  predicted durations and `pad_sequence`s the batch.
- **Padding traps (all five, cumulative).**
  1. `SeriesPredictor`'s bidirectional GRU consumes every padded token; its
     per-token outputs then feed `pitch_proj`/`energy_proj` (kernel 3) *before*
     the length regulator, so token padding reaches valid frames.
  2. `CBHG.forward` (`harmonyspeech.py:63`, the prenet) runs a conv bank with
     `kernel = 1..K`, a `MaxPool1d(k=2, stride=1, padding=1)` and a second
     bidirectional GRU over the token axis.
  3. `LengthRegulator` expands padded tokens (garbage durations) unless they are
     zeroed first, and pads the frame axis to the batch's longest expanded row.
  4. `_generate_mel` runs `self.lstm(x)` — a **bidirectional LSTM without a
     length argument** (`harmonyspeech.py:445`) — over that padded frame axis.
  5. The postnet `CBHG` (`harmonyspeech.py:450`) mixes frames through a conv bank
     and a bidirectional GRU before `post_proj`.
- **Measured leak.** Tiny instance (`n_mels=6`, 7-token and 12-token rows in one
  padded call): the long row matches its solo run to 6e-8, the **short row's own
  frames differ by 0.155 peak** — the padded frames change its answer, exactly as
  in 8-2/8-3. Pinned by
  `tests/unit/modeling/test_batched_hs_voicefixer.py::test_harmonyspeech_synthesizer_generate_takes_a_batch_but_padding_leaks`.
- **Verdict: skip.** Making this padding-safe means threading masks/packing
  through `SeriesPredictor`, `CBHG` (twice), `LengthRegulator` and the frame LSTM
  — five coupled submodules of the vendored generative path — for a CPU-pinned
  model. The handler stays a per-item loop, `HarmonySpeechSynthesizer` is **not**
  in `BATCHED_INFERENCE_FAMILIES`, and `docs/batching.md` records the reason.

### 2. HarmonySpeech vocoder — `MelGANGenerator` (**batched**)

- **Class.** `HarmonySpeechVocoder` → `("harmonyspeech", "MelGANGenerator")`
  (`models/__init__.py:22`).
- **Signature (before).** `inference(self, c, normalize_before=False)`
  (`harmonyspeech.py:713`): `c` is `[T, in_channels]`, transposed to
  `[1, C, T]` and run through `self.melgan`; returns `[T*prod(upsample_scales), 1]`.
- **Padding traps.** `self.melgan` starts and ends with
  `ReflectionPad1d((kernel-1)//2)` + `Conv1d(kernel)`; between them are
  `ConvTranspose1d(stride=u, kernel=2u)` stages and `ResidualStack`s whose
  dilated convs (`dilation = 3**j`, pad `(k-1)//2*d`) read up to `d` frames past
  a row's end. Unmasked, an 11-frame padded row's output drifted **2e-3 peak**.
- **Output length semantics.** Each transposed conv multiplies the time axis by
  exactly its stride (`out = T*u` for both even and odd `u`), so row `i`'s valid
  samples are `frames_i * prod(upsample_scales)` — read from the produced tensor
  (`output_len // frames`), not from a config constant.
- **Chosen path.** `MelGANGenerator.inference_batched(c, frame_lengths=None,
  normalize_before=False)` (`harmonyspeech.py:733`) returns `(waveform, sample_lengths)`.
  `B == 1`/all-equal rows call `self.melgan` unchanged (bit-identical); otherwise
  `run_masked` (`modeling/models/batch_padding.py`) multiplies the mask after
  every stage, folds `ReflectionPad1d` into its `Conv1d` so each row reflects its
  own tail, and `_masked_residual_stack` replays each `ResidualStack.stack`
  through the same helper. PQMF multi-band and causal variants raise → handler
  falls back to the loop.

### 3. VoiceFixer restorer — `VoiceFixerGenerator` UNet (**batched**)

- **Classes.** `VoiceFixerRestorer` (`voicefixer/voicefixer.py:157`) →
  `self.generator = VoiceFixerGenerator` (`:63`) → `denoiser` (BN + Linear + two
  `BN_GRU`) + `UNetResComplex_100Mb` (`modules.py:323`).
- **Signature (before).** `VoiceFixerRestorer.forward(self, audio_tensor)`
  (`:375`): `[B, C, samples]` in, `[B, 1, frames, 128]` out (`exp(log-mel)`);
  internally slices the clip into 30 s segments and concatenates the per-segment
  mels along `dim=3` (preserved verbatim, see Deviations 4). The mel frontend is
  `_pre` → `FDomainHelper.wav_to_spectrogram_phase` (centered STFT, reflect pad)
  → `MelScale`; `sp` (the magnitude spectrogram) is an **unused argument** of
  `VoiceFixerGenerator.forward` (audited; the batched path drops it).
- **Padding traps.** The denoiser's two `BN_GRU`s are **bidirectional over
  frames** — trailing padding changes every valid frame (the 8-2 lesson), so they
  are packed per row (`BN_GRU.forward_batched`, `modules.py:108`). The UNet then
  pads time to a multiple of 64 and runs 6 average-pool encoders / 6 pruned
  transposed-conv decoders with `BatchNorm2d` + `Conv2d(k=3)` at every level, so a
  padded row contaminates both the pooling boundary and the decoder tail:
  measured **26.7 peak** on a short row's own frames without isolation.
- **Output frame-count semantics.** `[B, 1, T_max, 128]`, cropped to the input's
  `T_max`; row `i`'s valid frames are its own `T_i` (the internal 64-frame
  alignment is not observable in the output).
- **Chosen path.** `VoiceFixerRestorer.preprocess(segment)` (`:333`, per-clip mel
  frontend) + `VoiceFixerRestorer.forward_batched(mels, frame_lengths)`
  (`:344`) → `VoiceFixerGenerator.forward_batched` (`:108`): pack the denoiser
  GRUs at `T_i`, mask `unet_in` to `T_i` (the historical path pads *after*
  building it, so a solo row's UNet input is plain zeros there), then
  `UNetResComplex_100Mb.forward_batched` (`modules.py:438`) with a
  `[B, 1, T, 1]` mask at each row's own `ceil(T_i/64)*64` extent, halved through
  the pools and doubled through the decoders (`ConvBlockRes`/`EncoderBlockRes`/
  `DecoderBlockRes.forward_batched`, `modules.py:195/:234/:299`). `B == 1`/all-
  equal rows call `forward` (bit-identical).

### 4. VoiceFixer vocoder — `VoiceFixerVocoderGenerator` (**batched**)

- **Classes.** `VoiceFixerVocoder` (`voicefixer/voicefixer.py:617`) →
  `self.model = VoiceFixerVocoderGenerator` (`:456`).
- **Signature (before).** `forward(self, mel_spectrogram)` (`:632`):
  `[B, 1, T, 128]` in; `/ weight_torch` → `tr_amp_to_db` → `tr_normalize` →
  `tr_pre` → `condnet` (5× `Conv1d(k=3)`) → `generator` (ReflectionPad + Conv1d →
  4× `UpsampleNet` + `ResStack` → ReflectionPad + Conv1d + Tanh); returns
  `[B, 1, T_cond * 441]`.
- **Length semantics.** `tr_pre` appends `frames % 2 + 4` condition frames **per
  row**, so the frontend cannot be applied to a padded batch: the condition tail
  and the resulting sample count (`T_cond * 441`) are row-specific.
- **Padding traps.** `ReflectionPad1d(3)` at both ends + transposed convs + dilated
  `ResStack`s; unmasked, a 10-frame row in a 14-frame batch drifted **0.28 peak**.
  `UpsampleNet`'s residual-skip branch is a second, subtler trap: it reshapes
  `(x + sin x).repeat(1, factor, 1)` from `[B, factor*C, T]` to `[B, C,
  factor*T]`, so its output frame `j` reads input frame `j % T` — the branch
  **wraps at the row's own frame count** and cannot be batched on padded rows at
  all. `_masked_upsample` (`voicefixer.py:24`) therefore replays it per row on
  that row's valid slice (a 1×1 conv) and masks once after the transposed conv.
- **Chosen path.** `VoiceFixerVocoder.forward_batched(mel, frame_lengths)`
  (`:661`) returns `(audio, sample_lengths)`: per-row `tr_pre` frontend, then
  `VoiceFixerVocoderGenerator.forward_batched` (`:575`) with `run_masked` over
  `condnet` + `generator` (`_VOCODER_MASKED_BLOCKS` = `UpsampleNet`, `ResStack`).
  All-equal rows call the plain graph (bit-identical).

### Evidence summary

| Check | Result |
|---|---|
| MelGAN `inference_batched` `B=1` vs `inference` | bit-identical (`torch.equal`) |
| MelGAN mixed lengths (5/11/7) vs solo | ≤8e-8 |
| MelGAN padding filled with 1e3 / -5e2 | identical output (padding-independent) |
| VoiceFixer vocoder `B=1` vs `forward` | bit-identical |
| VoiceFixer vocoder mixed lengths (5/9/7) vs solo | ≤3.3e-7 |
| VoiceFixer vocoder without masking (the trap) | 0.28 peak |
| Restorer `B=1` vs `forward` | bit-identical |
| Restorer mixed lengths (10/12/14 mel frames) vs solo | ≤1e-6 relative (`exp`-scaled) |
| Restorer 64-frame-misaligned rows (75/140 frames) | ≤2.2e-6 relative |
| Restorer without isolation (the trap) | 26.7 peak |
| HarmonySpeech synthesizer padded row (the skip) | 0.155 peak |

Tests: `tests/unit/modeling/test_batched_hs_voicefixer.py` (placement + all four
audits/B=1/mixed-length/padding-independence),
`tests/unit/inference_flow/test_batched_hs_voicefixer_handlers.py` (one forward per
batch, breaks pairing, per-request normalization, loop fallback, poisoned
request), and the 4-request engine tests in
`tests/integration/test_batched_synthesis.py`.

### VoiceFixer GPU device placement (verified mechanism)

- **Verified.** `torchlibrosa.stft.STFT.__init__` assigns
  `conv_real.weight.data = torch.Tensor(np.real(...))` / `conv_imag` likewise.
  `torch.Tensor(numpy)` is a CPU factory and `.data =` replaces the storage, so
  building the restorer inside the loader's `with torch.device(device)` block
  leaves exactly those two convs on the host:
  `{p.device for p in model.parameters()} == {"cuda:0", "cpu"}` (pinned by
  `test_voicefixer_restorer_built_under_a_cuda_context_leaves_the_stft_convs_on_the_host`).
- **Fix.** The loader's VoiceFixer branch now calls `model.to(device_config.device)`
  after construction *and* weight loading
  (`harmonyspeech/modeling/loader.py`, `_STFT_FRONTEND_MODEL_TYPES`). The STFT
  convs are registered parameters of the STFT module, so one recursive `.to()`
  places the whole tree — parameters, buffers and the STFT convs — on the
  configured device. On CPU it is a no-op, so CPU behaviour is unchanged.
- **Regression tests.** `get_model(..., load_format="dummy")` asserts every
  parameter/buffer (and the STFT convs) on CPU always and on CUDA when present,
  plus a real CUDA restorer forward (the bug's `input cuda vs weight cpu`
  symptom).

## Implementation steps

1. **VoiceFixer device fix**: locate the partial placement (loader `modeling/loader.py` VoiceFixer branch vs runner init), move the full model consistently, add a unit test asserting every parameter of the loaded (stubbed) model sits on the configured device. Re-run the family on GPU — it should now work at all (it hasn't since before Phase 4).
2. **Vendored audits** (hs1 synth/vocoder, voicefixer restorer/vocoder): document signatures; add batched variants only where missing, mirroring single-sample math exactly, with B=1 bit-equality tests (5-2 precedent).
3. **Handler batching** (four handlers): rows → pad on the mel-frame axis → `plan_batches` → batched forward → trim/unpack → per-request artifacts. Keep per-family quirks bit-faithful:
   - HarmonySpeech vocoder consumes `MelArtifact.breaks` (silence insertion points) — batching must not disturb the per-request breaks pairing (re-read the Phase 2 handler before editing; the breaks apply **after** decode, so pad/trim concerns don't touch them).
   - VoiceFixer restorer output mel feeds its vocoder as `MelArtifact` — trim per-request frames exactly.
4. **Config + gates**: `supports_batched_inference` += the four types; defaults on after gates; `batching:1-vs-4` harness axis extended to `harmonyspeech` (tts cases) and `voicefixer` (`restore_voice_a`) families.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/loader.py` (or runner init) | VoiceFixer full-device placement fix + test |
| `harmonyspeech/modeling/models/harmonyspeech/…`, `…/voicefixer/…` | batched forwards if missing (+ B=1 equality tests) |
| `harmonyspeech/task_handler/model_runner_base.py` | four batched handlers |
| `harmonyspeech/task_handler/batching.py` | supports-entries |
| `tests/unit/modeling/test_batched_hs_voicefixer.py` (new) | tests |

## Tests

- Unit: VoiceFixer placement (all params on configured device, stubbed model); B=1 bit-equality ×4 models; breaks pairing preserved under batching (hs1 vocoder).
- Integration: multi-request restoration + hs1 TTS with mixed-length inputs → per-request output lengths + pairing invariants.
- Harness: `batching:1-vs-4` green for `harmonyspeech` and `voicefixer` (GPU; the placement fix makes the VoiceFixer GPU axis runnable for the first time — record the family's dtype axis too while it's finally testable, appending to `docs/benchmarks/audio-regression.md`).

## Checklist

- [x] VoiceFixer GPU placement bug fixed + regression-tested (family runs on GPU) — loader `.to(device)` after construction/weight load; the STFT convs are registered parameters, so the whole tree moves; CPU no-op. Unit tests cover CPU always and CUDA (including a real or a dummy-weight forward); the actual GPU family gate run is deferred to the benchmark host (Deviations 1).
- [x] Four handlers batched (or evidence-based skip documented) with loop fallback — `_execute_harmonyspeech_vocoder`, `_execute_voicefixer_restorer`, `_execute_voicefixer_vocoder` batch; `_execute_harmonyspeech_synthesizer` is a documented skip (audit above) and keeps the per-item loop. All batched paths keep `len == 1`, `batched_inference: false` and per-sub-batch failure as loop fallbacks.
- [x] B=1 gates + family harness axes green — B=1 bit-equality and mixed-length equality at the model and handler tiers; `harmonyspeech` and `voicefixer` moved from `BATCHING_AXIS_PENDING` to `BATCHING_AXIS_ENABLED`; unit + integration tiers green, the GPU harness run itself is deferred (Deviations 1).
- [ ] VoiceFixer dtype axis recorded now that GPU runs work — **deferred** to the benchmark host (Deviations 2).
- [x] `summary.md` Implementation Status updated

## Deviations

1. **No real-model GPU gate run in this task.** The VoiceFixer placement bug is
   fixed and verified mechanically (every parameter/buffer/STFT conv on the
   configured device, plus a CUDA forward in the unit tier), and both families are
   in the `batching:1-vs-N` allow-list, but the harness run against downloaded
   weights is deferred to the benchmark host exactly like Phase 7-3/8-2/8-3. The
   unit and integration tiers back it: B=1 bit-equality, mixed-length rows equal
   to their solo runs, padding-independence, handler loop-vs-batch equality and
   the 4-request engine tests.
2. **The VoiceFixer dtype axis is deferred** (the checklist item above). Recording
   it needs the real weights on a GPU host; the placement unit test is what lands
   here. `voicefixer` is not in `DTYPE_AXIS_EXEMPT`, so it is a plain
   `--families voicefixer --axis dtype:float32-vs-float16 --gpu` run on the
   benchmark host once the weights are available (the doc table in
   `docs/benchmarks/audio-regression.md` now says "not re-run" instead of
   "blocked by the placement bug").
3. **The HarmonySpeech synthesizer is a documented skip**, not a batched handler:
   the vendored forward is `ForwardTacotron` (three `SeriesPredictor`s, a `CBHG`
   prenet, a duration-driven `LengthRegulator`, a bidirectional frame LSTM and a
   `CBHG` postnet), not the `SynthesizerTrn` this plan expected. Padding isolation
   would require masks/packing in five coupled submodules; the measured leak is
   0.155 peak. `HarmonySpeechSynthesizer` is therefore **not** in
   `BATCHED_INFERENCE_FAMILIES`, and its handler keeps the loop with the audit
   linked in the code comment.
4. **The restorer's multi-segment concatenation is preserved verbatim.** The
   historical `VoiceFixerRestorer.forward` concatenates per-30 s-segment mels
   along `dim=3` of `[B, 1, frames, bins]` — the *bin* axis, not the frame axis.
   The batched path reproduces that exact reduction per request (rows are
   `(request, 30 s slice)`), so a >30 s clip produces the same shapes and values
   as before. Fixing that upstream oddity would be a behaviour change outside this
   task.
5. **The `VoiceFixerGenerator.forward(sp, ...)` `sp` argument is dead** (audited:
   never read). The batched entry point therefore takes only the mel map; the
   solo path is untouched.
6. **`BATCHED_INFERENCE_FAMILIES` gains three types, not four**:
   `HarmonySpeechVocoder`, `VoiceFixerRestorer`, `VoiceFixerVocoder`. Shipped
   `max_batch_size` stays `1` for all of them — 8-6 owns the honest capacity
   values and the config files are deliberately untouched.
7. **`BATCHING_AXIS_PENDING` is now empty.** `scripts/audio_regression.py` keeps
   the dict (it carries the refusal message) and
   `tests/unit/benchmarks/test_audio_regression.py` drives the refusal path
   through `monkeypatch`, so the mechanism stays covered.
