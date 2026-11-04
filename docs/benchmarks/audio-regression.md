# Audio regression gate

Threshold-based A/B audio comparison for changes that should not alter what the
models say or how they sound: dtype (fp32 ↔ fp16), weight sharing
(deduplicated ↔ duplicated backbones), and later `torch.compile`/CUDA-graph
capture. Implemented by [`scripts/audio_regression.py`](../../scripts/audio_regression.py),
with fixtures in [`tests/audio_fixtures/`](../../tests/audio_fixtures).

## How it works

1. For each family the harness boots **one engine per side of the axis in a
   separate process** (`--worker`), runs the family's fixed fixture set, and
   writes the rendered clips.
2. The parent process loads both sides and computes per-clip metrics. No golden
   audio is committed: bit-exactness across torch versions is unrealistic, so
   both sides are always produced by the same build.
3. Exit code is non-zero when any family/case fails, so it can gate a config
   flip.

### Fixtures and provenance

| Fixture | Content | Provenance |
|---|---|---|
| `tests/audio_fixtures/ref_voice_a.wav` | 8 s, 24 kHz, mono PCM-16 | **Synthetic** — `generate_references.py` (harmonic glottal source + formant resonators, fixed seeds). No third-party audio. |
| `tests/audio_fixtures/ref_voice_b.wav` | 8 s, 24 kHz, mono PCM-16 | Synthetic (different formants/pitch). |
| `tests/test-data/samples/wanda4.wav`, `wanda5.wav` | real speech, pre-existing | Repository e2e fixtures (committed before this phase). Used by the Whisper-VAD-driven toolchains (OpenVoice, Harmony Speech) because synthetic vowel audio is not detected as speech; also the two sides of the real-speech VC case. |
| `tests/audio_fixtures/ref_voice_de.wav` | **optional**, not committed | User-provided short German speech clip for the native-accent German case. While the file is absent, `chatterbox_multilingual/tts_de_native` is skipped with one log line and excluded from metrics. |
| Texts | 2× EN, 1× DE, 1× ZH short sentences | `TEXTS` in `tests/audio_fixtures/__init__.py`. `en_wanda4` is the faster-whisper-tiny transcript of `wanda4.wav` ("It's time to go home."), used as the VC WER reference. |

### Metrics

| Metric | Applies to | Implementation |
|---|---|---|
| Log-mel spectral distance (MSE over aligned mel frames, dB²) | deterministic families | `librosa.feature.melspectrogram` (80 mel, n_fft 1024, hop 256) |
| Duration delta (%) | all | sample counts at a common 24 kHz rate |
| RMS loudness delta (%) | all | `sqrt(mean(x²))` |
| Sample correlation | deterministic families | `np.corrcoef` over the common length |
| Long-term average spectrum RMSE (dB) and `max\|Δsample\|` | stochastic families | time-mean of the log-mel; per-sample max difference |
| Word error rate (WER) | VC cases with a `transcript` | faster-whisper-tiny transcribes each rendered side; Levenshtein over case/punctuation-normalised words, `errors / \|reference words\|` |

**Deterministic** families (HarmonySpeech, VoiceFixer, KittenTTS) are compared
sample-wise. **Stochastic** families (Chatterbox sampling, OpenVoice/MeloTTS
`noise_scale`) are compared with distribution metrics: two independent samples
of the same distribution are not expected to match frame-by-sample.

### Voice-conversion cases and the WER gate

VC converts the *content* of the source clip, so the source must be real
speech: converting a synthetic vowel-like clip yields humming with the target's
timbre, which cannot demonstrate (or gate) intelligibility. The primary case is
therefore `vc_speech_a_to_b` (`wanda4` → `wanda5`, real speech both sides).
`vc_speech_to_synth_voice` (`wanda4` → synthetic `voice_b`) is kept as a
secondary, explicitly labelled timbre-porting case.

The harness transcribes **both** rendered sides with the resident
faster-whisper-tiny model and scores WER against `TEXTS["en_wanda4"]`
("It's time to go home."). WER is always reported (JSON `wer` object and the
markdown `WER` column). Above **0.4** is a warning; above **0.7** is a hard
failure (the `vc_speech_to_synth_voice` case sets `wer_gate: false` — a
non-speech target cannot be expected to transcribe, so it is report-only).
Transcription is language-aware via the case's `language`; clips in German (or
any non-English language) need a transcript in that language and a matching
`language` before WER is meaningful.

### Multilingual accent semantics

In pinned `chatterbox-tts==0.1.6`,
`mtl_tts.generate(text, language_id, audio_prompt_path, …)` uses `language_id`
only for tokenization. Accent and prosody come entirely from the conditionals
(the reference audio, or the bundled conds when none is given), and the bundled
conds are English-accented. `tts_de_en_accent` is therefore **German text with
an English accent by construction**, not a native German voice. The optional
`tts_de_native` case supplies `tests/audio_fixtures/ref_voice_de.wav` as the
conditional when that user-provided clip exists; otherwise it is skipped.

### Thresholds (current, in the script)

| Family kind | log-mel MSE | duration Δ | RMS Δ | correlation | LTAS RMSE |
|---|---|---|---|---|---|
| deterministic | < 1.0 | < 1 % | < 5 % | > 0.98 | — |
| stochastic | — | < 12 % | < 15 % | — | < 4 dB |
| `sharing` axis (all) | < 1e-2 | = 0 % | — | — | max\|Δ\| < 1e-3 (−60 dBFS) |

The `sharing` axis additionally requires **identical length**: a changed token
stream is a wiring bug, not noise. Exact (bit-identical) equality is asserted
in-process by
[`tests/e2e/tts/test_chatterbox_sharing_identity.py`](../../tests/e2e/tts/test_chatterbox_sharing_identity.py);
across two processes this host's CUDA kernel selection (cuBLAS/cuDNN workspace
heuristics) introduces last-bit noise in the autoregressive path, measured at
`max|Δsample| ≈ 1.5e-4`.

### Listening pack

On any failure or marginal case the harness copies the A/B clips to
`out/regression-pack/{family}/{case}__A.wav` / `__B.wav`. A human listening
pass over that folder is part of the sign-off for dtype and sharing changes; it
is not automated.

## Recorded runs

Host: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 610.88, torch 2.9.1+cu130,
Python 3.12.12, Windows 11. GPU 1 of 3 (otherwise idle), `CUDA_VISIBLE_DEVICES=1`.

### Identity self-test — `dtype:float32-vs-float32`

| Family | Case | Verdict | duration Δ% | RMS Δ% | log-mel MSE | corr | identical |
|---|---|---|---|---|---|---|---|
| kittentts (CPU) | tts_en | PASS | 0.0 | 0.0 | 0.0 | 1.0 | True |

### `dtype:float32-vs-float16`

| Family | Case | Verdict | duration Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | corr | identical |
|---|---|---|---|---|---|---|---|---|
| chatterbox | tts_bundled_voice | PASS | 0.0 | 0.366 | 8.91 | 0.18 | 0.76 | False |
| chatterbox | tts_clone_voice_a | PASS | 4.69 | 1.97 | 141.55 | 1.87 | 0.05 | False |
| chatterbox | vc_a_to_b | PASS | 0.0 | 1.37 | 1.57 | 0.19 | 0.85 | False |
| chatterbox_turbo | tts_bundled_voice | PASS | 0.0 | 0.14 | 0.004 | 0.009 | 0.9999 | False |
| chatterbox_multilingual | tts_en | PASS | 0.0 | 0.17 | 0.003 | 0.007 | 1.0 | False |
| chatterbox_multilingual | tts_de | PASS | 0.0 | 0.15 | 0.002 | 0.010 | 1.0 | False |
| openvoice_v1 | clone_voice_a | **FAIL (structural)** | — | — | — | — | — | — |
| openvoice_v2 | clone_voice_a | **FAIL (structural)** | — | — | — | — | — | — |
| melotts | tts_en | **FAIL (structural)** | — | — | — | — | — | — |
| harmonyspeech | — | skipped | CPU-pinned: `dtype` only selects bf16 on CPU |
| voicefixer | — | not run | pre-existing GPU device-placement bug (see below) |
| kittentts | — | skipped | ONNX runtime ignores torch dtype |

Boot VRAM in the same run (Chatterbox TTS + VC + embedding):

| Side | torch allocated | torch reserved |
|---|---|---|
| float32 | 3.04 GiB | 3.39 GiB |
| float16 | 2.05 GiB | 2.37 GiB |

Failures and why they are per-entry `float32` opt-outs:

* `openvoice_v1`, `openvoice_v2`, `melotts` — `RuntimeError: expected scalar
  type Float but found Half`: the adapted VITS code builds float32 tensors
  inside the flow/text encoder, so a half-precision module tree cannot be fed.
  MeloTTS additionally hits `AssertionError` in
  `transforms.rational_quadratic_spline` (spline bin widths underflow in fp16).
  Casting the prepared BERT features to the model dtype (now done in the
  MeloTTS handler) is necessary but not sufficient.
* `voicefixer` — GPU inference currently raises `Input type
  (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` (a submodule is
  never moved to the device); the dtype axis cannot be measured until that
  pre-existing bug is fixed. Kept `float32`.
* `harmonyspeech` is CPU-pinned in the shipped GPU config (the CPU executor
  rewrites fp16 → bf16), and KittenTTS/Silero are ONNX — torch `dtype` is a
  no-op there by design.

### `sharing:dedup-vs-dup` (Chatterbox)

Watermarking is disabled for this axis: PerthNet's conv kernels are not
bit-reproducible across processes either, and the gate is about the synthesis
path that sharing touches.

| Case | Verdict | duration Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | corr | identical | max\|Δsample\| |
|---|---|---|---|---|---|---|---|---|
| tts_bundled_voice | PASS | 0.0 | 0.0 | 9e-06 | 0.0002 | 1.0 | False | 6.1e-05 |
| tts_clone_voice_a | PASS | 0.0 | 0.0 | 4e-06 | 0.0001 | 1.0 | False | 1.5e-04 |
| vc_a_to_b | PASS | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | **True** | 0.0 |

In-process (same GPU, same allocator state) the same comparison is
**bit-identical** — see the e2e test above.

Boot VRAM of the three-role set (TTS + VC + embedding, fp32 weights):

| Side | torch allocated | torch reserved | shared registry |
|---|---|---|---|
| duplicated | 7.10 GiB | 7.45 GiB | — |
| deduplicated | 3.04 GiB | 3.39 GiB | 2.97 GiB |

### `batching:1-vs-4` (Phase 5)

The padded-batch axis drives the engine step-by-step: side `1` schedules every
case as its own one-request step (`max_batch_size: 1`, `batched_inference:
false`), side `4` schedules all four cases in **one** step
(`max_batch_size: 4`, `batched_inference: true`). Both sides run with the
models' sampling noise switched off (`generation_options.noise_scale = 0`,
`noise_scale_w = 0`) and with TF32 disabled in the worker, so the comparison
isolates padding/unpadding rather than sampling noise or batch-shape-dependent
kernel choice. Verdict: duration Δ ≤ 0.1 %, max |Δsample| ≤ 2e-3, log-mel MSE
≤ 1e-2, corr ≥ 0.999.

| Family | Case | Verdict | duration Δ% | max\|Δsample\| | log-mel MSE | corr | identical |
|---|---|---|---|---|---|---|---|
| melotts | batch_short | PASS | 0.0 | 1.59e-05 | 0.0 | 1.0 | False |
| melotts | batch_medium | PASS | 0.0 | 1.59e-05 | 0.0 | 1.0 | False |
| melotts | batch_long | PASS | 0.0 | 1.59e-05 | 0.0 | 1.0 | False |
| melotts | batch_shortest | PASS | 0.0 | 1.58e-05 | 0.0 | 1.0 | False |
| openvoice_v1 | batch_short | PASS | 0.0 | 0.0 | 0.0 | 1.0 | True |
| openvoice_v1 | batch_medium | PASS | 0.0 | 0.0 | 0.0 | 1.0 | True |
| openvoice_v1 | batch_long | PASS | 0.0 | 0.0 | 0.0 | 1.0 | True |
| openvoice_v1 | batch_shortest | PASS | 0.0 | 0.0 | 0.0 | 1.0 | True |

MeloTTS's 1.6e-5 residual is below one 16-bit LSB (the case only differs after
rounding to PCM16); OpenVoice V1 is byte-identical. Throughput numbers for the
same phase are in [throughput.md](throughput.md).

### Numeric noise floor

The `1.59e-05` in the batching tables is expected floating-point noise, not a
measurement bug. B=1 and in-batch outputs differ because float32 batched kernels
(GEMMs and convolutions over padded `[B, T_max]` shapes) accumulate in a
different order than the single-sample shapes. At full scale one 16-bit PCM LSB
is `1/32768 ≈ 3.05e-5`, so a ~1.6e-5 delta is below the smallest step the
PCM-16 WAV can represent — inaudible by construction. OpenVoice V1 comes out
byte-identical because its (float) deltas fall below PCM-16 rounding. The gate
treats sub-LSB deltas as passes: `BATCHING_MAX_ABS_DIFF = 2e-3` is over four
orders of magnitude above the observed noise.

### Post-review re-run (2026-11-05)

Re-run after the review fixes (real-speech VC cases, WER gate, accent-labelled
multilingual cases, optional native-German skip). GPU 1, same host/torch as
above. The old `vc_a_to_b` (synthetic source) and `tts_de` names are superseded
by `vc_speech_a_to_b` / `vc_speech_to_synth_voice` and `tts_de_en_accent`.

`dtype:float32-vs-float16`:

| Family | Case | Verdict | duration Δ% | RMS Δ% | log-mel MSE | LTAS RMSE dB | corr | WER |
|---|---|---|---|---|---|---|---|---|
| chatterbox | tts_bundled_voice | PASS | 0.0 | 0.366 | 8.91 | 0.18 | 0.76 | — |
| chatterbox | tts_clone_voice_a | PASS | 4.69 | 1.97 | 141.55 | 1.87 | 0.05 | — |
| chatterbox | vc_speech_a_to_b | PASS | 0.0 | 0.007 | 0.35 | 0.06 | 0.96 | **0.000** |
| chatterbox | vc_speech_to_synth_voice | PASS | 0.0 | 0.329 | 0.51 | 0.09 | 0.97 | 1.600 (report-only) |
| chatterbox_multilingual | tts_en | PASS | 0.0 | 0.17 | 0.003 | 0.007 | 1.0 | — |
| chatterbox_multilingual | tts_de_en_accent | PASS | 0.0 | 0.15 | 0.002 | 0.010 | 1.0 | — |
| chatterbox_multilingual | tts_de_native | skipped | optional `ref_voice_de.wav` absent | | | | | excluded |

`sharing:dedup-vs-dup`:

| Case | Verdict | duration Δ% | log-mel MSE | LTAS RMSE dB | corr | identical | max\|Δsample\| | WER |
|---|---|---|---|---|---|---|---|---|
| tts_bundled_voice | PASS | 0.0 | 2e-06 | 0.0001 | 1.0 | False | 3.1e-05 | — |
| tts_clone_voice_a | PASS | 0.0 | 5e-06 | 0.0001 | 1.0 | False | 1.5e-04 | — |
| vc_speech_a_to_b | PASS | 0.0 | 0.0 | 0.0 | 1.0 | **True** | 0.0 | **0.000** |
| vc_speech_to_synth_voice | PASS | 0.0 | 4e-06 | 0.0001 | 1.0 | False | 2.7e-04 | 2.200 (report-only) |

The real-speech VC output transcribes exactly to the source sentence on both
dtype and sharing axes, so the conversion path is healthy. The synthetic-target
variant's high WER is the measured consequence of a non-speech acoustic target
(Whisper hallucinates words on the hum-like output), which is why it is
reported but not gated.

## Reproducing

```bash
# identity self-test (CPU, runs anywhere)
python scripts/audio_regression.py --families kittentts --axis dtype:float32-vs-float32

# dtype gate
python scripts/audio_regression.py --families chatterbox,chatterbox_turbo,chatterbox_multilingual \
    --axis dtype:float32-vs-float16 --gpu \
    --json-out out/audio-regression/dtype.json --md-out docs/benchmarks/audio-regression.md

# sharing gate
python scripts/audio_regression.py --families chatterbox --axis sharing:dedup-vs-dup --gpu

# padded-batch equivalence gate (Phase 5)
python scripts/audio_regression.py --families melotts,openvoice_v1 \
    --axis batching:1-vs-4 --gpu
```
