# Batching & concurrency

This page documents what actually happens when several requests arrive for the
same model, and what `max_batch_size` means in the shipped configuration.

## `max_batch_size` is a capacity signal

`max_batch_size` (per model entry in `config.yml` / `config.gpu.yml`) is **the
maximum number of requests the engine admits for that model in one engine step,
and that the executor then processes together**.

That second half is the important one. Before Phase 5 the field was decorative
for most models: a synthesizer declared `10`, the scheduler admitted ten
requests, and the handler then ran them in a `for` loop, one at a time. The
value now matches the executor's real behaviour, so it can be used for capacity
planning.

## The batching matrix

This is the phase's contract with operators: every model type the loader
registers (`harmonyspeech/modeling/loader.py` `_MODEL_CONFIGS`) appears exactly
once, with the mechanism its handler really uses, the unit of its batching
budget, what happens when that mechanism is bypassed, and why `max_batch_size`
holds the shipped value. `tests/unit/test_batching_docs_matrix.py` parses this
table, fails when a registered type is missing or listed twice, and checks that
every shipped config entry declares exactly the `max_batch_size` this table
promises; `tests/unit/initialization/test_config.py` asserts the shipped configs
against the same policy (batched family, per-request loop, or queue/lane depth).

| Registry model type | Shipped entries | `max_batch_size` | Mechanism | Budget axis unit | Fallback | Why this value / skip reason |
|---|---|---|---|---|---|---|
| `OpenVoiceV1Synthesizer` | `ov1-synthesizer-en`, `-zh` | `8` | padded batch, one `infer_batched` forward per length-similar sub-batch | phone tokens | per-item loop (single request, or whole-step degrade) | Batched forward validated in Phase 5-2; `8` gives a sentence-level client real depth |
| `MeloTTSSynthesizer` | `ov2-synthesizer-{en,zh,es,fr,ja}` | `8` | padded batch, one `infer_batched` forward per sub-batch (phones/tones/language ids + BERT features) | phone tokens | per-item loop | Same machinery as OpenVoice V1 (Phase 5-2) |
| `OpenVoiceV1ToneConverterEncoder` | `ov1-tone-converter-encoder` | `8` | padded batch on `(request, VAD segment)` rows | mel frames | per-item loop | Batched forward in Phase 8-2; `8` is the initial capacity (tuning pending the Phase 8 benchmark-host run, `docs/benchmarks/throughput.md`) |
| `OpenVoiceV2ToneConverterEncoder` | `ov2-tone-converter-encoder` | `8` | padded batch on `(request, VAD segment)` rows | mel frames | per-item loop | As the V1 encoder (Phase 8-2/8-6) |
| `HarmonySpeechEncoder` | `hs1-encoder` | `8` | batched LSTM over `(request, fixed-length partial)` rows | mel frames | per-item loop | As the OpenVoice encoders (Phase 8-2/8-6) |
| `OpenVoiceV1ToneConverter` | `ov1-tone-converter` | `8` | padded batch, one `voice_conversion_batched` forward per sub-batch | mel frames | per-sub-batch loop | Landed in Phase 8-3; `8` is the initial capacity (tuning pending the benchmark-host run) |
| `OpenVoiceV2ToneConverter` | `ov2-tone-converter` | `8` | padded batch, one `voice_conversion_batched` forward per sub-batch | mel frames | per-sub-batch loop | As the V1 tone converter (Phase 8-3/8-6) |
| `HarmonySpeechVocoder` | `hs1-vocoder` | `8` | padded batch, one `inference_batched` forward per sub-batch | mel frames | per-sub-batch loop | Landed in Phase 8-4; `8` is the initial capacity (tuning pending the benchmark-host run) |
| `VoiceFixerRestorer` | `voicefixer-restorer` | `8` | padded batch on per-clip mel rows | mel frames | per-sub-batch loop | Landed in Phase 8-4; `8` is the initial capacity (tuning pending the benchmark-host run) |
| `VoiceFixerVocoder` | `voicefixer-vocoder` | `8` | padded batch, one batched forward per sub-batch | mel frames | per-sub-batch loop | Landed in Phase 8-4; `8` is the initial capacity (tuning pending the benchmark-host run) |
| `HarmonySpeechSynthesizer` | `hs1-synthesizer` | `1` | **loop** — no batch path | — (no padding-safe axis) | the loop is the only path | Evidence-based skip: the vendored ForwardTacotron drifts by 0.155 peak under padding (Phase 8-4 audit); padding-safe means rewriting three SeriesPredictors, a CBHG prenet, the LengthRegulator and a CBHG postnet |
| `FasterWhisper` | `faster-whisper-large-v3-turbo`, `faster-whisper-tiny` | `16` | native CT2 batch **inside one audio** (faster-whisper's own 30 s clip batch, `batch_size=16`) | 30 s audio clip | per-request loop across requests | Audit Phase 8-5: `BatchedInferencePipeline.transcribe()` takes one audio, not a list, so there is no cross-request batch to fold. `16` is the admission/queue depth; the clip batch is internal |
| `SileroVAD` | `silero-vad` | `16` | **loop** — no batch path, and lanes measured below break-even | — (single-stream 512-sample window) | the loop is the only path | Skip evidenced in Phase 8-5: `get_speech_timestamps` is a single-stream API; lanes on per-lane wrappers measured 1.10× / 0.95× / 0.92× at 4 / 8 / 16. `16` is queue depth only |
| `KittenTTSSynthesizer` | `kitten-tts-mini`, `-micro`, `-nano`, `-nano-int8` | `4` | **parallel lanes** — one request per lane on one shared ONNX session | — (no batch axis: every graph has a static batch dimension of 1) | one lane, inline, when the step holds a single request | Static batch dim 1 makes batching impossible without a re-export. `4` *is* the lane count; measured 1.71× at 4 lanes and 1.91× at 8 lanes (mini, this machine) |
| `ChatterboxTTS` | `chatterbox` | `1` | **loop** — no batch path | — (autoregressive token loop) | the loop is the only path | Explicit product decision, 2026-11-05 (see "Chatterbox is out of scope by decision") |
| `ChatterboxTurboTTS` | `chatterbox_turbo` | `1` | **loop** — no batch path | — (autoregressive token loop) | the loop is the only path | Same product decision |
| `ChatterboxMultilingualTTS` | `chatterbox_multilingual` | `1` | **loop** — no batch path | — (autoregressive token loop) | the loop is the only path | Same product decision |
| `ChatterboxEmbedding` | `chatterbox_embedding` | `1` | **loop** — no batch path | — | the loop is the only path | Same product decision — its non-autoregressive members are deferred with the family |
| `ChatterboxVC` | `chatterbox_vc` | `1` | **loop** — no batch path | — | the loop is the only path | Same product decision |

An entry that declares a value its handler cannot honour is a bug: it makes
`/metrics`, dashboards and capacity plans lie.

Both shipped configs (`config.yml` and `config.gpu.yml`) declare the same
capacities; they differ only in device placement and dtype. On CPU the padded
forwards still run (they are plain tensor ops), so the capacity values are not
per-device guesses — tune them per host with the throughput benchmark.

## Padded batching for one-shot synthesizers

OpenVoice V1 `SynthesizerTrn` and the MeloTTS synthesizers are fixed-graph
one-shot models (text → phones → mel → waveform in one pass), so a step can be
forwarded as a single padded batch:

1. The step is flattened into `(request, sentence)` rows.
2. Rows are grouped into length-similar sub-batches: never more than
   `max_batch_size` rows, never a `T_max / T_min` ratio above `4`, and never
   more than `batch_token_budget` phone tokens (default
   `max_batch_size × median length` of the step).
3. Each group is padded to `[B, T_max]` (phones/tones/language ids with the
   blank symbol `0`, BERT features with zeros to `[B, D, T_max]`) and forwarded
   once via `SynthesizerTrn.infer_batched`.
4. The decoder output is unpadded with the model's **own** per-sample frame
   counts (`y_lengths × hop_length`) and reassembled per request in input
   order.

`batched_inference` in the model entry controls it:

```yaml
- name: "ov2-synthesizer-en"
  model_type: "MeloTTSSynthesizer"
  max_batch_size: 8
  batched_inference: true   # default for these two families
```

* `true` / unset → padded batch for steps with more than one request.
* `false` → legacy per-item loop (set `max_batch_size: 1` as well).
* A single request per step always takes the per-item path: it is the fast path
  and the fallback if a batched forward raises for any reason (the engine logs a
  warning and re-runs the step request by request).

**Boundary note.** The decoder is convolutional, so the last frame of each
sample sees a slightly different right-hand context inside a padded batch
(explicit zeros plus conv bias) than it does alone (the conv's own implicit
zero padding). With the shipped float32 weights this is ≤1e-4 peak — four
orders of magnitude below the models' own sampling noise. The
`batching:1-vs-N` regression axis bounds it (`max |Δsample| < 2e-3`).

### One pattern, three length units

Every step that pads uses the same *pad → one forward → trim by explicit
lengths* machinery; only the **unit** of the length vector changes:

| Axis unit | One row | Steps that pad on it |
|---|---|---|
| tokens (phones) | phone / tone / language-id ids | synthesizers (OpenVoice V1, MeloTTS) |
| frames (mel) | `[bins, T]` mel or feature matrix | tone converters, restorers, vocoders, HarmonySpeech encoder |
| samples | 1-D waveform | speaker embedders |

The length-similarity rule and `batch_token_budget` are unit-agnostic: the
budget is measured in whatever unit the step passes in (phone tokens, mel
frames or audio samples, `max_batch_size × median length` by default). Which
of those steps actually batches today is the table at the top of this page.

## Padded batching for speaker embedders (Phase 8-2)

The two embed steps that a cloning request runs before synthesis use the same
machinery, on their own row unit:

* **OpenVoice tone-converter encoder** (`ov1/ov2-tone-converter-encoder`,
  `ReferenceEncoder`): the VAD step produces **several segments per request**, so
  the rows are `(request, segment)` pairs — exactly like `(request, sentence)` in
  the synthesizers. Each row is turned into its mel spectrogram by the historical
  per-segment STFT frontend, the rows are padded on the **frame** axis, forwarded
  for their real frame counts, and each request's segment embeddings are stacked
  and averaged **in segment order**, which is the arithmetic the per-item loop
  always used.
* **HarmonySpeech encoder** (`hs1-encoder`, `SpeakerEncoder`): the vendored
  model already accepts `[B, frames, mels]`, and its frontend produces
  fixed-length partial utterances, so the rows are `(request, partial)` pairs on
  the **frame** axis. A request's partial embeddings are averaged in partial
  order before the L2 normalisation.

Two boundary rules make a batch produce the solo value per row:

* The OpenVoice conv stack zeroes each row's padding region between layers.
  Without it the conv **bias** — not zeros — would feed the next layer's edge
  frames and drift the embedding (~1e-3), because a solo run only ever sees its
  own frames plus implicit zero padding.
* Both GRUs/LSTMs run over each row's valid frames only (packed sequences), so
  trailing padding can never reach the final hidden state, which *is* the
  embedding.

With `B == 1`, or a batch whose rows are all full length (the normal
HarmonySpeech case), the plain single-sample forward is called, which keeps the
historical path bit-identical. Cache hits (Phase 7) are resolved before the batch
is built, so a hit never enters a forward.

## Padded batching for tone converters (Phase 8-3)

The tone-conversion step (`ov1/ov2-tone-converter`) consumes the synthesis step's
output audio plus the target speaker embedding. Its row unit is the request:

1. The mel frontend runs **per clip**: `spectrogram_torch` reflect-pads each clip
   before the STFT, so a padded waveform batch would change the shorter rows'
   edge frames. The batch is built on the mel frames it produces.
2. Rows are grouped into length-similar sub-batches on the **mel-frame** axis
   (same `plan_batches` rules and per-step budget as the synthesizers).
3. Each group is one `SynthesizerTrn.voice_conversion_batched` forward with the
   padded `[B, bins, T_max]` mel, the per-row frame lengths and the per-row
   `[B, gin, 1]` source/target embeddings.
4. The waveform is unpadded in the **sample** domain with the model's own frame
   counts: `y_lengths × prod(upsample_rates)` (the decoder emits one frame's
   worth of samples per mel frame; this equals the configured hop length for both
   shipped converters). A single-sample run returns that same full decoder
   output.

The conversion noise is still the historical constant (`tau = 0.3`) and each row
draws its own noise, so a row's waveform is statistically — not sample-for-sample
— the loop's waveform. What the batched path does guarantee is that **padding
never reaches the valid tail**: every decoder stage holds each row's padding
region at zero, because a transposed conv would otherwise spread the padding
frames' conv bias back into the last valid samples (and, through the resblocks,
much further). With `tau = 0` a padded row is therefore equal to its solo run up
to batch-kernel float accumulation, and with `B == 1` the batched entry point is
bit-identical to the single-sample one.

## Padded batching for vocoders and the VoiceFixer restorer (Phase 8-4)

The two vocoders (`hs1-vocoder`, `voicefixer-vocoder`) and the VoiceFixer
enhancement step (`voicefixer-restorer`) use the same machinery on the
**mel-frame** axis:

1. Each request turns its own mel (or, for the restorer, its own audio slice) into
   rows with the historical per-clip frontend. The frontends stay per clip: the
   restorer's centered STFT and the vocoder's mel normalization/`tr_pre` tail are
   both defined on a clip's own frame count.
2. Rows are grouped into length-similar sub-batches (`plan_batches`, frame axis).
3. Each group is one batched forward per model, and each row's padding region is
   held at **zero after every stage**. That is what keeps a padded row equal to
   its solo run: a convolutional decoder would otherwise spread the padding
   frames' biases back into the valid tail (measured 0.28 peak on the VoiceFixer
   vocoder and 2e-3 on the MelGAN vocoder before the masking). Reflection pads
   see each row's own last frames instead of the batch padding, and the
   VoiceFixer vocoder's residual-skip branch — whose reshape mixes frames by the
   row's own length — is replayed per row.
4. Outputs are unpadded with each model's own length semantics: the MelGAN
   decoder returns `frames × prod(upsample_scales)` samples per row, the
   VoiceFixer vocoder the same for its condition frames, and the restorer returns
   mel frames (the 64-frame alignment of its internal down/up-sampling is part of
   the mask).

Per-family behaviour is preserved verbatim: the Harmony Speech vocoder's
inter-chunk silence (`breaks`) is inserted **after** decoding, per request, from
that request's own frame counts, and the VoiceFixer vocoder's peak normalization
uses **each request's own** maximum (a batch maximum would rescale a quiet clip by
its neighbour's loudest sample).

The Harmony Speech **synthesizer** stays a per-item loop. Its vendored
`ForwardTacotron` runs three duration/pitch/energy predictors, a CBHG prenet, a
duration-driven length regulator and a CBHG postnet, each with a bidirectional
GRU/LSTM over the padded axis; a padded row changes by ~0.16 peak on its own
frames, and making it padding-safe would mean rewriting five coupled submodules.
The per-item loop is the honest answer there; `max_batch_size` stays `1`.

## Parallel lanes: KittenTTS (Phase 8-5)

KittenTTS has no batch axis at all: all four cached variants
(`mini`, `micro`, `nano-fp32`, `nano-int8`) export a **static batch dimension of
1** — `input_ids: [1, sequence_length]`, `style: [1, 256]`, `speed: [1]`
(probed 2026-11-05 and re-confirmed while measuring here). Batching would need a
re-export with a dynamic batch axis, which is KittenML's pipeline, not ours. The
model is one-shot and non-autoregressive (tokens + style + speed → waveform in a
single `session.run`), so the step instead runs **one request per lane**:

* `max_batch_size` **is** the lane count — no new config key. The scheduler
  already admits up to that many kitten requests per step (`4` in the shipped
  configs), and the handler then dispatches each request's `generate` to a
  bounded thread pool of `min(len(inputs), max_batch_size)` lanes.
* One shared `InferenceSession`: `run()` is thread-safe and releases the GIL, so
  lanes really execute on separate cores without N copies of the weights; the
  session is built with `inter_op_num_threads = 1` and
  `intra_op_num_threads = 0` (ORT's own default — see below).
* Outputs are reassembled by request in input order, and each lane's exception
  becomes **only that request's** error result (the pairing invariants of Phases
  0-3 and 5-1 hold unchanged).
* The espeak phonemizer is **serialised by a lock**: `phonemizer`'s
  `EspeakBackend` is not thread-safe (concurrent `phonemize` calls on one backend
  raise `RuntimeError: number of lines in input and output must be equal, we
  have: input=1, output=2` — reproduced here while measuring). A call costs
  ~0.3 ms, so holding the lock around it is free next to a ~1.5 s inference.
* **Load-time warmup**: the wrapper runs one tiny `generate` at construction so
  the first real request skips the cold start.

### Measured lanes × `intra_op` matrix

Mini variant, 12-core dev machine, 8 distinct sentences (~5.9 s of audio each).
"seq" is the pre-lane loop over the same requests, "par" the best of two lane
passes through a bounded pool; `intra` is `intra_op_num_threads` (`def` = ORT's
own default).

| Lanes | `intra_op` | seq wall | par wall | Speedup | seq req/s | par req/s | RTF (par) |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 9.53 s | 3.76 s | 2.53× | 0.42 | 1.06 | 0.178 |
| 4 | 2 | 7.42 s | 3.85 s | 1.93× | 0.54 | 1.04 | 0.182 |
| 4 | 4 | 6.16 s | 3.40 s | 1.81× | 0.65 | 1.18 | 0.161 |
| 4 | def | 5.74 s | 3.35 s | 1.71× | 0.70 | 1.19 | 0.159 |
| 8 | 1 | 19.57 s | 6.50 s | 3.01× | 0.41 | 1.23 | 0.153 |
| 8 | 2 | 13.87 s | 6.45 s | 2.15× | 0.58 | 1.24 | 0.152 |
| 8 | 4 | 12.12 s | 6.27 s | 1.93× | 0.66 | 1.28 | 0.148 |
| 8 | def | 11.89 s | 6.22 s | 1.91× | 0.67 | 1.29 | 0.147 |

Single-request latency for the same sentence (warm, best of three):

| `intra_op` | Latency | RTF |
|---|---|---|
| 1 | 2.61 s | 0.443 |
| 2 | 1.71 s | 0.290 |
| 4 | 1.74 s | 0.296 |
| def | 1.61 s | 0.273 |

Reading of the matrix, and the baked-in constant:

* Throughput is **flat across `intra_op`** at 8 lanes (1.22–1.29 req/s,
  reproduced three times): the suspected oversubscription (8 lanes × default
  intra-op threads) does not cost anything on this host, because each run is one
  long C++ call and the OS schedules the short parallel phases fine. Pinning
  `intra_op` to 1 or 2 *loses* throughput instead of gaining it.
* Single-request latency is where the pin hurts: `intra_op=1` doubles it
  (2.61 s vs 1.61 s, RTF 0.443 vs 0.273).
* The session therefore uses `intra_op_num_threads = 0`, i.e. ORT's own default
  of one intra-op thread per physical core — set **explicitly** in the wrapper
  (`KITTEN_TTS_INTRA_OP_NUM_THREADS`) so the behaviour is pinned rather than
  inherited from a future ORT default, and deliberately **not** configurable.
  It also scales down on small hosts, which is this model's deployment target.
* Before/after: single-request RTF is unchanged (0.27) and a 4-lane step
  (the shipped `max_batch_size: 4`) improves from 0.70 req/s to 1.19 req/s
  (1.71×); 8 lanes reach 1.29 req/s (1.91×). The listed 2.5–3.0× speedups at
  `intra_op=1`/`2` are against their *own*, slower sequential baseline, not
  against the shipped configuration.

**Warmup.** On this machine the mini variant's first generate measured 1.52 s
against 1.47–1.53 s warm, i.e. no cold-start outlier left to remove (the ~2 s
cold start in the 2026-11-05 probe belonged to a freshly downloaded session).
The warmup stays because it costs one short inference at load time and removes
the first-request outlier on hosts where the graph *is* cold; a failed warmup is
logged and never fails the load.

**GPU execution providers — not pursued.** The pinned hse environment ships
CPU-only `onnxruntime` (`['AzureExecutionProvider', 'CPUExecutionProvider']`).
Adding `onnxruntime-directml` or `onnxruntime-gpu` would mean installing packages
into that pinned environment, which project policy forbids (Windows Application
Control blocks several newer wheels, and the CUDA EP risks DLL coexistence with
torch's CUDA 13 libraries). It is also not a product requirement: KittenTTS's
stated purpose is ultra-light CPU deployment, and a 74 MB one-shot graph may or
may not beat CPU per run. Opportunistic GPU support remains a benchmark-host
question, not a pending engine feature.

## Silero VAD: batching skip and lane verdict (Phase 8-5)

`get_speech_timestamps` is a **single-stream** API: it walks the waveform in
512-sample windows, calls the model once per window and post-processes the
resulting probability list into segments. A batch axis would mean
re-implementing that windowing loop and the upstream ONNX graph's state
protocol, for a step that is already cheap. Verdict: **no batching**.

Parallel lanes were evaluated and **rejected** on evidence, so the handler keeps
its sequential loop:

* The installed `silero_vad` package keeps the streaming LSTM state on the
  **wrapper instance** (`OnnxWrapper._state`, `._context`) and calls
  `reset_states()` at the start of every `get_speech_timestamps`. Concurrent
  calls on the shared model therefore corrupt each other: with 8 lanes on one
  wrapper, **7 of 16** concurrent results differed from their sequential answer.
  The lane design assumed per-call state; that assumption is false here.
* Lanes with *per-lane* wrappers (correct, but N sessions of a small model)
  measured 16 × 30 s clips:

  | Lanes | Wall | req/s | Speedup |
  |---|---|---|---|
  | 1 (sequential) | 3.94 s | 4.06 | 1.00× |
  | 4 | 3.58 s | 4.47 | 1.10× |
  | 8 | 4.15 s | 3.86 | 0.95× |
  | 16 | 4.28 s | 3.74 | 0.92× |

  Sublinear and then negative, exactly as expected for *many tiny* inferences
  with GIL-bound Python glue between them (contrast KittenTTS's one long call
  per request). At 8 lanes the requirement was ≥2×; the measured result is
  0.95×, so sequential execution stays and `max_batch_size: 16` is queue depth
  only. VAD is a minor hop in the cloning pipeline anyway — lanes would only
  matter for floods against `/v1/audio/vad` itself, and even then not
  profitably.

## Faster-whisper: why the STT step loops (Phase 8-5)

`BatchedInferencePipeline` (faster-whisper 1.2.1) is batched **within one
audio**, not across requests: `transcribe(audio, ...)` accepts a single path,
file object or mono ndarray, runs faster-whisper's own VAD to cut it into ≤30 s
clips, and feeds those clips through `forward()` in groups of `batch_size`
(`_batched_segments_generator` → `forward` → `generate_segment_batched`).
Neither `transcribe()` nor `WhisperModel.generate_segments()` accepts a list of
audios.

Every request on this model type carries its own audio — STT requests and the
pipeline's whisper-segmentation VAD step alike, because
`prepare_faster_whisper_inputs` returns one `AudioArtifact` per request — so the
handler transcribes one request at a time and builds **one pipeline object per
step** (it is a thin view over the shared model, and its word-timestamp cursor is
reset per request). `batch_size=16` is the pipeline's intra-audio clip batch;
`max_batch_size: 16` is the scheduler's admission width.

Folding requests into one call would mean re-implementing `transcribe()`:
per-audio VAD chunking, feature extraction, **per-audio language detection**
(and therefore a per-request tokenizer and prompt) and timestamp restoration, on
top of upstream internals. That trades a stable public API for a copy that
drifts, so the loop is the documented answer, not a gap.

## Chatterbox is out of scope by decision

> **Chatterbox family (all five model types)**: **explicitly out of scope** per
> the product decision (2026-11-05) — AR TTS needs a KV-cache story; its non-AR
> members (VC, embedding) are deferred with it to the dedicated later Chatterbox
> workstream.

So the `1` in the matrix for every Chatterbox type is a decision, not an
oversight. Chatterbox's T3 backbone generates autoregressively (one token at a
time, with a KV cache per sequence). Batching it properly needs a per-sequence
KV-cache scheduler; until that workstream lands, its executors process one
request at a time.

Operators who need more Chatterbox throughput should run **more engine
replicas** (more processes, each pinned to a GPU or sharing one with
`CUDA_VISIBLE_DEVICES` and per-process VRAM headroom) instead of raising
`max_batch_size`. Measured on one RTX 5060 Ti, Chatterbox throughput is flat at
~0.6 requests/s from concurrency 1 to 8 while p50 latency grows linearly — with
`max_batch_size: 1` extra concurrency queues instead of speeding anything up
(`docs/benchmarks/throughput.md`).

## What concurrency to expect

Phase 5 measurement below (RTX 5060 Ti, float32 for these families,
`requests/level = 16`). The clone row is **pre-Phase-8**: its tone-converter and
encoder hops were the `max_batch_size: 1` loops that Phase 8 now batches, so the
post-Phase-8 curve is `_tbd_` until the benchmark-host run
([`throughput.md`](benchmarks/throughput.md) §Phase 8).

| Family | c=1 | c=2 | c=4 | c=8 | Limiting factor |
|---|---|---|---|---|---|
| MeloTTS synthesizer (short text) | 8.3 req/s | 8.6 | 11.5 | **14.4** | padded batch, launch-bound |
| OpenVoice V2 clone (full pipeline, Phase 5 config) | 1.8 req/s | 1.9 | 1.9 | **1.9** | pre-8: per-item tone-converter hops; post-8 `_tbd_` |
| Chatterbox TTS | 0.6 req/s | 0.6 | 0.6 | **0.6** | autoregressive, one request per step |
| Whisper tiny STT | 2.7 req/s | 2.8 | 2.8 | **2.8** | batch inside one request; queueing above c=1 |

Practical guidance:

* **Sentence-level clients** (short texts) benefit most from synthesis batching;
  throughput keeps rising up to `max_batch_size`.
* **Full cloning toolchains** were bound by the tone-converter/encoder hops in
  Phase 5; those steps now batch (`max_batch_size: 8`, initial), so end-to-end
  clone throughput should rise with `max_batch_size` instead of staying flat —
  the benchmark-host run confirms the size. Do not size a fleet from the Phase 5
  clone row.
* **Long texts** are compute-bound: a single utterance already fills the GPU, so
  batching gives little to nothing. Do not size your fleet assuming batch
  scaling for long-form synthesis.
* **KittenTTS on CPU** scales with lanes, not with batch: 1.19 req/s at 4 lanes
  and 1.29 req/s at 8 lanes on a 12-core host, against 0.67–0.70 req/s for the
  sequential loop (mini variant, ~5.9 s of audio per request).

## Fairness and head-of-line behaviour

The scheduler admits up to `max_batch_size` requests per model per step, and a
request that does not fit stays in the waiting queue for the next step (Phase
5-1 fixed a bug where requests above the batch limit were dropped). While a
model is busy, newly arriving requests for the same model wait for the next
step — one model is always processed one step at a time. There is no
preemption and no per-request prioritisation.

A pipeline request that advances to its next step re-enters the waiting queue as
a fresh request under the same id, so a forwarded step and a new request for the
same model can share one cycle when the budget allows it.
