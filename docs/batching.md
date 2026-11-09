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

## Execution-model classes

Every model type belongs to exactly one class, and its shipped config must match
that class's semantics (owner decision 2026-11-07; the authoritative table is in
`.current_work/engine-modernization/10-5-WhisperGatesBenchmarksExit.md`):

| Class | Execution semantics | `max_batch_size` means | Additional knobs | Members (post-11-0) |
|---|---|---|---|---|
| `padded_batch` | one padded forward per length-similar sub-batch | sub-batch width | `batched_inference`, `batch_token_budget` | OpenVoice V1/V2 synthesizers and tone converters, their encoders, HarmonySpeech encoder/vocoder, VoiceFixer restorer/vocoder, Chatterbox VC (source tokenize + S3Gen vocode, Phase 11-1) |
| `loop` | per-request sequential | must be 1 | — | HarmonySpeech synthesizer |
| `loop_admission` | per-request sequential (serial) but several requests per engine step | admission depth | — | Chatterbox embedding (4; Phase 11-0 dated decision 2026-11-08) |
| `queue_depth` | admission/backpressure only | queue depth | — | Silero VAD (16) |
| `lanes_windows` | M parallel lanes × cross-request window batches | admission/lane depth | `num_workers`, `whisper_window_batch_size` | Faster-Whisper (M workers, B=8 windows shipped on); KittenTTS (lanes only, 4) |
| `ar_decode` | token-level AR decode on the shared KV pool | admission depth | pool slots, token budget | Chatterbox English + Turbo + Multilingual (4 requests each; 2 CFG rows per English/Multilingual request) |

`ar_decode`'s English and Turbo Chatterbox members landed in Phase 9-3 (the
token-level runner) and moved into this class in Phase 9-6; Multilingual joined
in Phase 12-7 once its analyzer became opt-in and its decode moved onto
`ChatterboxT3AR`. The shipped configs declare an **initial admission width of
4**, justified from the shared 2.75 GiB pool's arena table at each contract's
preferred slot cap: 8 English slabs × 135 MiB (1152 tokens) + 8 Multilingual
slabs × 135 MiB (1152 tokens) + 4 Turbo slabs × 144 MiB (1536 tokens) = 2736 MiB
≤ 2816 MiB. English/Multilingual use Perceiver resampling, so `len_cond` stays
at 34 tokens and 1152 covers 34 cond + ~118 text + 1000 speech; Turbo's
un-resampled 376-token prompt keeps the full 1536. The width is a capacity
decision pending the benchmark-host `batching:1-vs-N` seeded-equality gate;
`batched_inference: false` (or width 1) forces the proven sequential loop. A
pool refusal degrades that request to the sequential loop instead of failing it.

## The batching matrix

This is the phase's contract with operators: every model type the loader
registers (`harmonyspeech/modeling/loader.py` `_MODEL_CONFIGS`) appears exactly
once, with the mechanism its handler really uses, the unit of its batching
budget, what happens when that mechanism is bypassed, and why `max_batch_size`
holds the shipped value. `tests/unit/test_batching_docs_matrix.py` parses this
table, fails when a registered type is missing or listed twice, and checks that
every shipped config entry declares exactly the `max_batch_size` this table
promises; `tests/unit/initialization/test_config.py` asserts the shipped configs
against the same policy (the six execution-model classes above).

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
| `FasterWhisper` | `faster-whisper-large-v3-turbo`, `faster-whisper-tiny` | `16` | **lanes M × windows B** — M CTranslate2 workers (`num_workers`, parallel lanes) and, with `batched_inference: true`, one `encode` + one `generate` per formed batch of the admitted requests' ≤30 s windows (`whisper_window_batch_size`, default 8); the per-request pipeline's own intra-audio CT2 batch is 16 | 30 s audio clip | per-request `pipeline.transcribe` (lane pool) | Phase 11-0 final: `16` is the admission/queue depth (two-lane fairness ≤30 s vs >30 s at 3:1), M is `num_workers` (turbo 2 / tiny 4), B is the per-model `whisper_window_batch_size` (default 8, owner-approved initial), and the 16-clip intra-audio batch belongs to the per-request pipeline. The window batcher is **shipped on** in both configs (owner decision 2026-11-08: the B=1 real-model gate passed on text/segments/timings; `language_probability` moves 0.2-0.8 pp and is informational); fallbacks are `len == 1`, flag off, and any batch-level failure |
| `SileroVAD` | `silero-vad` | `16` | **loop** — no batch path, and lanes measured below break-even | — (single-stream 512-sample window) | the loop is the only path | Skip evidenced in Phase 8-5: `get_speech_timestamps` is a single-stream API; lanes on per-lane wrappers measured 1.10× / 0.95× / 0.92× at 4 / 8 / 16. `16` is queue depth only |
| `KittenTTSSynthesizer` | `kitten-tts-mini`, `-micro`, `-nano`, `-nano-int8` | `4` | **parallel lanes** — one request per lane on one shared ONNX session | — (no batch axis: every graph has a static batch dimension of 1) | one lane, inline, when the step holds a single request | Static batch dim 1 makes batching impossible without a re-export. `4` *is* the lane count; measured 1.71× at 4 lanes and 1.91× at 8 lanes (mini, this machine) |
| `ChatterboxTTS` | `chatterbox` | `4` | **AR decode** — token-level T3 decode on the shared KV pool (two CFG rows per request) plus the Phase 9-5 padded S3Gen hop for finished rows; the per-request loop is the fallback | one speech token per row per engine step (decode); speech tokens (vocode) | sequential loop (`len == 1`, `batched_inference: false`, pool refusal, per-row error) | Phase 9-6: `4` is the initial admission depth justified from the 1 GiB arena table (8 English rows × 90 MiB); pending the benchmark-host seeded-equality gate |
| `ChatterboxTurboTTS` | `chatterbox_turbo` | `4` | **AR decode** — token-level T3 decode on the shared KV pool (one row, Turbo sampler); finished rows vocode per row (meanflow hop skip); the per-request loop is the fallback | one speech token per row per engine step | sequential loop (`len == 1`, `batched_inference: false`, pool refusal, per-row error) | Phase 9-6: `4` is the initial admission depth (4 Turbo rows × 72 MiB); Turbo AR decode is committed scope, the meanflow vocode hop is the documented Phase 9-5 skip; pending the host gate |
| `ChatterboxMultilingualTTS` | `chatterbox_multilingual` | `4` | **AR decode** — same token-level T3 decode and Phase 9-5 padded S3Gen hop as English (two CFG rows per request; SDPA by default, analyzer opt-in) | one speech token per row per engine step (decode); speech tokens (vocode) | sequential loop (`len == 1`, `batched_inference: false`, pool refusal, per-row error) | Phase 12-7: `4` is the initial admission depth (8 Multilingual rows × 135 MiB at the 1152-token Perceiver-resampled slot); the analyzer hook leak/eager penalty and the `repetition_penalty=1.2` cutoff were fixed at the source |
| `ChatterboxEmbedding` | `chatterbox_embedding` | `4` | **loop with admission depth** — one `prepare_conditionals` at a time, up to 4 requests handed to the executor per engine step | — (serial execution; the depth is admission, not a batch axis) | the loop is the only path | Phase 11-0 (dated decision 2026-11-08): the embed step executes serially, but a depth of 1 serialised the clone flood at embed admission so the TTS AR step never saw more than one request. `4` is an admission depth, not a batch width |
| `ChatterboxVC` | `chatterbox_vc` | `8` | **padded batch** — one `tokenize_batched` pass over the step's 16 kHz source clips (per-row STFT, one encoder/quantizer forward) + the Phase 9-5 padded S3Gen vocode; the per-request loop is the fallback | mel frames (tokenize); speech tokens (vocode) | per-item loop (`len == 1`, `batched_inference: false`, audit-hostile/failure); a poisoned request errors alone | Phase 11-1: `8` is the initial width; the target voice comes from the Phase 7-2 cache, and the Perth watermark stays a per-request post-step (serial, measured in `docs/benchmarks/throughput.md`) |

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
   `max_batch_size × median length` of the step, capped at
   `DEFAULT_PHONE_BATCH_BUDGET = 1024`). The cap is what keeps a homogeneous
   long-sentence batch from spilling into Windows WDDM Shared GPU Memory: on
   GPU 2, `B=2` of 453-phone rows (`total_phones = 906`) already saturates
   compute (0.180 s/item vs 0.654 s at `B=1`), while `B=8` (3624 phones) is no
   faster and reserves 6512 MiB. Short/chunked rows stay under the cap and keep
   the full width.
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
frames or audio samples, `max_batch_size × median length` by default). Each
axis also has a measured area cap so a homogeneous batch of huge rows cannot
spill into Windows WDDM Shared GPU Memory: `DEFAULT_PHONE_BATCH_BUDGET = 1024`
for the synthesizers and `DEFAULT_MEL_FRAME_BATCH_BUDGET = 7500` for the
tone converters (a `B=2` 43 s paragraph already saturates compute; `B=8` of
3660-frame rows reserved 9796 MiB and stalled 400 s). Which of those steps
actually batches today is the table at the top of this page.

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

## Faster-whisper: lanes (Phase 10-1) on top of the intra-audio batch

`BatchedInferencePipeline` (faster-whisper 1.2.1) is batched **within one
audio**, not across requests: `transcribe(audio, ...)` accepts a single path,
file object or mono ndarray, runs faster-whisper's own VAD to cut it into ≤30 s
clips, and feeds those clips through `forward()` in groups of `batch_size`
(`_batched_segments_generator` → `forward` → `generate_segment_batched`).
Neither `transcribe()` nor `WhisperModel.generate_segments()` accepts a list of
audios. Folding requests into one call would mean re-implementing
`transcribe()` — per-audio VAD chunking, feature extraction, **per-audio
language detection** (and therefore a per-request tokenizer and prompt) and
timestamp restoration — on top of upstream internals, so the per-request call
stays.

What Phase 10-1 adds is execution **width** around that call:

* **`num_workers`** (per FasterWhisper entry) is passed straight to
  `WhisperModel(...)`, which maps it to CTranslate2 `inter_threads`. Same-device
  workers **share the loaded weights** and run on separate CUDA streams, so the
  cost is activations/workspace, not a second copy of the model. `1` is the
  pre-Phase-10 behaviour (CT2 serialises); the shipped initial values are
  `2` for `faster-whisper-large-v3-turbo` and `4` for `faster-whisper-tiny`,
  pending the Phase 10-5 benchmark. The CT2 docs warn the gain is
  workload-dependent, so the benchmark decides, not the knob.
* **M-worker dispatch.** A step of N requests is dispatched to the runner's own
  `ThreadPoolExecutor(max_workers=M)` (M = `num_workers`) with a semaphore
  keeping concurrent transcribes at or below M — never an executor/anyio
  default. Each worker consumes its own segment generator to the end. A step
  holding one request is a direct call on the calling thread.
* **Two-lane fairness.** Whisper admission is capacity-based and fills the step
  from a `<=30 s` and a `>30 s` lane drained at **3 short : 1 long**. Duration
  is already known at admission (the artifact bus decodes once in
  `add_request`), so lane selection costs no extra decode. The width composes
  three bounds: `max_batch_size` (capacity; `16` here is queue depth),
  the Phase 10-3 memory governor's allowed windows, and the lane credits.
  Every other model type keeps its boolean admission rule unchanged.
* **Segment-bound cancellation.** faster-whisper has no mid-`generate` abort,
  so the worker checks the request's abort/finish state between segments. An
  aborted request errors on its own; the rest of the step completes.
* **Metrics.** `/metrics` carries `hse_whisper_workers`,
  `hse_whisper_in_flight`, `hse_whisper_lane_short_depth`,
  `hse_whisper_lane_long_depth`, a per-request queue-wait histogram
  (`hse_whisper_queue_wait_seconds_*`) and, when the CT2 binding exposes them,
  `hse_whisper_ct2_queued_batches` / `hse_whisper_ct2_active_batches`
  (CTranslate2 4.4.0 does).
* **Per-request state discipline is preserved.** Each pool thread owns its
  pipeline instance (a thin view over the shared model) and resets
  `last_speech_timestamp` before every request, so one request's word-timestamp
  cursor can never shift another's.

`batch_size=16` remains the pipeline's intra-audio clip batch;
`max_batch_size: 16` is the scheduler's admission width; `num_workers` is the
number of transcribes that actually overlap.

### Cross-request window batching (Phase 10-2, shipped on as of 11-0)

Setting `batched_inference: true` on a `FasterWhisper` entry adds a second
execution path for steps with **more than one** request. **Phase 11-0 shipped it
on (owner decision 2026-11-08):** the real-model B=1 parity gate passed on all
three measured cases — text, segments, timings and language identical — with
only `language_probability` moving 0.2-0.8 pp (the batch encoder output feeds
language detection; the number is informational and the harness bounds it at
`1e-2`). Both shipped configs set `batched_inference: true`; an operator can
still set it `false` to force the proven per-request path.

* **Window formation reuses upstream.** The collector runs the same
  `get_speech_timestamps` call as `BatchedInferencePipeline.transcribe`
  (`max_speech_duration_s=30`, `min_silence_duration_ms=160`), then
  `collect_chunks` / `pad_or_trim` / the model's feature extractor. A 65 s
  speech request forms three windows (30 + 30 + 5 s), each tagged with its
  request, window index and concatenated-speech offset.
* **One encode + one generate per formed batch.** Rows from different requests
  share a batch (capped at `whisper_window_batch_size` windows, default 8,
  owner-approved initial value). CT2's
  `generate(encoder_output, prompts, …)` takes one prompt per row, so requests
  with different detected languages batch together; the KV cache and native
  beam search stay inside CT2 (vLLM's client-side beam search is explicitly not
  ported).
* **Language detection is free.** It reads the encoder output the batch is about
  to decode, dropping the second encoder pass `WhisperModel.detect_language`
  performs. One documented parity edge: when the first VAD window is shorter
  than 30 s and a second window exists, upstream detects on the concatenation
  of the first windows while the batcher detects on the first window alone.
* **Stitching matches the sequential payload.** Segments are built with
  upstream's `_split_segments_by_timestamps` + `restore_speech_timestamps`, then
  `{"text", "segments", "info"}` is serialised exactly like the per-request
  path, so VadArtifact and the wire layer are untouched.
* **Fallbacks.** `len == 1`, `batched_inference: false`, and any batch-level
  batcher failure use the per-request `pipeline.transcribe` path. A single
  poisoned request/window errors alone (collection or decoding); a batch-level
  CT2 failure falls back for the whole step, where the poisoned request errors
  alone.
* **Admission counts windows.** The Phase 10-3 governor's
  `allowed_windows` is now consumed in true 30 s window units
  (`ceil(duration / 30)` per request) instead of one window per request; a
  single long request is still admitted when nothing is in flight (progress
  floor) so its own size cannot starve it.

`hse_whisper_batch_fallbacks_total` counts batcher failures that fell back to
the per-request path.

## Chatterbox AR decode (Phase 9-3)

> **English, Turbo, and Multilingual Chatterbox are AR-batched as of Phase
> 12-7.** The runner decodes one speech token per row per engine step on the
> shared 9-2 KV-cache pool, across engine steps (9-4's "unfinished"
> scheduling). The per-request loop remains the fallback. Phase 9-6 moved
> English/Turbo into the `ar_decode` class; Multilingual joined in Phase 12-7.
> The benchmark-host seeded `batching:1-vs-N` gate is the flip condition for
> tuning the admission width.

**Mechanism.** A step prefills any newly admitted requests (one forward each —
prefill is short and per-request keeps the seeded equality gates exact), then
runs one batched forward for every active row. English and Multilingual requests
carry two CFG rows (conditional / unconditional), Turbo one. Rows are dropped on
EOS or at the token cap; finished rows hand their token stream to the Phase 9-5
padded S3Gen hop (below). Phase 12-8 releases a finished row's KV slots
immediately but defers its S3Gen vocode until every active row in the batch
finishes T3 decoding, so staggered-EOS chunks vocode together in one padded
`inference_batched` call; continuing rows also skip `prepare_inputs`.

**Reproducibility.** Every row owns a `torch.Generator` seeded from the request's
`seed`. A generator seeded like the global RNG reproduces the sequential loop
exactly (the B=1 bit-equality gate), and a row's tokens are identical whether it
decodes alone or in a batch (per-row draws, never one batched draw). With no
`seed`, the runner derives one from the model seed and a per-runner counter, so
a fresh engine replays the same request sequence.

**Fallbacks.** `len == 1` for a new request, `batched_inference: false`, a
disabled/absent pool, a refused slot grant, and a per-row hard error all keep the
sequential loop or error only that request. A poisoned row never de-batches its
neighbours (8-3 Deviation 6 discipline).

**Multilingual uses the same contract.** `ChatterboxMultilingualTTS` shares
English's `Llama_520M` decode contract, so it runs the same `ARBatchRunner`
(Phase 12-7). Its `AlignmentStreamAnalyzer` is now opt-in
(`HSE_CHATTERBOX_MTL_ANALYZER`, default off): the default path keeps native SDPA,
registers zero forward hooks and uses `repetition_penalty=2.0`, which avoids the
analyzer's premature 2-token-repeat EOS. When enabled, it switches to eager for
that call only and removes every hook in a `finally` block. VC is non-AR and
moved to the `padded_batch` class in Phase 11-1 (batched source tokenize + S3Gen
vocode); the embedder is the `loop_admission` member (Phase 11-0): it executes
serially but is admitted at depth 4 so a clone flood reaches the TTS AR step as a
real batch of requests.

**Conditioning-state deletion.** The AR path never assigns `model.conds`: the
conditionals are projected into each row's prefill, so the Phase 0-2
`finally`-restore is superseded on this path (the sequential fallback keeps it).
The decode's conditionals live on the runner's row.

AR-0 (Phase 9-1) removed two pieces of pure-loss work from the loop first: the
TTS role now decodes through HSE's cleaned path (`ChatterboxT3AR`) — no per-step
attention-output request when no analyzer is attached and no per-token progress
bar — with a seeded bit-equality gate against upstream. The before/after numbers
are `_tbd_` in [`benchmarks/throughput.md`](benchmarks/throughput.md) §Phase 9.

### Padded S3Gen vocode hop (Phase 9-5)

Finished AR rows are vocoded together instead of one at a time. The hop is the
upstream S3Gen graph (tokens + reference prompt → CFM mel → HiFT waveform); it is
**not** autoregressive, so it uses the Phase 8 padded-batch machinery rather than
the KV pool. The padding audit (`9-5-S3GenPaddedBatch.md`) measured, on a tiny
seeded instance of the real classes:

| Stage | Padding verdict | Measured leak (unmasked) |
|---|---|---|
| `UpsampleConformerEncoder` | not safe: mask after `embed`, the causal lookahead and `up_layer` | **0.4075 peak** |
| `ConditionalDecoder` (CFM) | safe by construction (per-stage masks, causal convs); only the mask *length* is rebuilt over the padded axis | ≤2e-6 (batch-kernel floor) |
| `HiFTGenerator` (f0 predictor + vocoder) | not safe: per-stage masks, plus per-row STFT/ISTFT | **0.44 peak** |
| Noise (CFM start, `SineGen` phase/noise, source noise) | per-row `torch.Generator` from the request seed | — |

The HSE-side `inference_batched` (`modeling/models/chatterbox/s3gen_batch.py`)
draws each row's noise from its own generator, pads the combined `[prompt |
speech]` token sequence and the per-row prompt mel, holds each row's padding at
zero after every stage, runs the STFT/ISTFT per row (they reflect/overlap at the
row's own end) and trims in the sample domain. A row's audio is identical alone
or in a batch; `B=1` with the same seed is bit-identical to upstream.

One 9-5 validation was too strict for VC and was relaxed in 11-1: upstream
`S3Gen.embed_ref` trims reference tokens to `mel_frames // 2` and warns, so a
real target's prompt mel is **not** exactly `2 × prompt_token_len` (measured
181 frames for 90 tokens). The flow itself takes the prompt length from
`prompt_feat.shape[1]`, so the hop now only requires the prompt to leave at
least one generated frame; the old strict check had made every real VC target
fall back to the loop.

Handler wiring: finished rows are grouped with `plan_batches` on the
**speech-token** axis (`batch_token_budget` reused) and one forward runs per
sub-batch. `len == 1`, `batched_inference: false`, an unsupported graph, or any
batch failure falls back to the per-row upstream call (seeded, so the fallback is
audio-identical); a poisoned row errors alone. The VC vocode step is the same
graph; Phase 11-1 wires it into the multi-request VC path and adds the batched
source tokenizer (see [Chatterbox VC batching](#chatterbox-vc-batching-phase-11-1)).

**Turbo (meanflow) is the documented hop skip.** Its solver is a different graph
with two batch-1 assumptions (`flow_inference` hard-codes `torch.randn(1, 80, …)`
and the `noised_mels` placement uses one prompt length); the Turbo AR decode is
9-3's committed scope, the non-AR vocode hop stays per row.

Operators who need more Chatterbox throughput than the shipped admission depth
**today** can still run more engine replicas (more processes, each pinned to a
GPU or sharing one with `CUDA_VISIBLE_DEVICES` and per-process VRAM headroom).
The shipped width is `4` requests per step; the shared 2.75 GiB pool can grant
the slots for it at the contracts' preferred caps (1152 tokens for
English/Multilingual, 1536 for Turbo), and surplus requests fall back to the
sequential loop rather than queueing forever. Measured on one RTX 5060 Ti,
Chatterbox throughput was flat at ~0.6 requests/s from concurrency 1 to 8 under
the pre-Phase-9 per-request loop (`docs/benchmarks/throughput.md`); the AR-runner
curve is the Phase 9 benchmark block.

## Chatterbox VC batching (Phase 11-1)

Voice conversion is source audio → S3 speech tokens → the same S3Gen vocode the
TTS path uses. Before 11-1 a multi-request VC step still ran the **source
tokenizer** one forward per request (`_chatterbox_vc_tokens`), so even a wide
admission had serial model forwards. Phase 11-1 adds the audited batched source
tokenizer and flips the family to `padded_batch` with an initial width of `8`.

The padding audit (`11-1-ChatterboxVCBatching.md`) measured, on a tiny seeded
instance of the real upstream classes:

| Stage | Padding verdict | Measured leak (unmasked) |
|---|---|---|
| `torch.stft` / `log_mel_spectrogram` | not bit-safe: `torch.stft` reflects at the row's own end, so a padded row's last valid mel frame reads zeros instead of its reflection → run **per row** | **0.21–0.51** max \|Δmel\| at that one frame (longest row exact) |
| `AudioEncoderV2` (conv1/conv2 + FSMN conformer) | safe by construction: masks each row's padding before `conv1` and `conv2`, zeroes padded FSMN inputs/outputs and masks padded attention keys | **0.0** hidden-state delta at valid positions |
| `FSQVectorQuantization` | safe: a per-position `Linear`, no cross-position mixing; padded positions are sliced off by the tokenizer's own token length | **0/36** rows with a token flip over 12 seeds |
| Long-audio path (`mel_len > 3000`, 30 s) | safe: same padded encoder, true segment lengths; short rows are padded to the window and masked | identical (775-token >30 s row) |
| Poisoned mel padding (`1e3`) | never read | bit-identical tokens |

The tokenizer returns `(tokens, token_lens)`: the second value is the per-row
**token length** (not a hidden state), and it is what `tokenize_batched` slices
with — never a padded-shape assumption.

The HSE-side `tokenize_batched` (`modeling/models/chatterbox/s3gen_batch.py`)
therefore runs the STFT/log-mel **per row** (exactly upstream's own call),
pads the mels on the frame axis, and runs **one** encoder/quantizer forward for
the whole step. `B=1` is bit-identical to upstream `tokenizer(clip[None])`;
mixed-length rows produce their solo token streams. The per-row `librosa`
decode is deliberately kept (decode hygiene is fenced out of scope: `librosa`'s
resampler differs numerically from `torchaudio`'s, so changing it must first
prove token-level equivalence).

Handler wiring (`_try_batched_chatterbox_vc`): the target voice is resolved from
precomputed conditionals / the Phase 7-2 cache / `set_target_voice`; every
source clip is decoded per row and tokenized in one pass; rows are vocoded in
length-similar sub-batches through the Phase 9-5 `inference_batched` hop. The
request seed is threaded into each `S3GenRow` (and the loop fallback seeds the
global RNG the same way), so a seeded request's audio is identical alone or in
a batch, and a batch failure is a performance fallback, not an audio change.
`len == 1`, `batched_inference: false`, an audit-hostile tokenizer graph or any
batch failure uses the per-item loop; a poisoned request errors alone.

**Watermark stays serial.** Perth runs once per output audio in the handler
(`_artifact_from_waveform`), after the batched hop; it is not part of the batch.
**Perth verdict (owner decision 2026-11-08):** no batch audit — it is a
per-request post-step and its measured cost is ~1.5 % of request service time
([`benchmarks/throughput.md`](benchmarks/throughput.md#perth-watermark-cost-owner-request-2026-11-08)).
Multilingual stays the documented AR skip; the meanflow Turbo vocode hop stays
the Phase 9-5 documented skip.

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
| Chatterbox TTS | 0.6 req/s | 0.6 | 0.6 | **0.6** | pre-9: autoregressive, one request per step; post-9 AR runner `_tbd_` |
| Whisper tiny STT | 2.7 req/s | 2.8 | 2.8 | **2.8** | pre-10: batch inside one request; post-10 lanes/windows `_tbd_` |

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

Long-form TTS chunking (Phase 12-2) does not add a new scheduling path: every
chunk of a `split_text` request is submitted as an ordinary admitted request
with its own id, so the chunks co-batch under exactly the rules above. All chunk
sub-requests are queued before any of their results are consumed, which lets a
single multi-chunk request fill one admission batch by itself; the serving layer
then stitches the per-chunk results in chunk-index order. Chunk sub-requests
carry the same model, voice and generation options as the original request (with
a deterministic per-chunk seed derived from the request seed), and the clone
reference embedding is resolved once (not once per chunk).
