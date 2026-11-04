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
planning:

| Family / handler | `max_batch_size` | What the executor really does |
|---|---|---|
| `ov1-synthesizer-*`, `ov2-synthesizer-*` (OpenVoice V1, MeloTTS) | `8` | One padded `[B, T_max]` forward per step (`batched_inference: true`) |
| `faster-whisper-*` | `16` | `BatchedInferencePipeline(batch_size=16)` (unchanged) |
| `silero-vad` | `16` | Cheap CPU loop; admission is honest, latency scales linearly |
| `kitten-tts-*` | `4` | Cheap CPU ONNX loop |
| `chatterbox*` (TTS / Turbo / Multilingual / VC / Embedding) | `1` | Autoregressive per-request loop — see "Chatterbox scales by replicas" |
| `hs1-*` (HarmonySpeech encoder/synth/vocoder) | `1` | Per-item loop |
| `ov1/ov2-tone-converter*` | `1` | Per-item loop |
| `voicefixer-*` | `1` | Per-item loop |

An entry that declares a value its handler cannot honour is a bug: it makes
`/metrics`, dashboards and capacity plans lie. `tests/unit/initialization/test_config.py`
asserts the shipped configs against this table.

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

## Chatterbox scales by replicas, not batch

Chatterbox's T3 backbone generates autoregressively (one token at a time, with
a KV cache per sequence). Batching it properly needs a per-sequence KV-cache
scheduler; until that exists, its executors process one request at a time and
`max_batch_size` is `1` for every Chatterbox entry.

Operators who need more Chatterbox throughput should run **more engine
replicas** (more processes, each pinned to a GPU or sharing one with
`CUDA_VISIBLE_DEVICES` and per-process VRAM headroom) instead of raising
`max_batch_size`. Measured on one RTX 5060 Ti, Chatterbox throughput is flat at
~0.6 requests/s from concurrency 1 to 8 while p50 latency grows linearly — with
`max_batch_size: 1` extra concurrency queues instead of speeding anything up
(`docs/benchmarks/throughput.md`).

## What concurrency to expect

From `docs/benchmarks/throughput.md` (RTX 5060 Ti, float32 for these families,
`requests/level = 16`):

| Family | c=1 | c=2 | c=4 | c=8 | Limiting factor |
|---|---|---|---|---|---|
| MeloTTS synthesizer (short text) | 8.3 req/s | 8.6 | 11.5 | **14.4** | padded batch, launch-bound |
| OpenVoice V2 clone (full pipeline) | 1.8 req/s | 1.9 | 1.9 | **1.9** | per-item tone-converter hops |
| Chatterbox TTS | 0.6 req/s | 0.6 | 0.6 | **0.6** | autoregressive, one request per step |
| Whisper tiny STT | 2.7 req/s | 2.8 | 2.8 | **2.8** | batch inside one request; queueing above c=1 |

Practical guidance:

* **Sentence-level clients** (short texts) benefit most from synthesis batching;
  throughput keeps rising up to `max_batch_size`.
* **Full cloning toolchains** are bound by the tone-converter hops
  (`max_batch_size: 1` by design in Phase 5), not by synthesis: raising the
  synthesizer batch alone does not raise end-to-end throughput.
* **Long texts** are compute-bound: a single utterance already fills the GPU, so
  batching gives little to nothing. Do not size your fleet assuming batch
  scaling for long-form synthesis.

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
