# Availiable models and Speech Frameworks

To be able to use a model for Inference tasks, Harmony Speech Engine needs to load the model on startup.
As soon as the model has been loaded, it is eglible for processing direct requests or requests as part of the internal
routing chain of a framework.

The following Models and Speech Frameworks are currently supported. YAML config examples are being provided below.

### Startup validation of model toolchains

A *model toolchain* (pipeline) is only usable when every mandatory step it needs
can be resolved to a configured model — for example `openvoice_v2` needs a VAD
model (Faster-Whisper or Silero VAD), the OpenVoice V2 tone-converter encoder,
MeloTTS and the OpenVoice V2 tone converter. On startup the engine checks each
declared pipeline and:

- logs a warning naming every incomplete pipeline *and the missing step* (with
  the model types that would satisfy it),
- exposes the list of complete multi-step toolchains as `available_families`,
- advertises **only complete toolchains** on the model endpoints
  (`/v1/audio/speech/models`, `/v1/embed/models`, …). Individual model cards are
  still advertised for every configured executor, so a partially configured
  deployment can still be addressed by concrete model name.

Startup does not fail on an incomplete toolchain by default: operators may
deliberately run a subset of the catalogue. Requests routed to an incomplete
toolchain are rejected immediately with an error response instead of waiting in
the queue. Pass `--strict-pipelines` (or set `HSE_STRICT_PIPELINES=1`, or
`strict_pipelines: true` in the config file) to turn those warnings into a boot
error.

Model cards are generated from the same declaration: each card lists its
`capabilities` (`tts`, `embed`, `vc`, `stt`, `translation`, `vad`, `restore`),
its `models` (the configured member executors) and the languages of its
configured variants (for example every configured MeloTTS language).

#### `input_vad_mode`

The OpenVoice pipelines accept `input_vad_mode: "whisper" | "silero"` to choose
the VAD backend for the reference-audio segmentation. `"whisper"` (or omitting
the field) uses Faster-Whisper; `"silero"` uses Silero VAD when one is
configured and otherwise falls back to Faster-Whisper with a warning. See
[Request routing](routing.md).

The following table should help with evaluating the requirements (footprint
figures are **measured** on the shipped GPU config after Phase 4 — see
[VRAM profiles](benchmarks/vram-profiles.md) for the full per-model table):

| Model Toolchain   | Type         | Implementation | Embedding requires Whisper | CPU only Performance | Voice / Processing Quality | Measured VRAM / RAM |
|-------------------|--------------|----------------|----------------------------|----------------------|----------------------------|---------------------|
| Harmony Speech V1 | TTS          | Adapted        | No                         | Very Good            | ++                         | 173 MiB host RAM    |
| OpenVoice V1      | TTS          | Adapted        | Yes                        | Good                 | +++                        | ~405 MiB per language (synth + converters) |
| OpenVoice V2      | TTS          | Adapted        | Yes                        | Good                 | +++                        | ~452 MiB per language (synth + converters) |
| Faster-Whisper    | STT <br>Translation <br>VAD* | Native         | N / A                      | Medium               | STT: +++ <br>VAD*: +       | ~75 MiB (tiny) – 1.6 GiB (large-v3-turbo) |
| Silero-VAD        | VAD          | Native         | N / A                      | Very Good            | +++                        | ~2 MiB host RAM     |
| VoiceFixer        | Converter    | Adapted        | N / A                      | Bad                  | input-dependent            | 402 MiB (restorer + vocoder) |
| Chatterbox        | TTS <br>VC   | Native         | N / A                      | n/a (GPU)            | +++                        | 2068 MiB (TTS) · 3413 MiB (Turbo) · 1025 MiB (multilingual) · 0 MiB marginal (VC/embedding, shared) |
| KittenTTS         | TTS          | Native         | N / A                      | Very Good            | ++                         | 25–80 MiB host RAM per variant |

\* While Whisper (or STT models in general) can work for pure VAD use cases, too, they require significantly more
processing time and the precision isn't as good. Reason for this is because they don't just check for voice activity,
but try to actually understand what is being said, which sometimes can also cause hallucinations (e.g. "understanding
words" in audio which has no speech in it; for example, breathing into a microphone) 

### Native rates and the resample contract

Every model pipeline decodes input audio once at the API boundary and hands the
model the rate it expects (16 kHz for Faster-Whisper/Silero VAD and the Harmony
Speech encoder, 44.1 kHz for VoiceFixer, the model's `sampling_rate` for
OpenVoice/MeloTTS, 24 kHz for Chatterbox). Output audio is produced at the
model's native rate:

| Model / toolchain | Native output rate |
|---|---|
| KittenTTS (all variants) | 24 kHz |
| Chatterbox TTS / Turbo / Multilingual / VC | 24 kHz |
| MeloTTS & OpenVoice V2 synthesizer | model config `sampling_rate` |
| OpenVoice V1 synthesizer | model config `sampling_rate` |
| Harmony Speech V1 | model config `sampling_rate` |
| VoiceFixer | 44.1 kHz |

Clients that need a specific rate should set `output_options.sample_rate` (one of
8000, 16000, 22050, 24000, 32000, 44100, 48000); the engine resamples once with
SoXR at the API boundary. See the [Audio I/O contract](audio-io-contract.md).

### Precision (`dtype`) and shared backbones

`dtype` is per config entry (`auto`, `float16`, `bfloat16`, `float32`) and is
applied to *torch* model weights only:

* **Chatterbox** (`ChatterboxTTS/TurboTTS/MultilingualTTS/VC/Embedding`):
  `float16` casts the **T3 backbone** (the large autoregressive model). S3Gen —
  the flow-matching decoder plus HiFiGAN vocoder — and the VoiceEncoder stay
  `float32`. Measured: casting S3Gen to half precision audibly degrades
  synthesis (sampled tokens change *after* the AR stage), so it is pinned; the
  VoiceEncoder's LSTM path is float32-only by construction. The Perth
  watermarking network is also kept float32.
* **OpenVoice V1/V2 and MeloTTS** stay `float32`: the adapted VITS code creates
  float32 tensors inside the flow/text encoder, and the MeloTTS flow splines
  assert under fp16. These are documented opt-outs, not oversights; batching is
  orthogonal to dtype (the padded-batch forward works in float32).
* **VoiceFixer** stays `float32` (precision-sensitive STFT/log-mel restoration;
  GPU inference has a pre-existing device-placement bug).
* **Harmony Speech V1** runs on CPU in the shipped GPU config; on CPU a requested
  `float16` is rewritten to `bfloat16` by the CPU executor.
* **KittenTTS** (ONNX) and **Silero VAD** (ONNX) ignore torch `dtype` — the
  variant is chosen by the `model` name. **Faster-Whisper** is CTranslate2 and
  selects its precision with `compute_type` (default `float16` on CUDA,
  `int8` on CPU); `dtype` does not affect it.

`backbone_group` (optional, any entry) makes configs that point at the same
checkpoint + revision + dtype **share backbone components** instead of loading a
copy each. The shipped `config.gpu.yml` puts `chatterbox`,
`chatterbox_multilingual`, `chatterbox_vc` and `chatterbox_embedding` in the
`chatterbox-base` group (one S3Gen + one VoiceEncoder; the embedding role also
shares T3, which it never synthesizes with). Every role keeps its own wrapper
instance, so per-request conditioning (`conds` / `ref_dict`) remains isolated.
`chatterbox_turbo` stays ungrouped — it synthesizes from a different
(`s3gen_meanflow`) checkpoint.

#### Batching

`max_batch_size` states how many requests a model processes per engine step, and
`batched_inference` (default on for the OpenVoice V1 and MeloTTS synthesizers, the
two speaker-embedding steps (Phase 8-2), the two tone converters (Phase 8-3) and
the two vocoders plus the VoiceFixer restorer (Phase 8-4)) selects the
padded-batch forward for one-shot synthesis, embedding, tone conversion,
vocoding and restoration. For **Faster-Whisper** the same flag is **opt-in** and
selects the cross-request window batcher (one CT2 `encode` + one `generate` for
all admitted requests' ≤30 s windows, up to 8 per formed batch); without it the
per-request pipeline path (and the `num_workers` lanes) is used. The Harmony
Speech synthesizer keeps a per-item loop (its vendored generator is not
padding-safe; see [Batching & concurrency](batching.md)). Chatterbox is
autoregressive and runs one request per step, so its capacity comes from
replicas. See [Batching & concurrency](batching.md) for the per-family table, the
length-similar sub-batching rules and the measured throughput.

#### Measured footprint (RTX 5060 Ti 16 GiB, full `config.gpu.yml` catalogue)

| Model toolchain | Implementation | Measured VRAM (after Phase 4) | Notes |
|---|---|---|---|
| Harmony Speech V1 | Adapted | 173 MiB host RAM (3 executors) | CPU-pinned |
| OpenVoice V1 | Adapted | ~153 MiB per language synth + 126 MiB × 2 converters | float32 |
| OpenVoice V2 / MeloTTS | Adapted | ~199 MiB per language synth + 126–127 MiB × 2 converters | float32 |
| Faster-Whisper | Native | ~1.6 GiB (large-v3-turbo) / ~75 MiB (tiny) | CT2, outside the torch allocator |
| Silero VAD | Native | ~2 MiB host RAM | ONNX, CPU |
| VoiceFixer | Adapted | 273 MiB restorer + 129 MiB vocoder | float32 |
| KittenTTS (4 variants) | Native | 25–80 MiB host RAM each | ONNX, CPU |
| Chatterbox (TTS) | Native | 2068 MiB | fp16 T3 + deduped S3Gen/VE |
| Chatterbox Turbo | Native | 3413 MiB | fp16 T3; own meanflow S3Gen |
| Chatterbox Multilingual | Native | 1025 MiB marginal | fp16 T3; S3Gen/VE shared |
| Chatterbox VC / Embedding | Native | 0 MiB marginal | fully shared backbones |

Whole catalogue: **8.52 GiB of torch allocations (8.83 GiB via `nvidia-smi`) on
a 15.93 GiB card**, versus 16.45 GiB (fp32, duplicated) before Phase 4 — which
overflowed the card into shared system memory. Details and the acceptance-floor
analysis: [VRAM profiles](benchmarks/vram-profiles.md); quality gates:
[audio regression](benchmarks/audio-regression.md).

### Model Types
- **Native** / **Third-Party** Models: build on top of other Repositories. This is the preferred way of adding new
  models or frameworks to Harmony Speech Engine, since it minimizes implementation effort.
- **Adapted** models: have their Neural Network Module code and parts of their repository's code copied over into 
  Harmony Speech Engine's codebase to allow for optimizations and supporting these models independent of the original
  group or individual who released it. Not all models or Frameworks are well maintained and released with processing
  optimizations ot toolchain interoperability in mind. So if the License allows it, we take the existing codebase and
  apply the necessary adaptions to allow for best performance when doing inference.

## Individual Models

### Transcription

#### Faster Whisper / Distil-Whisper
Whisper is a SOTA Audio Transcription model developed by OpenAI. ([Original Repo](https://github.com/openai/whisper))

Faster-Whisper is a Performance-Improved Version of Whisper, reimplemented usinc Ctranslate2.

Harmony Speech Engine is compatible with all Faster Whisper and Faster Distil-Whisper models supported by 
SYSTRAN's Implementation.

Supports Local model path or Huggingface-Resolving by Model Name as described in the github repo below. Just put the
name of the desired model into the `model` field of the configuration.

**Note on VAD**: While Faster Whisper can perform Voice Activity Detection, it is significantly less suited for this task compared to specialized VAD models like Silero VAD. Faster Whisper often exhibits worse performance and a higher rate of false detections for VAD purposes due to its primary focus on transcription. For dedicated VAD tasks, using Silero VAD is highly recommended.

Implementation Type: Native / Third-Party

Links: 
- [Github: Code, Model Overview, Benchmarks](https://github.com/SYSTRAN/faster-whisper)
- [Availiable Model Tags+Sizes](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py#L12)

##### YAML Config Example
```
model_configs:
  - name: "faster-whisper"
    model: "large-v3-turbo"
    model_type: "FasterWhisper"
    max_batch_size: 16
    dtype: "float32"
    device_config:
      device: "cuda:0"
```

##### Whisper VRAM governance

CTranslate2 has no memory-cap knob of its own and allocates outside PyTorch's
caching allocator, so Whisper's device memory is invisible to the torch VRAM
counters. The engine therefore measures it directly and bounds it on its own
side.

Whisper's capacity is deliberately three separate numbers: `max_batch_size: 16`
is the **admission/queue depth** (two fairness lanes, ≤30 s vs >30 s at 3:1),
`num_workers` is the number of transcribes that actually **overlap** (M
CTranslate2 workers on shared weights), and the **window cap** (8) is the memory
bound on one formed cross-request batch. The per-request pipeline's own
intra-audio clip batch stays 16. `docs/batching.md` holds the final matrix.

* On startup, each CUDA-placed Whisper config runs a **max-shape probe**
  (`num_workers` concurrent workers × `window_batch` windows × the longest
  admitted audio). The device-level `torch.cuda.mem_get_info` delta is recorded
  as that config's `peak_activation_bytes` and reported per model on `/health`
  with a "measured probe" annotation (a skipped probe says why it was skipped —
  it is never replaced by a guess).
* The boot check compares the projected Whisper footprint (checkpoint weights +
  probe peaks) with the measured free device headroom. Set
  `whisper_vram_budget_bytes` to turn an overshoot into a boot error; without it
  the check only warns.
* At runtime, admission shrinks the number of concurrent Whisper windows as the
  device-level used memory approaches the budget (or the device total when no
  budget is configured). The decision is a pure function driven by
  `torch.cuda.mem_get_info`, so it accounts for every resident family, not just
  Whisper.
* `whisper_max_audio_seconds` (default `300`) is the probe's **max shape** (the
  longest audio the startup probe validates) and the whisper-side duration bound
  used for capacity planning. It is not an admission rejection; set it to at
  least the longest audio you intend to admit so the measured probe peak is
  honest. The general endpoint upload limit remains a separate concern.
* `num_workers` (per Whisper model entry) is the CTranslate2 worker count
  (`inter_threads`) and the width of the execution lanes (Phase 10-1). Workers
  share the loaded weights, so this raises concurrency at the cost of
  activations, not a second model copy. Default `1` reproduces the pre-Phase-10
  behaviour; the shipped initial values are `2` (large-v3-turbo) and `4` (tiny),
  pending benchmark tuning.
* `batched_inference: true` (opt-in) enables the cross-request window batcher
  (Phase 10-2): the step's ≤30 s windows are folded into one CT2 `encode` +
  one `generate` per formed batch (cap 8 windows), with per-request language
  prompts and the same segment/timestamp stitching as the sequential path.
  `max_batch_size` remains the admission width; a batch-level failure falls back
  to the per-request pipeline, and `/metrics` exposes
  `hse_whisper_batch_fallbacks_total`. The shipped configs keep it **unset**
  until the benchmark-host B=1 parity gate passes (dated decision 2026-11-07),
  so the default remains the proven per-request path.

```yaml
# config.yml (top level, next to request_timeout_s)
whisper_vram_budget_bytes: 12884901888   # optional 12 GiB device-level ceiling
whisper_max_audio_seconds: 300

# per model entry
- name: faster-whisper-large-v3-turbo
  model_type: FasterWhisper
  max_batch_size: 16
  num_workers: 2
```

`hse_vram_headroom_bytes` is now true free device memory (a breaking change: it
used to be `device total − torch reserved`). `hse_vram_device_used_bytes` is the
device-level counterpart and includes CTranslate2. See
[VRAM profiles](benchmarks/vram-profiles.md#phase-10-3--device-level-truth).

---

### Voice Activity Detection (VAD)

#### Silero VAD
Silero VAD is a state-of-the-art (SOTA) voice activity detection model, optimized for detecting speech in audio samples. It is available in ONNX format for high-performance inference.

Silero VAD is integrated as a native model, allowing for efficient CPU-based processing. It supports dynamic parameters via the API for fine-tuning detection sensitivity.

Implementation Type: Native / Third-Party

Links:
- [Github: PyTorch Source Code](https://github.com/snakers4/silero-vad)
- [Huggingface: ONNX Model Sources](https://huggingface.co/onnx-community/silero-vad)

##### YAML Config Example
```
model_configs:
  - name: "silero-vad"
    model: "silero-vad" # Model name (not used for native models)
    model_type: "SileroVAD"
    load_format: "onnx" # Use ONNX version for better performance
    max_batch_size: 16
    dtype: "float32"
    device_config:
      device: "cpu" # Silero VAD is CPU-optimized
```

---

### Audio Restoration

#### VoiceFixer
VoiceFixer is a neural network-based system for restoring degraded speech audio developed by Haohe Liu. It can handle various types of audio degradation including background noise, reverberation, low resolution audio, audio clipping, and general audio artifacts.

VoiceFixer processes audio internally at 44.1kHz and uses a two-stage pipeline:
1. **VoiceFixerRestorer**: Audio enhancement and noise reduction
2. **VoiceFixerVocoder**: Mel-spectrogram to audio conversion

The models are accessible through the `/v1/audio/convert` endpoint for audio restoration workflows. 

Both stages run on the configured device (CPU or CUDA) and, when more than one
restoration request reaches the same model in one engine step, as one padded
batch per step (Phase 8-4).

Implementation Type: Adapted

Links:
- [Original Paper](https://arxiv.org/abs/2109.13731): "VoiceFixer: Toward General Speech Restoration With Neural Vocoder"
- [Original Repository](https://github.com/haoheliu/voicefixer)
- [Huggingface Models](https://huggingface.co/jlmarrugom/voice_fixer)

##### YAML Config Example
```
model_configs:
  - name: "voicefixer-restorer"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerRestorer"
    max_batch_size: 4
    dtype: "float32"
    device_config:
      device: "cuda:0"

  - name: "voicefixer-vocoder"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerVocoder"
    max_batch_size: 4
    dtype: "float32"
    device_config:
      device: "cuda:0"
```

---

## TTS Frameworks

### Harmony Speech V1
Harmony Speech is a High Performance One-Shot-Voice-Cloning TTS Engine developed by Project Harmony.AI, which has been
optimized for CPU-Only, Faster-Than-Realtime Inference. It currently only supports an English speaking Synthesizer.

It consists of 3 Models:
- Speaker Encoder Model (Based on [CorentinJ's G2E Implementation](https://github.com/CorentinJ/Real-Time-Voice-Cloning))
- Speech Synthesis Model (Based on [Forward-Tacotron](https://github.com/as-ideas/ForwardTacotron))
- Vocoder Model (Based on [kan-bayashi's MultiBand-MelGAN implementation](https://github.com/kan-bayashi/ParallelWaveGAN))

Harmony Speech V1 model weights are being released under the Apache License.

Supports Local model path or Huggingface-Resolving by Model Name.

Implementation Type: Adapted

Links: 
- [Huggingface](https://huggingface.co/harmony-ai/harmony-speech-v1)

##### YAML Config Example
```
model_configs:
  - name: "hs1-encoder"
    model: "harmony-ai/harmony-speech-v1"
    model_type: "HarmonySpeechEncoder"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "hs1-synthesizer"
    model: "harmony-ai/harmony-speech-v1"
    model_type: "HarmonySpeechSynthesizer"
    max_batch_size: 1
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "hs1-vocoder"
    model: "harmony-ai/harmony-speech-v1"
    model_type: "HarmonySpeechVocoder"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"
```
(Speaker Encoder is only needed if you want to create new embeddings or do a full voice clone)

---

### OpenVoice V1
OpenVoice is a TTS Framework created by MyShell AI.
V1 focuses on voice cloning for English and Chinese language.
Also it supports emotional voice styles for English.

It consists of 2 Models:
- Speaker Encoder & Voice Conversion Model
- Single Speaker TTS Model

The Voice conversion model is trained to convert Voices between two Speakers using Vocal alignment. Technically, it is
possible to do that between arbitrary speakers, but best results can be achieved when using the Single Speaker TTS
model outputs.

Additionaly, the embedding step requires a VAD / ASR model like Whisper to function.

Supports Local model path or Huggingface-Resolving by Model Name.

Implementation Type: Adapted

| Supported Language Tags |                           Supported Voice IDs / Styles                            |
|:-----------------------:|:---------------------------------------------------------------------------------:|
|           EN            | default, whispering, shouting, excited, cheerful, terrified, angry, sad, friendly |
|           ZH            |                                      default                                      |

Links: 
- [Github](https://github.com/myshell-ai/OpenVoice)
- [Huggingface](https://huggingface.co/myshell-ai/OpenVoice)

##### YAML Config Example
```
model_configs:
  - name: "ov1-synthesizer-en"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1Synthesizer"
    language: "EN"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-synthesizer-zh"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1Synthesizer"
    language: "ZH"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-tone-converter"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1ToneConverter"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-tone-converter-encoder"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1ToneConverterEncoder"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "faster-whisper"
    model: "large-v3"
    model_type: "FasterWhisper"
    max_batch_size: 16
    dtype: "float32"
    device_config:
      device: "cuda:0"
```
(For best performance, Harmony Speech Engine runs OpenVoice models "task-based", which means we define different model
types for Encoding and Voice Conversion, despite it's the same weights. Since the TTS model weights differ per language,
we need to load all languages we want to support in parallel.)

---

### OpenVoice V2
OpenVoice is a TTS Framework created by MyShell AI.
V2 focuses on voice cloning for multiple languages, building on top of a new TTS System called MeloTTS
Also it supports multiple dialect styles for English.

It consists of 2 Models:
- Speaker Encoder & Voice Conversion Model
- Single Speaker TTS Model

The Voice conversion model is trained to convert Voices between two Speakers using Vocal alignment. Technically, it is
possible to do that between arbitrary speakers, but best results can be achieved when using the Single Speaker TTS
model outputs.

Additionaly, the embedding step requires a VAD / ASR model like Whisper to function.

Supports Local model path or Huggingface-Resolving by Model Name.

Implementation Type: Adapted

| Model Name                                                                 | Supported Language Tags |       Supported Voice IDs / Styles        |
|----------------------------------------------------------------------------|:-----------------------:|:-----------------------------------------:|
| [MeloTTS-English](https://huggingface.co/myshell-ai/MeloTTS-English)       |      EN (English)       | EN-Default, EN-US, EN-BR, EN-INDIA, EN-AU |
| [MeloTTS-English-v2](https://huggingface.co/myshell-ai/MeloTTS-English-v2) |      EN (English)       |       EN-US, EN-BR, EN-INDIA, EN-AU       |
| [MeloTTS-English-v3](https://huggingface.co/myshell-ai/MeloTTS-English-v3) |      EN (English)       |                 EN-Newest                 |
| [MeloTTS-Chinese](https://huggingface.co/myshell-ai/MeloTTS-Chinese)       |      ZH (Chinese)       |                    ZH                     |
| [MeloTTS-Spanish](https://huggingface.co/myshell-ai/MeloTTS-Spanish)       |      ES (Spanish)       |                    ES                     |
| [MeloTTS-French](https://huggingface.co/myshell-ai/MeloTTS-French)         |       FR (French)       |                    FR                     |
| [MeloTTS-Japanese](https://huggingface.co/myshell-ai/MeloTTS-Japanese)     |      JP (Japanese)      |                    JP                     |
| [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)         |       KR (Korean)       |                    KR                     |

Links: 
- [Github](https://github.com/myshell-ai/OpenVoice)
- [Huggingface](https://huggingface.co/myshell-ai/OpenVoiceV2)

##### YAML Config Example
ATTENTION: Only one model per language is currently supported; you can NOT route to English-v3 and English-v2 at the
same time currently.
```
model_configs:
  - name: "ov2-synthesizer-en"
    model: "myshell-ai/MeloTTS-English-v3"
    model_type: "MeloTTSSynthesizer"
    language: "EN"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-synthesizer-zh"
    model: "myshell-ai/MeloTTS-Chinese"
    model_type: "MeloTTSSynthesizer"
    language: "ZH"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-tone-converter"
    model: "myshell-ai/openvoicev2"
    model_type: "OpenVoiceV2ToneConverter"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-tone-converter-encoder"
    model: "myshell-ai/openvoicev2"
    model_type: "OpenVoiceV2ToneConverterEncoder"
    max_batch_size: 8
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "faster-whisper"
    model: "large-v3"
    model_type: "FasterWhisper"
    max_batch_size: 16
    dtype: "float32"
    device_config:
      device: "cuda:0"
```
(For best performance, Harmony Speech Engine runs OpenVoice models "task-based", which means we define different model
types for Encoding and Voice Conversion, despite it's the same weights. Since the TTS model weights differ per language,
we need to load all languages we want to support in parallel.)
