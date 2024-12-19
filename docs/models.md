# Availiable models and Speech Frameworks

To be able to use a model for Inference tasks, Harmony Speech Engine needs to load the model on startup.
As soon as the model has been loaded, it is eglible for processing direct requests or requests as part of the internal
routing chain of a framework.

The following Models and Speech Frameworks are currently supported. YAML config examples are being provided.

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
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "hs1-synthesizer"
    model: "harmony-ai/harmony-speech-v1"
    model_type: "HarmonySpeechSynthesizer"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "hs1-vocoder"
    model: "harmony-ai/harmony-speech-v1"
    model_type: "HarmonySpeechVocoder"
    max_batch_size: 10
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
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-synthesizer-zh"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1Synthesizer"
    language: "ZH"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-tone-converter"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1ToneConverter"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov1-tone-converter-encoder"
    model: "myshell-ai/openvoice"
    model_type: "OpenVoiceV1ToneConverterEncoder"
    max_batch_size: 10
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
| [MeloTTS-Chinese](https://huggingface.co/myshell-ai/MeloTTS-Chinese)       |      ZH (Chinese)       |                  default                  |
| [MeloTTS-Spanish](https://huggingface.co/myshell-ai/MeloTTS-Spanish)       |      ES (Spanish)       |                  default                  |
| [MeloTTS-French](https://huggingface.co/myshell-ai/MeloTTS-French)         |       FR (French)       |                  default                  |
| [MeloTTS-Japanese](https://huggingface.co/myshell-ai/MeloTTS-Japanese)     |      JP (Japanese)      |                  default                  |
| [MeloTTS-Korean](https://huggingface.co/myshell-ai/MeloTTS-Korean)         |       KR (Korean)       |                  default                  |

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
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-synthesizer-zh"
    model: "myshell-ai/MeloTTS-Chinese"
    model_type: "MeloTTSSynthesizer"
    language: "ZH"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-tone-converter"
    model: "myshell-ai/openvoicev2"
    model_type: "OpenVoiceV2ToneConverter"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "ov2-tone-converter-encoder"
    model: "myshell-ai/openvoicev2"
    model_type: "OpenVoiceV2ToneConverterEncoder"
    max_batch_size: 10
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