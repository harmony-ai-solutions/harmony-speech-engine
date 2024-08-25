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
<br>Link: [Github](https://github.com/SYSTRAN/faster-whisper)

##### YAML Config Example
```
model_configs:
  - name: "faster-whisper"
    model: "large-v3"
    model_type: "FasterWhisper"
    max_batch_size: 16
    dtype: "float32"
    device_config:
      device: "cuda:0"
```

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
<br>Link: [Huggingface](https://huggingface.co/harmony-ai/harmony-speech-v1)

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
    model: "models/harmony-ai-solutions/harmonyspeech"
    model_type: "HarmonySpeechSynthesizer"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"

  - name: "hs1-vocoder"
    model: "models/harmony-ai-solutions/harmonyspeech"
    model_type: "HarmonySpeechVocoder"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu"
```


### OpenVoice V1


### OpenVoice V2