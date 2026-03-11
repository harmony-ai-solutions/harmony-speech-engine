# Phase 4: E2E Testing - TTS Models - Context

**Gathered:** 2026-03-03 (updated 2026-03-04)
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **true end-to-end tests** for TTS inference pipelines (KittenTTS, MeloTTS/OpenVoice V2, OpenVoice V1, HarmonySpeech). Each test validates that a request travels the full application stack — from the endpoint serving handler, through the async engine and scheduler, down to the CPU executor and model runner — and produces valid audio output.

**This is NOT model-level unit testing.** Tests must exercise the full request path.

</domain>

<architecture>
## True End-to-End Request Path

Each E2E test must traverse this full path:

```
1. Test builds a protocol Request object (TextToSpeechRequest, EmbedSpeakerRequest, etc.)
        ↓
2. Serving handler (OpenAIServingTextToSpeech / OpenAIServingVoiceEmbedding / etc.)
   [harmonyspeech/endpoints/openai/serving_text_to_speech.py etc.]
   - validates model exists, checks mode/language/voice parameters
   - creates typed RequestInput (TextToSpeechRequestInput, SpeechEmbeddingRequestInput, etc.)
   - calls self.engine.generate(request_id, request_data)
        ↓
3. AsyncHarmonySpeech.generate()
   [harmonyspeech/engine/async_harmonyspeech.py]
   - calls add_request() → adds to request tracker
   - polls engine_step() until the result stream yields a finished result
        ↓
4. HarmonySpeechEngine.add_request()
   [harmonyspeech/engine/harmonyspeech_engine.py]
   - calls check_reroute_request_to_model() which calls the appropriate
     reroute_request_* method based on requested_model group name
     (e.g. "harmonyspeech", "openvoice_v1", "openvoice_v2", "kitten-tts-nano", etc.)
   - sets request.model to the first concrete ModelConfig name to use
   - adds EngineRequest to Scheduler.waiting queue
        ↓
5. HarmonySpeechEngine.step()
   - calls Scheduler.schedule() → returns SchedulerOutputs with scheduled_requests_per_model
   - dispatches via ThreadPoolExecutor to CPUExecutor.execute_model() per model
        ↓
6. CPUExecutor.execute_model()
   [harmonyspeech/executor/cpu_executor.py]
   - delegates to CPUWorker.execute_model()
        ↓
7. CPUWorker.execute_model()
   [harmonyspeech/task_handler/cpu_worker.py]
   - delegates to CPUModelRunner.execute_model()
        ↓
8. CPUModelRunner.execute_model()
   [harmonyspeech/task_handler/cpu_model_runner.py]
   - runs actual model inference (KittenTTSSynthesizer / MeloTTSSynthesizer / etc.)
   - returns ExecutorResult containing result_data (output bytes/embedding)
        ↓
9. HarmonySpeechEngine._process_model_outputs() + check_forward_processing()
   [harmonyspeech/engine/harmonyspeech_engine.py]
   - For single-stage models: wraps result in TextToSpeechRequestOutput, marks FINISHED
   - For multi-stage pipelines: mutates request_data with intermediate output,
     re-adds to scheduler (FINISHED_FORWARDED), loop continues until final stage
        ↓
10. Final output: TextToSpeechResponse / EmbedSpeakerResponse / SynthesizeAudioResponse / etc.
    with base64-encoded audio/embedding data
```

## Multi-Stage Pipeline Routing Details

### KittenTTS — single-stage
- `mode="single_speaker_tts"`, model=`"kitten-tts-nano"` (or any variant name from config)
- requires `voice` parameter (e.g. `"Jasper"`, `"Bella"`, `"Luna"`, `"Bruno"`, `"Rosie"`, `"Hugo"`, `"Kiki"`, `"Leo"`)
- routes directly to `KittenTTSSynthesizer`, no forwarding
- `config.yml` model name examples: `"kitten-tts-mini"`, `"kitten-tts-micro"`, `"kitten-tts-nano"`, `"kitten-tts-nano-int8"`
- HF repos: `KittenML/kitten-tts-mini-0.8`, `KittenML/kitten-tts-micro-0.8`,
  `KittenML/kitten-tts-nano-0.8-fp32`, `KittenML/kitten-tts-nano-0.8-int8`

### MeloTTS / OpenVoice V2 — multi-stage (voice cloning) or single-stage (single speaker)
**Single speaker** (`mode="single_speaker_tts"`):
- Directly target a synthesizer model (e.g. `model="ov2-synthesizer-en"`, `language="EN"`, `voice="EN-Newest"`)
- Routes to `MeloTTSSynthesizer` only — single step, returns audio

**Voice cloning** (`mode="voice_cloning"`):
- Target the toolchain group model `model="openvoice_v2"`, requires `input_audio` (base64 reference speaker)
- `reroute_request_openvoice_v2()` routing:
  1. If `SpeechEmbeddingRequestInput` or `TextToSpeechRequestInput` with `input_audio` and no embedding yet
     → `OpenVoiceV2ToneConverterEncoder` (embed reference speaker)
  2. If `SynthesisRequestInput` or `TextToSpeechRequestInput` with no `input_audio` and no `input_embedding`
     → `MeloTTSSynthesizer` (text → speech audio)
  3. After synthesis when `input_audio` set (forwarded synthesis result) and `mode=voice_cloning`
     → `OpenVoiceV2ToneConverter` (apply speaker voice)
- Each forward step updates `input_vad_data` or `input_embedding` or `input_audio` on the same request

### OpenVoice V1 — multi-stage (voice cloning) or single-stage (single speaker)
**Single speaker** (`mode="single_speaker_tts"`):
- Target a synthesizer directly: `model="ov1-synthesizer-en"`, `language="EN"`, `voice="default"`
- Routes to `OpenVoiceV1Synthesizer` only

**Voice cloning** (`mode="voice_cloning"`):
- Target toolchain group `model="openvoice_v1"`, requires `input_audio`
- `reroute_request_openvoice_v1()` routing:
  1. Embedding of reference speaker → `OpenVoiceV1ToneConverterEncoder`
  2. Synthesis → `OpenVoiceV1Synthesizer` (matched by `language_id`)
  3. Tone conversion → `OpenVoiceV1ToneConverter`

### HarmonySpeech — always multi-stage via toolchain
- Target `model="harmonyspeech"` (toolchain group ID), `mode="single_speaker_tts"` or `"voice_cloning"`
- Requires all three components loaded: `HarmonySpeechEncoder`, `HarmonySpeechSynthesizer`, `HarmonySpeechVocoder`
- `reroute_request_harmonyspeech()` routing:
  1. `SpeechEmbeddingRequestInput` or `TextToSpeechRequestInput` with `input_audio` → `HarmonySpeechEncoder`
  2. `SynthesisRequestInput` or `TextToSpeechRequestInput` with `input_embedding` → `HarmonySpeechSynthesizer`
  3. `VocodeRequestInput` or `TextToSpeechRequestInput` with `input_audio` (mel from synth) → `HarmonySpeechVocoder`

**Individual stage access** (explicit model addressing):
- Embed: `EmbedSpeakerRequest(model="hs1-encoder", input_audio=<base64>)` → `OpenAIServingVoiceEmbedding`
  - Note: `_EMBEDDING_MODEL_GROUPS["harmonyspeech"] = ["HarmonySpeechEncoder"]` — only this one component needed for direct addressing
- Synthesize: `SynthesizeAudioRequest(model="hs1-synthesizer", input=<text>, input_embedding=<base64>)` → dedicated synthesis handler
- Vocode: `VocodeAudioRequest(model="hs1-vocoder", input_audio=<mel_base64>)` → dedicated vocode handler

### OpenVoice V1 & V2 — Individual Stage Access
Beyond the full toolchain flows described above, each stage can also be invoked directly:

**Tone Converter Encoder (reference speaker embedding)**:
- `EmbedSpeakerRequest(model="ov1-tone-converter-encoder", input_audio=<base64>)` → `OpenAIServingVoiceEmbedding`
- `EmbedSpeakerRequest(model="ov2-tone-converter-encoder", input_audio=<base64>)` → `OpenAIServingVoiceEmbedding`
- Note: `_EMBEDDING_MODEL_GROUPS["openvoice_v1"] = ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder"]` —
  the **toolchain group** model (`openvoice_v1`) requires both FasterWhisper AND ToneConverterEncoder to be loaded.
  However the individual encoder model can be addressed directly by its config name `"ov1-tone-converter-encoder"`.

**Synthesis only** (single speaker, explicit synthesizer addressing):
- `TextToSpeechRequest(model="ov1-synthesizer-en", mode="single_speaker_tts", language="EN", voice="default")` → TTS handler
- `TextToSpeechRequest(model="ov2-synthesizer-en", mode="single_speaker_tts", language="EN", voice="EN-Newest")` → TTS handler

**Tone Conversion only** (voice conversion stage):
- `VoiceConversionRequest(model="ov1-tone-converter", source_audio=<synth_audio_base64>, target_embedding=<embedding_base64>)` → `OpenAIServingVoiceConversion`
- `VoiceConversionRequest(model="ov2-tone-converter", source_audio=<synth_audio_base64>, target_embedding=<embedding_base64>)` → `OpenAIServingVoiceConversion`
- `_VOICE_CONVERSION_MODEL_TYPES = ["OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"]` — these are exposed as individual models

</architecture>

<decisions>
## Implementation Decisions

### Test Architecture: Full Stack, No Shortcuts
- Tests instantiate `AsyncHarmonySpeech` + serving handler(s); they never call model classes directly.
- Engine is built programmatically from `EngineConfig(model_configs=[...])` — do NOT read from `config.yml`.
- Build minimal configs: only load the models needed for the test case being run.
- Session-scoped engine fixtures per model group so weights are loaded once per pytest session.

### E2E conftest.py expansion
- `tests/e2e/conftest.py` must be extended with engine fixtures for each model group.
- Each fixture creates `AsyncHarmonySpeech.from_engine_args_and_config(engine_args, engine_config)`.
- The engine's `CPUExecutor._init_worker()` triggers `CPUWorker.load_model()` at fixture creation time,
  so models are downloaded/cached into `models_cache_dir` during fixture setup.

### Test Cases (full coverage)

| Plan | Model Group | Test | Pipeline Stage | API |
|------|-------------|------|----------------|-----|
| **04-01 KittenTTS** ||||
| 04-01 | KittenTTS mini | `test_kittentts_mini_single_speaker` | Full, single stage | TTS `mode=single_speaker_tts`, `voice=Jasper` |
| 04-01 | KittenTTS micro | `test_kittentts_micro_single_speaker` | Full, single stage | TTS |
| 04-01 | KittenTTS nano | `test_kittentts_nano_single_speaker` | Full, single stage | TTS |
| 04-01 | KittenTTS nano-int8 | `test_kittentts_nano_int8_single_speaker` | Full, single stage | TTS |
| **04-02 MeloTTS + OpenVoice V1** ||||
| 04-02 | MeloTTS EN | `test_melotts_en_single_speaker` | Full, synthesizer only | TTS `mode=single_speaker_tts`, `model=ov2-synthesizer-en`, `language=EN`, `voice=EN-Newest` |
| 04-02 | MeloTTS EN | `test_melotts_en_voice_cloning` | Full, 3-stage toolchain | TTS `mode=voice_cloning`, `model=openvoice_v2`, `input_audio` |
| 04-02 | MeloTTS | `test_melotts_synthesize_stage` | Synthesizer stage only | TTS `model=ov2-synthesizer-en`, `mode=single_speaker_tts` — verifies synthesizer output audio |
| 04-02 | OV2 Tone Converter | `test_openvoice_v2_tone_transfer_stage` | Tone conversion stage | VoiceConversion `model=ov2-tone-converter`, `source_audio`, `target_embedding` |
| 04-02 | OV2 Tone Converter Encoder | `test_openvoice_v2_embed_stage` | Speaker embedding stage | EmbedSpeaker `model=ov2-tone-converter-encoder`, `input_audio` |
| 04-02 | OpenVoice V1 EN | `test_openvoice_v1_en_single_speaker` | Full, synthesizer only | TTS `mode=single_speaker_tts`, `model=ov1-synthesizer-en`, `language=EN`, `voice=default` |
| 04-02 | OpenVoice V1 EN | `test_openvoice_v1_en_voice_cloning` | Full, 3-stage toolchain | TTS `mode=voice_cloning`, `model=openvoice_v1`, `input_audio` |
| 04-02 | OV1 Synthesizer | `test_openvoice_v1_synthesize_stage` | Synthesizer stage only | TTS `model=ov1-synthesizer-en`, `mode=single_speaker_tts` |
| 04-02 | OV1 Tone Converter | `test_openvoice_v1_tone_transfer_stage` | Tone conversion stage | VoiceConversion `model=ov1-tone-converter`, `source_audio`, `target_embedding` |
| 04-02 | OV1 Tone Converter Encoder | `test_openvoice_v1_embed_stage` | Speaker embedding stage | EmbedSpeaker `model=ov1-tone-converter-encoder`, `input_audio` |
| **04-03 HarmonySpeech** ||||
| 04-03 | HarmonySpeech | `test_harmonyspeech_single_speaker` | Full, 3-stage toolchain | TTS `mode=single_speaker_tts`, `model=harmonyspeech` |
| 04-03 | HarmonySpeech | `test_harmonyspeech_voice_cloning` | Full, 3-stage toolchain + embed | TTS `mode=voice_cloning`, `model=harmonyspeech`, `input_audio` |
| 04-03 | HarmonySpeech | `test_harmonyspeech_embed_stage` | Encoder only | EmbedSpeaker `model=hs1-encoder` |
| 04-03 | HarmonySpeech | `test_harmonyspeech_synthesize_stage` | Synthesizer only | SynthesizeAudio `model=hs1-synthesizer`, `input_embedding` |
| 04-03 | HarmonySpeech | `test_harmonyspeech_vocode_stage` | Vocoder only | VocodeAudio `model=hs1-vocoder`, `input_audio` (mel spectrogram) |

### Output Validation
- **Full pipeline TTS tests:** `response.data` is non-None, non-empty base64 string; decode and verify length > 0.
- **Embed tests:** `response.data` is non-None base64 string (speaker embedding).
- **Synthesize tests:** `response.data` is non-None base64 string (mel spectrogram / audio).
- **Vocode tests:** `response.data` is non-None base64 string (final audio).
- Format check for audio: decoded bytes must be non-empty; optionally assert WAV header `b'RIFF'`.

### Input Audio for Voice Cloning Tests
- Voice cloning tests need a reference audio input (`input_audio` = base64-encoded WAV/MP3).
- Use a minimal synthetic audio file (e.g. 1-second 24kHz silence or a bundled test fixture).
- Store in `tests/test-data/reference_speaker.wav` — create as part of conftest or task setup.

### CI Execution
- Individual CI jobs per model group (KittenTTS, MeloTTS+OV1, HarmonySpeech).
- CPU-only: all tests with `--device=cpu --dtype=float32`.
- Each job only runs its model's tests via `-k` flag.

</decisions>

<code_context>
## Key Source Files to Read Before Implementing

### Engine / Async Layer
- `harmonyspeech/engine/harmonyspeech_engine.py` — `HarmonySpeechEngine`: `add_request()`, `step()`, `_process_model_outputs()`, `check_forward_processing()`, `reroute_request_*()`, `init_custom_executors()`
- `harmonyspeech/engine/async_harmonyspeech.py` — `AsyncHarmonySpeech`: `generate()`, `engine_step()`, `from_engine_args_and_config()`, `add_request()`
- `harmonyspeech/engine/args_tools.py` — `EngineArgs`, `AsyncEngineArgs`

### Serving Layer
- `harmonyspeech/endpoints/openai/serving_text_to_speech.py` — `OpenAIServingTextToSpeech`, `_TTS_MODEL_TYPES`, `_TTS_MODEL_GROUPS`
- `harmonyspeech/endpoints/openai/serving_voice_embed.py` — `OpenAIServingVoiceEmbedding`, `_EMBEDDING_MODEL_GROUPS`
- `harmonyspeech/endpoints/openai/serving_voice_conversion.py` — `OpenAIServingVoiceConversion`
- `harmonyspeech/endpoints/openai/serving_audio_conversion.py` — `OpenAIServingAudioConversion`
- `harmonyspeech/endpoints/openai/serving_engine.py` — `OpenAIServing._check_model()`, `model_cards_from_config_groups()`
- `harmonyspeech/endpoints/openai/api_server.py` — app construction, serving handler instantiation pattern

### Protocol / Requests
- `harmonyspeech/endpoints/openai/protocol.py` — `TextToSpeechRequest`, `EmbedSpeakerRequest`, `SynthesizeAudioRequest`, `VocodeAudioRequest`, `TextToSpeechResponse`, `EmbedSpeakerResponse`, `SynthesizeAudioResponse`, `VocodeAudioResponse`
- `harmonyspeech/common/inputs.py` — `TextToSpeechRequestInput`, `SpeechEmbeddingRequestInput`, `SynthesisRequestInput`, `VocodeRequestInput`
- `harmonyspeech/common/outputs.py` — `TextToSpeechRequestOutput`, `SpeechEmbeddingRequestOutput`, `SpeechSynthesisRequestOutput`, `VocodeRequestOutput`

### Config
- `harmonyspeech/common/config.py` — `EngineConfig`, `ModelConfig`, `DeviceConfig`
- `config.yml` — reference for actual model names, HF repo IDs, required `language`/`voices`/`model_type` fields

### Executor / Worker
- `harmonyspeech/executor/cpu_executor.py` — `CPUExecutor._init_worker()`, `execute_model()`
- `harmonyspeech/task_handler/cpu_worker.py` — `CPUWorker.load_model()`, `execute_model()`
- `harmonyspeech/task_handler/cpu_model_runner.py` — `CPUModelRunner.load_model()`, `execute_model()`
- `harmonyspeech/modeling/loader.py` — `_MODEL_CONFIGS` dict, model class resolution

### Model Specifics
- `harmonyspeech/modeling/models/kittentts/kittentts.py` — `KittenTTSSynthesizer`, `KITTENTTS_MODEL_REPOS`, available voices list
- `harmonyspeech/modeling/models/melo/melo.py` — `MeloTTSSynthesizer` constructor, language/voice handling
- `harmonyspeech/modeling/models/openvoice/openvoice.py` — `OpenVoiceV1Synthesizer`, language/voice handling
- `harmonyspeech/modeling/models/harmonyspeech/` — encoder, synthesizer, vocoder sub-modules

### Existing Test Infrastructure
- `tests/e2e/conftest.py` — `models_cache_dir` fixture (session-scoped), auto-marker — needs engine fixtures added
- `tests/conftest.py` — `device`, `dtype` fixtures; `--device`/`--dtype` CLI options
- `tests/integration/conftest.py` — `test_app` fixture pattern for reference

</code_context>

<specifics>
## What Must Change from Old Plans

The old plans (before 2026-03-04) had tests directly instantiating model classes (e.g. `KittenTTSSynthesizer()`). This is **NOT** end-to-end — it bypasses the entire engine stack. Each updated plan must:

1. Read every layer of the request path (serving → engine → executor → model runner) before writing tests.
2. Build `EngineConfig` + `AsyncHarmonySpeech` + serving handler(s) in session-scoped fixtures in `tests/e2e/conftest.py`.
3. Send typed protocol requests through the serving handler (not directly to the engine or model).
4. Test both single-speaker and voice-cloning modes for models that support both.
5. Test individual stage access where the API allows explicit stage requests (embed, synthesize, vocode).
6. Verify `response.data` at each stage.

</specifics>

<deferred>
## Deferred Ideas
- GPU E2E tests → future phase
- Stream output testing → feature not yet implemented in serving_text_to_speech.py
- Other languages for MeloTTS (ZH, ES, FR, JP, KR) → covered by same code path, defer to later
</deferred>

---

*Phase: 04-e2e-tts-models*
*Context gathered: 2026-03-03 / architecture analysis: 2026-03-04*
