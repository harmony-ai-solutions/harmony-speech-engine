# System Patterns: Harmony Speech Engine

## Architecture Overview

Harmony Speech Engine follows a **multi-executor, request-routing architecture** that enables parallel processing of different speech AI models while providing a unified API interface. The system is built on a foundation forked from Aphrodite Engine, adapted specifically for speech processing workflows.

### Core Architectural Principles

1. **Per-Request Processing**: Unlike traditional LLM engines that use token sequence batching, HSE processes complete requests individually
2. **Multi-Model Parallelism**: Multiple models can be loaded and executed simultaneously across different executors
3. **Intelligent Request Routing**: Automatic routing of requests to appropriate models based on request type and parameters
4. **Workflow Orchestration**: Support for complex multi-step processing pipelines

## System Components

### 1. Engine Core (`HarmonySpeechEngine`)

**Responsibilities:**
- Central coordination of all system components
- Request lifecycle management
- Model executor initialization and management
- Request routing and workflow orchestration

**Key Patterns:**
```python
# Multi-executor initialization pattern
for model_cfg in self.model_configs:
    if model_cfg.device_config.device_type == "cpu":
        executor_class = CPUExecutor
    else:
        executor_class = GPUExecutorAsync
    
    executor = executor_class(model_config=model_cfg)
    self.model_executors[model_cfg.name] = executor
```

### 2. Request Scheduler (`Scheduler`)

**Responsibilities:**
- Request queue management
- Resource allocation and optimization
- Batch processing coordination
- Request status tracking

**Key Patterns:**
- **Priority-based scheduling**: Requests are prioritized based on type and resource requirements
- **Resource-aware batching**: Batching decisions consider available executor capacity
- **Status lifecycle management**: Comprehensive request status tracking (WAITING → RUNNING → FINISHED)

### 3. Model Executors

**CPU Executor (`CPUExecutor`)**
- Handles CPU-based model inference
- Optimized for models that don't require GPU acceleration
- Thread-safe execution for concurrent requests

**GPU Executor (`GPUExecutorAsync`)**
- Manages GPU-accelerated model inference
- Asynchronous execution patterns for better resource utilization
- Memory management for GPU resources

**Executor Pattern:**
```python
class ExecutorBase:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self.load_model()
    
    def execute_model(self, requests: List[EngineRequest]) -> List[ExecutorResult]:
        # Process batch of requests
        pass
```

### 4. Request Routing System (declarative, Phase 3)

Routing is **data, not code**. `harmonyspeech/processing/pipeline_registry.py`
declares every toolchain as a `PipelineSpec`:

```python
PipelineSpec(
    id="openvoice_v2",
    steps=(vad, embed, synthesize, tone_convert),  # each Step declares
    # requires (model_type preference), needs, produces, clears, advertises
    final_step="tone_convert",
    capabilities=frozenset({"tts", "embed", "vc"}),
    entry_rules=(...),  # request shape -> starting step (+ optional stop_after)
)
```

Only two functions implement the behaviour:

* `next_step(spec, state, request, model_configs)` — resolves the concrete model
  for the current step (preference order, language variants, `input_vad_mode`),
  or returns `None` so the engine emits the typed "pipeline not fully
  configured" error naming the step.
* `advance(spec, engine_request, result)` — maps the finished step's **typed
  artifacts** onto the request fields declared in `produces`, clears superseded
  fields, increments the step index and reports completion.

The routing position travels on `EngineRequest.pipeline_state` and is resolved
once at admission from the entry rules; forwarded requests never re-resolve it.
The old `reroute_request_*` ladders and the output-class-sniffing forwarding
branches are gone (Phase 3-2). Startup validation
(`validate_registry`), `/v1/models` cards (`model_cards_from_specs`) and the
capability/language metadata all read the same registry, so adding a toolchain
touches one file.

#### Entry points

Entry rules replaced field-presence heuristics. Notable ones:

* embedding-only requests start at VAD and `stop_after` the embed step;
* `mode: "single_speaker_tts"` starts **and stops** at the synthesis step (the
  old FIXME field-coercion hack is gone);
* voice conversion starts at the tone-conversion step;
* Chatterbox starts at its embed step only when a dedicated embedding executor
  is configured (otherwise the TTS executor embeds inline);
* single-step executors (`kitten_tts`, `fasterwhisper`, `silero_vad`) and
  concrete model names bypass the pipeline (`advertise_toolchain=False`).

### 5. Request Processing Pipeline

#### Multi-Step Processing Pattern

The system supports complex workflows that require multiple models working in
sequence. A hop only ever writes what the step declared:

```python
def advance(spec, engine_request, result):
    for output_attribute, field_name in step.produces:  # e.g. ("vad_artifact", "input_vad_data")
        artifact = getattr(result, output_attribute, None)
        if artifact is None:
            raise MissingStepArtifactError(...)  # typed error naming the step
        engine_request.artifacts[field_name] = artifact
    state.step_index += 1
    return step.name == (state.stop_after or spec.final_step)
```

Intermediate values are typed artifacts (Phase 2), never re-serialized wire
payloads; `needs` is checked against the artifact store first.

#### Request Status Lifecycle

```
WAITING → RUNNING → FINISHED_FORWARDED → RUNNING → FINISHED_STOPPED
                 ↘ FINISHED_STOPPED
                 ↘ FINISHED_ABORTED
                 ↘ FINISHED_IGNORED
```

## Model Integration Patterns

### 1. Model Configuration Pattern

```yaml
model_configs:
  - name: "model-identifier"
    model: "huggingface/model-path"
    model_type: "ModelClassName"
    language: "EN"  # Optional language specification
    voices: ["voice1", "voice2"]  # Optional voice options
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu|cuda"
```

### 2. Model Type Hierarchy

**Base Model Types:**
- `HarmonySpeechEncoder`: Speaker embedding generation
- `HarmonySpeechSynthesizer`: Text-to-speech synthesis
- `HarmonySpeechVocoder`: Audio vocoding
- `OpenVoiceV1Synthesizer`: OpenVoice V1 TTS
- `OpenVoiceV1ToneConverter`: OpenVoice V1 voice conversion
- `MeloTTSSynthesizer`: MeloTTS multilingual synthesis
- `VoiceFixerRestorer`: Audio restoration and denoising
- `VoiceFixerVocoder`: Mel-spectrogram to audio conversion
- `FasterWhisper`: Speech recognition and VAD

### 3. Model Loading Pattern

```python
class ModelLoader:
    @staticmethod
    def load_model(model_config: ModelConfig):
        if model_config.model_type == "HarmonySpeechEncoder":
            return HarmonySpeechEncoder.from_pretrained(model_config.model)
        elif model_config.model_type == "FasterWhisper":
            return FasterWhisperModel(model_config.model)
        # ... additional model types
```

## API Design Patterns

### 1. OpenAI Compatibility Layer

The API layer provides OpenAI-compatible endpoints while supporting extended functionality:

```python
# Standard OpenAI TTS endpoint
POST /v1/audio/speech
{
    "model": "tts-1",
    "input": "Hello world",
    "voice": "alloy"
}

# Extended HSE endpoint with voice cloning
POST /v1/audio/speech
{
    "model": "openvoice_v2",
    "input": "Hello world",
    "mode": "voice_cloning",
    "input_audio": "base64_encoded_reference_audio",
    "language_id": "EN"
}

# VAD endpoint with dynamic parameters
POST /v1/audio/vad
{
    "model": "silero-vad",
    "input_audio": "base64_encoded_audio",
    "get_timestamps": true,
    "threshold": 0.5,
    "min_speech_duration_ms": 250
}
```

### 2. Request/Response Pattern

**Input Types:**
- `TextToSpeechRequestInput`: Main TTS requests
- `SpeechEmbeddingRequestInput`: Speaker embedding generation
- `SpeechTranscribeRequestInput`: Speech-to-text processing
- `VoiceConversionRequest`: Voice conversion operations
- `DetectVoiceActivityRequestInput`: VAD processing with dynamic parameters
- `AudioConversionRequestInput`: Audio restoration and processing

**Output Types:**
- `TextToSpeechRequestOutput`: TTS results
- `SpeechEmbeddingRequestOutput`: Embedding vectors
- `SpeechTranscriptionRequestOutput`: Transcription and VAD data
- `VoiceConversionRequestOutput`: Converted audio
- `DetectVoiceActivityRequestOutput`: VAD results with optional timestamps
- `AudioConversionRequestOutput`: Processed audio data

### 3. API Protocol Serving Architecture

**Serving Layer Structure:**
```
FastAPI Application
├── OpenAIServingEngine (base serving class)
├── OpenAIServingTextToSpeech (/v1/audio/speech)
├── OpenAIServingVoiceActivityDetection (/v1/audio/vad)
├── OpenAIServingSpeechToText (/v1/audio/transcriptions)
└── OpenAIServingEmbedding (/v1/embeddings)
```

**Model Type Registration Pattern:**
```python
# Each serving class defines supported model types
_VAD_MODEL_TYPES = [
    "FasterWhisper",
    "SileroVAD"
]

# Model filtering and availability
@staticmethod
def models_from_config(configured_models: List[ModelConfig]) -> List[ModelCard]:
    return OpenAIServing.model_cards_from_config_groups(
        configured_models,
        _VAD_MODEL_TYPES,
        _VAD_MODEL_GROUPS
    )
```

### 4. Request Processing Flow

**Complete Request Lifecycle:**
```
1. FastAPI Endpoint Reception
   ↓
2. Protocol Validation (Pydantic models)
   ↓
3. Request Input Conversion (.from_openai() methods)
   ↓
4. Engine Request Creation
   ↓
5. Model Routing & Executor Selection
   ↓
6. Input Preparation (prepare_inputs())
   ↓
7. Model Execution (execute_model())
   ↓
8. Result Processing & Response Generation
   ↓
9. HTTP Response Return
```

**Input Processing Pipeline:**
```python
# 1. Protocol layer defines request structure
class DetectVoiceActivityRequest(BaseRequest):
    input_audio: str
    get_timestamps: Optional[bool] = False
    threshold: Optional[float] = 0.5
    # ... dynamic parameters

# 2. Conversion to internal format
DetectVoiceActivityRequestInput.from_openai(request_id, request)

# 3. Input preparation with parameter extraction
def prepare_silero_vad_inputs(requests_to_batch):
    def prepare(request):
        # Audio processing
        audio_tensor = torch.FloatTensor(audio_ref)
        
        # Parameter extraction
        vad_params = {
            'threshold': getattr(request, 'threshold', 0.5),
            'min_speech_duration_ms': getattr(request, 'min_speech_duration_ms', 250),
            # ... other parameters
        }
        
        return (audio_tensor, vad_params)

# 4. Model execution with unpacked parameters
def _execute_silero_vad(self, inputs, requests_to_batch):
    def run_vad(input_params):
        audio_tensor, vad_params = input_params
        # Use parameters directly in model call
        timestamps = get_speech_timestamps(
            audio_tensor, 
            self.model,
            threshold=vad_params['threshold'],
            # ... other parameters
        )
```

## Performance Optimization Patterns

### 1. Parallel Execution Pattern

```python
# Parallel model execution using ThreadPoolExecutor
with ThreadPoolExecutor(len(scheduled_requests_per_model.keys())) as ex:
    futures = []
    for model_name, model_requests in scheduled_requests_per_model.items():
        futures.append(ex.submit(self.model_executors[model_name].execute_model, model_requests))
    
    for future in futures:
        model_results = future.result()
        output.extend(model_results)
```

### 2. Resource Management Pattern

**Memory Management:**
- Lazy model loading to reduce startup time
- Model unloading for memory pressure situations
- GPU memory optimization for CUDA models

**Batch Processing:**
- Dynamic batch size adjustment based on available resources
- Request grouping by model type and parameters
- Efficient tensor operations for batch processing

### 3. Caching Patterns

**Model Caching:**
- Persistent model loading across requests
- Shared model instances for identical configurations
- LRU eviction for memory management

**Result Caching:**
- Speaker embedding caching for repeated voice cloning
- VAD result caching for identical audio inputs
- Synthesis result caching for repeated text/voice combinations

## Error Handling and Resilience Patterns

### 1. Graceful Degradation

```python
def execute_with_fallback(self, request, primary_model, fallback_model):
    try:
        return self.execute_model(request, primary_model)
    except ModelExecutionError:
        logger.warning(f"Primary model {primary_model} failed, using fallback")
        return self.execute_model(request, fallback_model)
```

### 2. Request Retry Pattern

- Automatic retry for transient failures
- Exponential backoff for resource contention
- Circuit breaker pattern for persistent failures

### 3. Resource Recovery

- Automatic GPU memory cleanup on CUDA errors
- Model reloading on corruption detection
- Executor restart on critical failures

## Extensibility Patterns

### 1. Plugin Architecture

```python
class ModelPlugin:
    def register_model_type(self, model_type: str, model_class: Type):
        self.model_registry[model_type] = model_class
    
    def create_model(self, model_config: ModelConfig):
        model_class = self.model_registry[model_config.model_type]
        return model_class.from_config(model_config)
```

### 2. Custom Executor Pattern

```python
class CustomExecutor(ExecutorBase):
    def __init__(self, model_config: ModelConfig, custom_params: Dict):
        super().__init__(model_config)
        self.custom_params = custom_params
    
    def execute_model(self, requests: List[EngineRequest]) -> List[ExecutorResult]:
        # Custom execution logic
        pass
```

### 3. Middleware Pattern

```python
class RequestMiddleware:
    def process_request(self, request: RequestInput) -> RequestInput:
        # Pre-processing logic
        return request
    
    def process_response(self, response: RequestOutput) -> RequestOutput:
        # Post-processing logic
        return response
```

## Monitoring and Observability Patterns

### 1. Metrics Collection

```python
class Stats:
    def __init__(self, now: float, num_running: int, num_waiting: int):
        self.timestamp = now
        self.num_running = num_running
        self.num_waiting = num_waiting
        self.processing_latency = []
        self.model_utilization = {}
```

### 2. Logging Pattern

- Structured logging with request correlation IDs
- Performance metrics logging at configurable intervals
- Error tracking with stack traces and context
- Model-specific metrics and diagnostics

### 3. Health Check Pattern

```python
def check_health(self) -> HealthStatus:
    for model_name, executor in self.model_executors.items():
        try:
            executor.health_check()
        except Exception as e:
            return HealthStatus.UNHEALTHY(f"Model {model_name} failed: {e}")
    return HealthStatus.HEALTHY
