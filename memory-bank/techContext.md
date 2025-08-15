# Technical Context: Harmony Speech Engine

## Technology Stack

### Core Runtime Environment

**Python 3.12**
- Primary development and runtime language
- Chosen for extensive ML/AI ecosystem support
- Compatibility with PyTorch and HuggingFace libraries

**PyTorch 2.4.1**
- Deep learning framework for model inference
- CUDA 12.1 support for GPU acceleration
- ROCm 6.1 support for AMD GPUs
- CPU-only deployment option available

### Web Framework and API

**FastAPI**
- Modern, high-performance web framework
- Automatic OpenAPI/Swagger documentation generation
- Async/await support for concurrent request handling
- Type hints and Pydantic integration for request validation

**Uvicorn**
- ASGI server for FastAPI applications
- High-performance async request handling
- Production-ready deployment capabilities

### Model Integration Libraries

**HuggingFace Transformers & Hub**
- Model loading and management
- Standardized model interfaces
- Community model repository integration

**Faster-Whisper**
- Optimized Whisper implementation for speech recognition
- CTranslate2 backend for improved performance
- VAD (Voice Activity Detection) capabilities

**Custom Model Integrations**
- OpenVoice V1/V2 implementations
- MeloTTS multilingual synthesis
- Harmony Speech V1 proprietary models

### Frontend Technology

**React 18 + Vite**
- Modern React development with Vite build system
- Fast development server and hot module replacement
- TypeScript support for type safety

**TailwindCSS**
- Utility-first CSS framework
- Responsive design capabilities
- Consistent styling across components

**Node.js 18+**
- Frontend build system and development server
- Package management via npm
- Integration with Python backend via API calls

## Development Environment Setup

### System Requirements

**Minimum Requirements:**
- Python 3.8+ (3.12 recommended)
- 8GB RAM (16GB recommended)
- 10GB disk space for models
- Linux/WSL (Windows support limited)

**GPU Requirements (Optional):**
- NVIDIA GPU with CUDA 12.1+ support
- 6GB+ VRAM for most models
- AMD GPU with ROCm 6.1+ support

### Installation Methods

**1. Local Development Setup**
```bash
# Environment setup
conda create -n hse python=3.12
conda activate hse

# PyTorch installation (CUDA)
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
git clone https://github.com/harmony-ai-solutions/harmony-speech-engine
cd harmony-speech-engine
pip install -r requirements-cuda.txt

# Frontend setup
conda install conda-forge::nodejs
cd frontend
npm install
```

**2. Docker Deployment**
```bash
# Standard deployment
docker-compose up -d

# NVIDIA GPU support
docker-compose -f docker-compose.nvidia.yml up -d

# AMD GPU support
docker-compose -f docker-compose.amd.yml up -d
```

### Development Dependencies

**Core Python Dependencies:**
```
torch>=2.4.1
transformers>=4.40.0
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
loguru>=0.7.0
pyyaml>=6.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
```

**Development Tools:**
```
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
```

**Frontend Dependencies:**
```json
{
  "react": "^18.2.0",
  "vite": "^4.4.0",
  "tailwindcss": "^3.3.0",
  "typescript": "^5.0.0",
  "@types/react": "^18.2.0"
}
```

## Build and Deployment

### Build System

**Python Package Build**
- `setup.py` with CMake integration for native extensions
- Conditional compilation based on target device (CUDA/ROCm/CPU)
- Automatic commit hash embedding for version tracking

**CMake Build Process**
```cmake
# Key CMake configurations
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DHARMONYSPEECH_TARGET_DEVICE=cuda
-DHARMONYSPEECH_PYTHON_EXECUTABLE=/path/to/python
```

**Frontend Build**
```bash
cd frontend
npm run build  # Production build
npm run dev    # Development server
```

### Docker Configuration

**Multi-Stage Dockerfile Pattern**
```dockerfile
# Base stage with common dependencies
FROM python:3.12-slim as base
RUN apt-get update && apt-get install -y build-essential

# CUDA-specific stage
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as cuda
COPY --from=base /usr/local /usr/local

# Final runtime stage
FROM cuda as runtime
COPY . /app
RUN pip install -e .
```

**Docker Compose Services**
- `harmony-speech-engine`: Main API service
- `frontend`: React development server (dev only)
- `nginx`: Reverse proxy for production
- Volume mounts for model storage and configuration

### Configuration Management

**YAML Configuration Structure**
```yaml
model_configs:
  - name: "model-identifier"
    model: "huggingface/model-path"
    model_type: "ModelClassName"
    max_batch_size: 10
    dtype: "float32"
    device_config:
      device: "cpu|cuda"
      device_map: "auto"
```

**Environment Variables**
```bash
HARMONYSPEECH_TARGET_DEVICE=cuda
HARMONYSPEECH_LOCAL_LOGGING_INTERVAL_SEC=5
MAX_JOBS=8
NVCC_THREADS=4
VERBOSE=0
```

## Model Management

### Model Storage Structure
```
models/
├── harmony-ai-solutions/
│   └── harmony-speech-v1/
├── myshell-ai/
│   ├── openvoice/
│   ├── openvoicev2/
│   └── MeloTTS-*/
└── faster-whisper/
    ├── tiny/
    ├── medium/
    └── large-v3-turbo/
```

### Model Loading Patterns

**HuggingFace Integration**
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto"
)
```

**Custom Model Loaders**
```python
class HarmonySpeechLoader:
    @staticmethod
    def load_encoder(model_path: str):
        return HarmonySpeechEncoder.from_pretrained(model_path)
    
    @staticmethod
    def load_synthesizer(model_path: str):
        return HarmonySpeechSynthesizer.from_pretrained(model_path)
```

**Native Model Integration Pattern**
```python
# Native models bypass HuggingFace loading and use direct library imports
def get_model(model_config: ModelConfig, device_config: DeviceConfig):
    # Check for native model types first
    if model_class == "native" and hf_config == "native":
        if model_config.model_type == "FasterWhisper":
            from faster_whisper import WhisperModel
            model = WhisperModel(model_config.model)
            return model
        elif model_config.model_type == "SileroVAD":
            from silero_vad import load_silero_vad
            use_onnx = getattr(model_config, 'use_onnx', True)
            model = load_silero_vad(onnx=use_onnx)
            return model
    
    # Standard HuggingFace loading for other models
    return load_hf_model(model_config, device_config)
```

**Model Type Registration System**
```python
# Model configuration dictionaries define loading behavior
_MODEL_CONFIGS = {
    "SileroVAD": {"default": "native"},
    "FasterWhisper": {"default": "native"},
    "VoiceFixerRestorer": {"default": "native"},
    "HarmonySpeechEncoder": {"default": "encoder/config.json"}
}

_MODEL_WEIGHTS = {
    "SileroVAD": {"default": "native"},
    "FasterWhisper": {"default": "native"},
    "VoiceFixerRestorer": {"default": "vf.ckpt"},
    "HarmonySpeechEncoder": {"default": "encoder/encoder.pt"}
}
```

## Performance Optimization

### Compilation Optimizations

**Compiler Caching**
- sccache support for faster rebuilds
- ccache fallback for development
- Ninja build system for parallel compilation

**CUDA Optimizations**
- NVCC thread parallelization
- Optimized memory allocation patterns
- Kernel fusion for common operations

### Runtime Optimizations

**Memory Management**
- Lazy model loading to reduce startup time
- Model sharing across requests
- GPU memory pooling for CUDA operations

**Concurrency Patterns**
- ThreadPoolExecutor for parallel model execution
- Async/await for I/O operations
- Request batching for improved throughput

### Monitoring and Profiling

**Performance Metrics**
```python
class Stats:
    timestamp: float
    num_running: int
    num_waiting: int
    processing_latency: List[float]
    model_utilization: Dict[str, float]
    memory_usage: Dict[str, int]
```

**Logging Configuration**
- Structured logging with JSON format
- Request correlation IDs for tracing
- Configurable log levels per component
- Performance metrics at regular intervals

## Testing Infrastructure

### Test Categories

**Unit Tests**
- Model loading and initialization
- Request routing logic
- API endpoint functionality
- Configuration validation

**Integration Tests**
- End-to-end API workflows
- Multi-model processing pipelines
- Docker container functionality
- Frontend-backend integration

**Performance Tests**
- Load testing with concurrent requests
- Memory usage profiling
- GPU utilization monitoring
- Latency benchmarking

### Test Execution

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark

# Full test suite
pytest tests/ --cov=harmonyspeech
```

## Security Considerations

### API Security

**Input Validation**
- Pydantic models for request validation
- File size limits for audio uploads
- Content type verification
- Rate limiting capabilities

**Authentication & Authorization**
- API key authentication support
- Role-based access control (planned)
- Request logging and audit trails

### Model Security

**Model Integrity**
- Checksum verification for downloaded models
- Secure model storage and access
- Sandboxed model execution environment

**Data Privacy**
- Local processing without external API calls
- Temporary file cleanup after processing
- Memory clearing for sensitive audio data

## Development Workflow

### Code Quality Standards

**Code Formatting**
- Black for Python code formatting
- Prettier for JavaScript/TypeScript
- Pre-commit hooks for automated formatting

**Type Checking**
- MyPy for Python type checking
- TypeScript for frontend type safety
- Pydantic for runtime type validation

**Linting**
- Flake8 for Python code quality
- ESLint for JavaScript/TypeScript
- Custom rules for project-specific patterns

### Version Control

**Git Workflow**
- Feature branches for development
- Pull request reviews required
- Automated testing on CI/CD
- Semantic versioning for releases

**Release Process**
- Release candidate testing phase
- Docker image building and publishing
- Documentation updates
- Community announcement and feedback

### Documentation Standards

**Code Documentation**
- Docstrings for all public functions
- Type hints for function signatures
- Inline comments for complex logic
- Architecture decision records (ADRs)

**User Documentation**
- API documentation via OpenAPI/Swagger
- Setup and deployment guides
- Model configuration examples
- Troubleshooting guides
