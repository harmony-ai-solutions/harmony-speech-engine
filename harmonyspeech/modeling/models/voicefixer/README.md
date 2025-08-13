# VoiceFixer Integration for Harmony Speech Engine

This module integrates VoiceFixer models into Harmony Speech Engine for audio restoration and enhancement.

## Overview

VoiceFixer is a neural network-based system for restoring degraded speech audio. It can handle various types of audio degradation including:
- Background noise
- Reverberation
- Low resolution audio (2kHz to 44.1kHz)
- Audio clipping
- General audio artifacts

## Models

### VoiceFixerRestorer
- **Purpose**: Audio denoising and enhancement
- **Input**: Degraded audio tensor [batch, channels, samples]
- **Output**: Enhanced mel-spectrogram tensor
- **Architecture**: Combines denoising GRU networks with UNet enhancement
- **Checkpoint**: `vf.ckpt`

### VoiceFixerVocoder
- **Purpose**: Convert mel-spectrograms to audio waveforms
- **Input**: Mel-spectrogram tensor [batch, mel_bins, time] or [batch, 1, time, mel_bins]
- **Output**: Audio waveform tensor [batch, 1, samples]
- **Architecture**: Neural vocoder with upsampling layers
- **Checkpoint**: `model.ckpt-1490000_trimed.pt`

## Configuration

Both models use hardcoded parameters that match their pre-trained checkpoints:
- Sample rate: 44.1kHz
- Mel bins: 128
- Window size: 2048
- Hop size: 441

Example configuration:
```yaml
model_configs:
  - name: "voicefixer-restorer"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerRestorer"
    device_config:
      device: "cuda"
      
  - name: "voicefixer-vocoder"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerVocoder"
    device_config:
      device: "cuda"
```

## Model Sources

The models can be loaded from several HuggingFace repositories:
- `jlmarrugom/voice_fixer` (recommended)
- `cqchangm/voicefixer`
- Future: `harmony-ai-solutions/voicefixer` (planned)

## Usage

The models are automatically integrated into HSE's model loading system:

1. **Model Registration**: Models are registered in the HSE model registry
2. **Automatic Loading**: Checkpoints are downloaded and cached automatically
3. **Device Management**: Supports both CPU and GPU inference
4. **Memory Management**: Includes segmented processing for large audio files

## Technical Details

### Architecture Adaptations
- Ported core VoiceFixer architectures to HSE patterns
- Adapted checkpoint loading to HSE's weight loading system
- Integrated with HSE's device management and tensor operations
- Added proper error handling and fallback initialization

### Optimizations
- **Memory Efficiency**: Processes audio in 30-second segments
- **Batch Processing**: Supports batch processing of multiple audio files
- **Device Flexibility**: Works on both CPU and GPU
- **Lazy Loading**: Models are loaded only when needed

### Dependencies
- PyTorch (for neural network operations)
- librosa (for audio processing)
- numpy (for numerical operations)
- soundfile (for audio I/O)

## Original VoiceFixer

This integration is based on the original VoiceFixer implementation by Haohe Liu:
- Paper: "VoiceFixer: Toward General Speech Restoration With Neural Vocoder"
- Original repository: https://github.com/haoheliu/voicefixer

## License

This integration follows the same license as the original VoiceFixer project and Harmony Speech Engine (AGPLv3).
