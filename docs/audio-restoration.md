# Audio Restoration with VoiceFixer

Harmony Speech Engine includes VoiceFixer models for audio restoration and enhancement. This guide covers how to use the `/v1/audio/convert` endpoint for audio restoration workflows.

## Overview

VoiceFixer can restore degraded speech audio by handling various types of audio degradation:
- Background noise
- Reverberation  
- Low resolution audio (upsampling from 2kHz to 44.1kHz)
- Audio clipping
- General audio artifacts

## API Endpoint

### `/v1/audio/convert`

The audio conversion endpoint supports VoiceFixer models for audio restoration.

**Base URL:** `http://127.0.0.1:12080/v1/audio/convert`

**Method:** `POST`

**Content-Type:** `application/json`

### Request Format

```json
{
  "model": "voicefixer",
  "source_audio": "<base64_encoded_audio>"
}
```

**Parameters:**
- `model` (string, required): Model name to use for restoration
  - `"voicefixer"`: Full audio restoration pipeline
- `source_audio` (string, required): Base64-encoded audio data

### Response Format

```json
{
  "data": "<base64_encoded_restored_audio_wav_44.1khz>",
  "model": "voicefixer"
}
```

**Response Fields:**
- `data` (string): Base64-encoded restored audio data
- `model` (string): Model that was used for processing


## Usage Examples

### Python Example

```python
import requests
import base64
import json

def restore_audio(input_file, output_file, model="voicefixer"):
    """Restore audio using VoiceFixer models."""
    
    # Load degraded audio file
    with open(input_file, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()
    
    # Prepare request
    payload = {
        "model": model,
        "source_audio": audio_data
    }
    
    # Send request to HSE
    response = requests.post(
        "http://127.0.0.1:12080/v1/audio/convert",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Save restored audio
        restored_audio = base64.b64decode(result["data"])
        with open(output_file, "wb") as f:
            f.write(restored_audio)
        
        print(f"Audio restored successfully!")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Model used: {result['model_used']}")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage
restore_audio("degraded_audio.wav", "restored_audio.wav")
```

## Configuration

Ensure VoiceFixer models are properly configured in your HSE configuration file:

```yaml
model_configs:
  - name: "voicefixer-restorer"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerRestorer"
    max_batch_size: 4
    dtype: "float32"
    device_config:
      device: "cuda:0"  # or "cpu" for CPU-only inference

  - name: "voicefixer-vocoder"
    model: "jlmarrugom/voice_fixer"
    model_type: "VoiceFixerVocoder"
    max_batch_size: 4
    dtype: "float32"
    device_config:
      device: "cuda:0"  # or "cpu" for CPU-only inference
```

## Troubleshooting

### Common Issues

**"Model not found" Error:**
- Ensure VoiceFixer models are configured and loaded
- Check model names match configuration (`voicefixer-restorer`, `voicefixer-vocoder`)
- Verify models downloaded successfully from HuggingFace

**Memory Issues:**
- Reduce `max_batch_size` in configuration
- Use CPU device for very long audio files
- Split long audio files into shorter segments

**Poor Restoration Quality:**
- Ensure input audio contains speech (not music)
- Try different input formats (WAV recommended)
- Check that input audio isn't already high quality

**Slow Processing:**
- Use GPU acceleration when available
- Reduce audio length for faster processing
- Check system resources (CPU/GPU utilization)

### Getting Help

For additional support:
- Check the [Interactive API Documentation](http://127.0.0.1:12080/docs) when HSE is running
- Review the [Models Documentation](models.md) for configuration details
- Consult the [VoiceFixer original paper](https://arxiv.org/abs/2109.13731) for technical details
