# API Documentation

The API of Harmony Speech Engine is built using FastAPI, which provides automatic interactive documentation.

## API Endpoint

By default, the API is accessible at:

http://127.0.0.1:12080

If you decide to use a different port or you're hosting the API,
please make sure to use the proper endpoint instead.


## Interactive Documentation

### Swagger UI

Swagger UI is available at, also Providing OpenAPI Spec for client generation:

http://127.0.0.1:12080/docs


### ReDoc

ReDoc documentation is available at:

http://127.0.0.1:12080/redoc


Both Swagger UI and ReDoc provide interactive documentation for exploring and testing the API endpoints.


## Endpoint Details

### Voice Activity Detection (VAD) Endpoint

**POST /v1/audio/vad**

This endpoint is used to detect human speech activity in a provided audio file. It supports both FasterWhisper and Silero VAD models.

**Request Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | The VAD model to use (e.g., "silero-vad", "faster-whisper") |
| `input_audio` | string | Yes | - | Base64-encoded audio data |
| `get_timestamps` | boolean | No | false | Whether to return speech segment timestamps |
| `threshold` | float | No | 0.5 | Speech detection sensitivity (Silero VAD only) |
| `min_speech_duration_ms` | integer | No | 250 | Minimum speech chunk duration in milliseconds (Silero VAD only) |
| `min_silence_duration_ms` | integer | No | 100 | Minimum silence duration to separate speech chunks in milliseconds (Silero VAD only) |
| `speech_pad_ms` | integer | No | 30 | Padding around speech segments in milliseconds (Silero VAD only) |
| `return_seconds` | boolean | No | false | Return timestamps in seconds instead of milliseconds (Silero VAD only) |

**Example Request:**

```bash
curl -X POST "http://localhost:12080/v1/audio/vad" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "silero-vad",
    "input_audio": "base64_encoded_audio_data",
    "get_timestamps": true,
    "threshold": 0.3,
    "min_speech_duration_ms": 500
  }'
```

## Helpful Resources

A recently generated OpenAPI definition file and scripts to generate our Golang and JavaScript clients can be found
in [docs/api](api)
