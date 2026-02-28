# Phase 6: Configuration Examples

## Objective

Add KittenTTS model configuration entries to [`config.yml`](config.yml) (and [`config.gpu.yml`](config.gpu.yml) if applicable).

## File to Modify

### `config.yml`

Add the following entries to the `model_configs` list. All KittenTTS models are CPU-optimized, so `device: "cpu"` is appropriate for all variants.

```yaml
  # KittenTTS - Ultra-lightweight ONNX TTS models (English only)
  # Choose one variant based on your quality/size requirements:

  # kitten-tts-mini: Highest quality, 80MB
  - name: "kitten-tts-mini"
    model: "KittenML/kitten-tts-mini-0.8"
    model_type: "KittenTTSSynthesizer"
    voices: ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
    max_batch_size: 4
    dtype: "float32"
    device_config:
      device: "cpu"

  # kitten-tts-micro: Good quality, 41MB
  # - name: "kitten-tts-micro"
  #   model: "KittenML/kitten-tts-micro-0.8"
  #   model_type: "KittenTTSSynthesizer"
  #   voices: ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
  #   max_batch_size: 4
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # kitten-tts-nano: Lightweight, 56MB fp32
  # - name: "kitten-tts-nano"
  #   model: "KittenML/kitten-tts-nano-0.8-fp32"
  #   model_type: "KittenTTSSynthesizer"
  #   voices: ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
  #   max_batch_size: 4
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"

  # kitten-tts-nano-int8: Most compact, 25MB (quantized)
  # - name: "kitten-tts-nano-int8"
  #   model: "KittenML/kitten-tts-nano-0.8-int8"
  #   model_type: "KittenTTSSynthesizer"
  #   voices: ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
  #   max_batch_size: 4
  #   dtype: "float32"
  #   device_config:
  #     device: "cpu"
```

## Notes

- The `voices` field is informational only for KittenTTS (the voices list is embedded in the model's `voices.npz` file downloaded from HuggingFace). However, including them in config follows the existing convention used by `OpenVoiceV1Synthesizer` and `MeloTTSSynthesizer`.
- Only one variant should be active at a time per deployment. Comment out the others.
- The `max_batch_size: 4` is conservative given that ONNX inference is already fast on CPU.
- `dtype: "float32"` is set even though ONNX runtime uses its own type management; this follows the pattern of other native models.

## Progress Checklist

- [ ] Add KittenTTS mini entry (active) to `model_configs` in [`config.yml`](config.yml)
- [ ] Add KittenTTS micro/nano/nano-int8 entries (commented out) to [`config.yml`](config.yml)
