# Phase 8: Documentation Update

## Objective

Update [`docs/models.md`](docs/models.md) and [`CHANGELOG.md`](CHANGELOG.md) to document KittenTTS support.

## Files to Modify

### `docs/models.md`

Add a new section for KittenTTS. Find the existing TTS models section and add:

```markdown
### KittenTTS

KittenTTS is an ultra-lightweight, CPU-optimized TTS system by [KittenML](https://kittenml.com) using ONNX Runtime for fast inference.

**Model Type**: `KittenTTSSynthesizer`

**Language Support**: English (EN)

**Available Model Variants**:

| Config Model Name | HuggingFace Repo | Params | Size |
|-------------------|-----------------|--------|------|
| `KittenML/kitten-tts-mini-0.8` | [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) | 80M | ~80MB |
| `KittenML/kitten-tts-micro-0.8` | [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) | 40M | ~41MB |
| `KittenML/kitten-tts-nano-0.8-fp32` | [KittenML/kitten-tts-nano-0.8-fp32](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) | 15M | ~56MB |
| `KittenML/kitten-tts-nano-0.8-int8` | [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) | 15M | ~25MB |

**Available Voices**: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`

**Audio Output**: 24,000 Hz, mono, WAV

**Example Configuration**:
```yaml
- name: "kitten-tts-mini"
  model: "KittenML/kitten-tts-mini-0.8"
  model_type: "KittenTTSSynthesizer"
  voices: ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
  max_batch_size: 4
  dtype: "float32"
  device_config:
    device: "cpu"
```

**Example API Request**:
```json
{
  "model": "kitten-tts-mini",
  "input": "Hello, this is KittenTTS speaking!",
  "voice": "Jasper"
}
```
```

---

### `CHANGELOG.md`

Add a new changelog entry under the current release candidate version. Find the most recent version header and add an entry:

```markdown
### Added
- **KittenTTS TTS support**: Added integration for KittenTTS ultra-lightweight ONNX TTS models by KittenML. Supports four model variants (mini, micro, nano fp32, nano int8) with 8 English voices. CPU-optimized, requires no GPU.
```

## Progress Checklist

- [ ] Add KittenTTS section to [`docs/models.md`](docs/models.md)
- [ ] Add KittenTTS entry to [`CHANGELOG.md`](CHANGELOG.md)
