# Phase 6: Config Examples

## Objective

Add commented example configurations for vllm-omni TTS models to `config.gpu.yml`. This gives users a ready-to-use starting point for Qwen3-TTS (CustomVoice, Base/voice-clone, VoiceDesign) and documents the `stage_memory` field.

## File to Modify

**`config.gpu.yml`**

## Where to Add

Append the new entries at the end of the `model_configs` list, after the existing `kitten-tts-nano-int8` commented block.

## Configuration Entries to Add

```yaml
  # =========================================================================
  # vllm-omni TTS Models
  # =========================================================================
  # These models require vllm-omni to be installed (GPU-only):
  #   pip install vllm-omni
  #
  # stage_memory controls GPU memory allocation per stage (optional).
  # Format: human-readable strings like "3G", "1.5G", "512M".
  # If omitted, vllm-omni uses its built-in defaults from the stage YAML.
  #
  # Memory planning guide (adjust based on your GPU VRAM):
  #   Qwen3-TTS 1.7B: Stage 0 (LLM talker) ~4-6G, Stage 1 (Code2Wav) ~3-4G
  #   Qwen3-TTS 3B:   Stage 0 ~8-10G, Stage 1 ~3-4G
  #   On a 24GB GPU, leaving ~8G free for other HSE models is recommended.
  #
  # Request routing: set model="vllm_omni_tts" in your TTS API request.
  # =========================================================================

  # --- Qwen3-TTS CustomVoice: Built-in named speakers (no reference audio needed) ---
  # Voices: Vivian, Ryan, Ethan, Chelsie, Serena, Aria, and others (model-dependent)
  # Use case: Fast, high-quality TTS with consistent named voices
  # - name: "qwen3-tts-customvoice"
  #   model: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  #   model_type: "VllmOmniTTS"
  #   stage_memory: ["6G", "4G"]   # Stage 0: LLM talker; Stage 1: Code2Wav vocoder
  #   voices: ["Vivian", "Ryan", "Ethan", "Chelsie", "Serena", "Aria"]
  #   max_batch_size: 4
  #   dtype: "bfloat16"
  #   device_config:
  #     device: "cuda"

  # --- Qwen3-TTS Base: Voice cloning from reference audio ---
  # Voices: [] (empty — any voice via input_audio in the request)
  # Use case: Clone any speaker's voice by providing reference audio
  # API: include "input_audio": "<base64-WAV>" in the request body
  # - name: "qwen3-tts-base"
  #   model: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  #   model_type: "VllmOmniTTS"
  #   stage_memory: ["6G", "4G"]
  #   voices: []
  #   max_batch_size: 4
  #   dtype: "bfloat16"
  #   device_config:
  #     device: "cuda"

  # --- Qwen3-TTS VoiceDesign: Describe a voice in natural language ---
  # Voices: [] (voice is described via instructions in the request, not pre-defined)
  # Use case: Generate speech with a custom voice described textually
  # Note: VoiceDesign task type is not yet directly exposed in HSE's TTS API.
  #       Add "instruct" support to TextToSpeechRequestInput for full support.
  # - name: "qwen3-tts-voicedesign"
  #   model: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  #   model_type: "VllmOmniTTS"
  #   stage_memory: ["6G", "4G"]
  #   voices: []
  #   max_batch_size: 4
  #   dtype: "bfloat16"
  #   device_config:
  #     device: "cuda"

  # --- Larger Qwen3-TTS models (better quality, more VRAM required) ---
  # - name: "qwen3-tts-customvoice-3b"
  #   model: "Qwen/Qwen3-TTS-12Hz-3B-CustomVoice"
  #   model_type: "VllmOmniTTS"
  #   stage_memory: ["10G", "4G"]   # 3B model needs more VRAM for LLM stage
  #   voices: ["Vivian", "Ryan", "Ethan", "Chelsie", "Serena", "Aria"]
  #   max_batch_size: 2
  #   dtype: "bfloat16"
  #   device_config:
  #     device: "cuda"
```

## Notes

- All entries are **commented out** by default (prefixed with `#`). Users must uncomment the desired model.
- The `stage_memory` values in the examples are for a **24GB GPU** (e.g., RTX 3090, A10G). Users with smaller GPUs should reduce these values or omit `stage_memory` entirely to use vllm-omni defaults.
- The comment block explains the `stage_memory` format, memory planning guidance, and the `model="vllm_omni_tts"` routing instruction.
- VoiceDesign entry notes the current limitation (no `instruct` field in HSE's TTS request).

## Progress Checklist

- [ ] Add the vllm-omni section header comment block to `config.gpu.yml`
- [ ] Add Qwen3-TTS CustomVoice 1.7B example (commented out)
- [ ] Add Qwen3-TTS Base 1.7B example (commented out)
- [ ] Add Qwen3-TTS VoiceDesign 1.7B example (commented out, with limitation note)
- [ ] Add Qwen3-TTS CustomVoice 3B example (commented out)
- [ ] Verify all entries are syntactically valid YAML when uncommented
