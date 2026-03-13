# Phase 3: Input Preparation - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement input preparation functions for all Chatterbox model variants: `prepare_chatterbox_tts_inputs()`, `prepare_chatterbox_multilingual_tts_inputs()`, `prepare_chatterbox_turbo_tts_inputs()`, `prepare_chatterbox_embedding_inputs()`, and `prepare_chatterbox_vc_inputs()`. These functions live in `harmonyspeech/task_handler/inputs.py` and transform incoming API requests into parameter tuples ready for model execution methods.

This phase does NOT include model execution (Phase 4), routing logic (Phase 5), or tests (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### Unsupported Parameter Validation

- **Raise `ValueError`** when a caller explicitly passes a non-None value for a parameter the target model doesn't support.
- This applies symmetrically: Turbo-only params (`norm_loudness`, `top_k`) sent to base ChatterboxTTS raise `ValueError`; base-TTS params (`exaggeration`, `cfg_weight`, `min_p`) sent to ChatterboxTurboTTS raise `ValueError`.
- Only validate when the field is **non-None** — absent/default params do not trigger errors. Callers sending a full options object with None fields must not be broken.
- This is a deliberate departure from the existing silent-ignore pattern elsewhere in the codebase. Chatterbox establishes the new standard.
- Error messages should name the specific param(s) that are unsupported and the model type.

### Language Validation (Multilingual Model)

- `prepare_chatterbox_multilingual_tts_inputs()` does **not** duplicate language validation from the serving engine — the serving engine (`endpoints/openai/serving_engine.py`) already validates `language` against `model.languages` before the request reaches the prepare function. Phase 3 relies on this.
- If `language_id` is absent/None, **default to English** (`"en"`).
- The 23 supported Chatterbox Multilingual language codes live in the **model wrapper class** (in `harmonyspeech/modeling/models/chatterbox/chatterbox.py`), not in the prepare function. The prepare function imports the list from the model.
- Language discoverability at the API level must align with the existing `LanguageOptions` / `model.languages` pattern in `serving_engine.py`. Chatterbox Multilingual registers its supported languages into that existing structure so they surface via `/v1/models`.
- Whether to expose languages as "voices" in the model card is **OpenCode's Discretion** — research existing multilingual model patterns (e.g. MeloTTS, OpenVoice) and follow the same approach.

### VC Input Conflict Resolution

- `prepare_chatterbox_vc_inputs()` requires EITHER `target_audio` OR `target_embedding` (not both, not neither).
  - Both provided → **raise `ValueError`**: "Provide either target_audio or target_embedding, not both."
  - Neither provided → **raise `ValueError`**: "ChatterboxVC requires either target_audio or target_embedding."
- Same rule for TTS voice cloning: if both `input_audio` AND `input_embedding` are provided → **raise `ValueError`**.
- `source_audio` presence is NOT validated in the prepare function — trust that the API/request schema layer enforces it upstream.

### Default Values Ownership

- Chatterbox-specific generation option defaults are applied **inside each prepare function** when the field is None.
- Each prepare function defines its own model-specific defaults inline:
  - `prepare_chatterbox_tts_inputs()`: `exaggeration=0.5`, `cfg_weight=0.5`, `temperature=0.8`, `repetition_penalty=1.2`, `top_p=1.0`, `min_p=0.05`
  - `prepare_chatterbox_turbo_tts_inputs()`: `temperature=0.8`, `repetition_penalty=1.2`, `top_p=0.95`, `top_k=1000`, `norm_loudness=True`
  - `prepare_chatterbox_multilingual_tts_inputs()`: `exaggeration=0.5`, `cfg_weight=0.5`, `temperature=0.8`, `repetition_penalty=2.0`, `top_p=1.0`, `min_p=0.05`
- `TextToSpeechGenerationOptions` is extended (not a separate class) with all 8 Chatterbox fields, all typed `Optional[T] = None`. The dataclass does NOT set model-specific defaults — those remain in the prepare functions.
- `GenerationOptions` (Pydantic wire model in `protocol.py`) also gets the corresponding 8 fields added with `= None` defaults.

### OpenCode's Discretion

- Whether Chatterbox Multilingual languages are exposed as "voices" or as a separate language list in the model card — research existing patterns (MeloTTS, OpenVoice) and be consistent.
- Exact `BytesIO` wrapping approach for embedding deserialization (follow existing `torch.load` pattern from OpenVoice as reference).
- `ThreadPoolExecutor` parallelism — follow the KittenTTS pattern (all existing prepare functions use it).

</decisions>

<specifics>
## Specific Ideas

- Chatterbox establishes a new validation standard in this codebase: unsupported params raise `ValueError` rather than being silently ignored. This is intentional and should be consistent across all 5 Chatterbox prepare functions.
- The serving engine's language validation pipeline (`serving_engine.py` lines 89–120) is the authoritative validator for `language` — prepare functions should NOT re-implement this.
- Reference patterns to follow:
  - `prepare_kittentts_synthesizer_inputs()` — cleanest TTS prepare function structure (inner closure + ThreadPoolExecutor)
  - `prepare_openvoice_tone_converter_inputs()` — reference for BytesIO + torch.load embedding deserialization pattern
  - Existing `LanguageOptions` registration in `serving_engine.py` — for how to expose multilingual languages

</specifics>

<deferred>
## Deferred Ideas

- Shared language registry (a unified registry across all model families) — came up during discussion but is broader than Phase 3. The existing `LanguageOptions` pattern in `serving_engine.py` covers Phase 3 needs.
- Extending the unsupported-param validation pattern to existing models (KittenTTS, OpenVoice, MeloTTS) — this would change established behavior and belongs in a separate refactoring phase.
- Test modifications for validation coverage — explicitly noted for Phase 7, but the Phase 3 prepare functions must be structured to be unit-testable (no side effects, clear ValueError contracts).

</deferred>

---

*Phase: 03-input-preparation*
*Context gathered: 2026-03-13*
