# Phase 2: Model Registration & Loading - Context

**Gathered:** 2026-03-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Register all four Chatterbox model variants (ChatterboxTTS, ChatterboxTurboTTS, ChatterboxMultilingualTTS, ChatterboxVC) in HSE's ModelRegistry and make them loadable via `get_model()` using the native loader pattern.

This phase covers: `_MODELS` dict entries, `_MODEL_CONFIGS` / `_MODEL_WEIGHTS` entries, loader branches in `get_model()`, and wrapper class stubs in a new `chatterbox/` sub-package.

This phase does NOT cover: input preparation, execution logic, routing, or config examples (those are Phases 3–6).

</domain>

<decisions>
## Implementation Decisions

### Module structure & file layout
- Create a `harmonyspeech/modeling/models/chatterbox/` sub-package (matches `kittentts/` pattern exactly)
- All 4 wrapper classes live in a single `chatterbox.py` file inside that package
- `__init__.py` exists (standard Python package) but imports are done directly from `chatterbox.py`, not re-exported from `__init__.py`
  - Loader imports: `from harmonyspeech.modeling.models.chatterbox.chatterbox import ChatterboxTTS` etc.

### _MODELS dict key naming
- Use exact names from roadmap success criteria:
  - `"ChatterboxTTS"`: `("chatterbox", "native")`
  - `"ChatterboxTurboTTS"`: `("chatterbox", "native")`
  - `"ChatterboxMultilingualTTS"`: `("chatterbox", "native")`
  - `"ChatterboxVC"`: `("chatterbox", "native")`
- Class name field stays as `"native"` sentinel — consistent with KittenTTS, FasterWhisper, SileroVAD

### _MODEL_CONFIGS and _MODEL_WEIGHTS
- All 4 variants use `{"default": "native"}` in both dicts — same as all existing native models

### Loader branch design
- Add 4 separate `elif` blocks inside the `if model_class == "native" and hf_config == "native":` branch in `get_model()` — one per variant
- Each block instantiates its wrapper: `model = ChatterboxTTS(model_name_or_path=model_config.model, device=device_config.device)` (or equivalent)
- No shared dispatch block — mirrors the existing per-model-type pattern exactly

### Wrapper class design
- Each wrapper class calls `from_pretrained()` (or equivalent Chatterbox API) inside `__init__`, accepting `model_name_or_path` and `device`
- Device placement passed to `from_pretrained()` directly — do not call `.to(device)` after loading
- Each wrapper implements a no-op `load_weights()` method for compatibility (same as KittenTTS)
- Model name/path supports both HuggingFace repo ID and local path — pass `model_config.model` through without transformation

### Loading behavior
- Lazy loading — models load on first `get_model()` call, not on startup
- HF Hub downloads happen transparently inside the wrapper `__init__` (no pre-verification)
- Follows existing HSE pattern established by KittenTTS and FasterWhisper

### Ignored param handling
- Each wrapper only forwards the parameters it supports to the underlying model
- Unsupported params (e.g., exaggeration/cfg_weight/min_p on Turbo) are silently dropped — no warning emitted
- Researcher should verify this matches the existing HSE convention for other models (OQ-02 is open)
- Consistent across all 4 variants — each variant drops its own unsupported params silently
- Document which params each variant ignores in an inline comment (not a docstring), for future maintainer clarity

### OpenCode's Discretion
- Exact import structure inside `chatterbox/__init__.py` (empty vs minimal)
- How to handle `from_pretrained()` if the chatterbox-tts API uses different method names for different variants (researcher to verify)
- Whether a device string or `torch.device` object is passed — researcher to check Chatterbox API

</decisions>

<specifics>
## Specific Ideas

- The `kittentts/` sub-package is the canonical reference — model this implementation closely on that pattern
- Roadmap success criteria uses `_get_model_cls("chatterbox")` in the description but the `_MODELS` dict keys are PascalCase (e.g. `"ChatterboxTTS"`). Researcher should verify the test calls `_get_model_cls("ChatterboxTTS")` — the lowercase slug `"chatterbox"` is only the module name inside the tuple, not the registry key.

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-model-registration-loading*
*Context gathered: 2026-03-12*
