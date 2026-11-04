# Phase 3-1: PipelineSpec Model & Registry

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 3 — Declarative PipelineSpec routing
> **Depends on:** Phase 2 complete (advance() writes artifacts)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (routing/dispatch points; research §6.2 Phase 2 is the source design)

## Objective

Create the declarative routing model that replaces both `if/elif` ladders: `Step`/`PipelineSpec` dataclasses, a registry of all pipelines, and the two resolution functions `next_step()` (routing) and `advance()` (forwarding). This task delivers the mechanism; 3-2 migrates the families onto it.

## Context (self-contained)

What is being replaced (verified 2026-11-04, `harmonyspeech/engine/harmonyspeech_engine.py`):

- **Routing** `:109-361`: `check_reroute_request_to_model` → per-family `reroute_request_*` functions that infer the current pipeline position from *field presence* (`input_audio`/`input_vad_data`/`input_embedding`/`input_mel_spectrogram` set/unset) and scan `model_configs` for a matching `model_type`.
- **Forwarding** `:363-506`: `check_forward_processing` infers what just happened from the *output class* (`SpeechTranscriptionRequestOutput` → set `input_vad_data`; `SpeechEmbeddingRequestOutput` → set `input_embedding`, clear `input_audio`; `SpeechSynthesisRequestOutput` + `voice_cloning` → audio into `input_audio`; else final) and re-submits with `FINISHED_FORWARDED`.

Known consequences to design against: silent pipeline change on inconsistent fields (`single_speaker_tts` FIXME `:187`), decorative `input_vad_mode` (TODO `:167`/`:234`), hang-forever on unresolved routing (fixed interim in 0-4; the spec makes it structural).

## Implementation steps

### 1. Dataclasses (extend `harmonyspeech/processing/pipeline_registry.py` from 0-4)

```python
@dataclass(frozen=True)
class Step:
    name: str
    requires: tuple[str, ...]          # model_types in preference order; first resolvable wins
    needs: frozenset[str]              # request fields/artifacts that must be present to enter
    produces: frozenset[str]           # artifact/field keys the step's output maps to
    optional: bool = False             # step may be skipped (entry-point rules decide)

@dataclass(frozen=True)
class PipelineSpec:
    id: str                            # "openvoice_v2", "chatterbox", ...
    steps: tuple[Step, ...]
    final_step: str
    capabilities: frozenset[str]       # {"tts","embed","vc","vad","stt","restore"}
    entry_rules: tuple[EntryRule, ...] # request → starting step index (3-4)
```

### 2. Resolution functions

```python
class PipelineState:                  # lives on EngineRequest (common/request.py)
    spec_id: str
    step_index: int
    started_at: float

def next_step(spec, state, request, model_configs) -> ModelConfig | None:
    """Resolve the concrete model for steps[state.step_index]:
    first `requires` entry that matches an available config (language/variant aware)."""

def advance(spec, engine_request, result_output) -> bool:
    """Map the step's output artifacts into engine_request.artifacts per `produces`,
    increment step_index, return True when final_step has produced its output."""
```

Key semantics:

- `needs` checks the **artifact store first** (Phase 2 precedence) — for artifacts, presence = typed object; base64 fields count only at the boundary (`from_wire` populates artifacts immediately).
- `advance()` no longer sniffs output classes: it reads the artifact slots set by 2-3's typed outputs (`vad_artifact`, `embedding_artifact`, `audio_artifact`, `mel_artifact`) — one unambiguous mapping per step, declared in `produces` (e.g. `"input_vad_data": "vad_artifact"`).
- Unresolvable `requires` → `None` → engine produces the typed "family not fully configured" error from 0-4 (same plumbing).

### 3. Registry

Populate `PIPELINE_SPECS` with the seven families (definitions listed in 3-2; registry lives here, migration of engine call sites in 3-2). Export `get_spec(requested_model) -> PipelineSpec | None` (sentinel name → spec; unknown → None → existing "unknown model" 400 path).

### 4. EngineRequest integration

`common/request.py`: add `pipeline_state: PipelineState | None = None`. Engine sets it at admission (via `entry_rules`, 3-4; interim: step 0 or the family's declared entry for the request type).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/processing/pipeline_registry.py` | `Step`, `PipelineSpec`, `EntryRule`, `PipelineState`, `next_step`, `advance`, registry |
| `harmonyspeech/common/request.py` | `pipeline_state` field |
| `tests/unit/` (new) | `test_pipeline_spec.py` |

## Tests

- Unit: synthetic specs — `next_step` resolves preference order, language variants, returns None on missing config.
- Unit: `advance` maps artifacts per `produces` and stops at `final_step`.
- Unit: unknown sentinel → `get_spec` returns None.

## Checklist

- [ ] `Step`/`PipelineSpec`/`EntryRule`/`PipelineState` implemented
- [ ] `next_step`/`advance` semantics (artifact-first `needs`, declared `produces`)
- [ ] Registry populated with all families (definitions from 3-2 table)
- [ ] EngineRequest carries pipeline state
- [ ] Unit tests for resolution/advance
- [ ] `summary.md` Implementation Status updated
