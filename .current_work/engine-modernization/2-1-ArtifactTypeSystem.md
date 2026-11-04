# Phase 2-1: Artifact Type System

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 2 — Artifact bus
> **Depends on:** Phase 1 complete (boundary decoder exists)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, `CONVENTIONS.md` (research §6.2 Phase 1 design is the source of this structure)

## Objective

Introduce the typed artifact classes that pipeline steps will exchange instead of base64 strings. This task creates the type system only (no executor changes yet) — it is the vocabulary every later subtask speaks.

## Context (self-contained)

Research §1.4 (verified): every intermediate hop today is an API-shaped string — base64 pickle (embeddings, conditionals), base64 WAV (synthesis→TC), base64 JSON float lists (VoiceFixer mel: ~5 MB payload, ~35–45 MB host churn, ~40–80 ms per hop [research estimates]). The artifact bus replaces these with in-process typed objects; serialization remains only at the HTTP boundary.

Design rules from the research (§6.2 Phase 1):

- `RequestInput` gains internal artifact fields that **take precedence** over the base64 fields.
- Base64 fields remain accepted (backward compatibility, external callers).
- **Device discipline**: every artifact carries `.device`; cross-device moves are explicit (`artifact.to(device)`), never implicit.
- **Lifecycle**: artifacts are owned by the `EngineRequest` and dropped in `free_finished_requests()`.

## Implementation steps

### 1. New module `harmonyspeech/common/artifacts.py`

```python
from dataclasses import dataclass, field
from typing import Literal
import torch

@dataclass
class AudioArtifact:
    """Waveform artifact. waveform shape: [1, samples] float32, on artifact.device."""
    waveform: torch.Tensor            # or np.ndarray for CPU-only steps — decide once (see note)
    sample_rate: int
    device: torch.device | None = None

    @property
    def duration_s(self) -> float: ...
    def to(self, device: str | torch.device) -> "AudioArtifact": ...  # explicit move, returns self-like

@dataclass
class MelArtifact:
    """Mel spectrogram artifact (VoiceFixer / HarmonySpeech synth→vocoder hop)."""
    mel: torch.Tensor                 # [1, mel_bins, frames]
    sample_rate: int
    hop_length: int
    device: torch.device | None = None

@dataclass
class SpeakerArtifact:
    """Speaker conditioning, backend-polymorphic."""
    embedding: torch.Tensor | None = None     # OpenVoice sid/ref-encoder embedding
    conditionals: object | None = None        # Chatterbox Conditionals (opaque to the bus)
    backend: Literal["openvoice", "chatterbox"] | None = None
    device: torch.device | None = None

@dataclass
class VadArtifact:
    """VAD/transcription segments — whisper segment dicts or silero timestamps."""
    segments: list[dict]
    # Phase 3-4 normalizes silero output to the same schema
```

Note on waveform type: Whisper/VAD steps run on CPU (numpy world: faster-whisper, `librosa`), Chatterbox/OV run on torch. Decide **once**: standardize on `torch.Tensor` (CPU tensors are cheap to create from numpy via `torch.from_numpy(...)` — shares memory) and provide `AudioArtifact.from_numpy(wav, sr)` / `.numpy` accessor. Document the decision in the module docstring.

### 2. Device-move helper

`to()` must be explicit and cheap-checked: assert target differs or no-op; log at debug on cross-device moves (helps catch accidental per-hop transfers during Phase 2-3 reviews).

### 3. Ownership plumbing (minimal)

`harmonyspeech/common/request.py` `EngineRequest`: add `artifacts: dict[str, object] | None = None` — the per-request artifact store keyed by the same names as the request fields (`"input_audio"`, `"input_embedding"`, `"input_vad_data"`, `"input_mel_spectrogram"`). `Scheduler.free_finished_requests()` (`scheduler.py:244-245`) requires no change (requests are dropped wholesale → artifacts garbage-collected); add an explicit `request.clear_artifacts()` hook called from `update_request_status` when a request reaches any `FINISHED_*` state, so VRAM pins release deterministically rather than "eventually".

### 4. Unit tests for the types themselves

Round-trips: numpy→artifact→device move→numpy; duration property; `MelArtifact` shape preservation.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/artifacts.py` (new) | the four artifact dataclasses + helpers |
| `harmonyspeech/common/request.py` | `artifacts` store + `clear_artifacts()` |
| `harmonyspeech/processing/scheduler.py` | call `clear_artifacts()` on finish (in `update_request_status`) |
| `tests/unit/` (new) | `test_artifacts.py` |

## Checklist

- [ ] `artifacts.py` with Audio/Mel/Speaker/Vad artifacts, explicit `to(device)`
- [ ] Waveform representation decision made (torch.Tensor) and documented
- [ ] `EngineRequest.artifacts` store + deterministic release on finish
- [ ] Type unit tests passing
- [ ] `summary.md` Implementation Status updated
