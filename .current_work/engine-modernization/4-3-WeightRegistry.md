# Phase 4-3: Weight Registry

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 4 — VRAM: fp16 rollout + Chatterbox weight dedup
> **Depends on:** 0-2 (per-request state off shared objects), 4-1 (dtype finalized before sharing keyed on it)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (loader topology), research §6.2 Phase 3 item 2

## Objective

A ref-counted, load-once weight registry so multiple executor configs can **share backbone submodules** loaded from the same checkpoint files at the same dtype. Keyed by `(repo_id, revision, component, dtype)` per the research design. This task delivers the registry; 4-4 wires Chatterbox onto it.

## Context (self-contained)

Today `loader.py` builds one model tree per config entry — four Chatterbox configs each `from_pretrained("resemble-ai/chatterbox")` (plus turbo), duplicating S3Gen (~0.5–0.8 GiB fp32) and VE (~0.1 GiB) 2–3× (research §5.3(a): ≈3–4 GiB fp32 / ≈1.5–2.5 GiB fp16 recoverable).

Safety invariant (from 0-2): shared submodules must be **read-only** — per-request state (`conds`, `ref_dict`) lives on wrapper executor instances, never on shared backbone modules. VE and S3Gen forwards are stateless (verify when re-reading upstream sources).

## Implementation steps

### 1. Registry module `harmonyspeech/modeling/weight_registry.py`

```python
class WeightRegistry:
    """Ref-counted shared module store. Engine owns ONE instance, passed to the loader."""

    def __init__(self) -> None:
        self._store: dict[tuple, torch.nn.Module] = {}   # (repo, revision, component, dtype) -> module
        self._refs: dict[tuple, int] = {}

    def acquire(self, key: tuple, loader_fn: Callable[[], torch.nn.Module]) -> torch.nn.Module:
        """Return the shared module, loading via loader_fn on first acquire.
        Same object on every acquire with the same key."""

    def release(self, key: tuple) -> None:
        """Decrement; drop the module when the count hits zero (frees VRAM)."""

    def stats(self) -> dict[str, int]:
        """entries + total param bytes — feeds /metrics (4-5)."""
```

- Thread-safety: engine loads sequentially at boot today; add a simple `threading.Lock` anyway (cheap, future-proof for lazy loading in Phase 6).
- Move-to-device policy: the registry stores modules **on their target device** already cast to `key.dtype`; acquirers must not re-cast/re-move (assert in `acquire` via a debug check).
- Read-only discipline: document in the class docstring + provide `registry.assert_readonly(module)` no-op hook for future enforcement ideas (e.g. param-level `_requires_grad=False` audit — never `requires_grad_(True)` on shared weights).

### 2. Engine wiring

`HarmonySpeechEngine.__init__` creates `self.weight_registry = WeightRegistry()` and passes it through executor construction into `loader.get_model(...)` (extend the loader signature/context object — re-read how `model_config` reaches the loader today). Executors/registry release on model unload — no unload path exists today (eager resident), so `release()` is exercised only by tests for now; wire it into the Phase 6 governor later.

### 3. Config surface

`ModelConfig` gains `backbone_group: str | None = None` (`common/config.py` + `EngineConfig.load_config_from_yaml` already forwards unknown YAML keys via `**model_cfg` — add the explicit parameter so it validates). Semantics: entries with the same `backbone_group` **and** the same underlying checkpoint repo/revision/dtype share components (4-4 defines which components per group).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/weight_registry.py` (new) | registry |
| `harmonyspeech/common/config.py` | `backbone_group` param |
| `harmonyspeech/modeling/loader.py` | accept registry; pass-through plumbing |
| `harmonyspeech/engine/harmonyspeech_engine.py` | own + pass the registry |
| `tests/unit/modeling/test_weight_registry.py` (new) | registry tests |

## Tests

- Unit: two `acquire`s with same key → identical object; loader_fn called once.
- Unit: `release` twice → module dropped (weakref/identity check); re-acquire reloads.
- Unit: different dtype/repo in key → distinct modules.
- Unit: `stats()` counts entries/bytes correctly.

## Checklist

- [x] Registry implemented (ref-count, lock, stats, read-only docs)
- [x] Engine owns one registry; loader plumbing in place
- [x] `backbone_group` config param validated
- [x] Registry unit tests green
- [x] `summary.md` Implementation Status updated

## Outcome & deviations (2026-11-04)

* `harmonyspeech/modeling/weight_registry.py` — `acquire(key, loader_fn)` loads
  once under a `threading.Lock`, asserts the module's dtype matches the key,
  pins `eval()` + `requires_grad=False`; `release()` decrements and drops at
  zero; `stats()` returns entries/total bytes/refs per component;
  `assert_readonly()` is now enforced rather than a no-op hook.
* Plumbing: `HarmonySpeechEngine.weight_registry` →
  `ExecutorBase(model_config, weight_registry)` → worker → model runner →
  `get_model(..., weight_registry=)`.
* `ModelConfig.backbone_group` is validated (string or None).
* Key detail found during implementation: the component name must encode the
  checkpoint *file set* (`s3gen`, `s3gen:mtl`, `s3gen:turbo`, `ve`, `ve:mtl`,
  `ve:turbo`, `t3`, `t3:multilingual`, `t3:turbo`), otherwise a multilingual
  role would incorrectly reuse the standard `s3gen.safetensors` when it was
  supposed to load `s3gen.pt`.
* Unit tests: `tests/unit/modeling/test_weight_registry.py` (identity, refcount,
  release/reload, distinct keys, stats, read-only, dtype consistency, threaded
  acquire).
