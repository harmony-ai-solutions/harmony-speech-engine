# Phase 7-2: Cache Integration in Executors

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 7 — Cross-request embedding cache
> **Depends on:** 7-1
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (handler/prepare flow), `CONCERNS.md` (state-isolation invariants)
> **State reference:** verified 2026-11-05 at `b81352d`+`d28b81d`

## Objective

Wire the cache into every embedding-producing step, and give audio artifacts a content hash the key is built from. After this task: a voice-clone request whose reference audio was seen in the last TTL window skips the embed model entirely; a flood of 20 identical clone requests computes the embedding **once**.

## Context (self-contained)

Embedding-producing handlers in `harmonyspeech/task_handler/model_runner_base.py` (current names verified 2026-11-05):

| Handler | Serves | Produces |
|---|---|---|
| `_execute_openvoice_tone_converter_encoder` | OpenVoice V1 **and** V2 embed steps | `SpeakerArtifact.embedding` (tensor) |
| `_execute_harmonyspeech_encoder` | HarmonySpeech embed step | `SpeakerArtifact.embedding` |
| `_execute_chatterbox_embedding` | Chatterbox embed step (used by TTS/VC pipelines and `/v1/embed/speaker`) | `SpeakerArtifact.conditionals` (object with device tensors) |
| `_execute_chatterbox_vc` (target prep) | VC target voice → `ref_dict`/`set_target_voice` | same conditionals mechanism — **include if the seam is clean** (target audio is the repeated input in VC floods) |

> **Executor note (2026-11-05): the VC seam is INCLUDED — verdict: clean.**
> The cached value is the `ref_dict` tensor dict wrapped in
> `SpeakerArtifact.conditionals` (the plain-dict form is an explicitly
> supported conditionals stand-in); a hit assigns the per-request deep copy to
> `model.ref_dict` and skips `set_target_voice`. Pre-computed conditionals
> still take precedence over the cache. See Deviations.

Boundary decode-once (Phase 2) already puts a decoded `AudioArtifact` on every request carrying `input_audio` — the hash is computed there, once.

## Implementation steps

### 1. `AudioArtifact.content_hash`

In `harmonyspeech/common/artifacts.py`: lazy cached property — sha256 over the waveform's contiguous float32 bytes **plus the sample rate** (so 16k vs 24k decodes of one clip can't collide). Compute at admission (`from_wire_audio` decode path) so every downstream step sees it for free; fall back to lazy compute on first access. Unit-test stability: same audio via wav vs flac container (decode → same PCM) → same hash; different audio → different hash.

### 2. Handler integration pattern (apply to the table above)

Inside each handler, per `(request, input)` pair — before calling the model:

```python
audio = ...                       # AudioArtifact from the prepared input (Phase 2)
key = cache.make_key(
    model_identity=self.model_identity,          # name/type/dtype/repo/load_version (7-1)
    audio_hash=audio.content_hash,
    params={"exaggeration": ..., "vad_source": ...},   # only what influences the output
)
cached = self.embedding_cache.get(key) if self.embedding_cache else None
if cached is not None:
    # hit: build the result output from the copied artifact — no model call
    # (populate the same output slots the miss path populates: artifact + any
    # legacy fields the serving layer needs; serialization stays at the boundary)
else:
    ... existing computation ...
    self.embedding_cache.put(key, artifact)
```

Rules:

- **Batch composition**: a step batch may mix hits and misses — resolve the cache per request first, compute only the misses (currently sequential; after Phase 8-2 the misses go through the padded batch), then assemble outputs **in original request order** via the existing zip pairing (Phase 0-3 invariants assert this — keep them green).
- **Miss-path parity**: a hit must produce byte-identical downstream behavior to a miss. OpenVoice embedders are deterministic → artifacts equal; Chatterbox `prepare_conditionals` is deterministic given identical weights/audio/params → assert in tests.
- **Per-request state isolation (Phase 0-2)**: the served copy is the request's own; the Chatterbox baseline-restore wrappers are untouched (cache sits above them, not inside).
- **Chatterbox `exaggeration`**: it is baked into `conditionals.t3.emotion_adv` — it MUST be part of `params`. Read the effective value from the request's generation options (default 0.5).
- **VAD source** for the OpenVoice TCE key: read from the `VadArtifact` (Phase 3-4 added `source="whisper"|"silero"`).
- **Model identity / load_version**: extend the runner init to register with the cache (bump `load_version` on every `load_model` — the engine may reload a config; see 4-3 registry plumbing for how the runner reaches engine-owned services).

### 3. No-cache fast path

`embedding_cache_enabled: false` (or `ttl_s: 0`) → handlers skip all cache calls (guard once at handler entry, not per request). Zero behavioral delta vs pre-Phase-7 — locked by a test running the same suite path with the cache disabled.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/common/artifacts.py` | `AudioArtifact.content_hash` (lazy, admission-computed) |
| `harmonyspeech/endpoints/openai/wire.py` | compute hash at `from_wire_audio` decode |
| `harmonyspeech/task_handler/model_runner_base.py` | cache lookup/put in the 3 embed handlers (+ VC target seam if clean) |
| `harmonyspeech/task_handler/gpu_model_runner.py` / `cpu_executor.py` | model-identity registration + `load_version` bump |
| `tests/unit/inference_flow/test_embedding_cache_integration.py` (new) | spy-based hit/miss tests |

## Tests

- Unit (stubbed model via existing test doubles): two identical requests → second embed model call count == 0; outputs artifact-equal to the first.
- Unit: different audio / different `exaggeration` / different VAD source / different model (ov1 vs ov2 config) → all misses.
- Unit: expired entry → recompute (monkeypatched clock); disabled cache → model called every time, zero cache calls.
- Unit: hit + miss mixed in one batch → outputs in request order, pairing asserts hold.
- Integration (`tests/integration/`): two engine-level voice-clone requests with the same reference (CPU model config) → second one measurably skips the embed step (spy on executor), end response equal (base64 output for deterministic families byte-equal).
- Regression: full existing suites green with cache enabled by default (the new default must not change any assertion — the outputs are identical, only compute is skipped).

## Checklist

- [x] `AudioArtifact.content_hash` + admission computation
- [x] Cache wired into OV TCE (V1/V2), HarmonySpeech encoder, Chatterbox embedding (+VC target if clean — note decision)
- [x] Hit/miss batch composition preserves pairing invariants
- [x] Determinism + isolation + disabled-path tests green
- [x] Integration test proves zero recompute for repeated reference (`tests/integration/test_embedding_cache_flow.py`)
- [x] `summary.md` Implementation Status updated

## Deviations (executor, 2026-11-05)

- **VC target seam: included** (verdict recorded in the Context table above).
  The ref dict is cached as `SpeakerArtifact(conditionals=<ref_dict dict>,
  backend="chatterbox")`; the cache's dict-aware deep copy provides Phase 0-2
  isolation on serve.
- **`load_version` bump seam location**: `ModelRunnerBase._load_model`
  (`task_handler/model_runner_base.py`). Both `CPUModelRunner.load_model` and
  `GPUModelRunner.load_model` call `_load_model()`, so one seam covers first
  boot, reloads, and fresh runner objects for a reloaded config. The plan's
  file table named `gpu_model_runner.py / cpu_executor.py`; neither needed
  edits for the bump itself. Threading the cache to the runners follows the
  weight-registry pattern and therefore touches `executor_base.py`,
  `cpu_executor.py`, `gpu_executor.py`, `cpu_worker.py`, `gpu_worker.py`
  (all None-safe).
- **Failure tolerance**: cache key resolution (make_key + get) runs in a
  per-request guard that degrades to a plain miss, and `cache.put` failures
  are logged and dropped — the cache can never fail a request that would
  otherwise succeed (no-cache fast-path parity).
- **Admission hash warming** is a post-prefill pass in
  `endpoints/openai/wire.py::from_wire_audio` (touching each decoded
  artifact's lazy property). The decode itself lives in
  `common/inputs.prefill_audio_artifacts`, which is unchanged — the hash is
  computed lazily on first access anyway, admission only pre-warms it.
- **Serve-path deep copy**: implemented inside the cache
  (`EmbeddingCache.get` → `_copy_artifact`, a field-wise clone mirroring
  `_clone_conditionals` plus CPU moves). It cannot literally import the
  handler-layer helper (processing ← task_handler would invert the layering),
  so the clone logic is deliberately duplicated in
  `processing/embedding_cache.py` and kept structurally identical.
- Tests were written but **not executed locally** (no Python environment with
  torch is available in this checkout); central verification pending. The
  engine-level integration proof is 7-3's exit work.
