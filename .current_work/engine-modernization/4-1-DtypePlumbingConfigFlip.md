# Phase 4-1: dtype Plumbing & Config Flip

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 4 — VRAM: fp16 rollout + Chatterbox weight dedup
> **Depends on:** Phase 0 complete (0-2 state isolation especially); 4-2 harness should be built in parallel to gate the flip
> **Codebase mapping consulted:** `.planning/codebase/STACK.md`, `CONCERNS.md` (all-fp32 config), research §5.3(c)

## Objective

Make `dtype` real for **every** loader path (the Chatterbox native loaders ignore it today) and flip the shipped GPU config from `float32` to `float16`, halving the ~16.4 GiB weights bill. Quality is gated by the 4-2 audio regression harness — build the harness before or with this task; the config flip is the last commit.

## Context (self-contained)

- `ModelConfig.dtype` supports `auto/float16/bfloat16` (`common/config.py:144-186`); the generic HF load path applies it via `_set_default_torch_dtype(model_config.dtype)` (`modeling/loader.py:254`) — OpenVoice/Melo/VoiceFixer already honor dtype.
- **Chatterbox does not**: `loader.py:304-337` calls the wrapper `from_pretrained(device=device)` variants (`modeling/models/chatterbox/chatterbox.py:86-101, 113-151, 203-218, 241-256`), which take **no dtype** and load fp32 weights.
- `config.gpu.yml`: every entry is `dtype: "float32"` (verified 2026-11-04).
- Known sensitivities (research §5.3(c)): VoiceFixer STFT/log-mel + normalization (test audio, not loss); OpenVoice/Melo flow + HiFiGAN fine at fp16 but `noise_scale` sampling changes output (stochastic gate, not a bug); Chatterbox VE already force-casts inputs to float32 (`chatterbox.py:173-189`) — mixed precision needs care in `prepare_conditionals`.
- CPU executor already downcasts fp16→bf16 on CPU (`cpu_executor.py:48-50`); `config.yml` (CPU) untouched by the flip.
- Whisper is CTranslate2 — `compute_type` already defaults `float16` on CUDA (`loader.py:283`); not affected by `dtype`.

## Implementation steps

### 1. Chatterbox wrapper dtype support

`modeling/models/chatterbox/chatterbox.py`: each wrapper `from_pretrained(device, **kwargs)` gains `dtype: torch.dtype = torch.float32`. After the upstream model is constructed:

```python
model = model.to(dtype)                      # weights cast
# Exception: keep the VoiceEncoder (VE) path in float32 — its LSTM already requires
# float32-safe input handling (chatterbox.py:173-189 casts inputs back to float32).
# Concretely: cast per-submodule (t3, s3gen -> dtype; ve -> torch.float32) after
# re-reading the upstream attribute names in the pinned chatterbox-tts==0.1.6 source.
```

Pass `dtype=model_config.dtype` from `loader.py:304-337` (all five Chatterbox branches incl. `ChatterboxEmbedding`). Log the effective per-submodule dtypes at load.

### 2. KittenTTS / Silero (no-ops, document)

ONNX/CT2 paths ignore torch `dtype` — add a code comment + config-schema note so operators stop expecting `dtype` there (KittenTTS quantized variants are chosen by `model:` name; Whisper by `compute_type`).

### 3. Per-model opt-out surface

`ModelConfig` already accepts per-entry `dtype` — document the policy: **default `float16` on GPU entries; any entry may pin `"float32"`** (expected opt-outs initially: none until 4-2 says otherwise; VoiceFixer is the most likely candidate). The 4-2 harness produces the evidence for opt-outs.

### 4. The flip (last, gated)

`config.gpu.yml`: change all GPU entries `dtype: "float32"` → `dtype: "float16"`. Only after 4-2 fixtures pass within thresholds on each family (run harness on a GPU host; results recorded in 4-2). Keep `config.gpu.fp32.yml` copy of the previous file for A/B debugging (or rely on git — prefer git, avoid file clutter; decide at implementation).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/chatterbox.py` | dtype param on all four wrapper `from_pretrained`s + per-submodule cast (VE stays fp32) |
| `harmonyspeech/modeling/loader.py` | pass `model_config.dtype` into Chatterbox branches |
| `config.gpu.yml` | flip to float16 (gated by 4-2) |
| `tests/unit/modeling/` | dtype plumbing tests |
| `docs/models.md` | dtype policy + per-family sensitivities |

## Tests

- Unit: wrapper `from_pretrained(dtype=torch.float16)` (tiny/mock checkpoint) → submodule dtypes as specified (t3/s3gen fp16, ve fp32).
- Unit: loader passes config dtype (spy/monkeypatch on wrapper).
- 4-2 harness (GPU): per-family fp32-vs-fp16 spectral-distance within thresholds.
- Boot check (GPU host): engine starts with flipped config; `torch.cuda.memory_allocated` after load ≈ half of fp32 baseline for the torch families (record numbers for 4-5).

## Checklist

- [x] Chatterbox wrappers accept + apply dtype (per-submodule policy implemented & documented)
- [x] Loader passes dtype for all native branches
- [x] KittenTTS/Silero/Whisper dtype no-op documented (docs/models.md)
- [x] 4-2 harness green on every runnable family; Chatterbox entries flipped to float16
- [x] VRAM-before/after numbers recorded (feeds 4-5)
- [x] `docs/models.md` + `CHANGELOG.md` updated
- [x] `summary.md` Implementation Status updated

## Outcome & deviations (2026-11-04)

Measured on the RTX 5060 Ti (16 GiB, driver 610.88, torch 2.9.1+cu130).

* **T3 → float16, S3Gen stays float32.** Casting S3Gen (flow decoder / 16 kHz
  tokenizer / HiFiGAN) to half precision audibly degraded synthesis: with
  identical T3 weights and a pinned seed the transcribed output changed
  ("The quick-brown fog could jump so that the lazy dust…") while duration and
  long-term spectrum stayed close. Upstream itself carries an
  `estimator_dtype = "fp32"` hint. Casting only T3 keeps the ~1 GiB/role win
  with no quality change. The VoiceEncoder is float32 by construction, and the
  Perth watermark network is pinned float32 (it is built inside the loader's
  default-dtype context and otherwise ends up with half-precision conv weights).
* **Partial config flip.** `config.gpu.yml` flips the five Chatterbox entries to
  `float16`. OpenVoice V1/V2 and MeloTTS stay `float32`: their adapted VITS code
  builds float32 tensors in the flow/text encoder (`expected scalar type Float
  but found Half`) and the MeloTTS flow splines assert under fp16. VoiceFixer
  stays `float32` (precision-sensitive restoration; its GPU path currently has a
  pre-existing device-placement bug, so the dtype axis could not be measured).
  Each opt-out is recorded in `docs/models.md` and
  `docs/benchmarks/audio-regression.md`.
* **A latent CUDA bug was found and fixed.** Chatterbox's bundled `conds.pt`
  carries a `requires_grad=True` tensor; `.to("cuda")` turned it into a non-leaf,
  which broke the Phase 0-2 baseline `copy.deepcopy` on GPU. Conditionals are now
  detached at load and the baseline snapshot clones field-by-field
  (`_clone_conditionals`). This affected the unshared path too — CPU-only e2e
  had masked it.
* Boot VRAM (Chatterbox TTS+VC+embedding): 3.04 → 2.05 GiB allocated.
