# Phase 0-1: Safe Deserialization

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 0 — Correctness & security foundation
> **Depends on:** nothing (first task of the plan)
> **Codebase mapping consulted:** `.planning/codebase/CONCERNS.md` (critical finding: unauthenticated RCE via deserialization of request payloads), `ARCHITECTURE.md`, `STRUCTURE.md`

## Objective

Eliminate arbitrary code execution through deserialization of **client-supplied** and file-loaded pickle payloads. Today `torch.load(..., weights_only=False)` is executed on data that either originates from an HTTP request body or from model checkpoints. `weights_only=False` invokes `pickle` and will execute arbitrary `__reduce__` gadgets embedded in the payload — an unauthenticated RCE (CONCERNS critical #1).

## Context (self-contained)

Offending call sites (verified 2026-11-04):

1. `harmonyspeech/task_handler/model_runner_base.py:332-333` — the OpenVoice tone-converter step deserializes the **request field** `input_embedding` (base64 → bytes → `torch.load(weights_only=False)`). This field is client-controllable on `/v1/voice/convert` and on `/v1/audio/speech` (voice-cloning mode), and is also produced internally by the encoder step (`model_runner_base.py:302-306`, `torch.save` → base64).
2. `harmonyspeech/modeling/hf_downloader.py:159-163` — `load_or_download_model()` runs `torch.load(ckpt_path, map_location=device, weights_only=False)` on local/HF-downloaded checkpoints (`checkpoint.pth`). Lower severity (files are engine-side), but the same primitive; OpenVoice/Melo/VoiceFixer checkpoints flow through here.
3. Chatterbox conditionals: `model_runner_base.py:833-838` serializes `dict(t3=conditionals.t3.__dict__, gen=conditionals.gen)` via `torch.save` → base64 into the **request field** `input_embedding`; the TTS step later reconstructs it (see `inputs.py:827-876` raw-bytes path and the `prepare_conditionals` consumption in `_execute_chatterbox_tts*`).

Constraints:

- The internal pipeline (encoder → synthesizer/tone-converter, embed → TTS) currently round-trips these blobs through request fields, so the fix must keep the internal flow working **and** harden the external boundary. The structural fix (artifacts that never leave the process) lands in Phase 2; this task makes the payload itself safe.
- `weights_only=True` succeeds only if the pickle contains plain tensors / primitive containers. A stacked `torch.Tensor` (OpenVoice encoder output) qualifies. The Chatterbox `arg_dict` qualifies **only if** `conditionals.gen` and `conditionals.t3.__dict__` reduce to tensors and primitives — verify at implementation time; if `gen` is a custom object, change the serialization format rather than allowlisting the class.

## Implementation steps

### 1. Audit all `torch.load` / `torch.save` call sites

Grep `torch.load(` and `torch.save(` across `harmonyspeech/`. Expected sites: `model_runner_base.py` (embedding encode/decode, tone conversion), `hf_downloader.py:163`, possibly `modeling/models/*` internals (OpenVoice/VoiceFixer loaders). Record every site and classify: **request-controlled**, **checkpoint**, **internal-only**.

### 2. Convert request-controlled deserialization to `weights_only=True`

`model_runner_base.py`:

- Tone-converter input path (`:332-333`): `torch.load(buf, map_location="cpu", weights_only=True)` wrapped in try/except — on failure return a typed `ValueError("Unsupported embedding payload...")` so the request finishes with a 400-style error result (see also Phase 0-3 for error-result plumbing).
- Encoder output serialization (`:302-306`): ensure what is saved is a bare tensor stack (`torch.save(stacked_tensor.cpu(), buf)`), which loads fine under `weights_only=True`.

### 3. Make Chatterbox conditionals payload tensor-only

In `_execute_chatterbox_embedding` (`:823-838`):

- Serialize a plain dict of tensors/primitives. If upstream `Conditionals.gen` is not a tensor container, extract its tensor state explicitly (e.g. `{k: v for k, v in vars(conditionals.gen).items()}` filtered to `torch.Tensor`/primitives) and document the exact schema in a module docstring.
- On the consuming side (`_execute_chatterbox_tts` / `_execute_chatterbox_turbo_tts` / `_execute_chatterbox_multilingual_tts` / `_execute_chatterbox_vc`, and `inputs.py:827-876` where conditionals are parsed), reconstruct the `Conditionals` object from the plain dict instead of relying on pickled custom classes.
- If reconstruction is not feasible without class pickling, fall back to **safetensors** (`safetensors.torch.save/load`) for the blob and base64 that; safetensors is already an allowed `load_format` concept in `common/config.py:87`.

### 4. Harden checkpoint loading

`hf_downloader.py:163`: switch to `weights_only=True`. Checkpoint `checkpoint.pth` files for OpenVoice/Melo are expected to be plain state dicts; if any checkpoint legitimately requires full pickling (custom classes), gate it behind an explicit engine-side config flag (e.g. env `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1`, default off) with a loud warning log.

### 5. Fail closed with useful errors

Every hardened load site: on `pickle.UnpicklingError` / `RuntimeError` / `safetensors` errors, raise `ValueError` with a message naming the field and the supported formats. These bubble up through `_build_error_result` (already per-request, `model_runner_base.py`) into `finish_reason="error"` outputs — verify that path works for embedding/TC steps.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/model_runner_base.py` | `weights_only=True` at `:332-333`; tensor-only conditionals serialization `:823-838`; reconstruction on consuming handlers |
| `harmonyspeech/modeling/hf_downloader.py` | `weights_only=True` at `:163` (+ optional escape-hatch env flag) |
| `harmonyspeech/task_handler/inputs.py` | Conditionals parse path `:827-876` if format changes |
| `tests/unit/modeling/` (new) | `test_safe_deserialization.py` |
| `tests/e2e/` | extend an existing voice-clone e2e test with a hostile-payload case |

## Tests

- Unit: a crafted pickle payload with a `__reduce__` gadget (e.g. one that writes a marker file / sets an env var via `os.system`) placed in `input_embedding` must raise `ValueError` and **not** execute. Test both the OpenVoice TC path and the Chatterbox conditionals path.
- Unit: round-trip — encoder-produced embedding blob loads with `weights_only=True`; Chatterbox conditionals serialize → reconstruct → `generate()` still works (CPU, tiny fixture).
- Unit: checkpoint load with `weights_only=True` for one OpenVoice checkpoint fixture (use `tests/test-data`).

## Checklist

- [x] All `torch.load`/`torch.save` sites audited and classified
- [x] `model_runner_base.py` request-controlled loads use `weights_only=True` and fail closed with typed errors
- [x] Chatterbox conditionals payload is tensor/primitive-only (or safetensors) and round-trips
- [x] `hf_downloader.py:163` hardened (+ optional escape hatch)
- [x] Hostile-payload unit tests added and passing
- [x] `CHANGELOG.md` entry (security fix, no internal details)
- [x] `summary.md` Implementation Status updated

## Implementation notes (2026-11-04)

Audit result:

| Site | Classification | Action |
|---|---|---|
| `model_runner_base.py` tone-converter embed load | request-controlled | `weights_only=True` + typed `ValueError` (`_load_embedding_tensor`) |
| `model_runner_base.py` encoder embed save | internal-only | unchanged (already a bare tensor stack) |
| `model_runner_base.py` Chatterbox conditionals save/load | request-controlled | payload restricted to tensors/primitives (`_conditionals_to_payload`); consumer loads via `Conditionals.load` (`weights_only=True`) wrapped in a typed `ValueError` (`_load_conditionals_payload`) |
| `hf_downloader.load_or_download_model` | checkpoint | `weights_only=True`; `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1` escape hatch with loud warning |
| `melo/utils.py` `load_checkpoint`, `parallel_wavegan/utils/utils.py` | checkpoint (tooling) | explicit `weights_only=True` (already the torch>=2.6 default) |
| `harmonyspeech.py` `torch.save` (training checkpoints) | internal-only | untouched |

Deviations:

- Chatterbox conditionals already round-trip with `weights_only=True` — `T3Cond` holds only tensors/primitives, so no safetensors fallback was needed. The payload schema is documented in `_conditionals_to_payload`.
- `Conditionals.load` in the pinned `chatterbox-tts==0.1.6` already uses `weights_only=True`; the consumer-side wrapper adds the typed error contract and field naming.
- The e2e hostile-payload case was **not** added: the e2e voice-clone suites download real Chatterbox weights, so the payload path is covered by unit tests with stubs (`tests/unit/modeling/test_safe_deserialization.py`).
- No OpenVoice checkpoint fixture exists under `tests/test-data`; the checkpoint test builds a synthetic state dict and a synthetic hostile checkpoint instead.
