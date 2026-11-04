# Phase 4-4: Chatterbox Backbone Sharing

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 4 — VRAM: fp16 rollout + Chatterbox weight dedup
> **Depends on:** 4-1 (dtype), 4-3 (registry), 0-2 (state isolation)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`, research §5.3(a), risk #5

## Objective

Deduplicate the Chatterbox family's shared backbones: `ChatterboxTTS`, `ChatterboxEmbedding`, `ChatterboxVC` (and turbo where applicable) share one S3Gen + one VE loaded from `resemble-ai/chatterbox`; the multilingual variant shares what its checkpoint actually contains identically. Expected recovery: ≈1.5–2.5 GiB at fp16 (≈3–4 GiB fp32) on top of the 4-1 halving.

## Context (self-contained)

Component needs per role (research §5.3(a) table):

| Role/config | T3 | S3Gen | VE |
|---|---|---|---|
| `ChatterboxTTS` | yes | yes | yes |
| `ChatterboxMultilingualTTS` | yes (**different T3**) | same checkpoint → share | share |
| `ChatterboxTurboTTS` | own repo (`chatterbox-turbo`) | own repo → separate group | separate group |
| `ChatterboxVC` | no | yes | yes |
| `ChatterboxEmbedding` | — | yes | yes |

Loader call sites: `modeling/loader.py:304-337` — each branch calls the wrapper `from_pretrained` which internally builds full trees. Sharing therefore requires **component-level construction**: build one backbone set per `(repo, revision, dtype)` and wire the same module objects into each role's model instance.

Safety (0-2 invariant): `conds` lives on the TTS wrapper instance; `ref_dict` on the VC instance — per executor, never on shared S3Gen/VE. Verify upstream attribute topology in the pinned `chatterbox-tts==0.1.6` source before surgery (`.venv` or the vendored reference under `.current_work/chatterbox-tts/`).

## Implementation steps

### 1. Component builder

In `modeling/models/chatterbox/chatterbox.py` (or a sibling `factory.py`):

```python
def build_backbone(repo: str, revision: str | None, dtype: torch.dtype, device, registry: WeightRegistry):
    """Load the reference model once via the registry keyed by (repo, revision, {"s3gen","ve"}, dtype);
    return (s3gen_module, ve_module) — shared instances."""

def assemble_role(role: str, backbone, *, multilingual_t3=None, device, dtype):
    """Construct the role's model object with shared s3gen/ve references and its own T3
    (standard/multilingual) or none (VC/embedding), replicating exactly what the upstream
    from_pretrained would have produced (attribute names, eval mode, devices)."""
```

Implementation approach: call the upstream `from_pretrained` **once per (repo, revision, dtype)** to get a fully-built reference model, then `deepcopy`-free "graft": for each subsequent role, construct the role object and assign `role_model.s3gen = shared_s3gen; role_model.voice_encoder = shared_ve` (exact attribute names from upstream source) **before** moving to device. The reference model's own T3 is reused for the TTS role; VC/multilingual discard/replace what they don't need. Keep an explicit mapping comment of upstream attribute names.

### 2. Loader wiring

`loader.py:304-337`: for Chatterbox branches, when `model_config.backbone_group` is set (e.g. `"chatterbox-base"`), build via the factory; otherwise keep the current per-config `from_pretrained` path (opt-in per config — safe rollout). `config.gpu.yml`: add `backbone_group: "chatterbox-base"` to `chatterbox`, `chatterbox-embedding`, `chatterbox-vc`; `chatterbox_multilingual` joins the group **only if** its S3Gen/VE checkpoint hashes match the base repo's (verify by comparing the actual safetensors/ckpt files in the HF cache; document the evidence in this file when done). Turbo stays ungrouped.

### 3. Distinct wrapper instances

Each config still gets its **own** executor/model wrapper object (so `conds`/`ref_dict` per 0-2 stay per-role); only backbone submodules are shared. Assert at assembly: `assemble_role` never mutates the shared modules' parameters (`requires_grad` stays False; buffers untouched).

### 4. Gate + measure

- Re-run the 4-2 harness with axis `sharing:dedup-vs-dup` — outputs must be within the deterministic thresholds (sharing identical weights should be **bit-identical**; any divergence = wiring bug).
- Record boot VRAM (torch.cuda.memory_allocated after load) before/after in `docs/benchmarks/` (feeds 4-5 and the research's curated-set arithmetic).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/` | backbone factory + role assembly |
| `harmonyspeech/modeling/loader.py` | backbone_group branch in Chatterbox cases |
| `config.gpu.yml` | `backbone_group` assignments (post verification) |
| `tests/unit/modeling/` | sharing identity tests |

## Tests

- Unit: two roles in a group share S3Gen/VE object identity; T3 distinct where required.
- Unit: shared module params identical (bitwise) after both roles constructed; `requires_grad` False.
- Harness (GPU): `sharing` axis bit-identical outputs for TTS/VC/embedding paths.
- Boot measurement recorded (expected: Chatterbox family block drops by ≈1.5–2.5 GiB fp16).

## Checklist

- [ ] Factory + assembly implemented against verified upstream attribute topology
- [ ] `backbone_group` opt-in wired in loader; turbo ungrouped
- [ ] Multilingual grouping decided on checkpoint-hash evidence (documented here)
- [ ] 4-2 sharing axis passes (bit-identical)
- [ ] Boot VRAM before/after recorded
- [ ] `summary.md` Implementation Status updated
