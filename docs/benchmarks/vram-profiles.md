# VRAM profiles

Measured VRAM footprint of the shipped `config.gpu.yml` catalogue, before and
after Phase 4 (fp16 T3 rollout + Chatterbox backbone deduplication). Produced by
[`scripts/measure_vram.py`](../../scripts/measure_vram.py).

## Method

* **Host:** NVIDIA GeForce RTX 5060 Ti (16311 MiB = 15.93 GiB), driver 610.88,
  CUDA 13.3 UMD, torch 2.9.1+cu130, Python 3.12.12, Windows 11.
* GPU 1 of 3, otherwise idle (`CUDA_VISIBLE_DEVICES=1`); GPU 0 carries the
  desktop compositor and was left alone so the numbers are not polluted by
  unrelated allocations.
* `HSE_ALLOW_UNSAFE_CHECKPOINT_LOAD=1` (pre-existing requirement for the
  Harmony Speech checkpoints).
* Boot only (`--no-warmup`): all 28 configured executors resident, then sample
  `torch.cuda.memory_allocated/reserved/peak` and
  `nvidia-smi --query-gpu=memory.used`.
  Per-model `VRAM MiB` is the allocation delta measured while that model loads
  (`GPUModelRunner.model_memory_usage`), so shared backbones are attributed to
  the role that loaded them and show 0 for later acquirers.
* **"Before" measurement method:** the pre-flip config was extracted verbatim
  from git (`git show HEAD:config.gpu.yml`) into
  `out/vram/config.gpu.before.yml`; the current build was then pointed at it.
  Because `backbone_group` is opt-in and the factory is only used when it is
  set, that run reproduces the fp32 + duplicated behaviour of commit `85cb65c`
  without stashing code.
* **Headroom** = device total − `torch.cuda.memory_reserved` (and, separately,
  device total − `nvidia-smi` used). On Windows the display driver/WDDM also
  holds a little memory outside CUDA's accounting, so `nvidia-smi` is the
  stricter number.

## Result — full catalogue

| Metric | Before (fp32, duplicated) | After (fp16 T3 + dedup) | Change |
|---|---|---|---|
| torch allocated | 16.45 GiB | **8.52 GiB** | −7.93 GiB (−48 %) |
| torch reserved | 16.54 GiB | 8.62 GiB | −7.92 GiB |
| nvidia-smi used | 15.67 GiB (16042 MiB) | **8.83 GiB** (9040 MiB) | −6.84 GiB |
| headroom (total − reserved) | 0.00 GiB | **7.30 GiB** | +7.30 GiB |
| headroom (nvidia-smi) | 0.26 GiB | 7.10 GiB | +6.84 GiB |
| shared registry | — | 2.98 GiB in `chatterbox-base` (3.96 GiB not duplicated) | |
| resident models | 28 | 28 | — |

The fp32 catalogue **overflows the 16 GiB card** (16.45 GiB of live CUDA
allocations on a 15.93 GiB device). On Windows this does not crash: WDDM pages
the excess into shared system memory, which is why the "before" boot finishes
but with ~0 free VRAM — a performance cliff rather than a clean failure. After
the phase the same catalogue fits with ~7 GiB free.

### Chatterbox block (the phase's target)

| Role | Before (MiB) | After (MiB) | Notes |
|---|---:|---:|---|
| `chatterbox` | 3102 | 2068 | fp16 T3 + shared S3Gen/VE |
| `chatterbox_turbo` | 4243 | 3413 | fp16 T3; own meanflow S3Gen (ungrouped) |
| `chatterbox_multilingual` | 3116 | 1025 | fp16 T3 only; S3Gen/VE shared |
| `chatterbox_vc` | 1067 | 0 | fully shared S3Gen/VE |
| `chatterbox_embedding` | 3101 | 0 | shares S3Gen/VE/T3 |
| **block total** | **14.29 GiB** | **6.35 GiB** | −7.94 GiB |

### After — per-model actuals

| model | device | dtype | group | VRAM MiB | weights MiB |
|---|---|---|---|---:|---:|
| chatterbox | cuda | float16 | chatterbox-base | 2068 | 2028 |
| chatterbox_turbo | cuda | float16 | — | 3413 | 3373 |
| chatterbox_multilingual | cuda | float16 | chatterbox-base | 1025 | 2035 |
| chatterbox_vc | cuda | float16 | chatterbox-base | 0 | 1007 |
| chatterbox_embedding | cuda | float16 | chatterbox-base | 0 | 2028 |
| voicefixer-restorer | cuda | float32 | — | 273 | 285 |
| voicefixer-vocoder | cuda | float32 | — | 129 | 129 |
| ov1-synthesizer-en | cuda | float32 | — | 153 | 153 |
| ov1-synthesizer-zh | cuda | float32 | — | 155 | 153 |
| ov1-tone-converter | cuda | float32 | — | 126 | 125 |
| ov1-tone-converter-encoder | cuda | float32 | — | 126 | 125 |
| ov2-synthesizer-en/zh/es/fr/ja | cuda | float32 | — | 199–200 each | 198 each |
| ov2-tone-converter | cuda | float32 | — | 127 | 125 |
| ov2-tone-converter-encoder | cuda | float32 | — | 126 | 125 |
| faster-whisper-* (CT2) | cuda | — | — | outside torch allocator (≈1.6 GiB via nvidia-smi) | |
| hs1-encoder/synthesizer/vocoder | cpu | float32 | — | 0 (host RAM) | 48 / 117 / 8 MiB host |
| kitten-tts-*, silero-vad | cpu | float32 | — | 0 (host RAM) | |

`chatterbox_turbo` is the largest single model because its T3 is a GPT2-medium
with `n_positions ≈ 8196`: the upstream module tree carries a bool causal mask
per attention layer (~1.5 GiB of `torch.bool` buffers, visible in `weights`).
They are part of the checkpoint's module tree and not removable without upstream
surgery.

## Acceptance floor

The plan's floor was "full catalogue resident with ≥ 8 GiB headroom on a 16 GiB
card", derived from an estimated ~5.9 GiB resident after a *full* fp16 rollout.
This machine has exactly a 16 GiB card, so the floor applies as written:

* **Measured headroom: 7.10 GiB (nvidia-smi) / 7.30 GiB (torch reserved).**
* **Verdict: floor missed by ≈0.7–0.9 GiB**, with the gap fully explained:

| Item | Size | Why it is still fp32 / present |
|---|---|
| OpenVoice V1/V2 + MeloTTS (9 models) | ≈2.0 GiB | adapted VITS code creates float32 tensors in the flow/text encoder; MeloTTS flow splines assert under fp16. Kept `dtype: "float32"` (see [audio regression](audio-regression.md)). |
| VoiceFixer (2 models) | ≈0.4 GiB | GPU device-placement bug blocks fp16 measurement; kept `float32`. |
| Chatterbox Turbo GPT2 mask buffers | ≈1.5 GiB | upstream `n_positions`-sized bool masks per attention layer; removing them is upstream surgery. |

Fixing either of the first two groups (2.0 GiB) would clear the floor; both are
follow-ups outside Phase 4's scope (upstream/adapted-model work). The headline
claim of the phase — the Chatterbox block shrinking from 14.29 GiB to 6.35 GiB
— is met and measured.

## Reproducing

```bash
# after (shipped config)
python scripts/measure_vram.py --config config.gpu.yml --no-warmup \
    --label "after" --json-out out/vram/after.json

# before (pre-flip config from git)
git show 85cb65c:config.gpu.yml > out/vram/config.gpu.before.yml
python scripts/measure_vram.py --config out/vram/config.gpu.before.yml --no-warmup \
    --label "before" --json-out out/vram/before.json
```

With `--warmup` the script additionally issues one short TTS request per
advertised toolchain and records the activation peak. It was run without
warmup here because VoiceFixer's GPU inference currently fails (pre-existing
bug), which would abort its warmup request.
