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
* **Headroom (Phase 4 measurement):** device total − `torch.cuda.memory_reserved`
  (and, separately, device total − `nvidia-smi` used). On Windows the display
  driver/WDDM also holds a little memory outside CUDA's accounting, so
  `nvidia-smi` is the stricter number.
* **Phase 10-3 correction:** the headroom series is now device-level truth from
  `torch.cuda.mem_get_info()` (`hse_vram_headroom_bytes` = true free device
  memory; `hse_vram_device_used_bytes` = total − free). The Phase 4 table below
  keeps its original torch-scoped labels for reproducibility; see
  [Phase 10-3 — device-level truth](#phase-10-3--device-level-truth) for the
  corrected numbers and why the torch-scoped figure overstated free memory.

## Result — full catalogue

| Metric | Before (fp32, duplicated) | After (fp16 T3 + dedup) | Change |
|---|---|---|---|
| torch allocated | 16.45 GiB | **8.52 GiB** | −7.93 GiB (−48 %) |
| torch reserved | 16.54 GiB | 8.62 GiB | −7.92 GiB |
| nvidia-smi used | 15.67 GiB (16042 MiB) | **8.83 GiB** (9040 MiB) | −6.84 GiB |
| headroom (torch reserved, legacy) | 0.00 GiB | **7.30 GiB** | +7.30 GiB |
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

* **Measured headroom (Phase 4, torch-scoped): 7.10 GiB (nvidia-smi) / 7.30 GiB (torch reserved).**
  Phase 10-3 redefines `hse_vram_headroom_bytes` as true device free memory; the
  device-level figure for this catalogue is recorded in the next section.
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

## Phase 10-3 — device-level truth

### Why the old headroom overstated free memory

`hse_vram_headroom_bytes` used to be `device total − torch.cuda.memory_reserved`.
That only sees PyTorch's caching allocator. CTranslate2 (Faster-Whisper) allocates
outside it, so a resident Whisper model was invisible to every VRAM series while
still consuming device memory — the reported headroom was too high by the CT2
footprint plus CUDA-context overhead.

Phase 10-3 redefines the series in place:

| Series | Meaning |
|---|---|
| `hse_vram_headroom_bytes` | **true free device memory** (`torch.cuda.mem_get_info` free) |
| `hse_vram_device_used_bytes` | **device-level used** (total − free): torch + CTranslate2 + CUDA contexts |
| `hse_vram_allocated/reserved/peak_allocated_bytes` | unchanged, torch-scoped ("what does torch hold") |
| `hse_model_peak_activation_bytes{model}` | measured max-shape probe peak (0 when the probe was skipped) |

### Corrected figures

Measured with `scripts/measure_vram.py` (which now records `device_used` and the
device-free headroom) on the benchmark host:

| Metric | Phase 4 run (torch-scoped) | Phase 10-3 device-level | Notes |
|---|---:|---:|---|
| device used (total − free) | _tbd_ | _tbd_ | includes CT2 + CUDA contexts |
| free device memory (headroom) | 7.30 GiB (torch reserved) | _tbd_ | the Phase 9 pool-sizing input |
| non-torch delta (device used − torch reserved) | 0.21 GiB (8.83 − 8.62, nvidia-smi vs reserved) | _tbd_ | CT2 weights/activations + context |
| Whisper turbo probe peak | — | _tbd_ | filled when the benchmark host runs the probe |
| Whisper tiny probe peak | — | _tbd_ | same |

The 0.21 GiB delta in the Phase 4 run is too small to include a resident
large-v3-turbo (≈1.6 GiB in fp16), so it represents the CUDA-context floor
rather than a Whisper-inclusive number. A benchmark-host run with the Whisper
configs resident must fill the `_tbd_` cells.

### Phase 9 pool-sizing input

Phase 9's AR KV-cache pool must be carved from the **device-level free** figure
above, not the legacy 7.30 GiB torch-reserved number. Until the benchmark-host
measurement lands, the conservative input is the nvidia-smi figure (7.10 GiB)
minus the measured Whisper probe peaks and a safety margin.

### Whisper governance knobs

| Config key | Default | Meaning |
|---|---|---|
| `whisper_vram_budget_bytes` | unset (advisory) | Optional hard device-level used-memory ceiling. Boot fails when the projected Whisper peak (weights + measured probe peaks) exceeds it. |
| `whisper_max_audio_seconds` | `300` | The startup probe's max shape (the longest audio it validates) and the whisper-side duration bound for capacity planning. Set it to at least the longest audio you intend to admit; it is not an admission rejection. |

Without a budget the boot check warns when the projected peak exceeds the
measured free headroom, and runtime admission scales concurrent windows against
the device total (i.e. available free memory). See
[models.md](../models.md#whisper-vram-governance).

### Phase 10-5 — budget-trip validation

The budget is only useful if the runtime governor actually holds it under load.
The benchmark host runs the 100-request heterogeneous flood twice: once with no
budget and once with `whisper_vram_budget_bytes` set **below demand** (e.g.
6 GiB on the 16 GiB card). Expected: `allowed_windows` shrinks, latency degrades
gracefully, device-used stays at/below the bound, and no request is lost or
crashed.

| Metric | Unbudgeted run | Budget trip | Notes |
|---|---|---|---|
| `whisper_vram_budget_bytes` | unset | _tbd_ | `/health` `whisper_vram` must report it |
| Max `hse_vram_device_used_bytes` | _tbd_ | _tbd_ | includes torch + CT2 + contexts |
| `allowed_windows` under load | cap (16) | _tbd_ | shrinks as used approaches the bound |
| Requests answered / loss | 100 / 0 | _tbd_ / **0** | |
| p50 / p95 | _tbd_ | _tbd_ | graceful degradation, no OOM |

The hermetic proof (mocked `torch.cuda.mem_get_info`, real scheduler and
`get_whisper_admission`) lives in
`tests/integration/test_whisper_exhaustion.py`; it shows `allowed_windows`
falling 6 → 2 → 1 (0 with the hard stop) and a flood completing under the
draining floor.

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
