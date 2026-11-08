# Phase 11-3: `s3gen_precision` Config Knob (fp16 vocode promotion)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 11
> **Depends on:** 11-2 (experiment: fp16 autocast gate 7/7 PASS, VC c=8 1.51×; bf16 rejected)
> **Owner decision 2026-11-08:** promote now. **The shipped-config flip is gated on the owner's listening pass** — this task delivers the knob + artifacts; the one-line flip is a follow-up commit after approval.

## Objective

Promote fp16 S3Gen autocast from the 11-2 env gate (`HSE_S3GEN_AUTOCAST`) to a first-class, honest config knob — **without** flipping shipped configs yet — and produce the seeded fp32-vs-fp16 render pairs for the owner's listening pass.

## Steps

1. **Config knob:** `s3gen_precision: "fp32" | "fp16"` on ModelConfig (default `"fp32"` — zero behavior change), validated like other dtype-ish fields; plumbed through the loader/factory to the 11-2 seam (`harmonyspeech/modeling/models/chatterbox/s3gen_precision.py`). Precedence: `HSE_S3GEN_AUTOCAST` env var **overrides** config (kept for bench/testing; document it). Applies to the Chatterbox families whose vocode uses S3Gen (en, multilingual, VC, turbo meanflow site); per-model, so mixed deployments stay honest.
2. **Observability:** surface the effective value (e.g. `/health` model block or a `hse_s3gen_autocast_enabled`-style metric) so a running deployment can prove which precision its vocode uses.
3. **Honesty/config tests:** knob validation; default `fp32` byte-identical (existing bit-identity/poison gates keep passing); `fp16` path exercises the seam (mock/monkeypatched on CPU); docs-matrix/config-honesty table updated if it enumerates precision.
4. **Listening-pass artifacts:** a small render mode (extend `scripts/bench_s3gen_autocast.py` or a sibling) that writes seeded pairs `fp32.wav` / `fp16.wav` per family (chatterbox bundled voice, clone voice, VC speech→B, VC speech→synth, multilingual en + de-accent, turbo bundled) into `out/bench/listening-11-3/` (untracked) + a one-screen index (README.md) listing case → files → the 11-2 gate numbers. ~7–8 pairs, ~3–4 s each.
5. **Docs:** `docs/models.md` §Precision — fp16 autocast becomes an approved config option (reference 11-2 numbers; bf16 stays documented-rejected); config-field documentation (`docs/` wherever `dtype` is documented); CHANGELOG entry.
6. **No shipped-config flip in this task.** Exit = knob + tests + docs + artifacts green; the flip (config.gpu.yml chatterbox entries → `s3gen_precision: "float16"`) is the owner's call after listening.

## Environment facts

- Interpreter `C:\Users\sge20\miniconda3\envs\hse\python.exe`; weights cached; **GPU 1 for any real renders** (`CUDA_VISIBLE_DEVICES='1'`), GPU 0 carries the desktop.
- **Server guardrails (mandatory):** background start + log file + bounded health poll + PID cleanup. Never a foreground server.
- `out/bench/` untracked — reuse, never commit. 4 pre-existing e2e errors in `tests/e2e/tts/test_harmonyspeech.py` are known artifacts.
- Full `tests/unit` + `tests/integration` + `tests/e2e/wire_contract` green at the end. No commits/pushes.

## Checklist

- [x] `s3gen_precision` knob validated + plumbed to the seam (env override documented)
- [x] Effective value observable at runtime
- [x] Honesty/config tests; default fp32 byte-identical
- [x] Listening-pass render pairs + index under `out/bench/listening-11-3/`
- [x] Docs (models.md, config docs, CHANGELOG); suites green; `summary.md` updated

## Result (2026-11-08)

**Knob shipped, no config flipped.** `ModelConfig.s3gen_precision` is
`"fp32"` (default, byte-identical) | `"fp16"`, validated strictly (case/space
tolerant, no dtype aliases). It is stored on the loaded S3Gen seam
(`_hse_s3gen_precision`) by `apply_model_dtype` → `install_inference_autocast`,
so the solo (`s3gen.inference`), batched (`_flow_batched`) and AR per-row
(`_vocode_ar_row`) vocode paths all resolve the same per-model value. The
`HSE_S3GEN_AUTOCAST` env var **overrides** the config when present (including an
explicit `off`); absent env falls back to the config.

**Observability:** `/health` `models.<name>.s3gen_precision` reports the
effective value for the four S3Gen-vocode families (TTS, Turbo, multilingual,
VC), preferring the loaded S3Gen's seam value so a shared-backbone group reports
what actually runs. `ChatterboxEmbedding` has no field.

**Listening pass:** seven seeded pairs in `out/bench/listening-11-3/`
(`<case>_fp32.wav` / `<case>_fp16.wav`, 3.0–4.4 s each) + `README.md` index +
`report.json`. Live log-mel MSE 0.012–0.127 dB², corr ≥ 0.9914, RMS Δ ≤ 0.09 %
— inside the 1.0 dB² / 0.98 / 5 % audibility bounds. Rendered on GPU 1
(RTX 5060 Ti) with the fp16 side driven by `HSE_S3GEN_AUTOCAST=float16` (no
config flip).

**Tests:** unit 1182 passed; integration 84 passed + 2 opt-in skips;
wire-contract e2e 50 passed + 3 model-gated skips. New tests: knob
default/validation + shipped-config no-flip; config-vs-env precedence;
config-driven seam engagement on CPU; batched precision threading; `/health`
field; listening case set + README renderer.

## Deviations (2026-11-08)

1. **Per-model precision lives on the S3Gen instance, not a process global.**
   `install_inference_autocast(s3gen, precision=...)` stores
   `_hse_s3gen_precision`; `_flow_batched`/`_vocode_ar_row` read it. A
   shared-backbone group has one S3Gen, so the last `apply_model_dtype` wins —
   `/health` prefers the seam value for that reason. Mixed *ungrouped*
   deployments are fully per-model as specified.
2. **The health field prefers the loaded S3Gen's seam value** (falling back to
   the config value when no runner/model is reachable, as in the shell-engine
   tests), so it cannot claim a precision the vocode is not running.
3. **The listening mode is a new `--mode listening` on
   `scripts/bench_s3gen_autocast.py`** rather than a sibling script; it reuses
   the same harness metrics. The fp16 side uses the env override so no shipped
   config changes in this task (the `config.gpu.yml` flip remains the owner's
   follow-up).
4. **`docs/models.md` §Precision was rewritten** from "opt-in experiment" to
   the approved `s3gen_precision` table (bf16 still documented-rejected), and
   the mixed-device example config shows the default explicitly. No shipped
   `config.yml`/`config.gpu.yml` change.
5. **No commits/pushes** (task constraint). `out/bench/` (including the
   listening pairs and `report.json`) is untracked.


## Follow-up: shipped flip (2026-11-08)

Owner listened to the seven seeded pairs (`out/bench/listening-11-3/`) and approved: "everything sounds fine". The flip commit sets `s3gen_precision: "float16"` on the four S3Gen-vocode families in `config.gpu.yml` (embedding untouched; CPU config untouched) and re-pins the shipped-state honesty test (`test_shipped_s3gen_precision_matches_owner_flip`). Same commit aligns the knob's value vocabulary with `dtype` — `"fp32"/"fp16"` → `"float32"/"float16"` (owner review: identical meaning must use identical identifier); the old spellings remain tolerated aliases in the runtime parser only. `/health` therefore reports `float16`/`float32`.
