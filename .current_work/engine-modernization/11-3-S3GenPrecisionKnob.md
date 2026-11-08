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
6. **No shipped-config flip in this task.** Exit = knob + tests + docs + artifacts green; the flip (config.gpu.yml chatterbox entries → `s3gen_precision: "fp16"`) is the owner's call after listening.

## Environment facts

- Interpreter `C:\Users\sge20\miniconda3\envs\hse\python.exe`; weights cached; **GPU 1 for any real renders** (`CUDA_VISIBLE_DEVICES='1'`), GPU 0 carries the desktop.
- **Server guardrails (mandatory):** background start + log file + bounded health poll + PID cleanup. Never a foreground server.
- `out/bench/` untracked — reuse, never commit. 4 pre-existing e2e errors in `tests/e2e/tts/test_harmonyspeech.py` are known artifacts.
- Full `tests/unit` + `tests/integration` + `tests/e2e/wire_contract` green at the end. No commits/pushes.

## Checklist

- [ ] `s3gen_precision` knob validated + plumbed to the seam (env override documented)
- [ ] Effective value observable at runtime
- [ ] Honesty/config tests; default fp32 byte-identical
- [ ] Listening-pass render pairs + index under `out/bench/listening-11-3/`
- [ ] Docs (models.md, config docs, CHANGELOG); suites green; `summary.md` updated
