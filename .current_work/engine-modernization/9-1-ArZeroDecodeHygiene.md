# Phase 9-1: AR-0 — Chatterbox T3 Decode-Loop Hygiene + Baseline

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** nothing (independent; land first — it establishes the upstream-code seam every later 9-x task uses)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; design input: study #14 D6 (`../../soulbits-cloud-backend/.current_work/inference-engine-re-eval-26-11/14-vllm-whisper-kvcache-and-hse-ar-batching.md` §2.11)
> **State reference:** verified 2026-11-07 — the Chatterbox tree in HSE is wrappers only (`harmonyspeech/modeling/models/chatterbox/{chatterbox.py,factory.py}`); `T3` is imported from the **installed upstream package** `chatterbox-tts==0.1.6` (`pip install --no-deps`, see CONCERNS §"Chatterbox TTS Dependency Pinning"). All `t3.py` / `s3gen.py` line numbers below refer to upstream 0.1.6 as introspected for study #14 — **re-verify in the installed package before editing** (workspace rule: re-read the target before editing).

## Objective

Remove pure-loss work from the T3 autoregressive decode loop before any architectural change, and measure it. Study #14 found that upstream `T3.inference()` passes `output_attentions=True` to the Llama forward on **every decode step** even when no analyzer runs (English path), and calls `tqdm` once per token — a per-token eager-attention materialization plus a progress-bar hook in the hottest loop of the engine's slowest family (0.6 req/s flat at c=8, Phase 5-3).

## Context (self-contained)

- Upstream decode loop (`chatterbox/models/t3.py:352-408`): Python per-token loop over `t3.tfmr` (HF Llama) with an HF `DynamicCache`; `output_attentions=True` is passed at `t3.py:344` (prefill) and `t3.py:403` (decode); a `tqdm` iterator wraps the loop (`t3.py:279-288` area). Removing attention-output materialization is **numerically inert** (it only returns extra tensors nobody reads on this path) — outputs must stay bit-identical, which makes this a rare zero-risk Chatterbox change with a free equality gate.
- **The seam decision (executed here, reused by 9-3/9-5):** we cannot edit site-packages. Choose and implement one:
  1. **HSE-side subclass** (recommended): `harmonyspeech/modeling/models/chatterbox/t3_ar.py` defines `ChatterboxT3AR(T3)` overriding only the inference path, assembled by `factory.py`'s `assemble_role`/`build_role_model` in place of the plain class. No monkey-patching of the upstream module (contrast: the existing process-wide `torch.load` patch in `chatterbox.py:28-46`, which we do not extend).
  2. Full vendored copy of `t3.py` into HSE's tree — heavier (upgrade-checklist burden per CONCERNS), only if the subclass can't reach the needed internals.
- Turbo note: the turbo wrapper hardcodes its own T3 hyper-parameters and deletes `t3.tfmr.wte` (`factory.py:144-158`); 9-1 applies to the TTS role first, turbo after a verify-first check that its inference path shares the same loop.

## Implementation steps

1. **Verify-first:** in the installed package (conda env / CI), confirm the three findings (`output_attentions` at both call sites, per-token `tqdm`, loop shape) and capture the exact upstream line numbers for the phase-file deviation notes.
2. Implement the seam (subclass preferred) with the cleaned loop: no `output_attentions` when no analyzer is attached, no `tqdm` (keep an opt-in debug flag behind an env var if wanted, default off).
3. Graft via `factory.py` for the `chatterbox` (TTS) role; embedding role is unaffected (it never decodes); multilingual/turbo untouched in this task beyond the verify-first note.
4. Equality gate (CPU, tiny deterministic config): cleaned path vs upstream `T3.inference` with fixed seed, temperature/top-p pinned → **identical sampled tokens and identical output audio hash**. If not identical, stop and re-audit — the change was supposed to be numerically inert.
5. Measurement: single-request RTF and c=4 req/s, before/after, on the benchmark host — **piggyback on the deferred Phase 7/8 measurement session** (one session, three sets of numbers). Record in `docs/benchmarks/throughput.md` as the Phase 9 baseline (§Phase 9, "AR-0").
6. Update `docs/batching.md` only if the loop-cost fact is worth a line; CHANGELOG only if user-visible latency/throughput moves measurably.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/t3_ar.py` (new) | cleaned inference path (the 9-x seam) |
| `harmonyspeech/modeling/models/chatterbox/factory.py` | assemble TTS role with the cleaned class |
| `tests/unit/modeling/test_chatterbox_ar0.py` (new) | bit-equality gate, no-tqdm assertion, factory wiring |
| `docs/benchmarks/throughput.md` | §Phase 9 AR-0 baseline rows (benchmark host) |

## Tests / validation

- Unit: seeded bit-equality (tokens + audio hash) cleaned vs upstream on CPU; `capsys`-style assertion that no tqdm output is produced; Phase 0-2 baseline-restore invariants still hold through the factory path (`tests/unit/initialization/test_engine.py` + chatterbox import suite stay green).
- Benchmark host: before/after RTF + req/s recorded; no behavior change claims beyond speed.

## Checklist

- [ ] Upstream findings re-verified in the installed package (line numbers recorded here)
- [ ] Seam implemented (subclass vs vendored copy — record the choice and why)
- [ ] Seeded bit-equality gate green
- [ ] Factory wiring for the TTS role; import/state-isolation suites green
- [ ] AR-0 before/after numbers recorded in `throughput.md` (benchmark-host session)
- [ ] `summary.md` Implementation Status updated
