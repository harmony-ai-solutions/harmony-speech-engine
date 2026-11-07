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

## Verify-first findings (2026-11-07, installed `chatterbox-tts==0.1.6`)

Re-verified in `C:\Users\sge20\miniconda3\envs\hse\Lib\site-packages\chatterbox\models\t3\t3.py`
(note: in this release `chatterbox/models/t3` is a **package**, so the phase
file's `t3.py` path resolves to `t3/t3.py`):

| Finding | Exact location | Verdict |
|---|---|---|
| `output_attentions=True` at prefill | `t3/t3.py:344` | confirmed |
| `output_attentions=True` on **every** decode step | `t3/t3.py:403` | confirmed |
| per-token `tqdm` | `t3/t3.py:352` — `for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True)` | confirmed |
| decode-loop shape | `t3/t3.py:352-408` — Python per-token loop driving `T3HuggingfaceBackend` (`inference/t3_hf_backend.py:73-116`) with an HF `DynamicCache` (`past = output.past_key_values` at `t3/t3.py:349` and `:408`) | confirmed |
| turbo inference path | `t3/t3.py:415-490` (`inference_turbo`): raw `self.tfmr(...)`, a `LogitsProcessorList`, no CFG, no `T3HuggingfaceBackend`, and its **own** `tqdm` at `t3/t3.py:457` — **no** `output_attentions` at all | **structurally different** |

Extra verified fact that shapes the measurement story: with transformers
4.57.1 and the shipped English `attn_implementation="sdpa"`, `output_attentions=True`
is **ignored** (a `UserWarning` from `transformers/utils/generic.py:1006`, and
`attentions is None`), so on the shipped TTS path the flag costs a per-step
warning rather than an attention materialisation; with `eager` (which HSE forces
for multilingual, whose analyzer needs the attention hooks) it really does
materialise attention weights. The equality gate covers **both**.

## Seam decision (2026-11-07)

**HSE-side subclass, chosen.** `harmonyspeech/modeling/models/chatterbox/t3_ar.py`
defines `ChatterboxT3AR(T3)`, overriding only the inference path; `factory.py`
instantiates it for the English TTS role (`_t3_class_for_role`, used by
`load_t3`). No vendored copy: the subclass reaches every internal it needs
(`self.hp`, `self.tfmr`, `self.cfg`, `self.speech_emb`, `self.speech_head`,
`self.compiled`/`self.patched_model`) and reuses upstream's own
`T3HuggingfaceBackend`, `AlignmentStreamAnalyzer` and HF logits warpers, so
there is no upgrade-checklist burden and no duplicated math. Nothing upstream is
monkey-patched (contrast: the process-wide `torch.load` patch at
`chatterbox.py:28-46`, deliberately not extended).

Phase 9 reuse is designed in: the cleaned loop is decomposed into
`ar_prefill` (conditioning + text + BOS composition), `ar_forward` (the
attention-flag decision), `ar_token_embedding` / `ar_cfg_expand` / `ar_step`
(per-step forward), `T3ARSampler` / `ar_sample` (the sampler call) and
`ar_progress` (the loop iterator). 9-3 can override the per-step/sampler pieces
for a batched loop over a KV-cache pool without copying `inference`.

## Deviations (2026-11-07)

1. **Tiny deterministic config.** Upstream `T3.__init__` resolves its Llama
   topology by name from `chatterbox.models.t3.llama_configs.LLAMA_CONFIGS`, so
   the unit fixture registers a test-only `"tiny_ar0"` entry there via
   `monkeypatch.setitem` (removed on teardown) and instantiates the real upstream
   class with a tiny `T3Config`. No checkpoint is loaded and no upstream module
   is patched in production.
2. **"Identical output audio hash" → identical token digest.** The AR stage's
   only output is the speech-token tensor that S3Gen consumes; producing audio
   would require the multi-GB S3Gen checkpoint, and weight downloads are out of
   scope. The gate asserts bit-identical tokens **and** an equal SHA-256 digest
   of that tensor.
3. **Turbo deferred to 9-3.** Its inference path is structurally different
   (`inference_turbo`, `t3/t3.py:415-490`), so grafting the cleaned class there
   would mean duplicating a second loop; the only hygiene issue on that path is
   its own `tqdm` (`t3/t3.py:457`). Recorded, not forced.
4. **Seam scope is the grouped factory path.** The shipped GPU config
   (`config.gpu.yml`) opts into `backbone_group: chatterbox-base`, so its
   Chatterbox TTS role gets the cleaned class. A config without
   `backbone_group` still falls back to `ChatterboxTTSModel.from_pretrained`
   (upstream class); that path is outside the phase's file table and is noted
   for 9-3/9-6 rather than patched here.
5. **Benchmark cells are `_tbd_`.** Single-request RTF and c=4 req/s need the
   benchmark host (no Chatterbox weights locally); the table structure landed in
   `docs/benchmarks/throughput.md` §Phase 9, values deferred to the piggybacked
   Phase 7/8 measurement session.
6. **Pre-existing turbo-branch bug found (not fixed).** `factory.load_t3`'s
   turbo branch calls `T3Config(**_TURBO_HP)`, but upstream `T3Config.__init__`
   only accepts `text_tokens_dict_size`, so that branch raises `TypeError` if
   ever reached. It is unreachable today (`build_role_model` returns `None` for
   turbo and `assemble_role` raises), which is why it has never surfaced. Left
   untouched (turbo is 9-3 scope) but recorded for whoever wires turbo into the
   AR runner.

## Test evidence (2026-11-07)

All runs on the local env (`C:\Users\sge20\miniconda3\envs\hse\python.exe`,
torch 2.9.1+cu130, transformers 4.57.1, CPU):

| Command | Result |
|---|---|
| `python -m pytest tests/unit/modeling/test_chatterbox_ar0.py -q` | **20 passed** |
| `python -m pytest tests/unit/modeling/test_chatterbox_{ar0,dtype,registry,sharing}.py tests/unit/initialization/test_chatterbox_imports.py tests/unit/inference_flow/test_chatterbox_{inputs,no_tempfile,routing,state_isolation}.py tests/integration/test_chatterbox_flow.py -q` | **105 passed** |
| `python -m pytest tests/unit -q` | **994 passed** (974 before + 20 new) |
| `python -m pytest tests/integration -q` | **59 passed** |
| `python -m ruff check <changed python files>` | clean |

The new file covers: the seeded equality gate for `eager` **and** `sdpa`
(10-token sequence, token tensor + SHA-256 digest), EOS-break parity, the
attention-inertness forward check, the no-tqdm assertion plus its upstream
control and the debug env flag, decode-state non-leakage across calls, and the
factory wiring (`_t3_class_for_role`, `load_t3` class selection, assembled TTS
role, multilingual left on upstream). `tests/unit/initialization/test_engine.py`
and `tests/unit/inference_flow/test_chatterbox_state_isolation.py` are inside the
green `tests/unit` run, so the Phase 0-2 baseline-restore invariants still hold
through the factory path.

## Checklist

- [x] Upstream findings re-verified in the installed package (line numbers recorded above, 2026-11-07)
- [x] Seam implemented (HSE-side subclass `ChatterboxT3AR`; rationale above — no vendored copy needed)
- [x] Seeded bit-equality gate green (tokens + digest, `eager` and `sdpa`, 20 unit tests)
- [x] Factory wiring for the TTS role; import/state-isolation suites green (unit 994 / integration 59)
- [ ] AR-0 before/after numbers recorded in `throughput.md` (benchmark-host session) — **table structure landed (§Phase 9 AR-0); all cells `_tbd_`, run deferred to the piggybacked Phase 7/8 measurement session**
- [x] `summary.md` Implementation Status updated
