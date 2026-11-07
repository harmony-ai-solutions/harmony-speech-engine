# Phase 9-3: Chatterbox ARBatchRunner (T3 on the shared pool)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 9 — Chatterbox AR batching
> **Depends on:** 9-1 (the upstream-code seam + cleaned loop), 9-2 (arena), 9-4 (scheduler "unfinished" — the runner decodes across many engine steps)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; design input: study #14 D3/D5/D8; upstream facts `chatterbox-tts==0.1.6` (line numbers from study #14 — re-verify in the installed package)
> **State reference:** handler + factory verified 2026-11-07 — `_execute_chatterbox_tts` mutates `self.model.conds` with the Phase 0-2 baseline restore in `finally` (`model_runner_base.py:1956-2008, 444-457`); `factory.py` role table: TTS own `t3_cfg`, embedding shares TTS's T3 (`:410-419`), multilingual own `t3_mtl23ls_v2` (`:125-137`), turbo own + unshared (`:144`).

## Objective

A token-level `ARBatchRunner` that decodes N Chatterbox requests' speech tokens in one step: one prefill, then one batched decode forward per token, rows dropped on EOS. This is the "mini-vLLM" core — and it **deletes the `model.conds` mutation model** (conditioning becomes prefill content, per-request state is explicit, Phase 0-2's `finally`-restore is superseded on this path).

## Context (self-contained)

- **Prefill (study #14 D3):** conditioning tokens (~34 for the audio prefix) + text tokens + BOS per row. With CFG, duplicate rows (cond/uncond) or the family's cfg-strength gating — mirror upstream's exact semantics from `T3.inference` (re-read it; do not invent). Conditionals come from the Phase 7 embedding cache where applicable — cache hits feed prefill, they do not skip it.
- **Decode:** 1 speech token/row/step; positions via the learned fixed embedding — upstream `get_fixed_embedding` already accepts `[B, T]` indices (study #14; re-verify). Per-row sampler (upstream temperature/top-p semantics) with a **per-row `torch.Generator`** seeded from the request's seed → a row must produce the same tokens whether it decodes alone or in a batch (the reproducibility gate).
- **Skip — multilingual (study #14 D8):** per-request `AlignmentStreamAnalyzer` state + attention hooks + `output_attentions=True` on the shared Llama (`alignment_stream_analyzer.py:32-80`) are hostile to a shared batch. Multilingual keeps the per-item loop and its honest `max_batch_size: 1`; record in `docs/batching.md`.
- **Turbo — committed scope (owner decision 2026-11-07):** Phase 9 is not done without turbo AR-batched. Its T3 deletes `tfmr.wte` and its S3Gen is meanflow (non-AR); the audit is an implementation step, and a blocker found there is a deviation to escalate — not a documented skip. Only the meanflow S3Gen *vocode hop* may fall back to per-row (9-5's call); the AR decode itself is in scope.
- **Fallbacks (the 5-2/8-x pattern):** `len == 1`, `batched_inference: false`, pool refusal (9-2), and per-row hard errors degrade to the sequential loop; a poisoned row errors alone, never de-batches neighbours (8-3 Deviation 6 precedent).
- **Equality gates:** (a) B=1, cache-path vs 9-1's cleaned loop: **bit-identical** tokens + audio; (b) B>1 rows vs their own solo runs with the same seeds: identical tokens (per-row generators make this well-defined); (c) no `model.conds` write on the AR path — assert it stays at its baseline object identity through a full decode.

## Implementation steps

1. Verify-first audit of upstream `T3.inference` internals: exact prefill composition, CFG mechanics, sampler params, EOS token, position embedding API, turbo deltas. Record in this file before writing code.
2. `harmonyspeech/task_handler/ar_runner.py` (new): `ARBatchRunner` driving `t3.tfmr` directly against a 9-2 arena — we own the loop, upstream provides the modules (through the 9-1 seam).
3. Wire `_execute_chatterbox_tts`: type-partitioned (Phase 5-1) → prefill from prepared inputs + conditionals → decode across steps (9-4) → finished rows hand their token stream + mel to the S3Gen hop (9-5 when it lands; until then per-row S3Gen calls).
4. Watermark/loudness post-steps unchanged (they run on final audio).
5. Register the Chatterbox T3 descriptor with the pool (layers/kv-heads/head_dim from the checkpoint config; `cap` from the token-rate derivation).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/ar_runner.py` (new) | the batched prefill/decode runner |
| `harmonyspeech/task_handler/model_runner_base.py` | `_execute_chatterbox_tts` AR path + fallbacks |
| `harmonyspeech/task_handler/batching.py` | supports-entries / family flags |
| `tests/unit/modeling/test_chatterbox_ar_runner.py`, `tests/unit/inference_flow/test_chatterbox_ar_handler.py` (new) | gates below |
| `tests/integration/test_batched_synthesis.py` (extend) | 4-request chatterbox engine test, pairing invariants |

## Tests / validation

- Unit: B=1 bit-identity vs the 9-1 loop; multi-row seeded token-equality vs solo; per-row generator reproducibility; EOS row-drop mid-batch; poisoned row errors alone; conds object identity unchanged; pool-refusal → loop fallback.
- Integration: 4 mixed-length chatterbox requests end-to-end (seeds pinned) → per-request audio correct + paired; abort mid-decode frees the slot (with 9-4).
- GPU gate (`batching:1-vs-N`, benchmark host, real weights): batched vs sequential with identical seeds → identical outputs; throughput rows in `throughput.md` are 9-6's job.

## Checklist

- [ ] Upstream decode audit recorded here (prefill/CFG/sampler/EOS/positions/turbo deltas)
- [ ] ARBatchRunner landed; B=1 bit-identity + seeded multi-row equality green
- [ ] Handler wired with all fallbacks; conds-mutation deletion asserted
- [ ] Multilingual documented in `docs/batching.md`; turbo AR-batched per the committed scope (audit recorded here)
- [ ] `summary.md` Implementation Status updated
