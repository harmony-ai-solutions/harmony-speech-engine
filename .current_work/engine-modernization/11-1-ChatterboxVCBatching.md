# Phase 11-1: Chatterbox VC Batching — Source-Tokenizer Audit + Capacity Flip

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 11 — Chatterbox gap-closing
> **Depends on:** 9-5 (S3Gen batch machinery — landed), 7-2 (embedding cache — landed); nothing else
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; audit precedents 8-2/8-3/8-4 + 9-5 (`docs/batching.md` §audit tables)
> **State reference:** verified 2026-11-08 — `_execute_chatterbox_vc` (`model_runner_base.py:2903-2977`: cache-backed target prep; batched path exists but is dormant at the shipped width), `_try_batched_chatterbox_vc` (`:2988-3064`: resolves per-row target `ref_dict`, then tokenizes sources **one at a time** via `_chatterbox_vc_tokens`, then vocodes rows in length-similar sub-batches through `inference_batched`), `_chatterbox_vc_tokens` (`:2978-2989`: per-row `librosa.load` re-decode/resample to `S3_SR` + one `s3gen.tokenizer(audio_16)` forward per clip); shipped `chatterbox_vc` is `max_batch_size: 1`, `loop` class.

## Objective

Make a multi-request VC step execute with **zero serial model forwards**: batched source-tokenizer passes + the (already landed) batched S3Gen vocode, the shipped capacity flipped from 1 to an honest initial width, and the family moved from `loop` to `padded_batch` in the execution-model class table — gated the Phase 8/9 way.

## Context (self-contained)

- **Everything except one hop already exists and is dormant:** the Phase 7-2 cache serves repeated target voices without compute (`_lookup` at `:2921-2927`); Phase 9-5 audited and batched the S3Gen vocode (`inference_batched`, leak ≤ 2.1e-7 after per-stage masks); `plan_batches` + `batch_token_budget` govern sub-batch formation; the fallback discipline (any prep/batch failure → per-item loop, poisoned row errors alone) is in place. What keeps a VC step serial today: (a) `max_batch_size: 1` admits one request per step, and (b) even at width > 1, the source clips are tokenized **one forward per row** (`_chatterbox_vc_tokens` inside the prepare loop at `:3009-3027`).
- **The only new engineering is a padding audit + `tokenize_batched`:** the S3 tokenizer (installed upstream `chatterbox.models.s3tokenizer`) is a conformer-stack encoder + quantizer over 16 kHz audio — the same module family 9-5 just proved auditable (its sibling `UpsampleConformerEncoder` needed masks after `embed` and the causal lookahead). The audit methodology and `run_masked` machinery carry over directly.
- **Decode hygiene stays OUT of scope (deliberately):** `_chatterbox_vc_tokens` re-encodes the artifact to a WAV buffer and re-decodes/resamples via `librosa` per row (`:2979-2981`). It is a CPU wart, not a model forward — and changing the decode/resample path risks breaking token-level bit-identity with the sequential fallback (`librosa`'s resampler differs numerically from `torchaudio`'s). Record as an optional follow-up that must prove token-level equivalence first; this task keeps `librosa` per row.
- **Perth watermark is NOT in scope:** it stays a per-request post-step (`model_runner_base.py:1123`) and is to be profiled at the benchmark-host session (known cap on the felt win of any output-path batching).
- **Multilingual is NOT this phase** (owner direction 2026-11-08: VC only; the multilingual `AlignmentStreamAnalyzer` decomposition audit is deferred until volume justifies it).
- Seeding: TTS requests carry seeds into per-row generators; VC requests may carry a seed — where present it must seed the row's vocode noise (`S3GenRow(seed=…)`, currently `None` at `:3046`) so batched-vs-sequential equality is well-defined for the gate.

## Implementation steps

1. **Verify-first audit** of `s3gen.tokenizer` on a tiny seeded instance (no checkpoint download; small-geometry real classes, the 9-1/9-5 test precedent): mixed-length rows vs solo runs, stage-by-stage padding verdicts with measured leak numbers, recorded in this file + the `docs/batching.md` audit table. The tokenizer returns `(tokens, extra)` — confirm what `extra` is per-row and whether any stage needs `run_masked` (expect: mask after embed / around any causal stage, like the sibling encoder).
2. **`tokenize_batched`** HSE-side (extend `harmonyspeech/modeling/models/chatterbox/s3gen_batch.py`): padded source-clip batch → per-row token streams, lengths-driven unpad; **B=1 bit-identity** vs the upstream per-row call (identical tokens).
3. **Handler wiring:** in `_try_batched_chatterbox_vc`, collect the per-row decoded 16 kHz clips (keeping the per-row `librosa` decode), then one `tokenize_batched` pass for the step's rows instead of the per-row forward; vocode batching unchanged. Fallbacks unchanged: audit-hostile / batch failure → per-item loop; poisoned row errors alone.
4. **Capacity + class flip:** `chatterbox_vc` → `max_batch_size: 8`, `batched_inference: true` (explicit), `loop` → `padded_batch` in the execution-model class table (knobs: `batched_inference`, `batch_token_budget`); `docs/batching.md` matrix row final (two hops: batched tokenize + batched vocode; cache for targets; watermark serial, measured separately); docs-matrix lint in sync.
5. **Seed plumbing:** thread the request seed (when present) into `S3GenRow` so the batched and sequential paths are comparable; assert a seeded row's audio is identical alone or batched.
6. **Gates:** local — loop-vs-batch equality on seeded inputs (tokens bit-identical; audio ≤ the 9-5 tolerance), poisoned-row isolation, cache-miss path (first-time targets computed correctly inside a batched step), pairing invariants. Host (benchmark session) — GPU `batching:1-vs-N` real-model gate; extend `scripts/audio_regression.py` allow-list; `_tbd_` cells in `docs/benchmarks/throughput.md` §VC flood + `vram-profiles.md` VC peak row.
7. **Exhaustion/flood proportionate to scope:** local integration test — 4+ mixed-length VC requests through the real engine loop (stub/tiny S3Gen) with pairing markers, plus a poisoned-row and cache-mist case; optionally a `--scenario chatterbox-vc` recipe appended to `scripts/bench_throughput.py` for the host session.
8. **Knowledge:** `docs/batching.md`, `docs/benchmarks/{throughput,vram-profiles}.md`, `CHANGELOG.md` (user-visible: concurrent voice conversion), `memory-bank/{progress,activeContext}.md`; tick `summary.md`.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/modeling/models/chatterbox/s3gen_batch.py` | `tokenize_batched` + masks |
| `harmonyspeech/task_handler/model_runner_base.py` | `_try_batched_chatterbox_vc` uses batched tokenize + seed rows |
| `config.yml`, `config.gpu.yml` | `chatterbox_vc` width 8 + `batched_inference: true` |
| `tests/unit/modeling/test_chatterbox_s3gen_batch.py`, `tests/unit/inference_flow/test_chatterbox_vc_batch.py` (extend), integration | gates above |
| `tests/unit/initialization/test_config.py`, `tests/unit/test_batching_docs_matrix.py` | class flip in sync |
| `docs/batching.md`, `docs/benchmarks/{throughput,vram-profiles}.md`, `scripts/bench_throughput.py` (optional) | matrix, `_tbd_` cells, recipe |

## Tests / validation

- Unit: tokenizer audit gates (B=1 bit-identity, mixed-length within measured bounds, poisoned padding); `tokenize_batched` unpad correctness; seeded-row audio equality.
- Handler/integration: loop-vs-batch equality; poisoned row alone; cache-miss inside a batched step; 4-request engine run pairing; fallback on tokenizer failure.
- Full `tests/unit` + `tests/integration` green; host gate + benchmark cells `_tbd_` until the benchmark session.

## Checklist

- [ ] Tokenizer padding audit recorded here + `docs/batching.md` (leak numbers)
- [ ] `tokenize_batched` landed; B=1 bit-identity + mixed-length gates green
- [ ] Handler wired (batched tokenize + seeded rows); fallbacks proven
- [ ] Width 8 + class flip; honesty/docs-matrix tests in sync
- [ ] Exhaustion/pairing integration test green
- [ ] Host gate + benchmark cells landed `_tbd_`; knowledge updated; `summary.md` ticked
