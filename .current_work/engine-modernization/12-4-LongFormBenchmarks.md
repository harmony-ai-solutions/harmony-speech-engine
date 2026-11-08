# Phase 12-4: Long-Form Benchmarks + Gates (single & batched, chunking on/off)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio (exit task)
> **Depends on:** 12-2, 12-3 (the feature under measurement); benchmark-host session (GPU 1 discipline)
> **Codebase mapping consulted:** `scripts/bench_throughput.py` scenario framework (8-6/9-6/10-5 precedent), `scripts/audio_regression.py` (4-2), `docs/benchmarks/throughput.md` structure
> **State reference:** verified 2026-11-08 — bench scenarios carry one-sentence payloads only (`"The quick brown fox…"` at `bench_throughput.py:104-200`); no long-text scenario exists; `_tbd_` cell discipline per Phase 11 precedent.

## Objective

Measure what chunking actually buys (the owner's explicit curiosity): long-form single-request latency and concurrent throughput, chunking **on vs off**, across the execution-model classes — plus a listening gate for stitch artifacts and regression gates for determinism.

## Hypotheses under test (record verdicts, falsifiable)

- **H1 (flood):** chunked long-form floods gain more than single requests — chunk sub-requests fill admission batches (on `ar_decode` chatterbox/turbo a 4-chunk request = 4 rows/step; on `padded_batch` melo/VC-style paths the step batches them).
- **H2 (single request):** concurrent chunk submission (12-2) means even a *single* long-form HTTP request approaches batched-step speed on batched families (chunks co-batch intra-request) — AR families bounded by admission width.
- **H3 (overhead):** stitcher + filter CPU cost is noise vs GPU synthesis (measure crossfade time explicitly at large chunk counts).
- **H4 (filters):** Phase B filters cost < ~5% wall time and change no timing guarantees.

## Implementation steps

1. `scripts/bench_throughput.py` long-form scenarios: a paragraph corpus (~400–1200 chars, ≈4–8 sentences, a fixed seeded fixture per family), `split_text` on/off, families: `chatterbox` (AR, width 4), `chatterbox_turbo`, `chatterbox_multilingual` (loop class — control), `melo-tts` (padded), `openvoice_v2` (clone pipeline), `kitten-tts-mini` (lanes); single-request wall-time + floods at c={1,4,8,20}; record per-chunk step composition from the engine log (8-6 tail discipline).
2. Regression gates: `audio_regression.py` — (a) short-text `split_text: true` byte-identical to off (12-2 gate, re-run at family breadth); (b) seeded chunked determinism (same seed ⇒ same stitched bytes).
3. Listening pass: stitched vs unstitched pairs for 2–3 families rendered under `out/bench/listening-12/` (the 11-3 renderer precedent) — **owner gate, dated decision** on crossfade audibility before the docs claim "seamless".
4. Host session on GPU 1 (server-lifecycle guardrails: background start, log file, bounded health poll, PID cleanup; INFO logging): fill `docs/benchmarks/throughput.md` §Phase 12 (long-form matrix, all cells measured, no `_tbd_` left), VRAM note if chunk floods move peaks (expected: no — chunk sub-requests are ordinary requests).
5. Knowledge: `docs/benchmarks/throughput.md`, CHANGELOG (numbers summary line), `summary.md` tick, verdicts recorded against H1–H4 (a falsified H2 is a finding, not a failure).

## Files to create/modify

| File | Change |
|---|---|
| `scripts/bench_throughput.py` | long-form scenarios + `--print-longform-scenarios` |
| `scripts/audio_regression.py` | short-text-identity + chunked-determinism axes |
| `docs/benchmarks/throughput.md` | §Phase 12 structure then measured cells |
| `.current_work/engine-modernization/12-4-…md` (this file) | H1–H4 verdicts + listening-gate outcome |

## Tests / validation

- Local: full suites green; scenario harness self-check (fixture determinism).
- Host: all §Phase 12 cells measured; listening pass approved by owner.

## Checklist

- [ ] Long-form scenarios landed; local determinism self-check green
- [ ] Regression gates (byte-identity, seeded determinism) green at family breadth
- [ ] Host session measured; §Phase 12 cells filled
- [ ] Listening pass owner-approved (dated); H1–H4 verdicts recorded; `summary.md` ticked
