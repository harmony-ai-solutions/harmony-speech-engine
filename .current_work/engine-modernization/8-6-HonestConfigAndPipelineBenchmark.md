# Phase 8-6: Honest Config, Pipeline Benchmark & Phase Exit

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 8 — Batching across all steps
> **Depends on:** 8-2, 8-3, 8-4, 8-5
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`; benchmark precedents `docs/benchmarks/throughput.md` (5-3, 7-3)

## Objective

Set the newly batched models' `max_batch_size` to measured truths, and prove the phase's headline claim with the **flood scenario**: 20 (and 200) concurrent voice-clone requests moving through a pipeline where **every step batches**. Then close the phase.

## Context (self-contained)

- The motivating scenario (product decision 2026-11-05): "if I get 20 or 200 voice cloning requests in parallel, we already benefit a lot from parallel processing and can move through the pipeline faster." Post 8-2/8-3 the OpenVoice clone pipeline (VAD→embed→synth→TC) batches at every step; Phase 7's cache removes the embed compute entirely for repeated voices.
- Honesty note for the benchmark report: within one engine step, models still execute **sequentially** — the win is per-step batched forwards (fewer kernel launches, better GPU utilization, fewer host round-trips), not cross-step parallelism (that's the separate worker-concurrency workstream). State this in `docs/benchmarks/throughput.md` so nobody misreads the numbers.
- Baseline to beat (5-3, GPU 1): OV2 clone path 1.86 → 1.90 req/s at c=8 (flat — TC/embed loops bound it); MeloTTS synth 14.43 req/s at c=8.

## Implementation steps

### 1. Truthful config values

After gates pass, set measured values (start 8, tune on data) for every newly batched model type in **both** configs: OpenVoice V1/V2 TCEs + TCs, hs1 synthesizer/vocoder, voicefixer restorer/vocoder (+ whisper stays 16 if 8-5 unchanged). Loops that stayed loops keep 1 (hs1-encoder if 8-2 skipped it, silero 16 as cheap CPU concurrency per 8-5 verdict, kitten 4, Chatterbox family 1). Keep the config-honesty test (5-3) in sync with every change.

### 2. The flood benchmark

Extend `scripts/bench_throughput.py`:

- `--concurrency 1,4,20` (and one manual 200 run for the report — HTTP client-side pooling permitting; if the client saturates below 200, record the ceiling honestly).
- Scenario matrix for the OV2 clone pipeline: `{same-voice, distinct-voices} × {cache on, cache off}` (reusing 7-3's `--clone-reference`), plus hs1 TTS and voicefixer restoration curves at the new batch sizes, plus a **KittenTTS lanes curve** (c=8 with lanes vs the old sequential loop — the message-queue behavior proof for a non-batchable model, using the fixed `intra_op` from 8-5).
- Report per-step timings (engine DEBUG already logs sub-batch composition — reuse the 5-3 mechanism) + end-to-end p50/p95/req/s.

Expected shape (record actuals): distinct-voices flood shows the full 8-2+8-3+5-2 stacking (each hop ~1–2 sub-batches instead of N sequential calls); same-voice + cache collapses the embed step (7-3 numbers) on top.

### 3. Documentation & exit

- `docs/benchmarks/throughput.md`: Phase 8 section — all curves, engine commit, GPU, cache state, the sequential-steps honesty note, and the 200-connection observation.
- `docs/batching.md` final matrix cross-checked against final config values (must agree — 8-5's consistency lint).
- Knowledge updates: `memory-bank/progress.md` + `activeContext.md`; `CONCERNS.md` (retire/annotate the "clone path throughput flat" observation with the new numbers); `CHANGELOG.md` (user-visible: higher concurrent throughput + new cache knobs — no internals).
- Full suites green (`tests/unit` ≥ 593 + new, `tests/integration`, `tests/e2e/wire_contract`, real-model e2e for the touched families: openvoice v1/v2, harmonyspeech, voicefixer, whisper STT).
- Tick Phase 8 in [summary.md](summary.md); deviations noted in the 8-* files.

## Files to create/modify

| File | Change |
|---|---|
| `config.gpu.yml`, `config.yml` | truthful values for newly batched models |
| `scripts/bench_throughput.py` | concurrency sweeps + per-step reporting |
| `docs/benchmarks/throughput.md`, `docs/batching.md` | results + final matrix |
| `tests/unit/initialization/test_config.py` (extend) | config-honesty sync |

## Tests / validation

- Config-honesty test updated and green.
- Flood benchmark recorded (all four clone-path combinations + hs1 + voicefixer curves + the 200-connection observation).
- All suites green.

## Checklist

- [x] All newly batched models at measured truthful `max_batch_size`; honesty test in sync — **initial `8` published in both configs and enforced by the policy test; the measurement + tuning pass is deferred to the benchmark host (Deviation 1)**
- [ ] Flood benchmark recorded (20 + 200 concurrency; per-step timings; cache on/off × same/distinct) — **tooling, scenario matrix and `_tbd_` tables landed; number collection is deferred to the benchmark host (Deviation 2)**
- [x] Docs: throughput section + final batching matrix + honesty note
- [x] Knowledge updates (memory-bank, CONCERNS, CHANGELOG) done
- [ ] Full suites green; Phase 8 ticked in summary.md — **phase owner's final run (838-test suite was green at hand-off; 8-6 added 19 test cases)**

## Outcome (2026-11-06)

**Config truth.** Both `config.gpu.yml` and `config.yml` now declare
`max_batch_size: 8` + `batched_inference: true` for the eight newly batched
entries (`hs1-encoder`, `hs1-vocoder`, `ov1/ov2-tone-converter[-encoder]`,
`voicefixer-restorer`, `voicefixer-vocoder`). The loops stay loops with their
documented values: `hs1-synthesizer` 1 (padding-unsafe), Chatterbox family 1
(product decision), whisper 16 (intra-audio CT2 batch; admission width), Silero
16 (sequential queue depth), KittenTTS 4 (lane count). A file-header comment and
the `docs/batching.md` rationale mark `8` as the initial value pending the
benchmark-host tuning run.

**Honesty tests.** `test_shipped_config_batch_sizes_match_executor_reality` was
rewritten as the policy itself rather than a name→value dump: it partitions the
loader registry into `BATCHED_INFERENCE_FAMILIES` (padded forward, >1, flag on),
`SINGLE_REQUEST_MODEL_TYPES` (loop, exactly 1) and
`CONCURRENCY_ONLY_MODEL_TYPES` (admission width / queue depth / lane count) and
asserts each shipped entry against its class. The partition must equal the
registry, so a new model type cannot ship without a capacity decision.
`tests/unit/test_batching_docs_matrix.py` gained a config cross-check: every
shipped entry's `max_batch_size` must equal the value in its matrix row.

**Benchmark tooling.** `scripts/bench_throughput.py`:

* `--concurrency 1,4,20` parses a de-duplicated list (a single integer still
  works); the HTTP client's pool is sized from the highest level (httpx's default
  of 100 would silently cap c=200) with `--http-max-connections` to probe a lower
  client ceiling on purpose.
* Families added: `openvoice_v1`, `harmonyspeech`, `voicefixer`
  (`/v1/audio/convert`) and `kitten`.
* `--embedding-cache {auto,on,off}` verifies the server's `/health`
  `embedding_cache.enabled` against the run's label and aborts on a mismatch —
  the cache toggle is a server-side flag, so the four clone cells span two boots.
  `--print-scenarios` prints the whole two-boot/four-run recipe.
* `--engine-log` tails the engine's DEBUG log around each level and reports the
  per-step sub-batch composition (`embed: N steps, M sub-batches, avg M/N per
  step, rows/requests`) next to p50/p95/req/s, in the console, JSON and markdown.
* Script-unit tests cover concurrency parsing, pool sizing, cache verification,
  the scenario matrix (same/distinct × cache on/off), the recipe and the
  composition parser.

**Docs.** `docs/benchmarks/throughput.md` gained the Phase 8 section: method
(tooling, benchmark host, commit `dab303a`, config table), the four-cell OV2
clone matrix plus OV1, hs1 and VoiceFixer curves, the KittenTTS lanes
comparison (sequential `1` vs lanes `4`), the per-step composition table, the
200-connection observation slot, the sequential-steps honesty note and the
reproduce block. Every measured cell is `_tbd_`. `docs/batching.md`'s matrix now
states `8` (initial) for the batched rows and marks the pre-Phase-8 concurrency
table as such.

**Knowledge.** `memory-bank/progress.md` + `activeContext.md` have Phase 8
sections; `CONCERNS.md` retires the "clone path throughput flat" finding with a
Phase 8 resolution block (and the stale "no real batching" overview bullet);
`CHANGELOG.md` has the user-visible concurrent-cloning/restoration entry.

### Deviations

1. **`max_batch_size: 8` is initial, not measured.** The plan says "set measured
   values (start 8, tune on data)". No measurement host is reachable from this
   checkout and the local HF cache contains only BERT text encoders, so the
   configs carry the documented initial `8` and the tuning pass is deferred to
   the benchmark host. `docs/batching.md` and the config header both say so.
2. **The flood benchmark is tooling + table structure, not numbers.** All four
   clone cells, the hs1/VoiceFixer curves, the kitten lanes comparison and the
   200-connection observation are `_tbd_` in `docs/benchmarks/throughput.md`;
   the reproduce commands and `--print-scenarios` make the run a copy-paste on
   the benchmark host (the Phase 7-3 pattern).
3. **Real-model e2e and the GPU `batching:1-vs-N` gates for the touched families
   are deferred** for the same weights reason (no speech-model weights cached
   locally; downloading multi-GB checkpoints is out of scope). The phase owner's
   full-suite run is likewise not performed here.
4. **`docs/batching.md`'s concurrency table keeps the Phase 5 numbers** (only
   annotated as pre-Phase-8) because replacing them now would mean publishing
   unmeasured values; the Phase 8 section carries the `_tbd_` post-batch tables.

