# Phase 11-0: Benchmark-Findings Fixes + Multi-Device Placement

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 11 — fixes + Chatterbox gap-closing
> **Depends on:** 9-x, 10-x (landed); blocks 11-1 by owner decision (fixes first)
> **Codebase mapping consulted:** the host measurement run 2026-11-08 (commit `4d192ab`) — findings recorded in `docs/benchmarks/throughput.md` §open findings; owner decisions 2026-11-08 (fix all; multi-device placement must work; whisper flip approved)

## Objective

Repair everything the first real host run exposed so the shipped build actually delivers the measured Phase 9/10 wins, and make explicit secondary-device placement (`cuda:1`, `cuda:2`) work end-to-end — pool, runner, probe, metrics.

## Context (self-contained)

Findings (engine was frozen during the run; none fixed):

1. **Critical — AR pool device mismatch:** `ARCachePool` is created with device `cuda` while models sit on indexed devices (`cuda:0`); `ARBatchRunner.register()` treats the mismatch as "arena on another device" and **disables the runner on every CUDA model**. Phase 9 batching never engages in the shipped build (`pool_bytes 0`, `grants 0`). The +24–31 % / +15–27 % numbers were measured through a serve fixture with the device pinned — real, but gated on this fix.
2. **High — `chatterbox_embedding.max_batch_size: 1`:** clone floods serialize at the embed admission step, so the TTS step never receives >1 request and the AR path stays dark even with (1) fixed. Measurement: width 4 on the embedder engaged the runner (grants 6, tokens/step 2.9). **Policy note:** the honesty table's `loop` class asserts width 1; the embedder executes serially regardless — re-class it honestly (e.g. serial-execution-with-admission-depth semantics, or a documented `loop+depth` variant) with a dated decision rather than silently violating the policy test.
3. **High — whisper startup probe failure:** the probe's synthetic/zero audio yields no VAD clip timestamps (`RuntimeError: No clip timestamps found…`) → `per_window_bytes = 0` → the runtime budget governor is inert (a 1 GiB budget overshot to 1.95 GiB unchecked). Fix the probe (speech-like probe audio or direct clip-timestamp synthesis in probe mode), revive `allowed_windows`, and validate the runtime trip with real per-window bytes.
4. **Medium — finished-row re-sample:** `ARBatchRunner.step()` re-samples a row already marked `finished` when passed again; the engine's pattern (pop finished rows per step) masks it, but the seeded parity harness goes red. Guard it + unit test; harness must go green without loosening.
5. **Medium — window cap has no config key** (module constant 8): add the per-model knob (default 8, owner-approved initial), plumb to the batcher.
6. **Low — DEBUG log line-wrap** breaks `--engine-log` parsing (80-column wrap): disable wrap for the engine-log file handler.

**Owner requirement (2026-11-08):** secondary-device placement (`device_config: cuda:1` / `cuda:2`) must work when specified — the (1) fix is not "normalize to `cuda:0`" but "resolve and honor the requesting model's actual device throughout": pool/arena allocation on the model's device, per-device budget accounting within `kv_cache_budget_bytes` when models live on different devices, probe on the model's device, metrics labeled accordingly.

**Owner decision (2026-11-08):** flip whisper `batched_inference: true` in the shipped configs (parity substantively passed) and widen the parity harness tolerance to 1e-2 with `language_probability` marked informational (moves 0.2–0.8 pp between paths).

## Implementation steps

1. Device normalization utility (canonical indexed device resolution; `cuda` ≡ the active device only at the single-device default) + `ARCachePool` per-device arenas keyed by resolved device; runner registration compares resolved devices; budget split/validation across devices; boot arena table gains a device column.
2. `chatterbox_embedding` admission depth 4 + honesty-class re-classification (dated decision recorded; policy test updated; docs matrix row).
3. Whisper probe: real per-window bytes from a working probe; runtime trip validated (mocked + real spot); `vram-profiles.md` cells updated from the fixed probe.
4. Finished-row guard in `ARBatchRunner.step()` + regression unit test; parity harnesses must pass green.
5. `whisper_window_batch_size` ModelConfig key (default 8) plumbed to the window batcher; docs.
6. Whisper `batched_inference: true` in `config.yml` + `config.gpu.yml` + harness tolerance/informational change (dated decision reference).
7. Log wrap fix for `--engine-log`.
8. Multi-device verification: unit tests with device resolution (CPU + mocked CUDA indices); **real** `cuda:1` placement test on this host (GPU 1 is the idle card; GPU 0 carries the desktop — keep heavy runs on 1, light checks on 0) covering: model load on `cuda:1`, arena on `cuda:1`, runner engagement (grants > 0), whisper probe + workers on `cuda:1`, and a two-model two-device smoke (tiny model on each).
9. Re-run the affected gates on this host (weights are cached now): both parity harnesses green, `batching:1-vs-N` AR families, a three-way spot check proving the **shipped build** (no fixtures) engages the runner (grants > 0, tokens/step > 1 at c≥4), budget-trip with real per-window bytes. Update the throughput/vram docs' findings table (fixed ✓ annotations), not the headline numbers.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/engine/ar_cache.py` | device resolution + per-device arenas/budget |
| `harmonyspeech/task_handler/ar_runner.py` | finished-row guard; registration device check |
| `harmonyspeech/modeling/whisper_probe.py` | working probe audio/clip synthesis; device |
| `harmonyspeech/common/config.py` (+ configs) | `whisper_window_batch_size`; embedding depth; whisper flip |
| `harmonyspeech/task_handler/whisper_batching.py` | window-cap knob |
| engine logging setup | `--engine-log` wrap fix |
| honesty/docs tests + new device tests | class re-class + multi-device |
| `docs/{batching,models}.md`, `docs/benchmarks/*.md` | findings annotated fixed; knobs documented |

## Tests / validation

- Unit: device normalization (incl. `cuda`→`cuda:0`, explicit `cuda:1`); per-device budget validation; finished-row guard; window-cap plumbing; probe math with synthesized clips.
- Integration: `cuda:1` end-to-end placement (model + arena + runner grants); two-device smoke; clone flood at c=4 through the shipped config engages the AR runner.
- Host gates re-run green; full `tests/unit` + `tests/integration` + wire-contract suites green.

## Checklist

- [ ] Pool device fix + per-device arenas; `cuda:1` placement proven end-to-end
- [ ] Embedding admission depth + honest class re-classification
- [ ] Whisper probe fixed; runtime budget governor live + validated
- [ ] Finished-row guard; both parity harnesses green
- [ ] Window-cap config key; whisper `batched_inference` flipped (dated)
- [ ] Log-wrap fix; findings table annotated; suites green; `summary.md` ticked
