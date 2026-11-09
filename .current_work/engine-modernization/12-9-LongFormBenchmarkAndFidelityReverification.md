# Phase 12-9: Long-Form Benchmark & Whisper Fidelity Re-Verification (Keep Extended Harness, Re-Measure Fixed Engine)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio (exit task after audit fixes)
> **Depends on:** 12-6, 12-7, 12-8; benchmark-host session (GPU 1 discipline)
> **Codebase mapping consulted:** `scripts/bench_throughput.py`, `scripts/audio_regression.py`, `docs/benchmarks/throughput.md`
> **State reference:** verified 2026-11-09 — `scripts/bench_throughput.py` already carries the real-prose *The Villainess Malfoy* fixture (`LONGFORM_FIXTURE_TEXT`, `:1354`) and the `--fidelity` faster-whisper `large-v3-turbo` evaluation gate (`:1980+`). Both must be **kept**, with only the obsolete 150/75 per-model chunk-budget workarounds removed.

## Objective

Preserve the extended Phase 12 benchmark system (`--scenario longform` with `LONGFORM_FIXTURE_TEXT` and `--fidelity` with faster-whisper `large-v3-turbo`), update its expectations to reflect the unified 300-character default budget across all families (4 chunks on the Malfoy fixture), clean up the false-assumption notes in `docs/`, and re-verify text fidelity, join quality, and long-form throughput on the benchmark host once 12-6..12-8 are implemented.

> **Progress 2026-11-09:** code + docs items are complete (first two checklist
> entries ticked), and the host GPU 1 re-verification is now **complete** as
> well. `--fidelity` at the unified 300-character budget (4 chunks) speaks
> **138/138 words for every family when chunked**; every Chatterbox ON row is at/near the
> Whisper noise floor (chatterbox 2, Turbo **4** — was 15–23, Multilingual **4**
> — was 67), and Multilingual is **0 unchunked** (was 140). The long-form
> throughput matrix was re-measured and replaced in
> `docs/benchmarks/throughput.md` §Phase 12. The re-run also uncovered and fixed
> the `default_token_budget` WDDM Shared Memory paging cliff behind the
> 120–402 s `split=off` stalls. **Only the owner listening sign-off remains.**

## Host re-verification results (2026-11-09, GPU 1)

Fidelity — `out/bench/fidelity-12-9.json`, Whisper `large-v3-turbo` CPU int8,
beam 5, unified `chunk_size=300` (4 chunks on the 807-char / 138-word *Villainess
Malfoy* fixture), noise floor ≈2–6 edit distance:

| Family | off edit / words | on (4 chunks) edit / words | Pre-fix / note |
|---|---|---|---|
| chatterbox | 12 / 136 | **2 / 138** | ON was 2 (unchanged); OFF was ~7, now 12 (still drops the final clause) |
| chatterbox_turbo | 108 / 114 | **4 / 138** | ON was **15–23** at 300 chars |
| chatterbox_multilingual | **0 / 138** | **4 / 138** | OFF was **140 / 73**, ON was **67 / 73** |
| melo | 2 / 138 | 2 / 138 | unchanged (1–2) |
| openvoice_v2 | 2 / 138 | 0 / 138 | unchanged (1–2) |
| kitten | 8 / 138 | 6 / 138 | all words; at/just over the ~6 floor |

Throughput — `out/bench/phase12-9-*.json`, 4 chunks/request, AR families +
kitten 4 requests/level at c={1,4,8}, melo/openvoice_v2 8 requests/level at
c={1,2,4,8}:

| Family | off c=1 p50 | on c=1 p50 | off c=8 req/s | off c=8 p50 | on c=8 req/s | on c=8 p50 |
|---|---|---|---|---|---|---|
| chatterbox | 15,203.9 ms | **10,295.6 ms** | 0.13 | 30,040.6 ms | 0.09 | 33,374.4 ms |
| chatterbox_turbo | 6,527.0 ms | 7,027.8 ms | 0.24 | 16,683.0 ms | 0.14 | 21,127.5 ms |
| chatterbox_multilingual | 18,831.2 ms | **10,494.6 ms** | 0.10 | 40,858.5 ms | 0.09 | 33,558.0 ms |
| melo | 1,134.9 ms | **916.8 ms** | **1.35** | 5,890.2 ms | 1.19 | 4,296.0 ms |
| openvoice_v2 | 1,525.1 ms | **1,478.2 ms** | **0.89** | 8,964.6 ms | 0.76 | 8,174.5 ms |
| kitten | 17,856.7 ms | **7,748.2 ms** | 0.10 | 39,314.9 ms | 0.13 | 23,087.3 ms |

Notable pre-fix → post-fix recoveries (same fixture, unified 300 chars):

* **Turbo ON:** c=1 11,720.2 ms (at the removed @150 workaround) → 7,027.8 ms;
  c=4 44,128.4 → 20,873.6 ms; c=8 67,181.2 → 21,127.5 ms.
* **Multilingual ON:** c=1 33,874.3 → 10,494.6 ms (**3.2×**); c=4 163,335.5 →
  36,460.4 ms (**4.5×**); c=8 227,592.3 → 33,558.0 ms (**6.8×**).
* **Melo OFF:** c=4 119,634 ms / 0.04 req/s → 1.16 req/s (3,425.2 ms); c=8
  203,134 ms / 0.04 req/s → **1.35 req/s** (5,890.2 ms).
* **OpenVoice V2 OFF:** c=4 175,012 ms / 0.02 req/s → 0.79 req/s (4,998.0 ms);
  c=8 402,000+ ms watchdog trip → **0.89 req/s** (8,964.6 ms).
* **Kitten ON:** c=1 7,748.2 ms (2.3× faster than OFF's 17,856.7 ms).
* **Chatterbox ON:** c=1 10,295.6 ms (1.48× faster than OFF's 15,203.9 ms).

**Paging cliff (new finding, fixed here).** The pre-fix `split=off` melo /
openvoice_v2 floods were WDDM Shared Memory paging, not compute: one c=8 step
padded an unchunked paragraph against its neighbours and reserved 6.5–9.8 GiB,
crossing the dedicated-VRAM limit and stalling 120–402 s. The fix is a padded-area
`default_token_budget` in `harmonyspeech/task_handler/batching.py` —
`DEFAULT_PHONE_BATCH_BUDGET = 1024` (MeloTTS / OpenVoice V1) and
`DEFAULT_MEL_FRAME_BATCH_BUDGET = 7500` (OpenVoice V2 tone converter) — plus
per-contract `ARBatchRunner.preferred_cap_tokens` of **1152** (English /
Multilingual, Perceiver-resampled) and **1536** (`TurboContract`, 376-token
un-resampled cond). Post-fix `split=off` melo reaches 1.35 req/s at c=8 and
openvoice_v2 0.89 req/s, with reserved VRAM ≤1.8 / ≤2.6 GiB for the padded
forwards.

## What to keep vs what to update in the benchmark harness

1. **KEEP (from 12-4 / 12-5):**
   - `LONGFORM_FIXTURE_TEXT` (the 807-character, 8-sentence, 138-word extract from *The Villainess Malfoy* in `scripts/bench_throughput.py:1354`).
   - `--scenario longform` matrix (`chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`, `melo`, `openvoice_v2`, `kitten` × `{off, on}` × `c={1, 4, 8, 20}`).
   - `--fidelity` mode in `scripts/bench_throughput.py` (renders each family's fixture, saves WAVs to `out/bench/fidelity-12/`, transcribes with faster-whisper `large-v3-turbo` at beam 5, and computes normalized LCS word edit distance and words heard / 138 reference words).
   - `scripts/audio_regression.py` axes `longform:off-vs-on` and `determinism:run1-vs-run2`.

2. **UPDATE (remove the 12-5 false-assumption workarounds):**
   - In `scripts/bench_throughput.py`:
     - Remove `LONGFORM_TURBO_DEFAULT_CHUNK_SIZE = 150`, `LONGFORM_MULTILINGUAL_DEFAULT_CHUNK_SIZE = 75`, and `LONGFORM_DEFAULT_CHUNK_SIZE_BY_FAMILY`.
     - `longform_default_chunk_size(family)` returns `LONGFORM_DEFAULT_CHUNK_SIZE` (`300`) for every family, giving **4 expected chunks/request** on `LONGFORM_FIXTURE_TEXT` across all six families.
     - Update `LONGFORM_CLASS["chatterbox_multilingual"]` from `"AR loop class (control)"` to `"AR decode, admission width 4"` (matching 12-7).
     - Update markdown budget string in `_longform_markdown` from `"server default (300; turbo 150; multilingual 75) chars"` to `"server default (300) chars"`.
   - In `docs/api.md`, `docs/models.md`, `docs/batching.md`, `docs/benchmarks/throughput.md`, and `CHANGELOG.md`:
     - Remove the false claims that `ChatterboxTurboTTS` has a 150-char long-context limit or that `ChatterboxMultilingualTTS` is "budget-sensitive" and requires 75-char chunks.
     - Document the unified 300-character default budget, the two-sided pause crossfade (`CrossfadeStitcher`), and the `ar_decode` batching class for all three Chatterbox TTS families (`chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`).

3. **Benchmark-host re-verification (GPU 1):**
   - **Step 1 — Fidelity gate (`--scenario longform --fidelity --longform-split both --gpu`):**
     Confirm that with `DEFAULT_CHUNK_SIZE = 300` (4 chunks), **all three Chatterbox families (`chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`) plus `melo` and `openvoice_v2` pass within the transcription noise floor (~2–6 edit distance, ~138/138 words spoken)** — proving that fixing `ARBatchRunner`'s `cap_tokens` (12-8) and Multilingual's `repetition_penalty=2.0` + analyzer/SDPA path (12-7) eliminated the word dropping at 300 chars without sub-sentence chunking.
     *Status 2026-11-09: **PASS** for all five, plus `kitten` (6/8, 138/138); see the results table above.*
   - **Step 2 — Listening pass (`out/bench/listening-12/`):**
     Re-render the stitched vs unstitched listening pairs with the 12-6 two-sided pause fade + 12-5 edge conditioning (40 ms fade / 120 ms pause) and 300-char chunks for owner sign-off. *Status 2026-11-09: the fresh post-fix renders are the `out/bench/fidelity-12/*-split-{off,on}.wav` pairs produced by the 12-9 fidelity gate (18:53–18:57Z); owner sign-off still open.*
   - **Step 3 — Long-form throughput matrix (`--scenario longform`):**
     Re-run the `chatterbox`, `chatterbox_turbo`, and `chatterbox_multilingual` ON/OFF rows at `c={1, 4, 8, 20}` (and DEBUG composition rows) and replace the pre-fix tables in `docs/benchmarks/throughput.md` §Phase 12.
     *Status 2026-11-09: re-measured for all six families and recorded; AR families + kitten at `c={1,4,8}`, melo/openvoice_v2 at `c={1,2,4,8}` (c=20 dropped — post-fix admission behaviour is already flat by c=8). DEBUG composition was not re-captured.*

## Files to create/modify

| File | Change |
|---|---|
| `scripts/bench_throughput.py` | Unified 300-char default budget (4 chunks for all families); `chatterbox_multilingual` class updated to `ar_decode` |
| `tests/unit/benchmarks/test_bench_longform_scenarios.py`, `tests/unit/benchmarks/test_bench_fidelity.py` | Update pinned expected chunk counts (4 for all families at 300 chars) and markdown strings; keep all fidelity gate unit tests |
| `docs/api.md`, `docs/models.md`, `docs/batching.md`, `docs/benchmarks/throughput.md`, `CHANGELOG.md` | Remove false 150/75 budget claims; record post-fix fidelity and throughput tables |

## Checklist

- [x] `scripts/bench_throughput.py` and benchmark unit tests updated to unified 300-char budget while keeping `LONGFORM_FIXTURE_TEXT` and `--fidelity` intact
- [x] False 150/75 chunk-budget explanations removed from `docs/api.md`, `docs/models.md`, `docs/benchmarks/throughput.md`, and `CHANGELOG.md`
- [x] Padded-batch WDDM paging cliff fixed (`DEFAULT_PHONE_BATCH_BUDGET = 1024`, `DEFAULT_MEL_FRAME_BATCH_BUDGET = 7500`, per-contract `preferred_cap_tokens = 1152 / 1536`) — new 12-9 finding
- [x] Host Whisper fidelity gate (`--fidelity`) green at `chunk_size=300` for `chatterbox`, `chatterbox_turbo`, `chatterbox_multilingual`, `melo`, `openvoice_v2`, `kitten` (138/138 words; ON edit 0–6)
- [x] Host long-form throughput matrix re-measured for all six families and recorded in `docs/benchmarks/throughput.md`
- [ ] Owner listening pass approved (fresh post-fix pairs in `out/bench/fidelity-12/`); `summary.md` Phase 12 marked complete
