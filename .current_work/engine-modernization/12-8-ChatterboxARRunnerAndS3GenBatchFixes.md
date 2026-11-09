# Phase 12-8: Chatterbox `ARBatchRunner` & S3Gen Batching Fixes (KV-Slot Sizing, Deferred Batched Vocode, Per-Token Overhead)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio & Chatterbox batching (audit fix)
> **Depends on:** 9-2, 9-3, 9-4, 9-5, 12-2, 12-7
> **Codebase mapping consulted:** `.planning/codebase/{ARCHITECTURE,CONVENTIONS,TESTING}.md`
> **State reference:** verified 2026-11-09 — `harmonyspeech/engine/ar_cache.py:100-104`, `harmonyspeech/task_handler/ar_runner.py:517-549,692-728,752-839`, `harmonyspeech/task_handler/model_runner_base.py:860-1095,1233-1264`, `harmonyspeech/task_handler/inputs.py:765,812,877`, `harmonyspeech/processing/scheduler.py:513-544`, `harmonyspeech/endpoints/openai/serving_text_to_speech.py:72-100`, `config.gpu.yml:15-18`.

## Objective

Fix the four architectural and numerical bugs in the Chatterbox `ARBatchRunner` + `S3Gen` batching pipeline discovered during the Phase 12 audit:

1. **Silent mid-sentence KV-slot truncation (`cap_tokens = 768`):** size arena slots so `ChatterboxTurboTTS` (whose un-resampled conditioning prompt is up to 376 tokens!) and `ChatterboxTTS` / `ChatterboxMultilingualTTS` never truncate below upstream's 1000-speech-token generation budget, and remove the `DEFAULT_CHUNK_SIZE_TURBO = 150` workaround.
2. **S3Gen padded batching (`_finalize_ar_rows`) bypassed in practice:** stop vocoding individual rows mid-decode on the exact single token step they hit EOS (where `len(finished) == 1` almost always, stalling active T3 rows and bypassing `inference_batched`); hold finished rows until the step's decode burst / active batch completes so finished chunks vocode **together** in one padded `inference_batched` call.
3. **Per-token engine-step & `prepare_inputs` `ThreadPoolExecutor` churn:** decode a burst of tokens per engine step (instead of 1 token per `asyncio` round-trip) and skip `prepare_inputs` (and its per-call `ThreadPoolExecutor` creation) for continuing AR requests.
4. **Missing `exaggeration` update on the AR path & slot-vs-request unit mismatch in `Scheduler._ar_admission_width`.**

## False assumptions from earlier phases — DO NOT repeat

1. **False assumption (Phase 9-2/9-3 `ar_cache.py:100`, `9-3-ChatterboxARRunner.md:14,156`, Phase 12-5 `serving_text_to_speech.py:72-80`):** *"`default_cap_tokens: 768` gives ~30 s of speech at 25 Hz for every Chatterbox model because prefill conditioning is ~34 tokens; therefore Chatterbox Turbo dropping 15–20 words at `chunk_size=300` (while speaking the same sentences perfectly SOLO) is a long-context failure of the Turbo model requiring `DEFAULT_CHUNK_SIZE_TURBO = 150`."*
   - **Reality:**
     - **Solo (`len(inputs) == 1`) and chunked (`len(inputs) > 1`) did not run the same code path!** In `_execute_chatterbox_tts_ar` (`model_runner_base.py:928-929`), `if len(inputs) == 1 and new_indices: return sequential(...)` bypassed `ARBatchRunner` completely for solo sentences and ran `T3.inference_turbo` (`max_gen_len = 1000` new speech tokens *after* prefill).
     - When `split_text=True` submitted 4 chunks concurrently (`len(inputs) == 4`), `ARBatchRunner` ran.
     - Unlike English/Multilingual `ChatterboxTTS` (which sets `use_perceiver_resampler = True` and compresses 150 prompt tokens down to 32 tokens via `Perceiver`, `len_cond = 34`), `ChatterboxTurboTTS` sets `use_perceiver_resampler = False` and `speech_cond_prompt_len = 375` (`tts_turbo.py:153-154`, `factory.py:181-182`). Its conditioning prefix in `T3CondEnc` is **up to 376 raw tokens**!
     - With ~70 text tokens for a ~280-char chunk, Turbo's prefill `length` is **~445 tokens** — consuming nearly 60 % of the 768-token slot before decoding begins.
     - `ARBatchRunner._prefill` (`ar_runner.py:727`) sets `row.max_new_tokens = min(1001, self.arena.cap_tokens - length)`, clamping Turbo to **`768 - 445 = 323` speech tokens (~12.9 s)**. When a 280–300 char chunk (~45 words) needed ~375–450 speech tokens (~15–18 s), `ARBatchRunner.step()` (`ar_runner.py:820-824`) hit `row.length >= self.arena.cap_tokens` and **silently truncated the chunk mid-sentence (`finish_reason = "length"`)**, cutting off the last 5–7 words of every chunk (~15–23 words across the 4-chunk paragraph)!
     - For the same reason, `split=off` benchmarks at `c >= 4` (`docs/benchmarks/throughput.md:2197`) truncated 807-char Turbo requests after `768 - 556 = 212` speech tokens (~8.5 s), making `split=off` look artificially fast.

2. **False assumption (Phase 9-5 `model_runner_base.py:1017-1018`):** *"Calling `self._finalize_ar_rows(finished, ...)` at the end of each 1-token `_execute_chatterbox_tts_ar` step batches S3Gen vocoding across the batch's rows."*
   - **Reality:** In `_execute_chatterbox_tts_ar`, `runner.step(active)` advances active rows by **1 speech token** and collects only the rows that hit EOS on **that exact token step $t$** into `finished`. Because different text chunks (or different concurrent requests) almost never emit the exact same number of speech tokens down to the single token (1/25 s), `len(finished)` on any given token step $t$ is almost always `1`!
   - When `len(finished) == 1`, `_finalize_ar_rows` (`model_runner_base.py:1066-1068`) takes the `if len(group_rows) == 1: self._finish_ar_row(...)` branch — **completely bypassing `inference_batched`** and running a solo ~150–350 ms S3Gen vocode **synchronously in the middle of the T3 decode**, stalling all remaining active T3 rows!

3. **False assumption (Phase 9-3/9-4):** *"Returning `UnfinishedRequestOutput` to the engine loop after every single speech token has negligible overhead."*
   - **Reality:**
     - `UnfinishedRequestOutput` is never emitted to the client (`harmonyspeech_engine.py:420`), yet for a 400-token decode the engine performs 400 full round-trips through `asyncio.wait_for` → `Scheduler.schedule()` → `loop.run_in_executor` → `ModelRunnerBase.execute_model` → `prepare_inputs` → `ARBatchRunner.step` (1 token) → `asyncio.sleep(0)`.
     - Worse, on **every single one of those 400 token steps**, `_execute_partition` (`model_runner_base.py:1240`) calls `prepare_inputs(self.model_config, requests_to_batch)` for all continuing requests, and `prepare_chatterbox_*_tts_inputs` (`inputs.py:765, 812, 877`) constructs and tears down a brand-new `with ThreadPoolExecutor() as executor:` on every token step — even though `_execute_chatterbox_tts_ar` ignores `inputs` for continuing rows!

## Host verification (Phase 12-9, 2026-11-09, GPU 1)

The post-fix host re-run confirms all four fixes. Fidelity
(`out/bench/fidelity-12-9.json`; Whisper `large-v3-turbo`, unified 300 chars =
4 chunks, 138-word fixture) and throughput (`out/bench/phase12-9-*.json`) — full
tables in `docs/benchmarks/throughput.md` §Phase 12:

* **KV-slot sizing (fix 1):** Turbo chunked at 300 chars now scores **4** edit
  distance with **138/138 words** (pre-fix: 15–23, dropping ~15–20 words). Its
  *unchunked* row is still 108/114, proving the residual weakness is real
  long-context behaviour while the chunked path is faithful. Base chatterbox
  ON is 2, Multilingual ON is 4 and OFF is **0** (pre-fix OFF was 140 / 73 words,
  ON 67 / 73).
* **Deferred batched vocode + multi-token steps (fixes 2–3):** Turbo ON latency
  fell c=1 11,720.2 ms (@150 workaround) → 7,027.8 ms, c=4 44,128.4 →
  20,873.6 ms, c=8 67,181.2 → 21,127.5 ms; Multilingual ON c=1 33,874.3 →
10,494.6 ms, c=4 163,335.5 → 36,460.4 ms, c=8 227,592.3 → 33,558.0 ms.
* **Admission/exaggeration math (fix 4):** no regression; c=1 chunked beats
  unchunked on five of six families (e.g. chatterbox 15,203.9 → 10,295.6 ms,
  kitten 17,856.7 → 7,748.2 ms).

**Follow-on finding fixed in 12-9:** with Turbo no longer truncating, the
single-shot `split=off` floods for melo/openvoice_v2 exposed a WDDM Shared
Memory paging cliff — one padded c=8 step reserved 6.5–9.8 GiB and stalled
92–402 s. `harmonyspeech/task_handler/batching.py` now caps padded area with
`DEFAULT_PHONE_BATCH_BUDGET = 1024` / `DEFAULT_MEL_FRAME_BATCH_BUDGET = 7500`,
and `ARBatchRunner` contracts declare `preferred_cap_tokens` **1152**
(English/Multilingual) / **1536** (Turbo). Post-fix `split=off` reaches
1.35 req/s (melo c=8) and 0.89 req/s (openvoice_v2 c=8).

## Implementation steps

1. **Fix KV-cache slot capacity (`cap_tokens`) & pool budget so `ARBatchRunner` never truncates below upstream's 1000-token speech budget (`harmonyspeech/engine/ar_cache.py`, `harmonyspeech/task_handler/ar_runner.py`, `config.gpu.yml`, `config.yml`):**
   - Raise `DEFAULT_AR_CACHE_CAP_TOKENS` / `ar_inference.default_cap_tokens` from `768` to **`1536`** (enough for Turbo's worst-case `376` cond + `~150` text + `1001` speech tokens = `1527` tokens, and English/Multilingual's `34` cond + `~500` text + `1000` speech tokens = `1534` tokens), and size `ar_inference.kv_cache_budget_bytes` so all configured AR arenas (`chatterbox` 8 slots, `chatterbox_multilingual` 8 slots, `chatterbox_turbo` 4 slots) fit comfortably on the 16 GiB GPU alongside the 8.5 GiB resident catalogue (or size `cap_tokens` per contract: e.g. `1280` for Perceiver-resampled contracts where `len_cond == 34`, `1536` for `TurboContract` where `len_cond <= 376`).
   - In `ARBatchRunner.step()` (`ar_runner.py:820-824`), log a `logger.warning(...)` whenever a row stops with `finish_reason = "length"` (`row.tokens_emitted >= row.max_new_tokens` or `row.length >= self.arena.cap_tokens`) so silent truncation can never hide again.
   - Remove `DEFAULT_CHUNK_SIZE_TURBO = 150` and `DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE` from `harmonyspeech/endpoints/openai/serving_text_to_speech.py` and `scripts/bench_throughput.py`, restoring the unified `DEFAULT_CHUNK_SIZE = 300` across all families.

2. **Skip `prepare_inputs` for continuing AR requests and remove unnecessary `ThreadPoolExecutor` churn (`harmonyspeech/task_handler/model_runner_base.py`, `harmonyspeech/task_handler/inputs.py`):**
   - In `ModelRunnerBase._execute_partition` (`model_runner_base.py:1233-1264`), when the runner has an active `_ar_runner` and a request in `requests_to_batch` already has an active row (`self._ar_runner.has_row(req.request_id)`), do **not** call `prepare_inputs` for that request — pass `None` at its index in `inputs` and only run `prepare_inputs` on the subset of newly admitted requests (`not self._ar_runner.has_row(req.request_id)`). When all requests in the step are continuing AR rows (`new_indices` is empty), skip `prepare_inputs` completely!
   - In `harmonyspeech/task_handler/inputs.py`, avoid spawning a `ThreadPoolExecutor` when `len(requests_to_batch) <= 1` (run `[prepare(requests_to_batch[0])]` inline on the calling thread).

3. **Decode multi-token bursts per engine step & defer S3Gen vocoding until the batch's rows finish (`harmonyspeech/task_handler/ar_runner.py`, `harmonyspeech/task_handler/model_runner_base.py`):**
   - Add a multi-token step quantum to `ARBatchRunner` (`DEFAULT_AR_STEP_TOKENS = 16`, `ARBatchRunner.step_tokens`), and loop inside `_execute_chatterbox_tts_ar` for up to that many tokens per **continuation** engine step while active rows remain, yielding back to the scheduler every burst so newly arrived requests can still join and aborts/deadlines are checked far less often than once per token. An **admission** step (a row that just prefilled) stays at exactly one token so newly admitted rows return immediately, preserving the step-1 abort/deadline/poisoned-row semantics. The burst breaks immediately when any row finishes or errors, so its KV-cache slot is freed without extra token steps.
   - **Defer S3Gen vocoding until co-batched rows finish together:**
     - When a row hits EOS on token step $t$ while other rows in the runner are still decoding, immediately extract its `(tokens, conds, seed, progress)` and **release its KV-cache slot** (`_take_ar_row`), but hold it in a runner-owned `_pending_vocode` map (`request_id -> (tokens, conds, seed, progress)`) and return `UnfinishedRequestOutput` for it while the remaining active rows in the batch finish decoding (or flush `_pending_vocode` once all rows in the current batch have finished decoding, or when a row is aborted/timed out).
     - Once the batch's rows have all finished their T3 decode, pass all accumulated finished rows together into `self._finalize_ar_rows(finished, results, turbo=turbo)`:
       1. No S3Gen vocode ever runs mid-decode to stall active T3 rows!
       2. All 4 chunks of a chunked request (or 4 concurrent requests) reach `_finalize_ar_rows` **at the same time** (`len(finished) == 4`), so `inference_batched` actually executes a single padded S3Gen batch forward for all 4 chunks!

4. **Fix `_ar_conditionals` `exaggeration` & `Scheduler._ar_admission_width` slot-per-request math (`harmonyspeech/task_handler/model_runner_base.py:860-902`, `harmonyspeech/processing/scheduler.py:513-544`):**
   - In `cast_conditionals(conds, t3, exaggeration=None)` (`ar_runner.py:143-155`) / `_ar_conditionals` (`model_runner_base.py:860-902`), clone `conds.t3` before mutating it (so a shared `SpeakerArtifact` from `_create_chunked_response` or `EmbeddingCache` is never mutated across requests) and apply `exaggeration` to `conds.t3.emotion_adv` when `exaggeration is not None` and `getattr(conds.t3, "emotion_adv", None) is not None` (matching `_align_chatterbox_conditionals` at `model_runner_base.py:809-810`).
   - In `ARCacheArena.capacity()` (`ar_cache.py:293-314`) / `ARBatchRunner.register()` and `Scheduler._ar_admission_width` (`scheduler.py:513-544`), record `rows_per_request` (`2` for CFG contracts, `1` for Turbo) on the arena capacity dict and compute free request capacity as `capacity["slots_free"] // rows_per_request` so the scheduler never admits a 2-slot CFG request when only 1 slot is free.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/engine/ar_cache.py`, `config.gpu.yml`, `config.yml` | Raise `default_cap_tokens` (e.g. `1536`) and size `kv_cache_budget_bytes` so Turbo + English + Multilingual fit without truncating 1000-token decodes |
| `harmonyspeech/task_handler/ar_runner.py` | Multi-token step support, `rows_per_request` in arena capacity, warning log on `finish_reason == "length"`, `cast_conditionals` `exaggeration` update |
| `harmonyspeech/task_handler/model_runner_base.py` | Skip `prepare_inputs` for continuing AR rows; multi-token burst + deferred `_pending_vocode` batching so `_finalize_ar_rows` batches all finished rows together after T3 decode |
| `harmonyspeech/task_handler/inputs.py` | Skip `ThreadPoolExecutor` creation for single-request `prepare_*` calls |
| `harmonyspeech/processing/scheduler.py` | Divide `slots_free` by `rows_per_request` in `_ar_admission_width` |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `scripts/bench_throughput.py` | Remove `DEFAULT_CHUNK_SIZE_TURBO = 150` and `DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE`; restore unified 300-char default |
| `tests/unit/**`, `tests/integration/**` | Tests for deferred batched S3Gen vocode on staggered-EOS rows, `prepare_inputs` skipped on continuation steps, `exaggeration` on AR path, `rows_per_request` admission math, and unified 300-char default |

## Tests / validation

- Unit tests:
  - 4 AR rows that hit EOS on **different** token steps (e.g. steps 5, 8, 12, 15) are held until all 4 finish T3 decode and then vocoded in **one** `inference_batched` call (`calls == 1`, `len(rows) == 4`), never 4 solo `s3gen.inference` calls.
  - `prepare_inputs` is called **once** (at admission) for a multi-step AR request, never on continuation steps.
  - `TurboContract` with a 376-token conditioning prefix + 80-token text prefill gets the full `1001` `max_new_tokens` budget under the new `cap_tokens`.
  - `_ar_conditionals` applies `exaggeration` to `conds.t3.emotion_adv` even when `SpeakerArtifact` carries precomputed conditionals.
  - `Scheduler._ar_admission_width` divides `slots_free` by `rows_per_request`.
- Integration tests: full `tests/unit` + `tests/integration` + `tests/e2e/wire_contract` green.

## Checklist

- [x] `cap_tokens` and pool budget sized so Turbo (376-token cond prompt) and English/Multilingual get full 1000-token speech decode without truncation
- [x] `DEFAULT_CHUNK_SIZE_TURBO = 150` and `DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE` removed; unified 300-char default restored
- [x] Continuing AR rows skip `prepare_inputs`; continuation steps decode a multi-token burst (`DEFAULT_AR_STEP_TOKENS = 16`) that stops immediately on EOS/error, while admission steps stay at one token
- [x] Staggered-EOS AR rows defer S3Gen vocoding until the batch finishes and run in one `inference_batched` call
- [x] `_ar_conditionals` applies `exaggeration`; `Scheduler._ar_admission_width` respects `rows_per_request`
- [x] Host 12-9 re-verification confirms the fixes (Turbo 300-char fidelity 15–23 → 4, 138/138 words; Multilingual 67 → 4 / OFF 0; c=8 latency 67.2 s → 21.1 s and 227.6 s → 33.6 s) and the follow-on WDDM paging budget
- [x] Unit + integration tests green; `summary.md` updated
