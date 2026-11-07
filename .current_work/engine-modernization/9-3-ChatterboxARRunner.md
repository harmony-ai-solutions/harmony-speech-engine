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

## Verify-first audit (2026-11-07, installed `chatterbox-tts==0.1.6`)

Re-verified in `C:\Users\sge20\miniconda3\envs\hse\Lib\site-packages\chatterbox`
(torch 2.9.1+cu130, transformers 4.57.1). Two decode contracts share the same
`T3` modules but differ in prefill composition, CFG, sampler, EOS rule and
position handling.

### English `T3.inference` (`models/t3/t3.py:226-412`; HSE seam `ChatterboxT3AR`)

| Aspect | Verified behaviour | Location |
|---|---|---|
| Prefill composition | `cond_emb` (`T3CondEnc`) + `text_emb` (+`text_pos_emb`) + `speech_emb(start)`(+`speech_pos_emb` → pos 0) **then a second BOS** `speech_emb(start)+get_fixed_embedding(0)`, both duplicated for CFG | `t3.py:263-268`, `:317-327`; `prepare_input_embeds` `:103-131` |
| CFG mechanics | **Duplicate rows**, not cfg-strength gating. The wrapper duplicates `text_tokens` only when `cfg_weight > 0` (`tts.py:237-238`); `prepare_input_embeds` zeroes row 1's text emb when `cfg_weight > 0` (`t3.py:114-115`); the BOS embed is always duplicated (`t3.py:324`). Decode combines `cond + cfg·(cond−uncond)` (`t3.py:353-358`). | as listed |
| Sampler order | CFG combine → (analyzer, English: absent) → repetition penalty over `generated_ids[:1]` (which starts with the BOS token) → temperature (`≠1.0`) → **min_p** → **top_p** → softmax → `torch.multinomial` (global RNG) | `t3.py:360-382`; `T3ARSampler.pick` |
| EOS | `stop_speech_token = 6562`; checked **immediately** after sampling (including the first token) → break; the EOS token stays in the returned stream, and `ChatterboxTTS.generate` then runs `drop_invalid_tokens` (drops SoS `6561` and from EOS) + `speech_tokens < 6561` | `t3.py:387-390`; `tts.py:256-262`; `s3tokenizer.drop_invalid_tokens` |
| Position API | `speech_pos_emb.get_fixed_embedding(idx)` accepts an int, `(T,)` or `(B,T)` tensor and returns `(B,T,dim)` (`learned_pos_emb.py:21-32`) — per-row positions are a call-site change, not an API change. **RoPE positions are the KV index** (`cache_position`), distinct from the learned speech position (`i+1`). | verified |
| Max tokens | `max_new_tokens=1000` from `tts.py:249` (upstream arg, not config) | verified |

### Turbo `T3.inference_turbo` (`models/t3/t3.py:414-490`; structurally different)

| Aspect | Verified behaviour | Location |
|---|---|---|
| Prefill composition | `cond_emb` + `text_emb` + `speech_emb(start)` **only** (no extra BOS, no CFG, one row); `input_pos_emb=None` so **no learned text/speech position embeddings at all**; `is_gpt=True` (GPT2) so `prepare_conditioning` also skips the cond-prompt position emb | `t3.py:429-435`; `tts_turbo.py:149-163` |
| Forward | raw `self.tfmr` (GPT2Model) with an HF `Cache`; `tfmr.wte` is **deleted** (`del t3.tfmr.wte`) but `inputs_embeds` never touches `wte`, and positions come from `wpe(position_ids)` | `t3.py:439-468`; `tts_turbo.py:162` |
| Sampler | `LogitsProcessorList([TemperatureLogitsWarper, TopKLogitsWarper(top_k), TopPLogitsWarper(top_p), RepetitionPenaltyLogitsProcessor])` in that order (temperature first, penalty last — different from English); input ids are `speech_start_token` for the first draw, then `cat(generated_speech_tokens)` | `t3.py:418-426`, `:450`, `:470-477` |
| EOS | No immediate check on token 1 (it is forwarded); after each loop draw, `if all(next == stop): break`; the trailing EOS is **stripped** from the result; `all(-inf)` logits break without emitting | `t3.py:472-488` |
| Max tokens | samples 1 token before the loop, then loops `max_gen_len=1000` → up to **1001** tokens | `t3.py:452-457` |
| Output post-step | `speech_tokens < 6561` + three `S3GEN_SIL` appended; S3Gen called with `n_cfm_timesteps=2` | `tts_turbo.py:284-293` |
| No attention output | never passes `output_attentions` | verified |

**How the runner covers both.** One `ARBatchRunner` with two contract
strategies (`EnglishContract` / `TurboContract`) over the same modules
(`prepare_input_embeds`, `speech_emb`, `speech_head`, `tfmr`). They differ only in
rows-per-request (2 CFG rows vs 1), prefill composition (extra BOS vs none),
per-token embedding (learned speech position vs none), sampler
(CFG+min_p+top_p vs logits-processor list), EOS rule (immediate vs after-loop)
and the token cap (1000 vs 1001). Both reduce to the sequential loops when run
for one request with a per-row generator seeded like the global RNG (the B=1
gate).

**Position finding that shaped the design.** The learned speech position
(`get_fixed_embedding(k)`) is *not* the RoPE/absolute position: upstream passes
no `position_ids`, so transformers uses `cache_position` (the KV index) for
RoPE/GPT2 `wpe`. The runner therefore carries two per-row cursors: the KV write
position and the learned speech position.

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

## Implementation notes (2026-11-07)

**Runner shape.** `harmonyspeech/task_handler/ar_runner.py` owns two contract
strategies (`EnglishContract` / `TurboContract`) over the same upstream modules.
Each engine step (1) samples one token per active row with that row's
`torch.Generator`, (2) runs one batched forward for the rows that still need
their token cached, and (3) returns `UnfinishedRequestOutput` for live rows or a
final result for rows that hit EOS/the cap. New rows are prefilled when admitted
(one forward per request — see deviation 3). `ArenaBatchCache` is an HF `Cache`
whose `update` writes each row's new K/V into its 9-2 slot (write-before-read)
and returns a per-layer `[R, H, len, D]` view: zero-copy when the batch's slots
are a contiguous ascending run, a length-only gather otherwise. When rows have
different lengths the forward gets a per-row `attention_mask`; when they share
one length it gets none, which is what keeps the B=1 gate bit-identical to
upstream's unmasked forward.

**Position handling (the one subtle trap).** Upstream passes no `position_ids`,
so transformers uses `cache_position` (the KV index) for RoPE / GPT2 `wpe`,
while the learned speech position (`get_fixed_embedding(k)`) is added to the
input embedding separately. The runner therefore carries two per-row cursors:
`length` (KV write index → RoPE) and `position` (learned speech position).

**Sampler.** `T3ARSampler.pick` gained an optional `generator=` keyword (default
`None` keeps upstream's global RNG). The runner draws per row, never as one
batched draw, so a generator seeded like the global RNG reproduces the sequential
loop and a row is reproducible alone or in a batch. Turbo builds its own
`LogitsProcessorList` in upstream's order (temperature → top_k → top_p →
repetition penalty) per row.

**Handler.** `_execute_chatterbox_tts` / `_execute_chatterbox_turbo_tts` now
dispatch to the runner when the family flag, the pool/arena and `len > 1` (for a
new request) allow it; the previous bodies are `_execute_chatterbox_*_tts_sequential`
and remain the fast path/fallback. The AR path resolves conditionals without
leaving them on the model (pre-computed by reference; reference-audio via
upstream `prepare_conditionals` with the baseline object restored immediately;
the bundled default voice via a clone), and finalizes each finished row with the
per-row S3Gen hop + watermark (9-5 batches that hop later).

**Registration.** The runner registers at model load (`_init_ar_runner`, called
from both model runners) under `ModelConfig.name` — 9-4's capacity-admission key.
The arena is sized `min(max_batch_size × rows_per_request, remaining budget // slab)`
so the English and Turbo families share the pool instead of the first one eating
it, and a runtime refusal becomes impossible when the budget is adequate.

## Deviations (2026-11-07)

1. **`t3_ar.py` gained an optional `generator=` keyword** on
   `T3ARSampler.pick` / `ar_sample`. The files table does not list 9-1's module,
   but the per-row generator is 9-3's core reproducibility requirement and
   reusing the seam's sampler is exactly what 9-1 designed it for. Default
   `None` preserves upstream's global-RNG behaviour byte-for-byte.
2. **`executor_base.py`, `gpu_model_runner.py`, `cpu_model_runner.py` touched.**
   The engine calls `release_request` on the *executor* (9-4), so
   `ExecutorBase.release_request` delegates to `driver_worker.model_runner`;
   `ModelRunnerBase.release_request` forwards to the AR runner. Registration at
   model init required the `_init_ar_runner()` call in both model runners.
3. **Prefill is per request, not a padded batch.** The study suggested a batched
   prefill; the runner does one forward per admitted request. Prefill is short
   and happens once per request, and a per-request prefill keeps the seeded
   equality gates exact (a padded prefill changes GEMM shapes and can flip a
   sampled token). The decode — the part that matters — is batched.
4. **`max_new_tokens` is capped by the slot capacity.** English keeps upstream's
   1000, Turbo its 1001, but a row also stops when its KV write would exceed
   `cap_tokens`; a prefill that does not fit a slot returns `None` and the
   handler takes the sequential loop (no slot cap there). With the shipped
   `default_cap_tokens: 768` this bounds an utterance to roughly 30 s minus the
   text prefill; 9-6 owns the measured value.
5. **Turbo's sequential fallback keeps `T3.inference_turbo` (including its
   `tqdm`).** 9-3's turbo scope is AR-batching, not loop hygiene; the runner
   covers the batched path. The turbo `T3Config` load bug is fixed in
   `factory.load_t3` (attributes set after construction, mirroring
   `tts_turbo.py:149-155`).
6. **The conditioning-state deletion uses a transient `prepare_conditionals`
   for the inline reference-audio path.** The decode never reads or writes
   `model.conds` (the conditionals live on the runner's row) and the baseline
   object is restored immediately, so identity is unchanged for the whole
   decode (asserted at every step). Pre-computed conditionals (the common
   pipeline path) never touch the model at all.
7. **Arena dtype follows the T3 weights** (`next(t3.parameters()).dtype`), i.e.
   fp16 on the CUDA rollout and fp32 on CPU — the 9-2 pool still owns the device
   and the budget. A pool on a different device than the model disables the
   runner (sequential) instead of failing boot.
8. **`docs/batching.md` rows for English/Turbo now name the AR mechanism** while
   keeping the shipped `max_batch_size: 1` (9-6 moves the class). The docs-lint
   assertion that quoted the old "out of scope" decision was updated to the new
   AR-decode + multilingual-skip wording.
9. **A pre-existing `SIM103` in `batching.py::plan_batches._fits` was rewritten**
   as a negated return so `ruff check` stays clean on the changed file; no
   behaviour change.

## Test evidence (2026-11-07)

Local env (`C:\Users\sge20\miniconda3\envs\hse\python.exe`, torch 2.9.1+cu130,
transformers 4.57.1, CPU, tiny seeded in-process T3 models — no weights):

| Command | Result |
|---|---|
| `python -m pytest tests/unit/modeling/test_chatterbox_ar_runner.py -q` | **11 passed** |
| `python -m pytest tests/unit/inference_flow/test_chatterbox_ar_handler.py -q` | **10 passed** |
| `python -m pytest tests/integration/test_chatterbox_ar_engine.py -q` | **3 passed** |
| `python -m pytest tests/unit -q` | **1076 passed** (1054 before + 22 new) |
| `python -m pytest tests/integration -q` | **73 passed** (70 before + 3 new) |
| `python -m ruff check <changed python files>` | clean |

The runner tests cover B=1 bit-identity against 9-1's cleaned loop for both
`eager` and `sdpa`, Turbo B=1 identity against `inference_turbo`, multi-row
seeded token equality against solo runs for both contracts, per-row generator
reproducibility across batch compositions, EOS row-drop mid-batch, poisoned-row
isolation, pool refusal, disabled pool, and arena registration under the
scheduler key. The handler tests cover pairing, `model.conds` identity at every
step, `len == 1` / `batched_inference: false` / pool-refusal fallbacks,
`release_request`, and poisoned-row isolation. The integration tests cover four
mixed-length seeded requests end-to-end, abort mid-decode (row + 2 slots freed),
and deadline expiry mid-decode (errors alone, loop lives). The 9-1 factory test
gained the Turbo `T3Config` regression test.

## Checklist

- [x] Upstream decode audit recorded here (prefill/CFG/sampler/EOS/positions/turbo deltas)
- [x] ARBatchRunner landed; B=1 bit-identity + seeded multi-row equality green
- [x] Handler wired with all fallbacks; conds-mutation deletion asserted
- [x] Multilingual documented in `docs/batching.md`; turbo AR-batched per the committed scope (audit recorded here)
- [x] `summary.md` Implementation Status updated
