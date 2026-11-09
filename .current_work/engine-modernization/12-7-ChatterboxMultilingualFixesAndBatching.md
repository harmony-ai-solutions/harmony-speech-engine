# Phase 12-7: Chatterbox Multilingual — Fix `repetition_penalty`, `AlignmentStreamAnalyzer` Leaks/Cutoffs, SDPA & AR Batching

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio & Chatterbox batching (audit fix)
> **Depends on:** 9-1, 9-3, 9-5, 12-2, 12-5
> **Codebase mapping consulted:** `.planning/codebase/{ARCHITECTURE,CONVENTIONS,TESTING,CONCERNS}.md`
> **State reference:** verified 2026-11-09 — `harmonyspeech/task_handler/inputs.py:817-879` (`prepare_chatterbox_multilingual_tts_inputs`), `harmonyspeech/modeling/models/chatterbox/{chatterbox.py:357-366,factory.py:134-145,520-532,t3_ar.py}`, `harmonyspeech/task_handler/{ar_runner.py,batching.py:104,model_runner_base.py:2890-2946}`, `harmonyspeech/processing/scheduler.py:38`, `harmonyspeech/endpoints/openai/serving_text_to_speech.py:82-100`, and installed `chatterbox-tts==0.1.6` (`chatterbox/mtl_tts.py:233-301`, `chatterbox/models/t3/t3.py:273-298`, `chatterbox/models/t3/inference/alignment_stream_analyzer.py:32-178`).

## Objective

Fix the root causes of `ChatterboxMultilingualTTS` premature EOS truncation, progressive latency degradation, and exclusion from SDPA / `ARBatchRunner` / padded `S3Gen` batching — and remove the destructive `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75` workaround introduced in Phase 12-5.

## False assumptions from earlier phases — DO NOT repeat

1. **False assumption (Phase 12-5 `serving_text_to_speech.py:82-92`, `12-5-JoinQualityAndRealFixture.md:266-285`):** *"Chatterbox Multilingual silently skips half the text at `chunk_size=300` (speaking only 73 of 138 words) because the model itself is budget-sensitive on long prose and requires `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75`."*
   - **Reality:** `ChatterboxMultilingualTTS` was **not** skipping words in the middle of a chunk — its chunks were being **aborted mid-sentence by forced EOS tokens** from `AlignmentStreamAnalyzer.step()`! Shrinking `chunk_size` to `75` (16 chunks) only masked the symptom by restarting synthesis 16 times on tiny ~8-word slices, at the cost of forcing `_split_oversize` / `_hard_wrap` to chop 90–160 char sentences in half mid-phrase (where `punc_norm` appends a fake period `.` and the stitcher inserts a 120 ms pause mid-sentence).
   - **Why `AlignmentStreamAnalyzer` forced EOS mid-sentence:**
     1. **Wrong `repetition_penalty` default in HSE (`harmonyspeech/task_handler/inputs.py:859`):** Upstream `ChatterboxMultilingualTTS.generate()` (`chatterbox/mtl_tts.py:241`) defaults to **`repetition_penalty=2.0`** (whereas English `ChatterboxTTS.generate()` defaults to `1.2`). In HSE, `prepare_chatterbox_multilingual_tts_inputs` copy-pasted `1.2`, and `_execute_chatterbox_multilingual_tts` (`model_runner_base.py:2927`) explicitly passed `repetition_penalty=1.2` into `self.model.generate(...)`, overriding upstream's `2.0`.
     2. **Immediate EOS on any 2-token repeat (`alignment_stream_analyzer.py:154-175`):** In upstream `AlignmentStreamAnalyzer.step()`:
        ```python
        # Check for excessive token repetition (3x same token in a row)
        token_repetition = (
            # self.complete and 
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )
        ...
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15
        ```
        Upstream commented out `# self.complete and` and sliced `[-2:]` (any **2** identical consecutive speech tokens, not 3!), and placed the forced-EOS overwrite *after* the incomplete-sentence EOS suppression (`if cur_text_posn < S - 3: logits[..., self.eos_idx] = -2**15`). With HSE's `repetition_penalty=1.2` instead of `2.0`, sampling two identical speech tokens in a row happens constantly mid-sentence, immediately forcing EOS and cutting off the rest of the chunk!
     3. **Broken frame-vs-character masking and `discontinuity` in `AlignmentStreamAnalyzer.step()` (`alignment_stream_analyzer.py:104-140`):** `A_chunk[:, self.curr_frame_pos + 1:] = 0` masks text token columns using the 25 Hz *speech frame index* `curr_frame_pos` against character token columns (~14–15 chars/s). Around ~60 % through a 300-char chunk, `curr_frame_pos` exceeds `S - 3`, unmasking the final punctuation/`[STOP]` tokens (which act as attention sinks in Llama). Meanwhile `discontinuity = not(-4 < cur_text_posn - self.text_position < 7)` freezes `self.text_position` if attention ever jumps by $\ge 7$ characters, or flips `self.complete = True` early and triggers `alignment_repetition` / `long_tail` EOS within ~5–10 frames (~200–400 ms).

2. **False assumption (Phase 9-1 `t3_ar.py:66-69`, Phase 9-3 `9-3-ChatterboxARRunner.md:16`, `docs/batching.md:499`):** *"ChatterboxMultilingualTTS must force `_attn_implementation = 'eager'`, cannot use `ChatterboxT3AR`, and cannot be batched in `ARBatchRunner` (`max_batch_size: 1`) because its `AlignmentStreamAnalyzer` needs eager attention hooks."*
   - **Reality:**
     - `ChatterboxMultilingualTTS`'s T3 is `T3(T3Config.multilingual())` — the **exact same `Llama_520M` architecture** as base `ChatterboxTTS` (`t3_config.py:4-40`), with `text_tokens_dict_size = 2454` instead of `704`.
     - Worse, upstream `T3.inference` (`t3.py:273-287`) sets `self.compiled = False` on **every call** and creates a new `AlignmentStreamAnalyzer`, which calls `target_layer.register_forward_hook(attention_forward_hook)` on `self.tfmr.layers[9, 12, 13].self_attn` (`alignment_stream_analyzer.py:84`) and **never removes the hook handles**! Every multilingual synthesis call permanently leaks 3 GPU→CPU `.cpu()` attention hooks onto `self.tfmr`. After 500 chunk calls, 1,500 hooks fire on every decode step of every future request (causing the 55 s → 91 s → 227 s latency collapse recorded in `docs/benchmarks/throughput.md:1996-2008, 2208`).
     - In addition, `chatterbox/mtl_tts.py:292` omits the `speech_tokens = speech_tokens[speech_tokens < 6561]` OOV filter that `chatterbox/tts.py:262` and `ar_runner.py:348` apply before `s3gen.inference`.

## Implementation steps

1. **Fix `repetition_penalty` default for `ChatterboxMultilingualTTS` (`harmonyspeech/task_handler/inputs.py:859`):**
   - In `prepare_chatterbox_multilingual_tts_inputs`, change the default `repetition_penalty` when `opts.repetition_penalty is None` from `1.2` to **`2.0`** (matching `chatterbox/mtl_tts.py:241`).
   - Update unit tests in `tests/unit/inference_flow/test_chatterbox_inputs.py` (and any stub prepare helpers) to assert the `2.0` default for `ChatterboxMultilingualTTS` while keeping `1.2` for `ChatterboxTTS` and `ChatterboxTurboTTS`.

2. **Eliminate the `AlignmentStreamAnalyzer` hook leak, premature EOS cutoffs, and `eager` attention penalty (`harmonyspeech/modeling/models/chatterbox/{chatterbox.py,factory.py,t3_ar.py}`):**
   - Route `ROLE_MULTILINGUAL` through `ChatterboxT3AR` (`factory.py:_AR_T3_ROLES = frozenset({ROLE_TTS, ROLE_MULTILINGUAL})`) and also graft `ChatterboxT3AR` onto models loaded via `ChatterboxMultilingualTTSModel.from_pretrained` (`chatterbox.py`).
   - Do **not** force `model.t3.tfmr.config._attn_implementation = "eager"` at rest (`chatterbox.py:364-365`, `factory.py:529-530`): `ChatterboxMultilingualTTS` keeps `Llama_520M`'s native `attn_implementation = "sdpa"`, and the `_SafeAlignmentStreamAnalyzer` switches to eager **only for the forward that needs it**, restoring the config and removing every hook afterwards.
   - Run the `_SafeAlignmentStreamAnalyzer` **by default** for multilingual (`ChatterboxT3AR.analyzer_enabled()` is `True` unless `HSE_CHATTERBOX_MTL_ANALYZER` is `0`/`false`/`no`/`off`). With `repetition_penalty=2.0` + `T3ARSampler`, the analyzer's post-completion `long_tail` trigger is what stops a chunk end that never emits `stop_speech_token` naturally; without it the chunk babbles to `max_new_tokens`.
     - When hooks are attached (sequential path), switch `_attn_implementation = "eager"` *before* setting `output_attentions = True` (the config setter rejects SDPA with the `ValueError: The output_attentions attribute is not supported when using the attn_implementation set to sdpa`), capture the hook handles, and **always remove them (`handle.remove()`) and restore both config attributes in `close()`/`finally`** so zero hooks leak across calls.
     - Fix the 2x-vs-3x repetition bug (`len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-3:])) == 1` only after `self.complete`).
   - Ensure the sequential multilingual path filters `speech_tokens = speech_tokens[speech_tokens < 6561]` (or uses `ChatterboxT3AR` + `drop_invalid_tokens` + `< 6561`) so an OOV token can never crash `s3gen`.
   - On the batched path, attach one `_SafeAlignmentStreamAnalyzer(..., attach_hooks=False)` per `ARRow` (`harmonyspeech/task_handler/ar_runner.py`). `ARBatchRunner._run_with_alignment` registers temporary eager attention hooks for the duration of one batched forward, fills each row's `last_aligned_attns` from its conditional CFG slot (`step_attention[2 * r_idx, head_idx, :, :kv_len]`), and removes every hook in `finally` — zero hooks remain on the shared `tfmr` after `begin()`/`step()`.

3. **Enable `ARBatchRunner` + padded `S3Gen` batching for `ChatterboxMultilingualTTS` (`harmonyspeech/task_handler/{ar_runner.py,batching.py,model_runner_base.py}`, `harmonyspeech/processing/scheduler.py`, `config.gpu.yml`, `config.yml`):**
   - Add `multilingual_text_tokens(model, text: str, language_id: str | None) -> Tensor` in `harmonyspeech/task_handler/ar_runner.py` (mirroring `chatterbox/mtl_tts.py:268-275`: `punc_norm(text)` → `model.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None)` → pad with `start_text_token` and `stop_text_token`; note CFG duplication happens inside `EnglishContract.build_prefill`, so `multilingual_text_tokens` returns `[1, T]` just like `english_text_tokens`).
   - `ChatterboxMultilingualTTS` shares the exact same decode contract as `EnglishContract` (2 CFG rows, `prepare_input_embeds`, learned speech position embeddings, `T3ARSampler` with default `repetition_penalty=2.0`, immediate EOS, `drop_invalid_tokens` + `< start_speech_token`, and the shared standard `S3Gen` which already supports `inference_batched`!).
   - Add `"ChatterboxMultilingualTTS"` to `AR_INFERENCE_FAMILIES` (`harmonyspeech/task_handler/batching.py:104`) and `AR_INFERENCE_MODEL_TYPES` (`harmonyspeech/processing/scheduler.py:38`).
   - Update `_execute_chatterbox_multilingual_tts` in `harmonyspeech/task_handler/model_runner_base.py:2890-2946` to dispatch to `_execute_chatterbox_tts_ar` when `runner is not None and runner.enabled` (keeping `_execute_chatterbox_multilingual_tts_sequential` as the `len == 1` / fallback path).
   - Update `chatterbox_multilingual` in `config.gpu.yml` and `config.yml` from `max_batch_size: 1` (`loop`) to `max_batch_size: 4` (`ar_decode`), and update the execution-model class table in `tests/unit/initialization/test_config.py`, `tests/unit/test_batching_docs_matrix.py`, and `docs/batching.md`.

4. **Remove `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75` (`harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `scripts/bench_throughput.py`):**
   - Remove `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75` from `serving_text_to_speech.py:82-100` and `LONGFORM_MULTILINGUAL_DEFAULT_CHUNK_SIZE = 75` from `scripts/bench_throughput.py:1292-1306` so `ChatterboxMultilingualTTS` uses the standard `DEFAULT_CHUNK_SIZE = 300` (4 sentence-preserving chunks on the Malfoy fixture instead of 16 mid-sentence fragments).
   - Update the unit and integration tests that pinned the 75-char override (`tests/unit/wire_contract/test_long_form_fields.py`, `tests/unit/benchmarks/test_bench_longform_scenarios.py`, `tests/integration/test_tts_chunking.py`).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/task_handler/inputs.py` | Default `repetition_penalty = 2.0` for `ChatterboxMultilingualTTS` |
| `harmonyspeech/modeling/models/chatterbox/{chatterbox.py,factory.py,t3_ar.py}` | Keep SDPA on Multilingual, wire `ChatterboxT3AR`, prevent `AlignmentStreamAnalyzer` hook leaks |
| `harmonyspeech/task_handler/{ar_runner.py,batching.py,model_runner_base.py}` | `multilingual_text_tokens`, enable `ARBatchRunner` + batched S3Gen for `ChatterboxMultilingualTTS` |
| `harmonyspeech/processing/scheduler.py` | Add `ChatterboxMultilingualTTS` to `AR_INFERENCE_MODEL_TYPES` |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `scripts/bench_throughput.py` | Remove `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75`; restore 300-char default |
| `config.gpu.yml`, `config.yml`, `docs/{batching,models,api}.md` | Move `chatterbox_multilingual` to `ar_decode` (`max_batch_size: 4`), remove false 75-char fidelity notes |
| `tests/unit/**`, `tests/integration/**` | Update input defaults, AR runner/handler tests, config honesty table, and chunk-size tests |

## Tests / validation

- Unit tests: `prepare_chatterbox_multilingual_tts_inputs` defaults `repetition_penalty` to `2.0`; `multilingual_text_tokens` + `ARBatchRunner` B=1 and multi-row seeded equality on a tiny multilingual T3; the analyzer is **on by default** and `HSE_CHATTERBOX_MTL_ANALYZER=0` disables it; zero leaked forward hooks on `t3.tfmr` and `_attn_implementation == "sdpa"` after `build_ar_backend()`/`inference()`/`begin()`/`step()`; config honesty and docs matrix tests green.
- Integration tests: 4-chunk multilingual request co-batches in `ARBatchRunner` and vocodes through `inference_batched`.

### GPU-2 verification (2026-11-09): why the analyzer is default-on

Verified on GPU 2 with the real `ChatterboxMultilingualTTS` checkpoint
(`t3_mtl23ls_v2.safetensors`) against the Malfoy fixture (138 words):

| Configuration | Result |
|---|---|
| No analyzer, `repetition_penalty=2.0` (fixed) | Cutoff fixed, **but** `t3_mtl23ls_v2` rarely emits `stop_speech_token` naturally at a sentence end, so chunks that never emit EOS babble until `max_new_tokens=1000`: **240/138 words, 91.5 s** |
| `_SafeAlignmentStreamAnalyzer` (3-token repeat fixed, hooks removed) + `repetition_penalty=2.0`, sequential `ChatterboxT3AR`, `chunk_size=300` (4 chunks) | **edit_distance = 0, 138/138 words, 38.12 s audio, 0 leaked hooks** |
| Same analyzer per-row in `ARBatchRunner` (batched `inference_batched` S3Gen), `chunk_size=300` (4 chunks) | **edit_distance = 2, 138/138 words, 37.34 s audio in 12.10 s wall-clock, 0 leaked hooks** |

Conclusion: the analyzer's post-completion `long_tail` EOS trigger is required at
chunk ends, not an optional latency penalty. `analyzer_enabled()` therefore
defaults to `True`; the per-row batched analyzer keeps the yield identical to the
solo run (a seeded row's tokens match whether decoded `B=1` or `B=2`) while
`_run_with_alignment` guarantees zero hooks remain on the shared `tfmr`.

## Checklist

- [x] `prepare_chatterbox_multilingual_tts_inputs` default `repetition_penalty` fixed to `2.0`
- [x] `ChatterboxMultilingualTTS` uses SDPA at rest + `ChatterboxT3AR`; analyzer default-on, eager only for the call, zero forward-hook leaks
- [x] `ChatterboxMultilingualTTS` wired into `ARBatchRunner` + padded `S3Gen` (`ar_decode`, `max_batch_size: 4`) with a per-row `_SafeAlignmentStreamAnalyzer` (zero leaked hooks)
- [x] `DEFAULT_CHUNK_SIZE_MULTILINGUAL = 75` workaround removed; 300-char default restored
- [x] Unit + integration suites and docs/config-honesty tests green; `summary.md` updated
