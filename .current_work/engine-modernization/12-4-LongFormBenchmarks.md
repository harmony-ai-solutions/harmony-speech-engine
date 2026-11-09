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

- [x] Long-form scenarios landed; local determinism self-check green
- [ ] Regression gates (byte-identity, seeded determinism) green at family breadth
- [ ] Host session measured; §Phase 12 cells filled
- [ ] Listening pass owner-approved (dated); H1–H4 verdicts recorded; `summary.md` ticked

## Deviations / verification log

Dated 2026-11-09 (Phase 12-4 code half; GPU 1 measurement session is separate).

**Fixture design (deterministic, no RNG state)**

- `scripts/bench_throughput.py` generates one paragraph per family with
  `longform_fixture(family)`: seed `LONGFORM_SEED = 20261109`, a pure arithmetic
  word-index construction (no `random` state, so identical on every host and
  Python version). Each fixture is 7–8 sentences and 768–940 characters, in the
  spec's 400–1200 range. SHA-256 per family is pinned by
  `tests/unit/benchmarks/test_bench_longform_scenarios.py`; the same test
  asserts 4 chunks at the server's 300-character budget (so one request can fill
  a 4-row AR admission batch).
- The determinism axis uses a fixed 446-character, 6-sentence literal
  (`audio_regression.LONGFORM_CHUNKED_TEXT`), which splits into 6 chunks at the
  120-character axis budget.

**Scenario table**

`--scenario longform` runs six families × `{off,on}` × c={1,4,8,20}, 8 requests
per level (2 warmups). c=1 is the single-request latency. Labels are
`phase12-<family>-split-<off|on>-c<N>`; `--longform-split both` emits the pair on
one boot. Families and classes:

| Family | Class | Fixture chars | Chunks/req (ON) |
|---|---|---|---|
| chatterbox | AR decode, admission width 4 | 768 | 4 |
| chatterbox_turbo | AR decode, admission width 4 | 938 | 4 |
| chatterbox_multilingual | AR loop class (control) | 795 | 4 |
| melo | padded batch | 940 | 4 |
| openvoice_v2 | clone pipeline | 829 | 4 |
| kitten | lanes | 937 | 4 |

ON = `generation_options.split_text: true` (default 300-character budget); OFF =
`split_text: false`. Text, voice, mode, reference, seed and filter booleans are
identical; the unit self-check asserts the payloads differ only in
`split_text`/`chunk_size`/`crossfade_ms`/`sentence_pause_ms`.

**Per-chunk composition**

The existing 8-6 tail discipline is reused: the long-form runner reads the
engine DEBUG log (`EngineLogTail` + `parse_step_composition`) per level and
records steps, padded sub-batches, rows and requests. Expected chunks/request is
computed client-side with the same `split_text_into_chunks` the serving layer
uses, because the HTTP response exposes no chunk metadata (limitation, stated in
the docs block). The unit self-check validates every payload against
`TextToSpeechRequest`.

**Regression axes**

- `longform:off-vs-on`: short-text TTS cases at family breadth; side B sets
  `split_text: true`. Watermark is off so Perth's cross-process conv noise cannot
  mask the dispatcher identity. Deterministic families must be bit-identical;
  stochastic families use the documented CUDA kernel-noise floor
  (`SHARING_MAX_ABS_DIFF` / `SHARING_LOG_MEL_MSE`), same as the sharing axis.
- `determinism:run1-vs-run2`: both sides run the same chunked request at seed
  1234; chunk seeds are `seed + chunk_index`, so the stitched bytes must match.
- Local stub proof: the harness has no in-process stub synthesizer, so the axes
  are proven on synthesized WAVs (identical passes, a different signal fails,
  stochastic noise-floor case passes) in
  `tests/unit/benchmarks/test_audio_regression_longform.py`.

**Exact host-session runbook (GPU 1)**

1. Boot A (headline, INFO; DEBUG depresses the clone pipeline ~3×). Use
   `config.gpu.yml` or a reduced bench config that advertises the six families;
   the harness skips (and records) any family whose model is not advertised.

   ```bash
   CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=INFO \
       python -m harmonyspeech.endpoints.openai.api_server \
       --config-file-path config.gpu.yml --port 2242 \
       > out/bench/engine-phase12-info.log 2>&1
   # bounded health poll
   for i in $(seq 1 60); do curl -fsS http://127.0.0.1:2242/health && break; sleep 2; done
   ```

2. Matrix (one command per family; both variants, c=1,4,8,20):

   ```bash
   python scripts/bench_throughput.py --scenario longform --families chatterbox \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-chatterbox --json-out out/bench/throughput-phase12-chatterbox.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families chatterbox_turbo \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-chatterbox_turbo --json-out out/bench/throughput-phase12-chatterbox_turbo.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families chatterbox_multilingual \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-chatterbox_multilingual --json-out out/bench/throughput-phase12-chatterbox_multilingual.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families melo \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-melo --json-out out/bench/throughput-phase12-melo.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families openvoice_v2 \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-openvoice_v2 --json-out out/bench/throughput-phase12-openvoice_v2.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families kitten \
       --longform-split both --concurrency 1,4,8,20 --longform-requests 8 \
       --label phase12-kitten --json-out out/bench/throughput-phase12-kitten.json \
       --md-out docs/benchmarks/throughput.md
   ```

3. Composition cells (Boot B, DEBUG + engine log; ON rows only):

   ```bash
   CUDA_VISIBLE_DEVICES=1 HARMONYSPEECH_LOG_LEVEL=DEBUG \
       python -m harmonyspeech.endpoints.openai.api_server \
       --config-file-path config.gpu.yml --port 2242 \
       > out/bench/engine-phase12-debug.log 2>&1
   python scripts/bench_throughput.py --scenario longform --families chatterbox \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-chatterbox-debug \
       --json-out out/bench/throughput-phase12-chatterbox-debug.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families chatterbox_turbo \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-chatterbox_turbo-debug \
       --json-out out/bench/throughput-phase12-chatterbox_turbo-debug.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families chatterbox_multilingual \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-chatterbox_multilingual-debug \
       --json-out out/bench/throughput-phase12-chatterbox_multilingual-debug.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families melo \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-melo-debug \
       --json-out out/bench/throughput-phase12-melo-debug.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families openvoice_v2 \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-openvoice_v2-debug \
       --json-out out/bench/throughput-phase12-openvoice_v2-debug.json \
       --md-out docs/benchmarks/throughput.md
   python scripts/bench_throughput.py --scenario longform --families kitten \
       --longform-split on --concurrency 1,4,8,20 --longform-requests 8 \
       --engine-log out/bench/engine-phase12-debug.log --label phase12-kitten-debug \
       --json-out out/bench/throughput-phase12-kitten-debug.json \
       --md-out docs/benchmarks/throughput.md
   ```

4. Regression gates (the harness boots its own reduced engine; run on the same
   idle card):

   ```bash
   python scripts/audio_regression.py \
       --families chatterbox,chatterbox_turbo,chatterbox_multilingual,openvoice_v1,openvoice_v2,melotts,harmonyspeech,kittentts \
       --axis longform:off-vs-on --gpu --json-out out/bench/gate-12-4-identity.json
   python scripts/audio_regression.py \
       --families chatterbox,chatterbox_turbo,chatterbox_multilingual,openvoice_v1,openvoice_v2,melotts,harmonyspeech,kittentts \
       --axis determinism:run1-vs-run2 --gpu --json-out out/bench/gate-12-4-determinism.json
   ```

5. H3/H4 probes:

   ```bash
   python scripts/bench_throughput.py --scenario longform --families chatterbox \
       --longform-split on --concurrency 1 --longform-requests 4 \
       --longform-crossfade-ms 50 --longform-sentence-pause-ms 50 \
       --label phase12-crossfade-50 --json-out out/bench/throughput-phase12-crossfade.json
   python scripts/bench_throughput.py --scenario longform --families melo \
       --longform-split on --concurrency 1,4 --longform-requests 8 \
       --longform-remove-dc-offset --longform-trim-silence --longform-fix-internal-silence \
       --label phase12-filters-on --json-out out/bench/throughput-phase12-filters.json
   ```

   Outputs land in `out/bench/throughput-phase12-*.json` (per-run report),
   `out/bench/gate-12-4-*.json` (regression verdicts), the appended
   `docs/benchmarks/throughput.md` blocks, and the two engine logs. The
   listening pass (`out/bench/listening-12/`) is the owner gate and is not part
   of this code half.

**Limitations / concerns**

- The HTTP response has no chunk metadata; chunks/request is the client-side
  splitter's expected count, while observed batching comes from the engine DEBUG
  log. A response-side chunk count would need a protocol change (out of scope).
- The composition cells need the DEBUG boot; headline throughput must use INFO
  (the documented ~3× DEBUG penalty). Do not mix them.
- The matrix uses 8 requests per level (long-form requests are 30–60 s of
  audio). Pass `--longform-requests 20` to match the short-text flood
  convention if the host budget allows; the report records the value used.
- The cross-process identity axis disables the watermark (Perth conv kernels are
  not bit-reproducible across processes). The in-process 12-2 gate remains the
  bit-exact proof; this axis is the family-breadth re-run.
- `summary.md` and the listening pass are untouched per task scope; the host
  measurement, H1–H4 verdicts and the docs cells remain `_tbd_` until the GPU 1
  session.

**Full-suite result:** `1471 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, 2026-11-09; baseline 1450 + 21 new tests: 9 fixture /
scenario / payload / markdown / family-selection self-checks and 12
regression-axis unit tests).

---

### 2026-11-09 — host benchmark findings, verdicts and fixes (follow-up)

The first host run of the two new axes produced three failure findings. All
three were investigated on this machine with direct evidence (real KittenTTS
ONNX inference, real Silero VAD, the in-process engine stubs) before any code
changed. Verdicts first, then fixes.

#### Finding 1 — `longform:off-vs-on`, `kittentts/tts_en` FAIL

Reported: `max_abs_diff 2.455`, `corr 0.9922`, `logmel MSE 2.46`.

**Chunk-splitting hypothesis: REFUTED.** The hypothesis named
`LONGFORM_AXIS_CHUNK_SIZE=120`, but that constant belongs to the *determinism*
axis; the identity axis deliberately leaves `chunk_size` unset, so its ON side
splits at the serving layer's `DEFAULT_CHUNK_SIZE=300`. Every TTS case text used
by the axis was split with the serving layer's own `split_text_into_chunks`:
all 12 case texts are 64 chars / one sentence and yield **exactly one chunk**
at 300 (and at 120 — the longest is 64 chars), so `_resolve_chunks` returns
`None` and both sides take the untouched single-shot path. The axis premise held
for every family — the reported failure was not a synthesis-boundary difference.

**Real cause: the same model-level sampling as Finding 3.** With the text
one chunk on both sides, the only way the outputs differ is that the model
itself draws new noise per run. `kitten-tts-mini-0.8`'s ONNX graph contains
`RandomNormalLike` / `RandomUniformLike` nodes **without a seed attribute**;
two fresh processes on the same sentence differ by max |Δsample| 0.14 with
`corr 0.9919`, `logmel MSE 2.454` and identical duration — the reported
`logmel MSE 2.46` / `corr 0.9922` reproduce exactly. (The reported
`max_abs_diff 2.455` equals the log-mel MSE to three digits and is impossible
for PCM-16 clips decoded into [-1, 1], where |Δ| ≤ 2.0; it looks like a
mislabelled field in the host summary. The measured sample difference is
~0.14–0.23.)

**Fix (harness).** The identity axis now *checks its premise* instead of
assuming it: `_longform_identity_cases` splits every case text at
`LONGFORM_IDENTITY_CHUNK_BUDGET` (pinned to the serving `DEFAULT_CHUNK_SIZE` by
a unit test) and excludes any multi-chunk case with a recorded `skip_reason`
(`partition_optional_cases` now reports such cases separately; a family whose
cases are all excluded is reported as `skipped`, never as an empty green). It
also verifies that `LONGFORM_CHUNKED_TEXT` still splits at the determinism
budget. KittenTTS is now classified as a model that samples every run
(`MODEL_SAMPLES_EVERY_RUN`): its identity/determinism verdict is distributional
(hard duration invariant + stochastic RMS/LTAS bounds), and the report uses the
stochastic thresholds so it cannot contradict its own verdict. The 4-2 identity
row that claimed a bit-identical kitten run is annotated in
`docs/benchmarks/audio-regression.md`; it is not reproducible today (the
`onnxruntime` pin is open, and ORT exposes no RNG seed, so the engine cannot
make the model deterministic).

#### Finding 2 — `longform:off-vs-on`, `openvoice_v1/clone_voice_b` ERROR

Reported: `RuntimeError: request failed: stack expects a non-empty TensorList`.

**Confirmed, both halves.** Silero VAD at 16 kHz returns **zero** speech
segments for *both* synthetic fixtures (`voice_a` and `voice_b`), while
`repo_voice_a` / `repo_voice_b` return 1–2 segments (measured with the
installed `silero_vad` + `get_speech_timestamps`). The OV `vad → embed` step
therefore handed the tone-converter encoder an empty segment list, and
`torch.stack([])` raised the opaque torch error. The same is true of the
`clone_voice_a` fixture — it only escapes because that case already uses
`repo_voice_a`.

**Fix (engine).** `run_converter_encoder` now raises
`ValueError("reference clip contains no detectable speech (VAD segments
empty)")` before any STFT work. The serving layer turns a handler error into a
400 with that message, so a client gets a diagnosable error instead of a torch
stack. The batched path keeps its existing fall-through to the loop, which
produces the same error for that request only (the batched call is unaffected).

**Fix (fixtures + harness).** `tests/audio_fixtures` gained
`VAD_UNDETECTABLE_CLIPS` / `vad_detectable_reference()` and a dated provenance
note: the synthetic clips are tones, not speech, so they must not be used as a
reference for a VAD-carried path; their bytes are unchanged because other axes
pin them. On the identity axis, clone cases on `VAD_CARRIED_FAMILIES`
(`openvoice_v1`, `openvoice_v2`) are retargeted to the real-speech clip of the
same role (`voice_b → repo_voice_b`, `voice_a → repo_voice_a`) with the
substitution recorded in the case label. Chatterbox/HarmonySpeech condition on
the clip directly and keep their synthetic references.

#### Finding 3 — `determinism:run1-vs-run2`, `kittentts/tts_en` FAIL

Reported: `logmel MSE 2.99`, `max_abs_diff 2.46`, `corr 0.9923`, duration Δ 0.0%.

**Verdict: neither an engine chunk→result pairing bug nor a wrong-file
comparison — the model samples on every inference.** Evidence:

1. **Direct model probe.** `KittenTTSSynthesizer("KittenML/kitten-tts-mini-0.8")`
   on one sentence, twice in one process: not equal, max |Δ| 0.188. Two fresh
   processes: not equal, max |Δ| 0.142, `corr 0.9919`, identical length
   (147400 samples).
2. **The exact failing workload, model-side.** The six chunks of
   `LONGFORM_CHUNKED_TEXT` at 120 chars, synthesised twice and concatenated:
   `duration_pct 0.0`, `rms_pct 0.027`, `logmel MSE 3.065`, `corr 0.9922`,
   `ltas_rmse 0.097 dB`. That reproduces the reported spectral/correlation
   signature (`~3.0 dB²`, `0.9923`) and the 0.0 % duration, at the model level,
   with no engine or harness in the loop.
3. **Graph inspection.** `onnx.load` shows two random nodes
   (`RandomNormalLike`, `RandomUniformLike`) with no `seed` attribute. ORT
   exposes no RNG seed, so this is not something the engine can pin.
4. **Pairing invariant is intact.** `run_in_lanes` returns results in input
   order (its own unit tests pin that), and `_execute_kittentts_synthesizer`
   assembles `lane_results[i]` onto `requests_to_batch[i]`. A crossing would
   swap whole blocks and change the stitched length (chunk texts differ in
   length), which contradicts the measured 0.0 % duration. The reported
   `max_abs_diff 2.46` again exceeds what PCM-16 clips in [-1, 1] can produce
   and equals the log-mel MSE scale; the measured sample difference for the
   6-chunk pair is 0.23.

**Fix (harness).** Same `MODEL_SAMPLES_EVERY_RUN` classification as Finding 1:
the determinism axis keeps **duration as a hard invariant** (it is what proves
the chunk count, pairing and stitch length did not move — the model's duration
predictor is deterministic; measured 0.0 % across runs) and gates RMS/LTAS with
the stochastic bounds. A crossed or duplicated chunk changes the stitched
length and the LTAS, so the gate still catches the failure it exists for.

**Fix (engine regression test).** Added
`test_kitten_chunked_lanes_pair_by_index_and_repeat_byte_identically`
(`tests/integration/test_tts_chunking.py`): a text-encoding KittenTTS stub with
`max_batch_size=4` runs a 4-chunk request twice through the real engine. It
asserts the four chunk texts reach the model in index order on both runs, the
step really admitted 4 rows across ≥2 lane threads (the lane path, not the
loop), the stitched bytes are identical across runs, and each block of the
stitched buffer carries its own chunk's audio. A chunk→result crossing fails
it. (`KittenStubModel`'s constant output could not see a crossing, which is why
the earlier seed test did not catch one.)

#### Files changed

* `harmonyspeech/task_handler/model_runner_base.py` — honest VAD-empty
  `ValueError` in `run_converter_encoder` (and the batched-path comment).
* `harmonyspeech/modeling/models/kittentts/onnx_model.py` — dated note: the
  graphs are stochastic by construction; no ORT seed exists.
* `scripts/audio_regression.py` — `LONGFORM_IDENTITY_CHUNK_BUDGET`,
  `_longform_identity_cases` (1-chunk premise + VAD-capable reference
  retargeting), `MODEL_SAMPLES_EVERY_RUN` / `VAD_CARRIED_FAMILIES` /
  `REAL_SPEECH_REFERENCE`, `partition_optional_cases` `skip_reason` support,
  distributional identity/determinism verdict for sampling models, all-cases-
  excluded is a recorded skip not a green, determinism fixture split assertion.
* `tests/audio_fixtures/__init__.py` — `VAD_UNDETECTABLE_CLIPS`,
  `vad_detectable_reference`, dated VAD-detectability provenance note.
* `tests/unit/benchmarks/test_audio_regression_longform.py` — +8 tests: the
  1-chunk premise for every family, budget==serving constant, multi-chunk case
  exclusion with reason, VAD reference retargeting (and Chatterbox keeps its
  synthetic clip), `skip_reason` partitioning, distributional sampling-model
  verdict, hard-duration failure.
* `tests/unit/inference_flow/test_batched_embedder_handlers.py` — the
  no-segment error now asserts the honest message; added the single-request
  loop-path variant.
* `tests/integration/test_tts_chunking.py` — +1 lane-pairing/determinism
  regression test.
* `docs/benchmarks/audio-regression.md` — correction note under the identity
  table.

**Full-suite result (2026-11-09, follow-up):** `1495 passed, 5 skipped`
(`tests/unit tests/integration tests/e2e/wire_contract`; baseline 1485 + 10 new
tests). No GPU gates were run here; the orchestrator re-runs the two host axes.

**Local end-to-end validation (CPU, 2026-11-09).** The axes were re-run through
the real harness (subprocess workers, real weights) on this machine, not only
unit-tested:

* `--families kittentts --axis longform:off-vs-on` → **PASS**:
  duration Δ 0.0 %, RMS Δ 0.207 %, log-mel MSE 2.51, LTAS 0.113 dB,
  corr 0.9914, identical False.
* `--families kittentts --axis determinism:run1-vs-run2` → **PASS**:
  duration Δ 0.0 %, RMS Δ 0.095 %, log-mel MSE 3.068, LTAS 0.088 dB,
  corr 0.9922, identical False — the same signature the host reported, now
  scored by the gate that matches the model's behaviour.
* `--families openvoice_v1 --axis longform:off-vs-on` → **PASS**, both cases
  bit-identical (duration/RMS/log-mel/LTAS all 0.0, corr 1.0). `clone_voice_b`
  ran with the retargeted `repo_voice_b`.
* Raw `voice_b` reference (no retargeting) through the real serving layer now
  fails with `reference clip contains no detectable speech (VAD segments
  empty)` (HTTP 400 path) instead of `stack expects a non-empty TensorList`.


## Host session record — 2026-11-09 (GPU 1, benchmark host)

**Config:** `out/bench/config-phase12.yml` (six families at shipped settings + silero-vad so the `openvoice_v2` pipeline composes; `request_timeout_s: 600` bench-only headroom — the shipped default 120s stays). Host env: `CUDA_VISIBLE_DEVICES=1`, `HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S=600` for the levels below (see finding F1); OV2 off-mode levels re-run with `--timeout 900` (see F2). Listening pairs (crossfade 20 ms / pause 80 ms, deliberately audible) under `out/bench/listening-12/` — **owner gate pending**.

### Hypothesis verdicts

- **H1 (floods gain) — SPLIT verdict.** CONFIRMED for padded-batch classes: melo c=8 throughput 0.09 → 1.31 req/s (**14.6×**; unchunked falls off a cliff at c≥8, p50 92 s); openvoice_v2 clone c=8 0.02 → 0.18 req/s (**~10×**, request latency 402 s → 36 s). FALSIFIED for AR admission classes under flood: chatterbox 0.26 → 0.11 req/s, turbo 0.52 → 0.14 — chunk sub-requests from concurrent requests convoy on the width-4 admission (per-request p50 balloons to 48 s). Loop class (multilingual, control) degrades as expected (serial executor, nothing to co-batch).
- **H2 (single request gains) — CONFIRMED** on every batched family: chatterbox 16.2 → 10.0 s (1.62×; 4 chunks co-fill one AR batch), turbo 8.0 → 7.2 s, melo 1.13 → 0.84 s, openvoice_v2 1.40 → 1.23 s, kitten 21.2 → 7.1 s (**3.0×**). Multilingual (loop) degrades 5.7 → 19.3 s — chunking is a net loss for that class today.
- **H3 (stitch overhead negligible) — recorded, not isolated.** crossfade 50 ms + pause 50 ms probe on chatterbox c=1: p50 11.7 s vs 10.1 s baseline — within AR run-to-run variance; the probe is a sanity bound, not an isolation. CPU stitcher cost remains bounded by the 12-3 microbenches (~ms/audio-minute).
- **H4 (filters ≈ free) — CONFIRMED.** All three filters on, melo c=1/4: +2–3 % wall, within noise.

### Findings (F1–F7)

- **F1 — engine iteration watchdog vs unchunked long-form:** the 120 s default (`HARMONYSPEECH_ENGINE_ITERATION_TIMEOUT_S`) is exceeded by legitimate unchunked OV2 clone steps at c≥8 (one step batches 8 × 41.6 s tone conversions); after 3 consecutive trips the engine marks itself errored and everything after fails instantly — the "permanent degradation" first observed. Chunked mode never trips it (steps stay ¼-sized): an underrated robustness property of chunking. Measured with the watchdog at 600 s; default-config cliff documented in throughput.md. Backlog note: watchdog should scale with admitted work, not wall-clock alone.
- **F2 — bench client timeout:** `--timeout` default 300 s < unchunked OV2 c≥8 request times (p50 ~400 s). The first run recorded client timeouts as "failures"; re-ran with `--timeout 900` for true cells. Honest cells now in §Phase 12.
- **F3 — two 12-2 follow-up fixes during the session (in-tree, tested):** (a) chunk sub-requests inherited the global 120 s request timeout and expired in queue under load (`request_timeout_s × (chunks+1)` override now); (b) chunked clone on a *concrete synthesizer card* built a standalone-embed sub-request the card rejects (`ov2-synthesizer-en` via `--model` override) — conditioning mode now resolves per pipeline vs concrete card.
- **F4 — `openvoice_v2` bench needs the pipeline name:** the family's concrete-card override bypasses VAD/embed/tone-convert (a plain MeloTTS request). SileroVAD added to the bench config so the real pipeline composes and advertises; measurements use `openvoice_v2` proper.
- **F5 — KittenTTS stochasticity correction:** the ONNX graphs contain `RandomNormalLike`/`RandomUniformLike` with **no seed attribute** — the model samples every inference; ORT exposes no RNG seed. The 4-2 "deterministic" classification was wrong (docs/benchmarks/audio-regression.md carries the correction). Both gate "failures" were this; the harness now scores kitten with a distributional verdict (hard duration + stochastic spectral tolerances).
- **F6 — VAD-undetectable synthetic fixtures:** `voice_a`/`voice_b` (24 kHz, quiet) yield 0 Silero segments at 16 kHz (repo wanda4/wanda5 detect fine). The engine now fails honestly (`reference clip contains no detectable speech (VAD segments empty)`) and the longform axis retargets clone cases to repo clips. Fixture module carries a dated warning.
- **F7 — cosmetic:** the gate exits 1 on PASS when optional cases are skipped (pre-existing convention, the German clip) — verdicts read from the printed overall/JSON.
- **Turbo chunked length artifact:** chunked turbo emits ~35 % more audio (51.6 s vs 38.0 s for the same text) — pacing difference, flagged for the owner listening pass.
- **Gates:** identity `longform:off-vs-on` and determinism `determinism:run1-vs-run2` both **PASS** at family breadth on the host (8 families each; JSONs `out/bench/gate-12-4-*.json`).

## Owner listening verdict — 2026-11-09 (listening gate)

The owner listened to the `out/bench/listening-12/` pairs (crossfade 20 ms / pause 80 ms, deliberately audible) and returned three verdicts. Follow-up work is Phase 12-5.

1. **Chatterbox chunked length is CORRECT behaviour (recorded as expected/good, no code change).** The unchunked long-form baseline "loses it towards the end and blabbles before breaking up": the OFF render is degenerate on long text, which also explains OFF audio being shorter and the multilingual family's variable 6–29 s outputs. Chunking is therefore a **quality/coherence win for autoregressive long-form**, not only a throughput lever. Recorded here and in the `docs/benchmarks/throughput.md` §Phase 12 method; H1/H2 remain throughput claims, this is a quality note.
2. **Joins stutter** — "small hiccup like silence stuttering where clips get connected." Root cause: unset `crossfade_ms`/`sentence_pause_ms` defaulted to 0/0 (hard butt splices) and the stitcher did no per-chunk edge conditioning, so each join carried whatever ragged leading/trailing silence the synthesis run emitted. **Fixed in Phase 12-5**: stitcher-internal edge conditioning (10 ms guard, −45 dBFS relative gate, 150 ms per-edge bound) plus tuned 40 ms/120 ms defaults when chunking is on and the fields are unset. Explicit values still win; the join-gap uniformity test pins it.
3. **Word-salad fixtures are unlistenable.** The arithmetic long-form generator produced gibberish. **Fixed in Phase 12-5**: the benchmark fixture is now the owner-supplied real-prose extract from *The Villainess Malfoy* (807 chars, 8 sentences, 4 chunks at the 300-char budget), a module literal shared by every family; pinned SHA-256 and chunk count updated. `audio_regression.py`'s axis texts stay synthetic on purpose (numeric identity/determinism purpose, never listened to) — decision recorded in the 12-5 log.

---

## Post-implementation audit — false assumptions in the 12-4 AR/Multilingual verdicts (2026-11-09)

> ⚠️ **DO NOT trust the H1/H2 AR and Multilingual benchmark interpretations above without reading this section.** Code audit on 2026-11-09 found that the Chatterbox numbers in the 12-4 matrix were distorted by four engine bugs, which are addressed in **12-6..12-9**:

1. **`split=off` on `chatterbox` and `chatterbox_turbo` at `c >= 4` was artificially fast because `ARBatchRunner` silently truncated the audio at `cap_tokens = 768` (`ar_runner.py:727, 820`).**
   - At `c=1` (`len(inputs) == 1`), `_execute_chatterbox_tts_ar` bypassed `ARBatchRunner` and ran the sequential loop (`max_new_tokens = 1000`).
   - At `c >= 4` (`len(inputs) == 4`), `ARBatchRunner` ran with `self.arena.cap_tokens = 768` (total prefill + decode). For `chatterbox_turbo`, the un-resampled conditioning prompt (`speech_cond_prompt_len = 375`, `use_perceiver_resampler = False`) plus ~180 text tokens for the 807-char paragraph consumed ~556 tokens of the 768-token slot, leaving only **~212 speech tokens (~8.5 s of audio)** before `ARBatchRunner` hard-truncated the request with `finish_reason = "length"`! Comparing truncated ~8.5 s `split=off` renders against full ~40 s `split=on` renders made `split=off` req/s look artificially high and falsified H1 for the wrong reason. See [12-8-ChatterboxARRunnerAndS3GenBatchFixes.md](12-8-ChatterboxARRunnerAndS3GenBatchFixes.md).
2. **Phase 9-5's batched S3Gen vocode (`_finalize_ar_rows`, `model_runner_base.py:1017`) never actually batched on `split=on` chunks.**
   - `_execute_chatterbox_tts_ar` calls `_finalize_ar_rows` on **every single token step** for only the rows that hit EOS on that exact token index $t$. Because the 4 chunks of a paragraph have different lengths and finish on different token steps, `len(finished)` was almost always `1`, so `inference_batched` was bypassed and each chunk ran a solo ~150–350 ms S3Gen vocode synchronously in the middle of the T3 token decode, stalling the remaining active T3 rows. See [12-8](12-8-ChatterboxARRunnerAndS3GenBatchFixes.md).
3. **Per-token engine-step & `prepare_inputs` `ThreadPoolExecutor` churn (`model_runner_base.py:1240`, `inputs.py:765, 812`).**
   - `ARBatchRunner` decodes only 1 speech token per `execute_model` call and returns to the asyncio loop (`asyncio.wait_for` → `Scheduler.schedule()` → `loop.run_in_executor` → `execute_model` → `asyncio.sleep(0)`). On **every single token step**, `_execute_partition` re-ran `prepare_inputs` for all continuing requests, creating and destroying a brand-new `ThreadPoolExecutor` per token even though `_execute_chatterbox_tts_ar` ignores `inputs` for continuing rows. See [12-8](12-8-ChatterboxARRunnerAndS3GenBatchFixes.md).
4. **`chatterbox_multilingual` suffered from a wrong `repetition_penalty=1.2` default (`inputs.py:859`, upstream is `2.0`), an unbounded forward-hook leak in `AlignmentStreamAnalyzer` (`alignment_stream_analyzer.py:84`), and unnecessary exclusion from SDPA / `ARBatchRunner`.**
   - Every chunk call leaked 3 GPU→CPU `.cpu()` attention hooks onto `self.tfmr` forever, causing progressive latency collapse across runs (55 s → 91 s → 227 s). See [12-7-ChatterboxMultilingualFixesAndBatching.md](12-7-ChatterboxMultilingualFixesAndBatching.md).

The extended benchmark harness (`--scenario longform`) and Whisper fidelity gate (`--fidelity`) remain valid and are kept; [12-9-LongFormBenchmarkAndFidelityReverification.md](12-9-LongFormBenchmarkAndFidelityReverification.md) re-runs the matrix and fidelity gate once 12-6..12-8 land.


