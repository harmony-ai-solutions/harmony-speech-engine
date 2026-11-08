# Phase 12-2: Serving-Layer Chunking + Stitching (all TTS families)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 12 — Long-form audio
> **Depends on:** 12-1 (utils + fields); reuses 7-2 embedding cache (embed once, synthesize N chunks)
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md` (serving module pattern); March plan 3-1/4-1 sub-task files superseded
> **State reference:** verified 2026-11-08 — synthesis dispatch in `endpoints/openai/serving_text_to_speech.py` (re-verify exact lines before editing); perth watermark is a per-request post-step in `task_handler/model_runner_base.py` (~:1123 per 11-1 spec — re-verify); `speed` consumed natively by some model types only (`NATIVE_SPEED_MODEL_TYPES` concept from the March plan — verify which families read `speed_function` today).

## Objective

When `split_text: true` (or `chunk_size` set): split → synthesize chunks → stitch → return **one** response. Model-agnostic: every TTS family (chatterbox en/turbo/multilingual, hs1, kitten ×4, melo, openvoice v1/v2) gets it from the serving layer, zero model-runner changes.

## Context (self-contained)

- **Chunk submission is concurrent, stitching is ordered.** All chunk sub-requests are submitted at once (asyncio.gather-equivalent on the engine path) so a multi-chunk request can fill an admission batch **by itself** — on `ar_decode` families 4 chunks = 4 rows in one step; on `padded_batch` families the step batches them; this is 12-4's measured hypothesis, do not serialize "for simplicity". Stitch by chunk index, never by completion order.
- **Clone mode embeds ONCE.** `voice_cloning`/`input_audio` requests resolve the reference through the existing embed step **one time** (7-2 cache key unchanged), then every chunk synthesizes against the same conditioning. Assert this in tests — re-embedding per chunk is the silent failure mode.
- **Watermark exactly once, on the stitched buffer.** The per-request post-step would watermark every chunk (muddled watermark, crossfaded over). Design: per-chunk synthesis with watermark suppressed + one application on the final output (reuse the model's watermarker; serving layer needs a small helper or a runner-side flag). Pin with a test that the watermark hook runs exactly once per HTTP request.
- **Per-chunk failure fails the request** — honest all-or-nothing; no partial audio with a 200.
- **Seeding:** a request seed derives per-chunk seeds deterministically (`seed + chunk_index`) so chunked output is reproducible and the 9-3 AR runner rows stay comparable; assert re-run stability.
- **Speed/normalization ordering:** native-speed models keep their native handling; non-native families get `time_stretch` after stitching (whole-output, not per-chunk); `norm_loudness`/peak normalization after stitching. `sentence_pause_ms`/`crossfade_ms` during stitch.
- **Degenerate cases:** short text (0 or 1 chunks) must be **byte-identical** to today's path (no stitcher in the pipeline) — this is the regression gate.

## Implementation steps

1. Chunking dispatcher in `serving_text_to_speech.py` (guarded by the 12-1 fields; single-chunk fast path = existing code untouched).
2. Concurrent chunk submission + ordered stitch + the watermark/seed/speed rules above; `NATIVE_SPEED_MODEL_TYPES` frozenset pinned by test (verify current native consumers of `speed_function` first — do not guess the set).
3. Tests: integration — 2-chunk and 5-chunk texts through the real engine loop (stub/tiny models, the 8-x precedent): ordered stitch, clone-embeds-once, watermark-once, chunk-failure ⇒ request failure, seed determinism; e2e per family (the March list: chatterbox, turbo, multilingual, hs1, kitten ×4, melo, openvoice — `single_speaker_tts` + `voice_cloning` where supported) asserting valid WAV + expected duration growth vs chunk count; wire-contract additions for the new fields (default-off stays identical).
4. Docs: `docs/api.md` (fields now effective), `docs/batching.md` note (chunk sub-requests are ordinary admitted requests), `CHANGELOG.md` (with 12-1).

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | dispatcher + stitch + watermark/speed rules |
| `harmonyspeech/task_handler/model_runner_base.py` (or serving-side helper) | watermark-once mechanism; no other runner changes |
| `tests/integration/test_tts_chunking.py` (NEW), `tests/e2e/tts/*` (extend) | gates above |
| `docs/api.md`, `docs/batching.md`, `CHANGELOG.md` | honest user-facing docs |

## Tests / validation

- Byte-identity gate: short text with `split_text: true` ⇒ identical bytes to `split_text: false`.
- Full suites green; no new skips.

## Checklist

- [ ] Dispatcher + ordered stitch landed (concurrent submission)
- [ ] Clone-embeds-once + watermark-once + seed derivation pinned
- [ ] Per-family e2e + integration + wire contracts green
- [ ] Byte-identity gate green; docs/CHANGELOG; `summary.md` ticked
