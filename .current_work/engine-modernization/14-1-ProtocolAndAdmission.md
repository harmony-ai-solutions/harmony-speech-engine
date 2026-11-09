# Phase 14-1: Streaming Protocol + Admission Validation

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS (OpenAI `stream_format` contract)
> **Depends on:** **Phase 12-2** (chunk loop + carry stitcher — hard gate; 14-x lands after it)
> **Codebase mapping consulted:** streaming research 2026-11-08 (code-expert agent, this session); `.planning/codebase/ARCHITECTURE.md`
> **State reference:** verified 2026-11-08 (research; re-anchored 2026-11-09) — `stream` output option currently rejected at `serving_text_to_speech.py:200-204` (was `:42-46` before Phase 12 reshaped the file; "Stream output is not yet supported") and documented as a 400 at `docs/api.md`; `AudioOutputOptions` at `endpoints/openai/protocol.py:112-127`; commented `StreamingResponse` scaffolding at `api_server.py:291-292`; OpenAI reference: `/v1/audio/speech` streams with `stream_format: "audio"|"sse"` (default `audio`), SSE events `speech.audio.delta` (`audio` field — **not** `delta`) / `speech.audio.done`.

## Objective

The wire surface for streaming: one new output option, one deprecated alias, honest admission rules, and the response headers clients need — all validation **before** synthesis starts.

## Owner decisions (2026-11-08)

- **Both envelopes in the MVP**: `stream_format: "audio"` (chunked binary) and `"sse"` — one frame generator, two thin envelopes (14-3/14-4).
- **`stream: true` accepted as a deprecated alias** for `stream_format="audio"` (kokoro/vLLM-Omni client compat; the documented 400 becomes behavior — update `docs/api.md:81`).
- **Unsupported format + streaming ⇒ 400 before synthesis** (e.g. `format: "mp3"` with any `stream_format`), message naming the `wav`/`pcm` restriction. No silent downgrades.
- **`output_options.sample_rate` works while streaming** (stateful resampler in 14-2 — owner opt-in, telephony 8 k/16 k in mind; deviates from the research's defer recommendation, recorded here).
- **Whole-buffer-only post-options + streaming ⇒ 400 before synthesis** (owner decision 2026-11-09, second post-12 pass): `trim_silence: true`, `fix_internal_silence: true`, and `speed ∉ {None, 1}` on non-native-speed families (i.e. every Chatterbox family — `NATIVE_SPEED_MODEL_TYPES` in `serving_text_to_speech.py:86-93` is exactly the four non-Chatterbox synthesizers) are rejected up front, message naming the whole-buffer-only restriction. Grounded at `8b7d222`: the buffered chunked path runs trim/internal on the *final stitched buffer* with a silence gate relative to the **global** RMS peak (`audio_utils._silent_frame_mask`, `audio_utils.py:853-859`) and stretches non-native speed whole-buffer post-stitch (`serving_text_to_speech.py:492-501`, librosa phase vocoder) — none of it can apply sample-identically to incrementally-emitted frames. `remove_dc_offset` **stays allowed** (per-chunk, stateless, pre-stitch in both paths). Revisit only as a designed streaming variant (running-peak gate, stateful stretch) — not MVP.
- **Streamed audio is watermarked per emitted segment** (owner decision 2026-11-09): the buffered path's single whole-clip perth call (`_apply_final_watermark`, `serving_text_to_speech.py:714-723`) cannot exist in streaming; each stitched, crossfaded segment gets one perth `apply_watermark` call before PCM encoding. Documented divergence: segment-wise vs whole-clip embedding — the 14-5 equality axis is therefore scoped to **pre-watermark PCM** (see 14-3/14-5).

## Implementation steps

1. Protocol: `stream_format: Literal["audio","sse"] | None = None` on `AudioOutputOptions` (+ the three input layers, the 12-1 precedent); `stream: bool` accepted as deprecated alias ⇒ `"audio"`; a `stream_format` without streaming intent is just… streaming (OpenAI's body streams by default once a format is chosen — mirror that: setting `stream_format` at all selects streaming).
2. Admission validation (in serving, pre-synthesis, all failures ⇒ today's JSON error shape): streaming requires `format in {"wav","pcm"}`; `sample_rate` **allowed** (any rate the codec layer accepts — the 14-2 resampler handles it); `trim_silence`/`fix_internal_silence`/non-native `speed` + streaming ⇒ 400 (owner decision 2026-11-09 above — the 2026-11-08 research note claimed "stretch happens per chunk in the streaming path", which the shipped code contradicts: the buffered path stretches the whole stitched buffer and Chatterbox models ignore `speed` natively, so there is no per-chunk mechanism to compose with); `remove_dc_offset` + streaming allowed.
3. Response headers for both envelopes: `X-Audio-Format`, `X-Audio-Sample-Rate` (the **effective/target** rate), `X-Audio-Channels`, `X-Audio-Bits`, `X-Request-Id`, plus streaming hygiene (`X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`) — exact set pinned by 14-5 contract tests.
4. Docs: `docs/api.md` — replace the `stream: true` 400 line with the full streaming section (both envelopes, restrictions, header table); OpenAPI models updated.
5. Tests: validation matrix unit/integration (alias mapping; wav/pcm accepted; mp3+stream ⇒ 400 naming the restriction; `trim_silence`/`fix_internal_silence`/non-native-`speed` + stream ⇒ 400 naming the whole-buffer restriction; `remove_dc_offset` + stream accepted; `sample_rate` accepted; non-streaming requests byte-identical to today).

## Files to create/modify

| File | Change |
|---|---|
| `endpoints/openai/protocol.py` + `common/inputs.py` + `task_handler/inputs.py` | `stream_format` + alias |
| `endpoints/openai/serving_text_to_speech.py` | admission validation (replaces the :42-46 rejection) |
| `docs/api.md`, OpenAPI | streaming section |

## Checklist

- [ ] Protocol + alias landed; validation matrix green
- [ ] Headers defined; docs honest (400 line replaced)
- [ ] Non-streaming wire byte-identical; `summary.md` ticked
