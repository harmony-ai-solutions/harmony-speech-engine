# Phase 14-1: Streaming Protocol + Admission Validation

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS (OpenAI `stream_format` contract)
> **Depends on:** **Phase 12-2** (chunk loop + carry stitcher — hard gate; 14-x lands after it)
> **Codebase mapping consulted:** streaming research 2026-11-08 (code-expert agent, this session); `.planning/codebase/ARCHITECTURE.md`
> **State reference:** verified 2026-11-08 (research) — `stream` output option currently rejected at `serving_text_to_speech.py:42-46` ("Stream output is not yet supported") and documented as a 400 at `docs/api.md:81`; `AudioOutputOptions` at `endpoints/openai/protocol.py:112-127`; commented `StreamingResponse` scaffolding at `api_server.py:291-292`; OpenAI reference: `/v1/audio/speech` streams with `stream_format: "audio"|"sse"` (default `audio`), SSE events `speech.audio.delta` (`audio` field — **not** `delta`) / `speech.audio.done`.

## Objective

The wire surface for streaming: one new output option, one deprecated alias, honest admission rules, and the response headers clients need — all validation **before** synthesis starts.

## Owner decisions (2026-11-08)

- **Both envelopes in the MVP**: `stream_format: "audio"` (chunked binary) and `"sse"` — one frame generator, two thin envelopes (14-3/14-4).
- **`stream: true` accepted as a deprecated alias** for `stream_format="audio"` (kokoro/vLLM-Omni client compat; the documented 400 becomes behavior — update `docs/api.md:81`).
- **Unsupported format + streaming ⇒ 400 before synthesis** (e.g. `format: "mp3"` with any `stream_format`), message naming the `wav`/`pcm` restriction. No silent downgrades.
- **`output_options.sample_rate` works while streaming** (stateful resampler in 14-2 — owner opt-in, telephony 8 k/16 k in mind; deviates from the research's defer recommendation, recorded here).

## Implementation steps

1. Protocol: `stream_format: Literal["audio","sse"] | None = None` on `AudioOutputOptions` (+ the three input layers, the 12-1 precedent); `stream: bool` accepted as deprecated alias ⇒ `"audio"`; a `stream_format` without streaming intent is just… streaming (OpenAI's body streams by default once a format is chosen — mirror that: setting `stream_format` at all selects streaming).
2. Admission validation (in serving, pre-synthesis, all failures ⇒ today's JSON error shape): streaming requires `format in {"wav","pcm"}`; `sample_rate` **allowed** (any rate the codec layer accepts — the 14-2 resampler handles it); `speed` post-stretch on non-native models composes (stretch happens per chunk in the streaming path — verify with 14-3).
3. Response headers for both envelopes: `X-Audio-Format`, `X-Audio-Sample-Rate` (the **effective/target** rate), `X-Audio-Channels`, `X-Audio-Bits`, `X-Request-Id`, plus streaming hygiene (`X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`) — exact set pinned by 14-5 contract tests.
4. Docs: `docs/api.md` — replace the `stream: true` 400 line with the full streaming section (both envelopes, restrictions, header table); OpenAPI models updated.
5. Tests: validation matrix unit/integration (alias mapping; wav/pcm accepted; mp3+stream ⇒ 400 naming the restriction; `sample_rate` accepted; non-streaming requests byte-identical to today).

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
