# Phase 14-4: SSE Envelope (`stream_format: "sse"`)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-3 (the frame generator), 14-2 (event primitives)
> **State reference:** verified 2026-11-08 (research) — OpenAI contract: `speech.audio.delta {type, audio}` → `speech.audio.done {type, usage}` (vLLM-Omni's first impl used the wrong field name `delta` and was corrected — `audio` is authoritative); `speech.audio.error` exists in vLLM-Omni/speaches; `[DONE]` emission is inconsistent ecosystem-wide (AssemblyAI documents none) — we emit it, clients must not require it; `EventSource` is GET-only and cannot POST — browsers use `fetch` + `ReadableStream` (document this).

## Objective

A thin wrapper turning 14-3's frame generator into the event stream: the "contract-bearing" mode — boundaries, end-of-stream metadata (`sample_rate`, `duration_s`, `usage`), heartbeats, and **mid-stream error events** (binary mode's truncation weakness fixed here).

## Implementation steps

1. Envelope generator: same frame source as 14-3; each frame → `speech.audio.delta` with `audio` (base64 PCM, resampled if requested) + additive `chunk_index`; terminal `speech.audio.done` with `id/model/created/sample_rate/format/chunks/duration_s/usage` (duration from the stitcher's cumulative sample counts — consume 12-2's exposure, don't recompute); then `data: [DONE]`.
2. Error semantics: pre-first-byte failure ⇒ normal HTTP JSON error; post-first-byte ⇒ `speech.audio.error {error:{message,type,code}}` then close (exactly one terminal event, `done` xor `error` — pinned by test).
3. Keepalive: `: keepalive` comment line when a chunk takes >15 s (before the first delta too — browsers/proxies keep the connection warm during first-chunk synthesis).
4. Content-Type `text/event-stream; charset=utf-8` + the 14-1 headers; no gzip.
5. Tests (unit on the envelope + integration through the route): event ordering, delta `audio` b64 round-trip, exactly-one-terminal-event, error-after-N-deltas path, keepalive emission, done-payload fields.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py` | SSE branch over the shared generator |
| `tests/` | envelope gates |

## Checklist

- [ ] Envelope + error/keepalive semantics landed
- [ ] Terminal-event + ordering + round-trip gates green
- [ ] `summary.md` ticked
