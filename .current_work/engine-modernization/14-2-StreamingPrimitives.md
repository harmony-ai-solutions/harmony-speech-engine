# Phase 14-2: Streaming Primitives (frames, WAV header, SSE events, stateful resampler)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 14 — Streaming TTS
> **Depends on:** 14-1 (field names/validation)
> **State reference:** verified 2026-11-08 (research) — `audio_codec.encode_audio` is whole-buffer (`common/audio_codec.py:355-384`), `resample` is one-shot `soxr.resample` (`:177-193`) — per-chunk one-shot resampling clicks/drifts, hence the stateful stream below; ecosystem WAV-header precedent: OpenAI/Kokoro (`0xFFFFFFFF` sizes) / vLLM-Omni / Google (headerless PCM @24 k).
>
> ⚠️ **Carry reference superseded (Phase 12, verified 2026-11-09, commit `8b7d222`):** the `.current_work/Chatterbox-TTS-Server/server.py:1009-1067` carry is **not** the reference implementation any more — it has no per-chunk edge conditioning and butt-splices whatever ragged silence each render emits. The shipped carry is `harmonyspeech/common/audio_utils.py::CrossfadeStitcher` (per-chunk `condition_edges`, two-sided `cos²`/`sin²` pause fade, `feed()` + `flush()` proven sample-identical to the whole-buffer `crossfade_stitch`). 14-2's primitives feed **from** that class; do not re-implement a carry here.

## Objective

`harmonyspeech/endpoints/openai/streaming.py` — pure, I/O-free primitives both envelopes share. No FastAPI, no engine calls; everything unit-testable byte-for-byte.

## Implementation steps

1. `wav_stream_header(sample_rate, channels, bits) -> bytes`: the 44-byte RIFF header with **`0xFFFFFFFF` in both size fields** (RIFF size + data size) — the OpenAI/Kokoro-compatible streaming WAV; assert field-by-field in tests (magic, `fmt ` 16/PCM, byte rate = rate×channels×bits/8, block align). Known wart, documented: Python stdlib `wave` misreports duration; `soundfile`/`ffprobe`/browsers are fine.
2. `pcm16_frame(artifact_bytes) -> bytes`: strip container → s16le PCM payload (reuse codec-layer decoders; the artifact on the wire path is already raw-ish — verify what the serving layer actually holds per family and keep one canonical conversion).
3. **Stateful resampler** `ResampleStream` wrapper over `soxr.ResampleStream` (verify-first: confirm the installed `python-soxr` exposes `ResampleStream` and pin its exact API): `feed(pcm_chunk) -> bytes`, `flush() -> bytes`; **continuity pinned by test** — a sine sweep split at arbitrary boundaries, streamed through the wrapper, must match a one-shot `soxr.resample` of the concatenation within a tight tolerance (e.g. ≤1e-6 relative), and concatenated feed outputs + flush must equal the full buffer (no gaps/duplicates at boundaries). Owner decision 2026-11-08: this ships in the MVP (telephony 8 k/16 k).
4. SSE primitives: `sse_event(event_type, payload) -> str` (`event:` line + `data:` JSON + blank line — `event:` lines optional in the ecosystem, emit them), `delta_payload(audio_b64, chunk_index)`, `done_payload(id, model, created, sample_rate, format, chunks, duration_s, usage)` (`usage` extended with `input_characters`/`audio_seconds`, the KittenML/AssemblyAI additive precedent), `error_payload(err)`, `keepalive() -> ": keepalive\n\n"`. `data: [DONE]` emitted after `done` (chat-completions familiarity; never required by clients).
5. Unit tests: header bytes (golden hex fixture), frame extraction, resampler continuity/flush, SSE escaping (payloads with newlines/quotes), b64 round-trip.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/streaming.py` | NEW — all primitives |
| `tests/unit/endpoints/test_streaming.py` (or nearest convention) | NEW — byte-level gates |

## Checklist

- [ ] soxr `ResampleStream` verified + wrapped; continuity/flush gate green
- [ ] WAV header + frame + SSE primitives with golden-fixture tests
- [ ] `summary.md` ticked
