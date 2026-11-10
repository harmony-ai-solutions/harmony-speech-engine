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

- [x] soxr `ResampleStream` verified + wrapped; continuity/flush gate green
- [x] WAV header + frame + SSE primitives with golden-fixture tests
- [x] `summary.md` ticked

## Verification log (2026-11-10)

**Changed** — `harmonyspeech/endpoints/openai/streaming.py` extended in place
(no parallel module): the module docstring now covers both phases, `__all__`
grew additively, and a new "Binary envelope primitives" + "SSE envelope
primitives" section carries `wav_stream_header`, `pcm16_frame`,
`StreamResampler` (alias `ResampleStream`), `sse_event`, `delta_payload`,
`done_payload`, `error_payload`, `keepalive`, plus the constants
`WAV_STREAM_HEADER_BYTES`, `SSE_EVENT_{DELTA,DONE,ERROR}`, `SSE_KEEPALIVE`,
`SSE_DONE_FRAME`. The 14-1 symbols (`resolve_stream_format`,
`streaming_response_headers`, every constant) are byte-identical — only the
docstring and `__all__` changed. New tests:
`tests/unit/wire_contract/test_streaming_primitives.py` (105 tests; sibling of
14-1's `test_streaming_protocol.py` rather than a new
`tests/unit/endpoints/` package, which the repo does not have — the
wire-contract convention is the established home for this seam).

**Verified `soxr` API (environment assumption pinned).** Installed
`python-soxr` **1.0.0** exposes `ResampleStream`:

* constructor — `soxr.ResampleStream(in_rate: float, out_rate: float,
  num_channels: int, dtype='float32', quality='HQ')`;
* `resample_chunk(x: np.ndarray, last=False) -> np.ndarray` — `last=True`
  flushes the filter tail and must be called once at the end; a later
  `resample_chunk` raises `RuntimeError: Input after last input` (the wrapper
  pre-empts this with its own `RuntimeError`);
* `clear()` resets the same configuration; `delay` and `num_clips` are
  **methods**, not attributes;
* mono input may be `(frames,)` or `(frames, 1)`; `dtype='float32'` matches
  the serving layer's waveform representation.

Measured: with `quality='HQ'`, streaming `resample_chunk` output equals
one-shot `soxr.resample(x, in, out, quality='HQ')` **exactly** (max abs diff
0.0) for realistic lengths at 48 k→16 k, 44.1 k→8 k, 24 k→22.05 k and
16 k→48 k. Only *total* inputs of 1–2 samples differ (soxr's own edge
handling/length choice), which is not a chunk seam; the gate uses real-length
sweeps split at 0/1/3/odd/non-block-aligned boundaries. `tests/unit/wire_contract/test_streaming_primitives.py::TestStreamResampler::test_installed_soxr_exposes_the_pinned_resample_stream_api`
re-pins the signature/defaults, so an environment change fails loudly.

**Serving-layer verification (what `_chunked` holds).** The chunked path
gathers one `AudioArtifact` per chunk and reads `audio_artifact.numpy`
(float32 mono `[samples]`, `serving_text_to_speech.py:618-633`), then the
buffered encoder runs `to_wire_audio → encode_artifact →
audio_codec.encode_audio`. So `pcm16_frame` is the **one canonical
conversion**: `audio_codec.encode_audio(waveform, rate, "pcm")` (the existing
`_to_s16le`). Only the legacy pre-artifact payload (base64 WAV `str`) and
container `bytes` go through `audio_codec.decode_wav_b64` /
`decode_audio` — the codec layer's decoders, never a second decode path.
No codec-layer or `CrossfadeStitcher` change was made (the primitives consume
the stitcher's float32 output).

**Deviations**

1. `StreamResampler.feed()`/`flush()` exchange **float32 numpy samples**, not
   bytes. 14-3's frame loop resamples between the watermark and the PCM
   encoder, and the continuity gate is a float comparison against one-shot
   `soxr.resample`; a byte-level API would quantize twice and could not meet
   the ≤1e-6 tolerance. `pcm16_frame()` is the single quantizer at the end.
   The task-file name `ResampleStream` is exported as an alias for
   discoverability.
2. `pcm16_frame()` accepts an `AudioArtifact`-like object (the verified
   serving representation), a numpy waveform, container `bytes`, or the legacy
   base64 `str`; all routes end in the same `encode_audio(..., "pcm")`.
3. `done_payload()` keeps the task-file positional signature. `usage` is
   copied (never mutated) and extended with `audio_seconds` (from
   `duration_s`, rounded to 6 dp) and `input_characters` (from the caller's
   usage when supplied, else `0`). `duration_s` is documented as coming from
   the resampler's `emitted_samples ÷ rate`, never from wire bytes.
4. `error_payload()` returns exactly `{error: {message, type, code}}` as the
   task file specifies (no top-level `type`); the event name carries the type.
5. Added `SSE_DONE_FRAME = "data: [DONE]\n\n"` so 14-4 emits the final frame
   without hand-rolling it (still a separate emission after `done`).
6. The known `wave` wart is pinned by test: `soundfile` reads the stream to
   EOF (`frames == len(samples)`) while stdlib `wave.getnframes()` reports
   `0xFFFFFFFF // 2 == 2147483647`.

**Blast radius.** `gitnexus` was available but its index is at `1243ddb` (one
commit behind `HEAD`, and it predates the untracked 14-1/14-2 files), so the
graph cannot see this module. `gitnexus.detect_changes(scope="unstaged")`
reports **0 affected processes, risk low** for the uncommitted tree. Grep
blast radius: the only importers of `endpoints.openai.streaming` are
`serving_text_to_speech.py:62` (imports `STREAMING_AUDIO_FORMATS`,
`resolve_stream_format`) and `tests/unit/wire_contract/test_streaming_protocol.py`
— both unaffected by additive names; nothing existing was renamed or removed.

**Tests.** Command:
`& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header`

* before (14-1 baseline): **1749 passed, 5 skipped**
* after: **1854 passed, 5 skipped** (+105, all in
  `test_streaming_primitives.py`)
* `ruff check` clean on both touched files (the repo does not enforce
  `ruff format`; existing files are not format-clean, so no unrelated
  reformatting was applied).

**Reproduce**

```powershell
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -c "import soxr, inspect; print(soxr.__version__); print(inspect.signature(soxr.ResampleStream.__init__)); print(inspect.signature(soxr.ResampleStream.resample_chunk))"
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/wire_contract/test_streaming_primitives.py -q --no-header
```

**Blocker:** none (installed python-soxr 1.0.0 has `ResampleStream`; no new
dependencies).
