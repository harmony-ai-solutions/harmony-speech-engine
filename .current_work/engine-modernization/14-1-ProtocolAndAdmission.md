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

- [x] Protocol + alias landed; validation matrix green
- [x] Headers defined; docs honest (400 line replaced)
- [x] Non-streaming wire byte-identical; `summary.md` ticked

## Verification log (2026-11-10)

**Anchors re-verified against `7357358` (tree moved since 2026-11-09; Phase 13 landed)**

- The old rejection is at `serving_text_to_speech.py:235-239` (not `:200-204`;
  Phase 13 added the voice-pack admission above it), text `"Stream output is not
  yet supported"`.
- `AudioOutputOptions` is at `protocol.py:219-235`; `NATIVE_SPEED_MODEL_TYPES`
  at `serving_text_to_speech.py:91-98` (unchanged content: HarmonySpeech,
  OpenVoice V1, MeloTTS, KittenTTS).
- The documented 400 is `docs/api.md:164` (not `:81`); `docs/audio-io-contract.md:173`
  also carried it and was updated too.
- `docs/api/openapi.json` is **already stale** for the Phase 12 fields (it has
  no `split_text`/`chunk_size`/`trim_silence`/`remove_dc_offset`), i.e. 12-1 did
  not regenerate it. Left untouched (recorded as a deviation), since regenerating
  it would bundle unrelated Phase 12/13 schema drift into this task.

**What changed**

- NEW `harmonyspeech/endpoints/openai/streaming.py` (14-2 will extend this
  module): `STREAM_FORMATS`, `STREAMING_AUDIO_FORMATS` (`wav`/`pcm`),
  `STREAMING_AUDIO_CHANNELS`/`BITS`, the seven header-name constants,
  `STREAMING_RESPONSE_HEADERS` (canonical ordered set), `resolve_stream_format`
  (pure; accepts the pydantic or plain input-layer options object) and
  `streaming_response_headers(...)` (builds the full set, rejects non-wav/pcm).
- `protocol.py`: `AudioOutputOptions.stream_format: Literal["audio","sse"] | None
  = None`; `stream` kept and documented as the deprecated alias for `"audio"`.
- `common/inputs.py`: `TextToSpeechAudioOutputOptions.stream_format` mirrors the
  wire field. `task_handler/inputs.py` needs no change: `from_openai` passes the
  pydantic options object through whole (the 12-1 precedent recorded for
  generation options applies identically here).
- `serving_text_to_speech.py`:
  - pure module-level `streaming_admission_error(request, *, uses_native_speed)`
    implementing (a) wav/pcm-only, (c) `trim_silence`, (d)
    `fix_internal_silence`, (e) non-native `speed != 1`; `sample_rate` and
    `remove_dc_offset` deliberately pass;
  - `_validate_streaming_admission(request)` as the single pre-synthesis gate
    (14-3's dispatcher can call it directly);
  - `_request_uses_native_speed` split into `_uses_native_speed(requested_model,
    model_name)` + the existing wrapper, so admission can resolve the family
    from the wire request before `request_data` exists;
  - the old rejection replaced by admission + an explicit interim 501 (see
    deviations).
- Docs: `docs/api.md` AudioOutputOptions table now documents `stream_format` /
  deprecated `stream`, plus a `#### Streaming output` section (both envelopes,
  header table, final admission table); `docs/audio-io-contract.md` row updated.
- Tests: NEW `tests/unit/wire_contract/test_streaming_protocol.py` (44 tests:
  protocol/literal, alias + format resolution, header set, full pure admission
  matrix, HTTP admission through the wire client, native-speed wire case,
  non-streaming byte-identity), NEW
  `tests/integration/test_tts_streaming_admission.py` (11 tests through the real
  serving object + 12-2 `EngineHarness`: no synthesis on any rejection, native
  vs non-native speed from configured model types, alias equivalence, buffered
  path unchanged). `test_multipart.py::test_stream_output_still_rejected`
  renamed/updated to the new contract.

**Blast radius (gitnexus, index at `7357358`)**

- `impact(create_text_to_speech, upstream)` → LOW, 0 direct callers, 0 affected
  processes; `impact(_request_uses_native_speed, upstream)` → LOW, 0 direct.
  The route calls the serving object through `api_server.openai_serving_tts`, so
  the graph has no static caller edge; grep confirms the only call sites are
  `api_server.create_speech` and the tests that drive it.
- `detect_changes(unstaged)` → 45 changed symbols / 6 files, 0 affected
  processes, risk `low`. No renamed symbols (the new method split keeps
  `_request_uses_native_speed` intact for the chunked path).

**Deviations (with reasons)**

1. **Interim 501 guard.** The admission contract is final, but 14-2/14-3/14-4
   are explicitly out of scope here. Leaving the old rejection after admission
   would contradict the new contract; falling through would return a buffered
   JSON 200 for a request that asked to stream (a silent downgrade). An admitted
   streaming request therefore answers **501 `NotImplementedError`** ("Streaming
   output is not available in this build yet.") until 14-3's dispatcher replaces
   the guard. This is not an admission failure and not a 400.
2. **`speed` rule is literal `!= 1`.** Per the owner decision ("`speed` not in
   {None, 1}"), non-positive values are also rejected on non-native families
   while streaming. The buffered chunked path silently ignores `speed <= 0`; for
   streaming we refuse rather than silently ignore (no silent downgrades).
3. **`task_handler/inputs.py` untouched** — 12-1 precedent (whole-object
   pass-through); there is no output-options type in that layer.
4. **OpenAPI JSON not regenerated** — pre-existing staleness, 12-1 precedent
   (see anchors above). The pydantic models are the spec source, so the live
   `/openapi.json` reflects the new fields automatically.
5. **No CHANGELOG entry.** Streaming is not user-visible until 14-3/14-4; an
   entry now would advertise an unimplemented envelope. Land it with 14-3.
6. **`docs/api.md` section placement.** The full streaming section lives under
   `### AudioOutputOptions` (Common Types) next to the field table; the TTS
   endpoint links to it through `output_options`.

**Test counts**

- Before: `1694 passed, 5 skipped` (baseline in the task brief).
- After: `1749 passed, 5 skipped` (**+55**: 44 unit + 11 integration; the
  multipart test was renamed, not added).

```
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit tests/integration tests/e2e/wire_contract -q --no-header
```

Targeted reproduce:

```
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m pytest tests/unit/wire_contract/test_streaming_protocol.py tests/integration/test_tts_streaming_admission.py tests/unit/wire_contract/test_multipart.py -q --no-header
& "C:\Users\sge20\miniconda3\envs\hse\python.exe" -m ruff check harmonyspeech/endpoints/openai/streaming.py harmonyspeech/endpoints/openai/serving_text_to_speech.py harmonyspeech/endpoints/openai/protocol.py harmonyspeech/common/inputs.py tests/unit/wire_contract/test_streaming_protocol.py tests/integration/test_tts_streaming_admission.py
```

**API notes for 14-3/14-4**

- Call `resolve_stream_format(request.output_options)` to choose the dispatcher
  branch; `None` = buffered (unchanged).
- Call `self._validate_streaming_admission(request)` before synthesis; it
  returns the 400 message or `None`. Replace the interim 501 guard with the
  envelope dispatch.
- Emit `streaming_response_headers(audio_format=<wav|pcm>,
  sample_rate=<effective/target>, request_id=<id>)` on both envelopes; the
  header names/order are in `STREAMING_RESPONSE_HEADERS` for the 14-5 contract
  tests.
- 14-2 should extend `streaming.py` (not create a second module): it is already
  the pure, FastAPI-free home of the shared constants.

