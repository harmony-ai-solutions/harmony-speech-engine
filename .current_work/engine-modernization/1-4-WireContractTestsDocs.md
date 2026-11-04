# Phase 1-4: Wire-Contract Tests & Docs

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 1 — Wire contract
> **Depends on:** 1-1, 1-2, 1-3
> **Codebase mapping consulted:** `.planning/codebase/TESTING.md`, `CONVENTIONS.md`

## Objective

Consolidate Phase 1 into a coherent, documented **wire contract** — a single e2e matrix proving input formats × output formats × transports behave as specified — and close the phase's documentation duties (docs, README, CHANGELOG, memory bank).

## Steps

### 1. Contract matrix e2e test

New `tests/e2e/wire_contract/test_contract_matrix.py` (parametrized):

| Axis | Values |
|---|---|
| Input format | wav, mp3, aac (ffmpeg-gated), flac |
| Transport | JSON base64, multipart |
| Output format | default (passthrough), wav, flac, mp3, pcm |
| Output sample rate | none, 16000, 44100 |
| Endpoint | speech (cloning), transcriptions, voice/convert |

Assert: HTTP status, decodable output at expected format/rate, error cases (400 on unsupported format; 413 on oversized upload; typed error on garbage audio).

### 2. Documentation consolidation

- `docs/`: one "Audio I/O contract" page: input formats (native vs ffmpeg-dependent), upload transports with examples, output `output_options` semantics, `pcm` layout, sample-rate policy, error behavior. Link from endpoint pages.
- `README.md`: feature note (format support + multipart).
- `CHANGELOG.md`: user-facing entries for format honoring, codec support, multipart (no internal identifiers per workspace rules).

### 3. Knowledge updates

- `memory-bank/progress.md` + `activeContext.md`: Phase 1 complete.
- `.planning/codebase/CONCERNS.md`: retire any "output options ignored" / "input format undocumented" findings if listed.
- `docs/models.md`: note per-model native sample rates (feeds the resample contract) if not already accurate.

### 4. Exit review

- Full test suite run (manual, per workspace rules); record results in this file.
- Tick Phase 1 in [summary.md](summary.md).

## Checklist

- [x] Contract-matrix e2e test exists and passes (with documented ffmpeg-gated skips)
- [x] "Audio I/O contract" docs page written and linked
- [x] README + CHANGELOG updated
- [x] memory-bank + CONCERNS updated
- [x] Full suite recorded; Phase 1 ticked in summary.md

## Implementation notes & test record (2026-11-04)

### Contract matrix

`tests/e2e/wire_contract/test_contract_matrix.py` runs against the real FastAPI app with a
boundary-decoding FakeEngine (`tests/wire_contract_support.py`), so transports/codecs/formats are
verified without model downloads:

- input formats {wav, flac, mp3, aac*} × transports {json, multipart} × {transcriptions, speech cloning}
- output formats {default, wav, flac, ogg, mp3, pcm, aac*} × sample rates {none, 16000, 44100, 24000}
  for `/v1/audio/speech`, plus {flac, mp3, pcm}@16k for `/v1/voice/convert` and `/v1/audio/convert`
- errors: 400 on unsupported format/sample rate, 400 on garbage audio over both transports, 413 on oversized uploads
- aac cases are gated on `audio_codec.FFMPEG_AVAILABLE` (`*`) and skip cleanly without ffmpeg

Real-weight cases (`NEEDS_MODELS`, e.g. FasterWhisper on mp3/aac and KittenTTS flac@16k) are
`skipif`-guarded behind `HSE_E2E_MODEL_TESTS=1` and did **not** run here (no model downloads in this
task); they remain manual verification.

### Results on this machine (ffmpeg 8.0 on PATH)

| Suite | Result |
|---|---|
| `pytest tests/unit -q` | **281 passed** (162 baseline + 119 new) |
| `pytest tests/integration -q` | **23 passed** |
| `pytest tests/e2e/wire_contract -q` | **38 passed, 3 skipped** (skips = real-model cases) |

### Documentation

- New `docs/audio-io-contract.md` (transports, curl examples, input format matrix, output options,
  `pcm` layout, sample-rate policy, error behavior), linked from `docs/api.md`.
- `docs/api.md`: AudioOutputOptions rewritten, transport/format sections added, multipart examples
  for all six audio endpoints.
- `docs/models.md`: native output rates section.
- `README.md`: audio I/O feature note. `CHANGELOG.md`: user-facing entries.
- `memory-bank/{progress,activeContext}.md` updated; `CONCERNS.md` retired the "output options
  ignored" finding and marked multipart/size-limit items done.

### Deviations

- The full e2e suite still requires model downloads, so it was not executed here; only the new
  wire-contract e2e module (transport/format logic with mocked responses) plus the unit/integration
  suites were run. Real-model contract cases are skipped by default.
- Garbage-audio typed errors are asserted through the boundary-decoding FakeEngine (unit level);
  with model weights available the same path is exercised by the `NEEDS_MODELS` cases.
