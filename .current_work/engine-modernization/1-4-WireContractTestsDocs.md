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

- [ ] Contract-matrix e2e test exists and passes (with documented ffmpeg-gated skips)
- [ ] "Audio I/O contract" docs page written and linked
- [ ] README + CHANGELOG updated
- [ ] memory-bank + CONCERNS updated
- [ ] Full suite recorded; Phase 1 ticked in summary.md
