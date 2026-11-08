# Phase 13-3: `/v1/audio/voices` Endpoint + Docs (phase exit)

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 13 — Standard voice pack & voices endpoint
> **Depends on:** 13-1 (catalog), 13-2 (pack embeddings warmed — endpoint works either way, docs assume both)
> **Codebase mapping consulted:** `api_server.py` route + `ModelList` precedents (`/v1/audio/speech/models` :293, `/v1/embed/models` :320); upstream optional-model precedent (`277e930`)
> **State reference:** verified 2026-11-08 — no voices-list route exists; per-capability model-list routes are the house style; `serving_text_to_speech.show_available_models` is the listing precedent.

## Objective

One discovery endpoint for voice selection: `GET /v1/audio/voices` returns every configured model's voice catalog (native + standard pack entries, honestly labeled); `?model=<name>` narrows to one model (the upstream `277e930` shape). Clients stop reading our config files to know what they can ask for.

## Context (self-contained)

- **Response shape** (additive, `ModelList`-consistent): `{object: "list", data: [{model, model_type, cloning_capable, voices: [{name, source: "native"|"standard", languages: [...]?}]}]}` — exact field names finalized here during review, then pinned by wire-contract tests. `source` is the honesty label: a `standard` voice on chatterbox and a `native` voice on melo mean different things and the API must not blur them.
- Per-voice preview/sample audio is **explicitly out of scope** (noted as a possible follow-up; would need caching decisions of its own).
- Unknown `?model=` ⇒ 400 with the documented message (house style: honest admission errors).
- The endpoint is read-only catalog state — no warm-up dependency, works before 13-2 hydration completes (standard voices are listable even if their embeddings aren't warm yet).

## Implementation steps

1. Route + response model in `api_server.py` (house listing style), data from the 13-1 catalog module.
2. Wire-contract tests: full-list shape, `?model=` narrowing, unknown-model 400, source labels per model type (melo/kitten ⇒ no `standard` entries), stability of the pack name set.
3. Integration: catalog consistency between the endpoint and actual routing (every listed standard voice resolves; every routable standard voice is listed).
4. Docs: `docs/api.md` endpoint reference with examples (curl + JSON), `README.md` one-liner if it lists endpoints, CHANGELOG (Phase 13 block), `summary.md` tick.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/endpoints/openai/api_server.py` | GET `/v1/audio/voices` |
| `harmonyspeech/endpoints/openai/protocol.py` | response models |
| `tests/e2e/wire_contract/` + integration | contract pins |
| `docs/api.md`, `README.md`, `CHANGELOG.md` | honest docs |

## Tests / validation

- Full suites green; catalog↔routing consistency test green.

## Checklist

- [ ] Endpoint + response models landed
- [ ] Wire-contract + consistency tests green
- [ ] Docs/CHANGELOG/README; `summary.md` ticked (Phase 13 complete)
