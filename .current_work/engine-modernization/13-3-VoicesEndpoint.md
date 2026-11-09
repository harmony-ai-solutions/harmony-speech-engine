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

- [x] Endpoint + response models landed
- [x] Wire-contract + consistency tests green
- [x] Docs/CHANGELOG/README; `summary.md` ticked (Phase 13 complete)

## Verification log (2026-11-09)

**Landed.**

- `GET /v1/audio/voices` in `api_server.py` (house listing style, next to
  `/v1/audio/speech/models`), response models in `protocol.py`
  (`VoiceCatalogList`, `ModelVoiceCatalogEntry`, `VoiceDescriptor`), and the
  listing builder `voice_catalog_listing()` in `modeling/voice_catalog.py`.
- Response shape (pinned by wire-contract tests):
  `{object: "list", data: [{model, object, model_type, cloning_capable, vc_target_kind, voices: [{name, source, languages}]}]}`.
  `object` per entry is `"model"` or `"toolchain"`; `model_type` is null on a
  toolchain entry; `vc_target_kind` is `"target_audio"`, `"target_embedding"` or
  null; `languages` is null for standard pack clips.
- Tests: `tests/unit/modeling/test_voice_catalog_listing.py` (10),
  `tests/unit/wire_contract/test_voices_catalog_contract.py` (4),
  `tests/e2e/wire_contract/test_voices_endpoint.py` (8, incl. `?model=`
  narrowing, unknown-model 400, per-model-type source labels, pack-set
  stability, disabled pack), `tests/integration/test_voices_catalog_consistency.py`
  (24; endpoint↔catalog↔routing, both VC target kinds, every listed name
  resolves, every routable name listed).
- Docs: `docs/api.md` `### Voices Catalog` (field table, curl + JSON for the
  full list / `?model=` / 400, and `target_voice` examples on both VC kinds),
  `docs/models.md` pointer, README core-feature one-liner, CHANGELOG
  user-visible entry.

**Design decisions / deviations.**

1. **Pipeline ids are their own entries** (recommended in the task): they are
   routable `model` values clients can name. Presentation mirrors
   `/v1/audio/speech/models`: a configured model named after its pipeline
   (shipped `chatterbox`, `chatterbox_vc`) is **one merged entry** with
   `object: "model"`; a pipeline id with no configured model of that name
   (`openvoice_v1`, `harmonyspeech`) gets `object: "toolchain"`,
   `model_type: null`. Only complete, voice-capable toolchains are listed, the
   same completeness rule the model cards use.
2. **Capability is request-level, not model-type-level.** Individual and
   pipeline entries use `request_cloning_capable` /
   `request_vc_target_voice_kind` (13-1 addendum), so a target-embedding tone
   converter whose VAD/encoder is not configured is honestly not capable, has
   `vc_target_kind: null`, and lists no pack names. `voices_for_model()` gained
   an additive keyword-only `include_standard` (default preserves 13-1
   behaviour) so the listing can withhold pack names without a second catalog
   implementation.
3. **The pack switch hides names, not capability.** With
   `standard_voice_pack: false`, no `standard` entries are listed; the
   capability flags still describe the model because explicit reference clips
   keep working. Documented and pinned by tests.
4. **Native voices on a toolchain are merged across configured variants** by
   name with unioned languages: `default` on `openvoice_v1` reports
   `["EN", "ZH"]` once. Individual models carry their own config language, or
   `"default"` when the config exposes voices without one (KittenTTS).
5. **Catalog source.** The route reads `model_configs` + `standard_voice_pack`
   from the running engine (the same source the serving modules use), so the
   listing is exactly what requests route to and has no warm-up dependency. A
   missing engine (app built without `run_server`) yields an empty list, not an
   error.
6. **Unknown `?model=` ⇒ 400**, documented message: "The model `<name>` has no
   voice catalog entry. Call GET /v1/audio/voices without `model` to list every
   model with selectable voices." (`type: "BadRequestError"`, `param: "model"`).
   The wording also covers a configured id that has no voice catalog (e.g. a
   Whisper model) instead of pretending it does not exist.
7. `tests/wire_contract_support.py` `WireApp`/`FakeEngine` gained additive
   `model_configs` / `standard_voice_pack` kwargs so the wire tests carry a real
   catalog through the real FastAPI route. Default behaviour is unchanged.
8. **GitNexus:** the `harmony-speech-engine` index is 12 commits behind HEAD
   and predates the uncommitted 13-1/13-2 modules, so impact analysis was done
   by grep (the 12-6/13-2 precedent): `voices_for_model` has one in-module
   caller (`voice_catalog`) plus tests; `ModelVoiceCatalog` is constructed only
   in `voice_catalog`; both edits are additive with defaults, and
   `voice_catalog_listing` is new surface with no existing callers.
9. **Not blocked on 13-2's host measurement.** The 13-2 warm-up boot-cost
   measurement remains `_tbd_` (host session); this endpoint is read-only
   catalog state and does not depend on it.

**Test counts:** full suite (`tests/unit tests/integration tests/e2e/wire_contract`)
**1687 passed, 5 skipped** (baseline before 13-3: 1641 passed, 5 skipped;
+46 new tests: 10 catalog-listing unit, 4 protocol wire-contract, 8 endpoint
wire-contract, 24 catalog↔routing consistency incl. 18 `?model=` cases).
