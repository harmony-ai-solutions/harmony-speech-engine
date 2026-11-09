# Phase 13-1: Standard Voice Pack + Catalog + Request Routing

> **Plan:** [Engine Modernization](summary.md) · **Phase:** 13 — Standard voice pack & voices endpoint
> **Depends on:** 7-2 embedding cache (live — pack embeddings ride it); nothing else
> **Codebase mapping consulted:** `.planning/codebase/ARCHITECTURE.md`; upstream voice-pack layout read 2026-11-08
> **State reference:** verified 2026-11-08 — upstream pack `.current_work/Chatterbox-TTS-Server/voices/*.wav` (~20 clips, names `Abigail`…`Everett`…; MIT `LICENSE` in repo root); embed endpoint `POST /v1/embed/speaker` (`api_server.py:300`); `input_embedding` request path exists in protocol; VC takes `target_audio`/`target_embedding`; per-family `voices:` config keys exist for native-voice models.

## Objective

**Default speaker identities for the service that stay consistent across every voice-cloning pipeline**: the same named voice (e.g. `Abigail`) resolves to the same reference clip and therefore the same-sounding speaker on chatterbox en/turbo/multilingual (and as a VC target), OpenVoice v1/v2, and HarmonySpeech hs1 — plus a uniform catalog that also lists each model's **native** voices. Users keep supplying their own references/embeddings unchanged (owner decision 2026-11-08).

## Owner decisions (2026-11-08)

- **Keep upstream voice names** — MIT permits it; no renaming (earlier rename idea dropped).
- Scope = **voice-cloning pipelines only**; single-speaker models (melo, kitten-tts, hs1 single-speaker modes) physically cannot adopt external references — they list their native voices only, honestly.
- The consistency claim in docs is **speaker identity similarity** (same reference clip per family's own embedding space), *not* identical audio across families — each renderer sounds like itself.

## Context (self-contained)

- **Embedding spaces are per family** — chatterbox base group shares one VoiceEncoder, turbo owns its own, OpenVoice uses its speaker encoder, hs1 its encoder. So nothing is embedded "once globally": the pack ships **WAVs**, and each family embeds on demand through its **existing** cloning path (13-2 adds warm-up + disk cache). Zero new embedding logic in this task.
- **Routing is name resolution, nothing deeper:** on a cloning-capable model, `voice: "Abigail"` (where `voice` is currently validated/ignored per family — verify current validation per family first) resolves to the pack WAV and enters the existing `voice_cloning` path as the reference; the 7-2 cache keys on content hash, so repeat uses are compute-free even before 13-2. On VC, a `target_voice` name resolves the same way. On non-cloning models a pack name must **fail honestly** (400, unknown voice for this model) — never silently map to a native voice.
- **Vendoring:** copy the WAVs into the package (e.g. `harmonyspeech/assets/voices/`), verify sizes/durations first (~20 clips; if any clip is oddly long/short, note and keep verbatim anyway — they are the identity), include the upstream MIT license text + provenance note (`docs/models.md` or a `NOTICE` block in the pack dir). Confirm package-data inclusion in the build config.
- **Catalog module** (`harmonyspeech/modeling/voice_catalog.py` or `common/`): per model — `native` entries (from config `voices:` keys + code-level language options, verify where native voice lists live per family) + `standard` entries (pack names) for cloning-capable model types only. One source of truth, consumed by routing (here), warm-up (13-2) and the endpoint (13-3).

## Implementation steps

1. Verify-first: per-family `voice` validation today (which families validate against config lists, which ignore); native-voice sources (config keys vs `LanguageOptions`); exact set of cloning-capable model types (expected: `ChatterboxTTS`, `ChatterboxTurboTTS`, `ChatterboxMultilingualTTS`, `ChatterboxVC` (target), `OpenVoice` v1/v2 types, hs1 voice-clone mode — confirm each accepts reference audio/embedding on the wire).
   - Verified 2026-11-09 (commit `8b7d222`): all three Chatterbox TTS families are now `ar_decode` members in `AR_INFERENCE_FAMILIES` / `AR_INFERENCE_MODEL_TYPES`, including Multilingual (12-7). That changes only how a cloning request is *decoded*, not how its reference is embedded — the 7-2 embed step and its cache key are unchanged, so 13-2's warm-up targets the same artifacts as before. `ChatterboxVC` is a `padded_batch` member (11-1) and takes `target_audio`/`target_embedding` as a target, not a source.
2. Vendor the pack + license/provenance; package-data wiring; sizes recorded here.
3. Catalog module + resolution API (`voices_for_model(model_config)`, `resolve_voice(model_config, name) → {kind, wav_path | native_id}`).
4. Routing: `voice`/`target_voice` name resolution into the existing cloning paths (serving layer; no runner changes); honest 400 on non-cloning models; config knob `standard_voice_pack: true` (global, default on — it only adds resolvable names). **Resolution point (post-12, 2026-11-09):** resolve once per HTTP request **before** `from_wire_audio`/`_resolve_chunks` (`serving_text_to_speech.py:210-229`) — i.e. a pack name becomes the request's `input_audio` (+ cloning mode) right at dispatch. The 12-2 chunked dispatcher then reads the resolved reference from `input_audio`/artifacts exactly like a client-supplied clip, `voice_id` already rides chunk sub-requests (`_build_chunk_request:544`, `common/inputs.py:507`), and the embed-once `_reference_conditioning_mode` dispatch applies unchanged — no per-chunk resolution, no pack-WAV re-reads, single-shot and chunked paths see the same thing.
5. Tests: catalog unit (listing per model type; melo/kitten have no standard entries; names stable); integration — `voice="Abigail"` on chatterbox ⇒ 200 + embed step executed once + 7-2 cache hit on repeat; on VC as `target_voice` ⇒ 200; on melo ⇒ 400 with a clear message; **pack voice + `split_text: true` ⇒ one embed across all chunks** (extend the 12-2 embed-once pattern in `tests/integration/test_tts_chunking.py`); e2e smoke on 2 families asserting the same pack voice works on both.
6. Docs: `docs/models.md` (pack section: identity claim, MIT provenance, name list), `docs/api.md` (`voice` semantics per family), CHANGELOG with 13-2/13-3.

## Files to create/modify

| File | Change |
|---|---|
| `harmonyspeech/assets/voices/*` (+ LICENSE/provenance) | NEW — vendored pack |
| `harmonyspeech/modeling/voice_catalog.py` (or `common/`) | NEW — catalog + resolution |
| `harmonyspeech/endpoints/openai/serving_text_to_speech.py`, `serving_voice_conversion.py` | name resolution into cloning paths |
| `harmonyspeech/common/config.py` (+ both configs) | `standard_voice_pack` knob |
| tests (unit/integration/e2e), docs ×2, CHANGELOG | gates + honest docs |

## Tests / validation

- Full suites green; no new skips; wire contracts unchanged (additive field semantics only).

## Checklist

- [x] Per-family verification recorded here (validation today, cloning-capable set)
- [x] Pack vendored + provenance; package-data verified
- [x] Catalog + routing landed; honest 400 on non-cloning models
- [x] Cache-ride + cross-family smoke tests green; docs/CHANGELOG; `summary.md` ticked

## Deviations / verification log

Dated 2026-11-09 (Phase 13-1 implementation).

**Verified (step 1 — per-family `voice` validation and cloning capability)**

- **Where native voices live.** Native voice ids are the per-family `voices:`
  config keys: OpenVoice V1 styles (`default`, `whispering`, …), MeloTTS
  language speaker ids (`EN-Newest`, `ZH`, …), and the eight KittenTTS names.
  The Chatterbox families and Harmony Speech have **no** `voices:` keys (their
  cards expose no native presets); Chatterbox Multilingual's 23 entries are
  code-level *language* options, not voices.
- **Validation today.** `OpenAIServing._check_model` validates `language` and
  `voice` only against the model card's `languages[].voices` lists. Chatterbox
  and Harmony Speech cards have none, so `voice` was previously ignored there;
  OpenVoice/Melo/Kitten cards validate it. `ChatterboxVC` requests carry no
  `language`/`voice` field, so no card validation applies.
- **Cloning-capable set (verified on the wire).** All three Chatterbox TTS
  families accept `input_audio`/`input_embedding` (`Chatterbox*TTS.prepare_inputs`),
  `ChatterboxVC` accepts `target_audio`/`target_embedding`, the OpenVoice V1/V2
  pipelines clone via `vad → embed → synthesize → tone_convert`, and the
  `harmonyspeech` pipeline clones via `encoder → synthesizer → vocoder`.
  **Capability is a pipeline property**: a concrete synthesizer (MeloTTS, the
  OpenVoice V1 synthesizer, the Harmony Speech synthesizer, KittenTTS) never
  reads a reference on its own and is not capable, even though it is a member of
  a capable pipeline. The catalog models this structurally (pipeline offers
  `tts` and has a step that produces `embedding_artifact`), so it cannot drift
  into a hand-maintained model-type list.
- **OpenVoice VC nuance.** The OpenVoice tone converters consume
  `target_embedding` and ignore `target_audio`, so `target_voice` is supported
  only by target-audio VC models (`ChatterboxVC`). An OpenVoice VC
  `target_voice` is an honest 400 that says so. Supporting it would require
  running the pack WAV through the OpenVoice encoder first, which is new
  embedding logic and explicitly out of scope.

**Vendored pack (step 2)**

- 28 clips (not ~20 as the 2026-11-08 state reference estimated), 44.1 kHz mono
  16-bit PCM WAV, durations 7.255–9.729 s, sizes 639,894–858,156 bytes, 19.51 MiB
  total. No clip was oddly long/short; all are kept byte-for-byte.
- Vendored to `harmonyspeech/assets/voices/` with the upstream MIT `LICENSE` and
  a `PROVENANCE.md` (source, date, format, name list, byte-for-byte policy).
- Package data wired in **both** `pyproject.toml` (`[tool.setuptools.package-data]`)
  and `setup.py` (`package_data`): `assets/voices/*.wav`,
  `assets/voices/LICENSE`, `assets/voices/PROVENANCE.md`. A wheel build was not
  run (per the no-build-mid-task convention); the two manifests are the
  authoritative wiring.

**Catalog + routing (steps 3–4)**

- New `harmonyspeech/modeling/voice_catalog.py`: pack inventory
  (`standard_voice_names`, `standard_voice_path`), per-model listing
  (`voices_for_model`, `voice_catalog`), per-model resolution
  (`resolve_voice`), and request-level resolution
  (`request_cloning_capable`, `request_vc_target_audio_capable`,
  `request_native_voices`, `resolve_request_voice`, `lookup_standard_voice`).
- New `harmonyspeech/endpoints/openai/standard_voice.py`: the base64 wire
  payload helper. It deliberately lives at the boundary — the engine-core
  serialization test forbids wire encoding under `modeling/`.
- Routing resolves a pack name **once per HTTP request, before**
  `from_wire_audio`/`_resolve_chunks`: the resolved WAV is injected as the
  request's `input_audio` (VC: `target_audio`) and the mode is derived as
  `voice_cloning`. The 12-2 chunked dispatcher then sees a client-supplied
  reference (artifacts + `voice_id` on chunk sub-requests unchanged), so the
  embed-once path and the 7-2 content-hash cache apply without modification.
  No runner changes.
- Honest 400s: pack name on a non-cloning model (lists native voices), pack
  name with an explicit reference (`input_audio`/`input_embedding`, or
  `input_vad_data`, which describes a different clip), pack disabled
  (`standard_voice_pack: false`), `target_voice` on a non-target-audio VC model,
  and unknown model (deferred to the existing 404).
- Native names keep their pre-13 path. Native precedence over a pack name on a
  collision is pinned; pack names are matched exactly (capitalised upstream
  names).
- `standard_voice_pack: true` added to `EngineConfig`, `HarmonySpeechEngine`
  and both shipped configs (default on).

**Deviations**

- **`_check_model` was not modified.** GitNexus impact analysis reported
  **CRITICAL** risk (6 direct callers across every serving module; 12 affected
  execution flows). The pack voice is instead validated through an *effective
  request copy* (`input_audio` + `mode="voice_cloning"` + the native default
  voice id) that mirrors exactly what the serving layer dispatches. This keeps
  the shared admission method byte-identical for every other endpoint.
- **GitNexus index is stale.** The repo index was last built at `91bb954`
  (12 commits behind HEAD `cc2479b`), so impact results carry that caveat; the
  `_check_model` blast radius was additionally confirmed by grep
  (6 direct call sites) and by reading every serving module. 12-6 precedent.
- **Pack name implies cloning mode.** A pack name is accepted with either
  declared `mode` (and even with an omitted mode) and dispatched as
  `voice_cloning`; the documented mode contract is otherwise unchanged. This
  follows the task's "pack name becomes `input_audio` (+ cloning mode)" note.
- **Native source voice default.** OpenVoice V1/V2 and MeloTTS still need a
  native *source* voice id for their synthesizer step; a pack voice uses
  `"default"` when configured, else the first native voice for the request's
  language. Chatterbox/Harmony Speech ignore `voice_id`, so it is cleared.
- **`target_voice` on OpenVoice VC is unsupported** (see above) — honest 400
  rather than new embedding logic.
- **Real-model pack tests are gated.** `tests/e2e/tts/test_standard_voice_pack.py`
  runs only with `HSE_E2E_MODEL_TESTS=1` and is outside the default suite path
  (the `tests/e2e/wire_contract` convention). The hermetic cross-family test in
  `tests/integration/test_tts_chunking.py` runs in the default suite and is the
  gate that is actually exercised here.

**Full-suite result:** `1612 passed, 5 skipped` (`tests/unit tests/integration
tests/e2e/wire_contract`, 2026-11-09). 38 new tests are in that path (31 unit
catalog/wire + 7 integration); the same run collects no new skips. The Phase 12
figure quoted in the brief (1485) predates the Phase 12-9 test additions: HEAD
`cc2479b` was at 1574 passed / 5 skipped before this task (1612 − 38). The three
gated real-model pack tests live outside the run path.

## Addendum (2026-11-09)

**Owner decision.** Close the 13-1 gap recorded under "Deviations": `target_voice`
on an OpenVoice V1/V2 voice-conversion request must work. The wire asymmetry is
real — the tone converters consume `target_embedding` and ignore `target_audio` —
so the serving layer now resolves the pack WAV through the pipeline's own
`vad → embed` steps and uses the returned artifact as `target_embedding`. This is
**not new embedding logic**: it is the same encoder artifact the TTS clone path
computes, so it rides the Phase 7-2 in-memory cache and the 13-2 persistent
warm-up cache. Owner approved 2026-11-09; the previous "honest 400" is replaced.

**Capability model (introspectable, not a boolean).** `voice_catalog.py` now
keeps the two VC target-voice kinds apart:

- `VC_TARGET_AUDIO` (`"target_audio"`) — `ChatterboxVC` consumes the clip itself;
  the serving layer injects `target_audio` exactly as before (byte-identical).
- `VC_TARGET_EMBEDDING` (`"target_embedding"`) — the OpenVoice V1/V2 tone
  converters (`DIRECT_VC_TARGET_EMBEDDING_MODEL_TYPES`). A standard `target_voice`
  is embedded first.
- `ModelVoiceCatalog.vc_target_kind` exposes the kind per configured model for
  13-3's voices endpoint; `cloning_capable` stays "a pack name is meaningful".
- Request-level `request_vc_target_voice_kind(...)` returns the kind only when the
  target can actually accept a pack name; `resolve_target_embedding_route(...)`
  returns the pipeline id + configured encoder executor, and requires the whole
  mandatory embed path (VAD + encoder) to be configured. Without either, the
  request stays an honest 400 (no half-run).
- `StandardVoiceLookup.vc_target_kind` carries the decision to the serving layer;
  `resolve_request_voice(target=True)` accepts either kind.

**Routing.** `serving_voice_conversion.convert_voice` resolves the pack name once
per HTTP request, before the VC request enters the engine. For target-audio VC it
keeps the existing `target_audio` injection. For target-embedding VC it builds
**one** `SpeechEmbeddingRequestInput`, names the pipeline id as `requested_model`
(the engine's `embed_from_audio` entry rule needs the pipeline so the VAD step
runs; a concrete encoder name would skip it), consumes the stream, and puts the
returned `SpeakerArtifact` into the artifact store as `target_embedding`
(`target_audio` is never set). A directly named tone converter resolves its
sibling pipeline id the same way, so the cache key is identical whether the
client names `openvoice_v2` or `ov2-tone-converter`. Every existing honest 400 is
retained: pack disabled, unknown voice, explicit `target_audio`/`target_embedding`
plus `target_voice`, and non-VC/non-embed-capable models.

**Tests.** New hermetic integration coverage in
`tests/integration/test_tts_chunking.py`:

- pack `voice` on the `openvoice_v2` TTS clone path: resolves, one encoder
  forward, native source voice default (`EN-Newest`) applied, repeat is a 7-2 hit;
- pack `voice` on `harmonyspeech` clone mode: encoder → synth → vocoder, one
  encoder forward, repeat is a 7-2 hit;
- OV VC `target_voice`: cold = one encoder forward, warm = zero; the VC request
  carries `target_embedding` and never `target_audio`;
- a VC request after the 13-2 pack warm-up performs zero encoder forwards (the
  VC path shares the warm artifact, not just the in-request 7-2 entry);
- a directly named `ov2-tone-converter` routes its embedding through
  `openvoice_v2`;
- a lone tone converter with no encoder/VAD is an honest 400.

The type-strict harness gained a HarmonySpeech encoder stub and one-segment
encoder rows (the fake VAD yields one segment); the tone-converter stub now
mirrors the real VC prepare (`source_audio` + `target_embedding`, target used as
source). Catalog unit tests pin both kinds and the configured-path requirement;
the wire-contract error assertion moved to the new message.

**GitNexus.** The index is stale (built at `91bb954`, 12 commits behind), so the
13-1 symbols are absent from it. `convert_voice` impact was still available:
**LOW** (one direct caller, `api_server.convert_voice`; five execution flows).
The catalog/routing changes were validated by grep plus the full suite (12-6
precedent).

**Full-suite result (addendum):** `1641 passed, 5 skipped` (`tests/unit
tests/integration tests/e2e/wire_contract`, 2026-11-09) — +10 tests over the
13-2 baseline of 1631/5 (4 catalog unit additions after the parametrisation
change, 6 integration: OV2 TTS pack, harmonyspeech pack, OV VC cold/warm, OV VC
after pack warm-up, concrete OV2 VC, no-encoder 400). No new skips.

