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
4. Routing: `voice`/`target_voice` name resolution into the existing cloning paths (serving layer; no runner changes); honest 400 on non-cloning models; config knob `standard_voice_pack: true` (global, default on — it only adds resolvable names).
5. Tests: catalog unit (listing per model type; melo/kitten have no standard entries; names stable); integration — `voice="Abigail"` on chatterbox ⇒ 200 + embed step executed once + 7-2 cache hit on repeat; on VC as `target_voice` ⇒ 200; on melo ⇒ 400 with a clear message; e2e smoke on 2 families asserting the same pack voice works on both.
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

- [ ] Per-family verification recorded here (validation today, cloning-capable set)
- [ ] Pack vendored + provenance; package-data verified
- [ ] Catalog + routing landed; honest 400 on non-cloning models
- [ ] Cache-ride + cross-family smoke tests green; docs/CHANGELOG; `summary.md` ticked
