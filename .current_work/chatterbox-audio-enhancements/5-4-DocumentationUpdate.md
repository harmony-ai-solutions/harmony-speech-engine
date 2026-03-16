# Phase 5-4: Documentation Update

## Objective

Update the project documentation to reflect all new features added in Phases 1–4:
- New `split_text` / `chunk_size` request fields
- New `normalize_audio` / `crossfade_ms` / `sentence_pause_ms` `GenerationOptions` fields
- `norm_loudness` removal (replaced by `normalize_audio`)
- `speed` field behaviour clarification (post-synthesis for non-native-speed models)
- New `harmonyspeech/common/audio_utils.py` module

---

## Files to Modify

| File | Changes |
|---|---|
| `docs/api.md` | Document new `TextToSpeechRequest` fields and updated `GenerationOptions` fields |
| `docs/models.md` | Note which model types support native speed vs post-synthesis speed |
| `CHANGELOG.md` | Add a changelog entry for the new features |
| `README.md` | Check if any TTS quick-start examples reference `norm_loudness` and update them |

---

## 1. Changes to `docs/api.md`

Read the file first to identify the section covering `TextToSpeechRequest` and `GenerationOptions`.

### In the `TextToSpeechRequest` fields table, add:

| Field | Type | Default | Description |
|---|---|---|---|
| `split_text` | `bool` | `false` | When `true`, long input text is split into sentence-level chunks, each synthesized independently, then stitched with equal-power crossfade. Recommended for texts longer than ~200 characters. |
| `chunk_size` | `int` | `200` | Maximum character count per chunk when `split_text=true`. Splits are always made at sentence boundaries; `chunk_size` controls the target maximum. |

### In the `GenerationOptions` fields table:

**Remove** the `norm_loudness` row.

**Add / update**:

| Field | Type | Default | Description |
|---|---|---|---|
| `normalize_audio` | `bool` | `null` | When `true`, peak-normalizes the final audio to prevent clipping. If the peak amplitude exceeds 0.99, the audio is scaled to a peak of 0.95. Applies to all TTS model types. Replaces the former `norm_loudness` field. |
| `crossfade_ms` | `int` | `50` | Crossfade duration in milliseconds used when stitching chunked audio segments together. Only relevant when `split_text=true`. |
| `sentence_pause_ms` | `int` | `0` | Optional silence gap (in milliseconds) inserted between stitched audio chunks. Only relevant when `split_text=true`. |
| `speed` | `float` | `1.0` | Playback speed multiplier. For models with native speed support (HarmonySpeech, OpenVoice V1, MeloTTS, KittenTTS), speed is applied at synthesis time. For Chatterbox models, pitch-preserving time-stretch is applied post-synthesis via librosa. |

---

## 2. Changes to `docs/models.md`

Read the file first to locate the Chatterbox model section and the speed-related documentation.

### Under each Chatterbox model entry (ChatterboxTTS, ChatterboxTurboTTS, ChatterboxMultilingualTTS), add a note:

> **Speed control**: These models do not have native speed support. Setting `speed` in `GenerationOptions` applies a post-synthesis pitch-preserving time-stretch using `librosa.effects.time_stretch`. Values below 1.0 slow down speech; values above 1.0 speed it up.

### Under models with native speed support (HarmonySpeech, OpenVoice V1, MeloTTS, KittenTTS), add a note:

> **Speed control**: This model handles speed natively at synthesis time. Post-synthesis time-stretch is not applied.

### Add a general note about text chunking (or in the TTS section overview):

> **Text chunking**: All TTS models support `split_text=true` in `TextToSpeechRequest`. When enabled, the serving layer splits the input at sentence boundaries, synthesizes each chunk independently, then stitches them with equal-power crossfade. This improves quality and prosody for long-form text.

---

## 3. Changes to `CHANGELOG.md`

Add a new entry at the top of the changelog (after the `[Unreleased]` heading if present, otherwise as the latest entry). Follow the existing format.

```markdown
## [Unreleased]

### Added
- Text chunking and smart crossfade stitching for all TTS models: set `split_text=true` and optionally `chunk_size` in `TextToSpeechRequest` to split long input at sentence boundaries and stitch audio with equal-power crossfade.
- Post-synthesis speed factor for Chatterbox models: setting `speed` in `GenerationOptions` now applies pitch-preserving time-stretch via librosa. Other model types continue to use native speed control.
- Peak normalization option for all TTS models: set `normalize_audio=true` in `GenerationOptions` to prevent audio clipping on output.
- New `crossfade_ms` and `sentence_pause_ms` fields in `GenerationOptions` to control stitching behaviour.

### Changed
- `norm_loudness` field in `GenerationOptions` has been renamed to `normalize_audio`. The old field name is no longer accepted.
```

---

## 4. Changes to `README.md`

Read the file first to check for any occurrence of `norm_loudness`. If found:
- Replace all occurrences of `norm_loudness` with `normalize_audio` in code examples or descriptions.

If the README contains a TTS usage example, check whether `split_text` would be a useful addition to showcase.

---

## Progress Checklist

- [ ] Read `docs/api.md` to locate `TextToSpeechRequest` and `GenerationOptions` sections
- [ ] Add `split_text` and `chunk_size` to the `TextToSpeechRequest` fields table in `docs/api.md`
- [ ] Remove `norm_loudness` from `GenerationOptions` table in `docs/api.md`
- [ ] Add `normalize_audio`, `crossfade_ms`, `sentence_pause_ms` to `GenerationOptions` table in `docs/api.md`
- [ ] Update `speed` field description in `docs/api.md` to note native vs post-synthesis behaviour
- [ ] Read `docs/models.md` to locate Chatterbox and native-speed model sections
- [ ] Add speed-control notes to Chatterbox model entries in `docs/models.md`
- [ ] Add speed-control notes to native-speed model entries in `docs/models.md`
- [ ] Add text chunking note to `docs/models.md`
- [ ] Read `CHANGELOG.md` to determine existing format and insert new entry
- [ ] Read `README.md` and replace any `norm_loudness` occurrences with `normalize_audio`
