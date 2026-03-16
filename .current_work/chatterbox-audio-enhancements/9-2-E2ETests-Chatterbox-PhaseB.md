# Phase 9-2: E2E Tests — Chatterbox Suites (Phase B)

## Objective

Extend the **three existing Chatterbox e2e test files** with smoke tests for the Phase B post-processing features:

- `remove_dc_offset` — most useful with `split_text=True` (per-chunk DC removal before stitching)
- `trim_silence` — applied to final audio
- `fix_internal_silence` — applied to final audio
- Combined flag test — all Phase B flags enabled together

No new test files are created — tests are appended to the existing per-model files.

## Files to Modify

| File | Changes |
|---|---|
| `tests/e2e/tts/test_chatterbox.py` | Add: `remove_dc_offset`, `trim_silence`, `fix_internal_silence`, and combined flag tests |
| `tests/e2e/tts/test_chatterbox_turbo.py` | Add: same set of tests for ChatterboxTurboTTS |
| `tests/e2e/tts/test_chatterbox_multilingual.py` | Add: same set of tests for ChatterboxMultilingualTTS |

## Prerequisites

- Phase 5-2 (E2E Chatterbox Phase A) complete: `_decode_b64_wav()`, `LONG_TEXT`, and `GenerationOptions` are already imported/defined in each file

---

## 1. Changes to `tests/e2e/tts/test_chatterbox.py`

Read the file first to confirm:
- `_decode_b64_wav()` is present
- `LONG_TEXT` is present
- `GenerationOptions` is imported or must be added via inline import

The existing `chatterbox_engine` fixture returns `(engine, serving_tts, serving_embed, serving_vc)`.

Add the following at the end of the file:

```python
def test_chatterbox_tts_trim_silence(chatterbox_engine, mock_raw_request):
    """trim_silence=True on ChatterboxTTS: produces a valid shorter or equal-length audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    # Baseline without trimming
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts"),
        mock_raw_request,
    ))
    # With trimming
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    # Trimmed version should be <= baseline length
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


def test_chatterbox_tts_fix_internal_silence(chatterbox_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxTTS: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_tts_split_text_with_dc_offset_removal(chatterbox_engine, mock_raw_request):
    """split_text=True + remove_dc_offset=True: stitched audio is valid and non-empty."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(remove_dc_offset=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_tts_all_phase_b_flags(chatterbox_engine, mock_raw_request):
    """All Phase B flags enabled together: produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 2. Changes to `tests/e2e/tts/test_chatterbox_turbo.py`

Read the file first to confirm `_decode_b64_wav()`, `LONG_TEXT`, and fixture name.

The `chatterbox_turbo_engine` fixture returns `(engine, serving_tts, serving_embed)`.

Add at the end of the file:

```python
def test_chatterbox_turbo_tts_trim_silence(chatterbox_turbo_engine, mock_raw_request):
    """trim_silence=True on ChatterboxTurboTTS: produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts"),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


def test_chatterbox_turbo_tts_fix_internal_silence(chatterbox_turbo_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxTurboTTS: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_split_text_with_dc_offset_removal(chatterbox_turbo_engine, mock_raw_request):
    """split_text=True + remove_dc_offset=True on ChatterboxTurboTTS: stitched audio is valid."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(remove_dc_offset=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_all_phase_b_flags(chatterbox_turbo_engine, mock_raw_request):
    """All Phase B flags on ChatterboxTurboTTS: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 3. Changes to `tests/e2e/tts/test_chatterbox_multilingual.py`

Read the file first to confirm `_decode_b64_wav()`, `LONG_TEXT`, fixture name, and language parameter.

The `chatterbox_multilingual_engine` fixture returns `(engine, serving_tts, serving_embed)`.

Add at the end of the file:

```python
def test_chatterbox_multilingual_tts_trim_silence(chatterbox_multilingual_engine, mock_raw_request):
    """trim_silence=True on ChatterboxMultilingualTTS: produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="single_speaker_tts", language="en"
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            language="en",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


def test_chatterbox_multilingual_tts_fix_internal_silence(chatterbox_multilingual_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxMultilingualTTS: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            language="en",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_multilingual_tts_split_text_with_dc_offset_removal(chatterbox_multilingual_engine, mock_raw_request):
    """split_text=True + remove_dc_offset=True on ChatterboxMultilingualTTS: valid stitched audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            language="en",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(remove_dc_offset=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_multilingual_tts_all_phase_b_flags(chatterbox_multilingual_engine, mock_raw_request):
    """All Phase B flags on ChatterboxMultilingualTTS: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=LONG_TEXT,
            mode="single_speaker_tts",
            language="en",
            split_text=True,
            chunk_size=120,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 4. Voice Cloning Phase B Tests — `tests/e2e/tts/test_chatterbox.py`

Read the file first to confirm:
- `REFERENCE_AUDIO` constant is defined (or add it as `tests/test-data/samples/wanda4.wav` loaded as base64)
- `_decode_b64_wav()` is present

Add the following at the end of the file after the single-speaker Phase B tests:

```python
def test_chatterbox_tts_voice_cloning_trim_silence(chatterbox_engine, mock_raw_request):
    """trim_silence=True on ChatterboxTTS voice cloning: produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


def test_chatterbox_tts_voice_cloning_fix_internal_silence(chatterbox_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxTTS voice cloning: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_tts_voice_cloning_all_phase_b_flags(chatterbox_engine, mock_raw_request):
    """All Phase B flags on ChatterboxTTS voice cloning: produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 5. Voice Cloning Phase B Tests — `tests/e2e/tts/test_chatterbox_turbo.py`

Read the file first to confirm:
- `LONG_REFERENCE_AUDIO` constant is defined (or add it as `tests/test-data/samples/jerry_seinfeld_prompt.wav` loaded as base64)
- The embedding step pattern is present (embed first, then use `input_embedding`)
- `_decode_b64_wav()` is present

ChatterboxTurbo voice cloning requires a pre-computed embedding (the model requires audio >= 5 seconds). Use the `serving_embed` to embed first, then pass `input_embedding` to `create_text_to_speech`.

Add the following at the end of the file after the single-speaker Phase B tests:

```python
def test_chatterbox_turbo_tts_voice_cloning_trim_silence(chatterbox_turbo_engine, mock_raw_request):
    """trim_silence=True on ChatterboxTurboTTS voice cloning: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine
    # Pre-compute embedding (requires >= 5s audio)
    embed_resp = asyncio.run(serving_embed.embed_speaker(
        EmbedSpeakerRequest(model="chatterbox_turbo", input_audio=LONG_REFERENCE_AUDIO),
        mock_raw_request,
    ))
    assert isinstance(embed_resp, EmbedSpeakerResponse)
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_embedding=embed_resp.data,
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_voice_cloning_fix_internal_silence(chatterbox_turbo_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxTurboTTS voice cloning: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine
    embed_resp = asyncio.run(serving_embed.embed_speaker(
        EmbedSpeakerRequest(model="chatterbox_turbo", input_audio=LONG_REFERENCE_AUDIO),
        mock_raw_request,
    ))
    assert isinstance(embed_resp, EmbedSpeakerResponse)
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_embedding=embed_resp.data,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_voice_cloning_all_phase_b_flags(chatterbox_turbo_engine, mock_raw_request):
    """All Phase B flags on ChatterboxTurboTTS voice cloning: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine
    embed_resp = asyncio.run(serving_embed.embed_speaker(
        EmbedSpeakerRequest(model="chatterbox_turbo", input_audio=LONG_REFERENCE_AUDIO),
        mock_raw_request,
    ))
    assert isinstance(embed_resp, EmbedSpeakerResponse)
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_embedding=embed_resp.data,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 6. Voice Cloning Phase B Tests — `tests/e2e/tts/test_chatterbox_multilingual.py`

Read the file first to confirm:
- `SHORT_REFERENCE_AUDIO` or `REFERENCE_AUDIO` constant is defined (ChatterboxMultilingual uses `input_audio` directly with no minimum length)
- `_decode_b64_wav()` is present

Add the following at the end of the file after the single-speaker Phase B tests:

```python
def test_chatterbox_multilingual_tts_voice_cloning_trim_silence(chatterbox_multilingual_engine, mock_raw_request):
    """trim_silence=True on ChatterboxMultilingualTTS voice cloning: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="en",
            input_audio=SHORT_REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="en",
            input_audio=SHORT_REFERENCE_AUDIO,
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


def test_chatterbox_multilingual_tts_voice_cloning_fix_internal_silence(chatterbox_multilingual_engine, mock_raw_request):
    """fix_internal_silence=True on ChatterboxMultilingualTTS voice cloning: produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="en",
            input_audio=SHORT_REFERENCE_AUDIO,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_multilingual_tts_voice_cloning_all_phase_b_flags(chatterbox_multilingual_engine, mock_raw_request):
    """All Phase B flags on ChatterboxMultilingualTTS voice cloning: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="en",
            input_audio=SHORT_REFERENCE_AUDIO,
            generation_options=GenerationOptions(
                remove_dc_offset=True,
                trim_silence=True,
                fix_internal_silence=True,
                normalize_audio=True,
            ),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## Progress Checklist

**`test_chatterbox.py` — `single_speaker_tts` additions:**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `TEXT_INPUT` present
- [ ] Add `test_chatterbox_tts_trim_silence`
- [ ] Add `test_chatterbox_tts_fix_internal_silence`
- [ ] Add `test_chatterbox_tts_split_text_with_dc_offset_removal`
- [ ] Add `test_chatterbox_tts_all_phase_b_flags`

**`test_chatterbox.py` — `voice_cloning` additions:**
- [ ] Confirm `REFERENCE_AUDIO` constant is present (wanda4.wav as base64)
- [ ] Add `test_chatterbox_tts_voice_cloning_trim_silence`
- [ ] Add `test_chatterbox_tts_voice_cloning_fix_internal_silence`
- [ ] Add `test_chatterbox_tts_voice_cloning_all_phase_b_flags`

**`test_chatterbox_turbo.py` — `single_speaker_tts` additions:**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `TEXT_INPUT` present
- [ ] Add `test_chatterbox_turbo_tts_trim_silence`
- [ ] Add `test_chatterbox_turbo_tts_fix_internal_silence`
- [ ] Add `test_chatterbox_turbo_tts_split_text_with_dc_offset_removal`
- [ ] Add `test_chatterbox_turbo_tts_all_phase_b_flags`

**`test_chatterbox_turbo.py` — `voice_cloning` additions:**
- [ ] Confirm `LONG_REFERENCE_AUDIO` constant is present (jerry_seinfeld_prompt.wav as base64)
- [ ] Confirm `EmbedSpeakerRequest` and `EmbedSpeakerResponse` are imported
- [ ] Add `test_chatterbox_turbo_tts_voice_cloning_trim_silence`
- [ ] Add `test_chatterbox_turbo_tts_voice_cloning_fix_internal_silence`
- [ ] Add `test_chatterbox_turbo_tts_voice_cloning_all_phase_b_flags`

**`test_chatterbox_multilingual.py` — `single_speaker_tts` additions:**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `TEXT_INPUT` present
- [ ] Add `test_chatterbox_multilingual_tts_trim_silence`
- [ ] Add `test_chatterbox_multilingual_tts_fix_internal_silence`
- [ ] Add `test_chatterbox_multilingual_tts_split_text_with_dc_offset_removal`
- [ ] Add `test_chatterbox_multilingual_tts_all_phase_b_flags`

**`test_chatterbox_multilingual.py` — `voice_cloning` additions:**
- [ ] Confirm `SHORT_REFERENCE_AUDIO` constant is present (wanda4.wav as base64)
- [ ] Add `test_chatterbox_multilingual_tts_voice_cloning_trim_silence`
- [ ] Add `test_chatterbox_multilingual_tts_voice_cloning_fix_internal_silence`
- [ ] Add `test_chatterbox_multilingual_tts_voice_cloning_all_phase_b_flags`

**Validation:**
- [ ] Run `pytest tests/e2e/tts/test_chatterbox.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_chatterbox_turbo.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_chatterbox_multilingual.py -v -m "e2e"` — new tests pass
