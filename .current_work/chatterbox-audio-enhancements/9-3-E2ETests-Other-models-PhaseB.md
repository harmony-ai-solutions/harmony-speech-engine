# Phase 9-3: E2E Tests — Other Model Suites (Phase B)

## Objective

Extend the **four non-Chatterbox e2e test files** with smoke tests for the Phase B post-processing features that apply to them:

- **`trim_silence`** — applies to **all** model types (model-agnostic, applied in serving layer on final audio)
- **`fix_internal_silence`** — applies to **all** model types
- **`remove_dc_offset`** — primarily useful with `split_text=True` (per-chunk before stitching); also accepted for single-chunk synthesis (no-op if only one chunk)
- **Combined flag test** — all Phase B flags enabled together with `split_text=True`

No new test files are created — tests are appended to the existing per-model files.

## Files to Modify

| File | Changes |
|---|---|
| `tests/e2e/tts/test_harmonyspeech.py` | Add: `trim_silence`, `fix_internal_silence`, combined flags |
| `tests/e2e/tts/test_kittentts.py` | Add: `trim_silence`, `fix_internal_silence`, combined flags for all 4 variants |
| `tests/e2e/tts/test_melotts.py` | Add: `trim_silence`, `fix_internal_silence`, combined flags |
| `tests/e2e/tts/test_openvoice_v1.py` | Add: `trim_silence`, `fix_internal_silence`, combined flags |

## Prerequisites

- Phase 5-3 (E2E Other Models Phase A) complete: `_decode_b64_wav()`, `LONG_TEXT` already defined in each file

---

## 1. Changes to `tests/e2e/tts/test_harmonyspeech.py`

Read the file first. The fixture is `harmonyspeech_engine` returning `(engine, serving_tts, serving_embed)`.
Model: `"harmonyspeech"`, `mode="voice_cloning"`, requires `input_audio=REFERENCE_AUDIO`.
`TEXT_INPUT` and `REFERENCE_AUDIO` are already at module level.

Add at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_tts_trim_silence(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech TTS with trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = harmonyspeech_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="harmonyspeech", input=TEXT_INPUT, mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="harmonyspeech", input=TEXT_INPUT, mode="voice_cloning",
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


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_tts_fix_internal_silence(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech TTS with fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = harmonyspeech_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="harmonyspeech", input=TEXT_INPUT, mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_tts_all_phase_b_flags(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech TTS with all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = harmonyspeech_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="harmonyspeech", input=LONG_TEXT, mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            split_text=True, chunk_size=120,
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

## 2. Changes to `tests/e2e/tts/test_kittentts.py`

Read the file first. All four variants use `voice="Jasper"`, `language="default"`, `mode="single_speaker_tts"`.

Add the following at the end of the file (one `trim_silence`, one `fix_internal_silence`, and one combined test per variant):

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_trim_silence(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_mini_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-mini", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-mini", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_fix_internal_silence(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_mini_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-mini", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_all_phase_b_flags(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_mini_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-mini", input=LONG_TEXT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            split_text=True, chunk_size=120,
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


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_trim_silence(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_micro_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-micro", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-micro", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_fix_internal_silence(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_micro_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-micro", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_all_phase_b_flags(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_micro_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-micro", input=LONG_TEXT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            split_text=True, chunk_size=120,
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


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_trim_silence(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_fix_internal_silence(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_all_phase_b_flags(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano", input=LONG_TEXT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            split_text=True, chunk_size=120,
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


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_trim_silence(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_int8_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano-int8", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano-int8", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_fix_internal_silence(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_int8_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano-int8", input=TEXT_INPUT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_all_phase_b_flags(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_int8_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano-int8", input=LONG_TEXT, mode="single_speaker_tts",
            voice="Jasper", language="default",
            split_text=True, chunk_size=120,
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

## 3. Changes to `tests/e2e/tts/test_melotts.py`

Read the file first. Fixture `melotts_en_engine` returns `(engine, serving_tts, serving_embed, serving_vc)`.
Model: `"ov2-synthesizer-en"`, `language="EN"`, `voice="EN-Newest"`.

Add at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_trim_silence(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov2-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="EN-Newest",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov2-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="EN-Newest",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_fix_internal_silence(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov2-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="EN-Newest",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_all_phase_b_flags(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov2-synthesizer-en", input=LONG_TEXT, mode="single_speaker_tts",
            language="EN", voice="EN-Newest",
            split_text=True, chunk_size=120,
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

### MeloTTS EN — Voice Cloning (`mode="voice_cloning"`, toolchain `"openvoice_v2"`)

MeloTTS supports voice cloning via the `"openvoice_v2"` toolchain name with `input_audio=REFERENCE_AUDIO`.

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_trim_silence_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v2", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="EN-Newest",
            input_audio=REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v2", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="EN-Newest",
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


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_fix_internal_silence_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v2", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="EN-Newest",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_all_phase_b_flags_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v2", input=LONG_TEXT, mode="voice_cloning",
            language="EN", voice="EN-Newest",
            input_audio=REFERENCE_AUDIO,
            split_text=True, chunk_size=120,
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

## 4. Changes to `tests/e2e/tts/test_openvoice_v1.py`

Read the file first. Fixture `openvoice_v1_en_engine` returns `(engine, serving_tts, serving_embed, serving_vc)`.
Model: `"ov1-synthesizer-en"`, `language="EN"`, `voice="default"`.

Add at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_trim_silence(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov1-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="default",
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov1-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="default",
            generation_options=GenerationOptions(trim_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
    baseline_audio, _ = _decode_b64_wav(baseline.data)
    assert len(audio) <= len(baseline_audio)


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_fix_internal_silence(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov1-synthesizer-en", input=TEXT_INPUT, mode="single_speaker_tts",
            language="EN", voice="default",
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_all_phase_b_flags(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov1-synthesizer-en", input=LONG_TEXT, mode="single_speaker_tts",
            language="EN", voice="default",
            split_text=True, chunk_size=120,
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

### OpenVoice V1 EN — Voice Cloning (`mode="voice_cloning"`, toolchain `"openvoice_v1"`)

OpenVoice V1 supports voice cloning via the `"openvoice_v1"` toolchain name with `input_audio=REFERENCE_AUDIO`.

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_trim_silence_voice_cloning(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN voice cloning: trim_silence=True produces valid audio <= baseline length."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    baseline = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v1", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="default",
            input_audio=REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v1", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="default",
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


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_fix_internal_silence_voice_cloning(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN voice cloning: fix_internal_silence=True produces valid audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v1", input=TEXT_INPUT, mode="voice_cloning",
            language="EN", voice="default",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(fix_internal_silence=True, normalize_audio=False),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_all_phase_b_flags_voice_cloning(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN voice cloning: all Phase B flags + split_text: valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v1", input=LONG_TEXT, mode="voice_cloning",
            language="EN", voice="default",
            input_audio=REFERENCE_AUDIO,
            split_text=True, chunk_size=120,
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

**`test_harmonyspeech.py` additions:**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `TEXT_INPUT`, `REFERENCE_AUDIO` present
- [ ] Add `test_harmonyspeech_tts_trim_silence`
- [ ] Add `test_harmonyspeech_tts_fix_internal_silence`
- [ ] Add `test_harmonyspeech_tts_all_phase_b_flags`

**`test_kittentts.py` additions:**
- [ ] Read file to confirm `_decode_b64_wav` and `LONG_TEXT` present
- [ ] Add `test_kittentts_mini_trim_silence`
- [ ] Add `test_kittentts_mini_fix_internal_silence`
- [ ] Add `test_kittentts_mini_all_phase_b_flags`
- [ ] Add `test_kittentts_micro_trim_silence`
- [ ] Add `test_kittentts_micro_fix_internal_silence`
- [ ] Add `test_kittentts_micro_all_phase_b_flags`
- [ ] Add `test_kittentts_nano_trim_silence`
- [ ] Add `test_kittentts_nano_fix_internal_silence`
- [ ] Add `test_kittentts_nano_all_phase_b_flags`
- [ ] Add `test_kittentts_nano_int8_trim_silence`
- [ ] Add `test_kittentts_nano_int8_fix_internal_silence`
- [ ] Add `test_kittentts_nano_int8_all_phase_b_flags`

**`test_melotts.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `REFERENCE_AUDIO` present
- [ ] Add `test_melotts_en_trim_silence` (single_speaker_tts)
- [ ] Add `test_melotts_en_fix_internal_silence` (single_speaker_tts)
- [ ] Add `test_melotts_en_all_phase_b_flags` (single_speaker_tts)
- [ ] Add `test_melotts_en_trim_silence_voice_cloning`
- [ ] Add `test_melotts_en_fix_internal_silence_voice_cloning`
- [ ] Add `test_melotts_en_all_phase_b_flags_voice_cloning`

**`test_openvoice_v1.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file to confirm `_decode_b64_wav`, `LONG_TEXT`, `REFERENCE_AUDIO` present
- [ ] Add `test_openvoice_v1_trim_silence` (single_speaker_tts)
- [ ] Add `test_openvoice_v1_fix_internal_silence` (single_speaker_tts)
- [ ] Add `test_openvoice_v1_all_phase_b_flags` (single_speaker_tts)
- [ ] Add `test_openvoice_v1_trim_silence_voice_cloning`
- [ ] Add `test_openvoice_v1_fix_internal_silence_voice_cloning`
- [ ] Add `test_openvoice_v1_all_phase_b_flags_voice_cloning`

**Validation:**
- [ ] Run `pytest tests/e2e/tts/test_harmonyspeech.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_kittentts.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_melotts.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_openvoice_v1.py -v -m "e2e"` — new tests pass