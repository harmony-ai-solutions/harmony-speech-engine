# Phase 5-3: E2E Tests — Extend Other Model Test Suites

> **Note:** Where a model supports both `single_speaker_tts` and `voice_cloning` modes, new feature tests are provided for **both** modes. KittenTTS supports single speaker only — no voice cloning tests apply.

## Objective

Extend the **four non-Chatterbox e2e test files** with test cases for the new features that apply to them:
- **`split_text` (chunking)** — applies to **all** model types (chunking is model-agnostic, done in the serving layer)
- **`normalize_audio`** — applies to **all** model types (normalization is model-agnostic, done in the serving layer)
- **`speed` via post-synthesis time-stretch** — does **NOT** apply to these models because they are all in `NATIVE_SPEED_MODEL_TYPES` (HarmonySpeechSynthesizer, OpenVoiceV1Synthesizer, MeloTTSSynthesizer, KittenTTSSynthesizer handle speed natively in their model runner)

No new test files are created — one suite per model type, tests are added to the existing file.

## Why no speed post-synthesis tests here?

The models covered in these suites all belong to `NATIVE_SPEED_MODEL_TYPES`:
- `HarmonySpeechSynthesizer` — passes `speed` into the synthesis model directly
- `OpenVoiceV1Synthesizer` — passes `speed` into the tone converter
- `MeloTTSSynthesizer` — passes `speed` into the synthesis model directly
- `KittenTTSSynthesizer` — passes `speed` into the synthesis model directly

For these models the serving layer's post-synthesis `apply_speed_factor()` is intentionally skipped (guarded by `NATIVE_SPEED_MODEL_TYPES`). Therefore there is nothing new to test for speed in these suites — the existing single-speaker tests already exercise native speed indirectly.

---

## Files to Modify

| File | Changes |
|---|---|
| `tests/e2e/tts/test_harmonyspeech.py` | Add: chunking + normalize_audio tests for the TTS path |
| `tests/e2e/tts/test_kittentts.py` | Add: chunking + normalize_audio tests for all 4 variants |
| `tests/e2e/tts/test_melotts.py` | Add: chunking + normalize_audio tests |
| `tests/e2e/tts/test_openvoice_v1.py` | Add: chunking + normalize_audio tests |

---

## Shared helpers

Each test file needs:

```python
import io
import base64
import numpy as np
import soundfile as sf

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold."
)

def _decode_b64_wav(b64_data: str) -> tuple[np.ndarray, int]:
    """Decode a base64 WAV audio response to numpy float32 array + sample rate."""
    raw = base64.b64decode(b64_data)
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr
```

Also import `GenerationOptions` where it is not already imported:
```python
from harmonyspeech.endpoints.openai.protocol import GenerationOptions
```

---

## 1. Changes to `tests/e2e/tts/test_harmonyspeech.py`

The fixture is `harmonyspeech_engine` returning `(engine, serving_tts, serving_embed)`.
The existing `TEXT_INPUT = "Hello, world."` constant is at module level.
The existing model name used for full TTS is `"harmonyspeech"`.

Add the following tests at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_tts_split_text(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech TTS with split_text=True produces non-empty stitched audio."""
    engine, serving_tts, serving_embed = harmonyspeech_engine
    request = TextToSpeechRequest(
        model="harmonyspeech",
        input=LONG_TEXT,
        mode="voice_cloning",
        input_audio=REFERENCE_AUDIO,
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_harmonyspeech_tts_normalize_audio(harmonyspeech_engine, mock_raw_request):
    """HarmonySpeech TTS with normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = harmonyspeech_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="harmonyspeech",
            input=TEXT_INPUT,
            mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

**Note:** HarmonySpeech TTS requires `input_audio` (reference audio) for voice cloning mode. Both new tests use `mode="voice_cloning"` with `REFERENCE_AUDIO` to match the existing test pattern in this file.

---

## 2. Changes to `tests/e2e/tts/test_kittentts.py`

The file has 4 fixtures:
- `kittentts_mini_engine` → `(engine, serving_tts)`, model `"kitten-tts-mini"`
- `kittentts_micro_engine` → `(engine, serving_tts)`, model `"kitten-tts-micro"`
- `kittentts_nano_engine` → `(engine, serving_tts)`, model `"kitten-tts-nano"`
- `kittentts_nano_int8_engine` → `(engine, serving_tts)`, model `"kitten-tts-nano-int8"`

All use `voice="Jasper"` and `language="default"`.

Add these tests at the end of the file (one chunking + one normalize test per variant):

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_split_text(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: split_text=True produces non-empty stitched audio."""
    engine, serving_tts = kittentts_mini_engine
    request = TextToSpeechRequest(
        model="kitten-tts-mini",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        voice="Jasper",
        language="default",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_mini_normalize_audio(kittentts_mini_engine, mock_raw_request):
    """KittenTTS mini: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_mini_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-mini",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            voice="Jasper",
            language="default",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_split_text(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: split_text=True produces non-empty stitched audio."""
    engine, serving_tts = kittentts_micro_engine
    request = TextToSpeechRequest(
        model="kitten-tts-micro",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        voice="Jasper",
        language="default",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_micro_normalize_audio(kittentts_micro_engine, mock_raw_request):
    """KittenTTS micro: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_micro_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-micro",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            voice="Jasper",
            language="default",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_split_text(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: split_text=True produces non-empty stitched audio."""
    engine, serving_tts = kittentts_nano_engine
    request = TextToSpeechRequest(
        model="kitten-tts-nano",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        voice="Jasper",
        language="default",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_normalize_audio(kittentts_nano_engine, mock_raw_request):
    """KittenTTS nano: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            voice="Jasper",
            language="default",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_split_text(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: split_text=True produces non-empty stitched audio."""
    engine, serving_tts = kittentts_nano_int8_engine
    request = TextToSpeechRequest(
        model="kitten-tts-nano-int8",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        voice="Jasper",
        language="default",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_kittentts_nano_int8_normalize_audio(kittentts_nano_int8_engine, mock_raw_request):
    """KittenTTS nano-int8: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts = kittentts_nano_int8_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="kitten-tts-nano-int8",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            voice="Jasper",
            language="default",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 3. Changes to `tests/e2e/tts/test_melotts.py`

The fixture is `melotts_en_engine` returning `(engine, serving_tts, serving_embed, serving_vc)`.
Existing imports include `TextToSpeechRequest`, `TextToSpeechResponse`. Single-speaker model: `"ov2-synthesizer-en"`, `language="EN"`, `voice="EN-Newest"`.

Add at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_split_text(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: split_text=True produces non-empty stitched audio."""
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    request = TextToSpeechRequest(
        model="ov2-synthesizer-en",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        language="EN",
        voice="EN-Newest",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_normalize_audio(melotts_en_engine, mock_raw_request):
    """MeloTTS EN: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov2-synthesizer-en",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            language="EN",
            voice="EN-Newest",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

### MeloTTS EN — Voice Cloning (`mode="voice_cloning"`, toolchain `"openvoice_v2"`)

MeloTTS supports voice cloning via the `"openvoice_v2"` toolchain name with `input_audio=REFERENCE_AUDIO`.

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_split_text_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: split_text=True produces non-empty stitched audio."""
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    request = TextToSpeechRequest(
        model="openvoice_v2",
        input=LONG_TEXT,
        mode="voice_cloning",
        language="EN",
        voice="EN-Newest",
        input_audio=REFERENCE_AUDIO,
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_melotts_en_normalize_audio_voice_cloning(melotts_en_engine, mock_raw_request):
    """MeloTTS EN voice cloning: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = melotts_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v2",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="EN",
            voice="EN-Newest",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 4. Changes to `tests/e2e/tts/test_openvoice_v1.py`

The fixture is `openvoice_v1_en_engine` returning `(engine, serving_tts, serving_embed, serving_vc)`.
Existing imports include `TextToSpeechRequest`, `TextToSpeechResponse`. Single-speaker model: `"ov1-synthesizer-en"`, `language="EN"`, `voice="default"`.

Add at the end of the file:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_split_text(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN: split_text=True produces non-empty stitched audio."""
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    request = TextToSpeechRequest(
        model="ov1-synthesizer-en",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        language="EN",
        voice="default",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_normalize_audio(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="ov1-synthesizer-en",
            input=TEXT_INPUT,
            mode="single_speaker_tts",
            language="EN",
            voice="default",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

### OpenVoice V1 EN — Voice Cloning (`mode="voice_cloning"`, toolchain `"openvoice_v1"`)

OpenVoice V1 supports voice cloning via the `"openvoice_v1"` toolchain name with `input_audio=REFERENCE_AUDIO`.

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_split_text_voice_cloning(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN voice cloning: split_text=True produces non-empty stitched audio."""
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    request = TextToSpeechRequest(
        model="openvoice_v1",
        input=LONG_TEXT,
        mode="voice_cloning",
        language="EN",
        voice="default",
        input_audio=REFERENCE_AUDIO,
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_openvoice_v1_normalize_audio_voice_cloning(openvoice_v1_en_engine, mock_raw_request):
    """OpenVoice V1 EN voice cloning: normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed, serving_vc = openvoice_v1_en_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="openvoice_v1",
            input=TEXT_INPUT,
            mode="voice_cloning",
            language="EN",
            voice="default",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## Progress Checklist

**`test_harmonyspeech.py` additions (voice_cloning only):**
- [ ] Read file before editing to confirm `TEXT_INPUT`, `REFERENCE_AUDIO`, imports
- [ ] Add `io`, `base64`, `numpy`, `soundfile` imports + `_decode_b64_wav()` helper + `LONG_TEXT` constant
- [ ] Add `test_harmonyspeech_tts_split_text`
- [ ] Add `test_harmonyspeech_tts_normalize_audio`

**`test_kittentts.py` additions (single_speaker_tts only — no voice cloning):**
- [ ] Read file before editing to confirm `TEXT_INPUT`, imports
- [ ] Add `io`, `base64`, `numpy`, `soundfile` imports + `_decode_b64_wav()` helper + `LONG_TEXT` constant
- [ ] Add `test_kittentts_mini_split_text`
- [ ] Add `test_kittentts_mini_normalize_audio`
- [ ] Add `test_kittentts_micro_split_text`
- [ ] Add `test_kittentts_micro_normalize_audio`
- [ ] Add `test_kittentts_nano_split_text`
- [ ] Add `test_kittentts_nano_normalize_audio`
- [ ] Add `test_kittentts_nano_int8_split_text`
- [ ] Add `test_kittentts_nano_int8_normalize_audio`

**`test_melotts.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file before editing to confirm `TEXT_INPUT`, `REFERENCE_AUDIO`, imports
- [ ] Add `io`, `base64`, `numpy`, `soundfile` imports + `_decode_b64_wav()` helper + `LONG_TEXT` constant
- [ ] Add `test_melotts_en_split_text`
- [ ] Add `test_melotts_en_normalize_audio`
- [ ] Add `test_melotts_en_split_text_voice_cloning`
- [ ] Add `test_melotts_en_normalize_audio_voice_cloning`

**`test_openvoice_v1.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file before editing to confirm `TEXT_INPUT`, `REFERENCE_AUDIO`, imports
- [ ] Add `io`, `base64`, `numpy`, `soundfile` imports + `_decode_b64_wav()` helper + `LONG_TEXT` constant
- [ ] Add `test_openvoice_v1_split_text`
- [ ] Add `test_openvoice_v1_normalize_audio`
- [ ] Add `test_openvoice_v1_split_text_voice_cloning`
- [ ] Add `test_openvoice_v1_normalize_audio_voice_cloning`

**Validation:**
- [ ] Run `pytest tests/e2e/tts/test_harmonyspeech.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_kittentts.py -v -m "e2e"` — new tests pass
- [ ] Run `pytest tests/e2e/tts/test_melotts.py -v -m "e2e"` — new tests pass (including VC variants)
- [ ] Run `pytest tests/e2e/tts/test_openvoice_v1.py -v -m "e2e"` — new tests pass (including VC variants)
