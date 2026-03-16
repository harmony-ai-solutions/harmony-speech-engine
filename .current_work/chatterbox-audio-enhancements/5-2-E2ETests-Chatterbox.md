# Phase 5-2: E2E Tests — Extend Existing Chatterbox Test Suites

> **Note:** Where a Chatterbox model supports both `single_speaker_tts` and `voice_cloning` modes, new feature tests are provided for **both** modes so that each parameter is validated end-to-end regardless of mode.

## Objective

Extend the **three existing Chatterbox e2e test files** with test cases for the new features (chunking, speed factor, normalize_audio). Also update any `norm_loudness` references in the Turbo test file.

No new test files are created — one suite per model type, tests added to the existing file for that model.

## Files to Modify

| File | Changes |
|---|---|
| `tests/e2e/tts/test_chatterbox.py` | Add: chunking, speed, normalize_audio tests for standard ChatterboxTTS |
| `tests/e2e/tts/test_chatterbox_turbo.py` | Add: chunking, speed, normalize_audio tests; rename `norm_loudness` → `normalize_audio` |
| `tests/e2e/tts/test_chatterbox_multilingual.py` | Add: chunking, speed, normalize_audio tests |

## Shared helpers

Each test file already imports `TextToSpeechRequest`, `TextToSpeechResponse`, etc. Add a module-level helper at the top of each modified file (if not already present):

```python
import io
import numpy as np
import soundfile as sf

def _decode_b64_wav(b64_data: str) -> tuple[np.ndarray, int]:
    """Decode a base64 WAV audio response to numpy float32 array + sample rate."""
    import base64
    raw = base64.b64decode(b64_data)
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr
```

Also add at module level:
```python
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold."
)
```

---

## 1. Changes to `tests/e2e/tts/test_chatterbox.py`

Add the following tests at the end of the file, using the existing `chatterbox_engine` fixture which returns `(engine, serving_tts, serving_embed, serving_vc)`.

### `test_chatterbox_tts_split_text`
```python
def test_chatterbox_tts_split_text(chatterbox_engine, mock_raw_request):
    """split_text=True on a long text produces a non-empty response via chunking+stitching."""
    engine, serving_tts, _, _ = chatterbox_engine
    request = TextToSpeechRequest(
        model="chatterbox",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0
```

### `test_chatterbox_tts_split_text_longer_than_single_sentence`
```python
def test_chatterbox_tts_split_text_longer_than_single_sentence(chatterbox_engine, mock_raw_request):
    """Stitched multi-chunk output is longer than a single short-text synthesis."""
    engine, serving_tts, _, _ = chatterbox_engine

    short_response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts"),
        mock_raw_request,
    ))
    short_audio, _ = _decode_b64_wav(short_response.data)

    long_response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=LONG_TEXT, mode="single_speaker_tts",
            split_text=True, chunk_size=120,
        ),
        mock_raw_request,
    ))
    long_audio, _ = _decode_b64_wav(long_response.data)

    assert len(long_audio) > len(short_audio) * 2
```

### `test_chatterbox_tts_speed_slower`
```python
def test_chatterbox_tts_speed_slower(chatterbox_engine, mock_raw_request):
    """speed=0.5 produces ~2x longer audio via post-synthesis time-stretch."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts"),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts",
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5
```

### `test_chatterbox_tts_normalize_audio`
```python
def test_chatterbox_tts_normalize_audio(chatterbox_engine, mock_raw_request):
    """normalize_audio=True produces valid non-clipping audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=TEXT_INPUT, mode="single_speaker_tts",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse)
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 2. Changes to `tests/e2e/tts/test_chatterbox_turbo.py`

### Rename `norm_loudness` → `normalize_audio`

First, **read the full file** to find any direct usage of `norm_loudness` in request construction (e.g. `GenerationOptions(norm_loudness=True)`). Rename all occurrences to `normalize_audio`.

### Add new tests (same structure as above, using `chatterbox_turbo_engine` returning `(engine, serving_tts, serving_embed)`):

```python
def test_chatterbox_turbo_tts_split_text(chatterbox_turbo_engine, mock_raw_request):
    """split_text=True on ChatterboxTurboTTS: non-empty stitched audio."""
    engine, serving_tts, _ = chatterbox_turbo_engine
    request = TextToSpeechRequest(
        model="chatterbox_turbo",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_speed_slower(chatterbox_turbo_engine, mock_raw_request):
    """speed=0.5 on ChatterboxTurboTTS produces longer audio via post-synthesis time-stretch."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts"),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts",
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5


def test_chatterbox_turbo_tts_normalize_audio(chatterbox_turbo_engine, mock_raw_request):
    """normalize_audio=True on ChatterboxTurboTTS: valid non-clipping output."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_turbo_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo", input=TEXT_INPUT, mode="single_speaker_tts",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse)
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---

## 3. Changes to `tests/e2e/tts/test_chatterbox_multilingual.py`

Add the following tests using the `chatterbox_multilingual_engine` fixture returning `(engine, serving_tts, serving_embed)`:

```python
def test_chatterbox_multilingual_tts_split_text(chatterbox_multilingual_engine, mock_raw_request):
    """split_text=True on ChatterboxMultilingualTTS: non-empty stitched audio."""
    engine, serving_tts, _ = chatterbox_multilingual_engine
    request = TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=LONG_TEXT,
        mode="single_speaker_tts",
        language="en",
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_multilingual_tts_speed_slower(chatterbox_multilingual_engine, mock_raw_request):
    """speed=0.5 on ChatterboxMultilingualTTS produces longer audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(model="chatterbox_multilingual", input=TEXT_INPUT, mode="single_speaker_tts", language="en"),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="single_speaker_tts", language="en",
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5


def test_chatterbox_multilingual_tts_normalize_audio(chatterbox_multilingual_engine, mock_raw_request):
    """normalize_audio=True on ChatterboxMultilingualTTS: valid non-clipping output."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="single_speaker_tts", language="en",
            generation_options=GenerationOptions(normalize_audio=True),
        ),
        mock_raw_request,
    ))
    assert isinstance(response, TextToSpeechResponse)
    audio, _ = _decode_b64_wav(response.data)
    assert len(audio) > 0
    assert float(np.abs(audio).max()) <= 1.0
```

---
---

## 4. Voice Cloning Additions to `tests/e2e/tts/test_chatterbox.py`

The existing `test_chatterbox.py` already defines `REFERENCE_AUDIO = load_sample_audio_b64("wanda4")`. The fixture `chatterbox_engine` returns `(engine, serving_tts, serving_embed, serving_vc)`.

Append these voice-cloning variants for the new Phase A parameters after the existing tests:

```python
def test_chatterbox_tts_split_text_voice_cloning(chatterbox_engine, mock_raw_request):
    """split_text=True with voice cloning produces a non-empty stitched response."""
    engine, serving_tts, _, _ = chatterbox_engine
    request = TextToSpeechRequest(
        model="chatterbox",
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


def test_chatterbox_tts_speed_slower_voice_cloning(chatterbox_engine, mock_raw_request):
    """speed=0.5 with voice cloning produces ~2x longer audio via post-synthesis time-stretch."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=TEXT_INPUT, mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=TEXT_INPUT, mode="voice_cloning",
            input_audio=REFERENCE_AUDIO,
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5


def test_chatterbox_tts_normalize_audio_voice_cloning(chatterbox_engine, mock_raw_request):
    """normalize_audio=True with voice cloning: audio peak <= 1.0."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _, _ = chatterbox_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox", input=TEXT_INPUT, mode="voice_cloning",
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

## 5. Voice Cloning Additions to `tests/e2e/tts/test_chatterbox_turbo.py`

The existing `test_chatterbox_turbo.py` defines `LONG_REFERENCE_AUDIO = load_sample_audio_b64("jerry_seinfeld_prompt")`. The fixture `chatterbox_turbo_engine` returns `(engine, serving_tts, serving_embed)`.

ChatterboxTurbo voice cloning requires a **pre-computed embedding** from `LONG_REFERENCE_AUDIO` (>= 5 seconds). The test performs embed step first, then uses `input_embedding` with `mode="voice_cloning"`.

Append these after the existing tests:

```python
def test_chatterbox_turbo_tts_split_text_voice_cloning(chatterbox_turbo_engine, mock_raw_request):
    """split_text=True + Turbo voice cloning (pre-computed embedding): non-empty stitched audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine

    embed_request = EmbedSpeakerRequest(model="chatterbox_turbo_embedding", input_audio=LONG_REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed got: {embed_response}"

    request = TextToSpeechRequest(
        model="chatterbox_turbo",
        input=LONG_TEXT,
        mode="voice_cloning",
        input_embedding=embed_response.data,
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_turbo_tts_speed_slower_voice_cloning(chatterbox_turbo_engine, mock_raw_request):
    """speed=0.5 + Turbo voice cloning: produces ~2x longer audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine

    embed_request = EmbedSpeakerRequest(model="chatterbox_turbo_embedding", input_audio=LONG_REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed got: {embed_response}"

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo", input=TEXT_INPUT, mode="voice_cloning",
            input_embedding=embed_response.data,
        ),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo", input=TEXT_INPUT, mode="voice_cloning",
            input_embedding=embed_response.data,
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5


def test_chatterbox_turbo_tts_normalize_audio_voice_cloning(chatterbox_turbo_engine, mock_raw_request):
    """normalize_audio=True + Turbo voice cloning: audio peak <= 1.0."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, serving_embed = chatterbox_turbo_engine

    embed_request = EmbedSpeakerRequest(model="chatterbox_turbo_embedding", input_audio=LONG_REFERENCE_AUDIO)
    embed_response = asyncio.run(serving_embed.create_voice_embedding(embed_request, mock_raw_request))
    assert isinstance(embed_response, EmbedSpeakerResponse), f"Embed got: {embed_response}"

    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_turbo", input=TEXT_INPUT, mode="voice_cloning",
            input_embedding=embed_response.data,
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

## 6. Voice Cloning Additions to `tests/e2e/tts/test_chatterbox_multilingual.py`

The existing `test_chatterbox_multilingual.py` defines `SHORT_REFERENCE_AUDIO = load_sample_audio_b64("wanda4")`. The fixture `chatterbox_multilingual_engine` returns `(engine, serving_tts, serving_embed)`.

ChatterboxMultilingual supports voice cloning with `input_audio=SHORT_REFERENCE_AUDIO` (no minimum audio length enforced).

Append these after the existing tests:

```python
def test_chatterbox_multilingual_tts_split_text_voice_cloning(chatterbox_multilingual_engine, mock_raw_request):
    """split_text=True + Multilingual voice cloning: non-empty stitched audio."""
    engine, serving_tts, _ = chatterbox_multilingual_engine
    request = TextToSpeechRequest(
        model="chatterbox_multilingual",
        input=LONG_TEXT,
        mode="voice_cloning",
        language="en",
        input_audio=SHORT_REFERENCE_AUDIO,
        split_text=True,
        chunk_size=120,
    )
    response = asyncio.run(serving_tts.create_text_to_speech(request, mock_raw_request))
    assert isinstance(response, TextToSpeechResponse), f"Got: {response}"
    audio, sr = _decode_b64_wav(response.data)
    assert len(audio) > 0 and sr > 0


def test_chatterbox_multilingual_tts_speed_slower_voice_cloning(chatterbox_multilingual_engine, mock_raw_request):
    """speed=0.5 + Multilingual voice cloning: produces ~2x longer audio."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine

    normal = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="voice_cloning",
            language="en", input_audio=SHORT_REFERENCE_AUDIO,
        ),
        mock_raw_request,
    ))
    slow = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="voice_cloning",
            language="en", input_audio=SHORT_REFERENCE_AUDIO,
            generation_options=GenerationOptions(speed=0.5),
        ),
        mock_raw_request,
    ))
    normal_audio, _ = _decode_b64_wav(normal.data)
    slow_audio, _ = _decode_b64_wav(slow.data)
    assert len(slow_audio) > len(normal_audio) * 1.5


def test_chatterbox_multilingual_tts_normalize_audio_voice_cloning(chatterbox_multilingual_engine, mock_raw_request):
    """normalize_audio=True + Multilingual voice cloning: audio peak <= 1.0."""
    from harmonyspeech.endpoints.openai.protocol import GenerationOptions
    engine, serving_tts, _ = chatterbox_multilingual_engine
    response = asyncio.run(serving_tts.create_text_to_speech(
        TextToSpeechRequest(
            model="chatterbox_multilingual", input=TEXT_INPUT, mode="voice_cloning",
            language="en", input_audio=SHORT_REFERENCE_AUDIO,
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

**`test_chatterbox.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file before editing to confirm `TEXT_INPUT` constant name
- [ ] Add `_decode_b64_wav()` helper and `LONG_TEXT` constant
- [ ] Add `test_chatterbox_tts_split_text`
- [ ] Add `test_chatterbox_tts_split_text_longer_than_single_sentence`
- [ ] Add `test_chatterbox_tts_speed_slower`
- [ ] Add `test_chatterbox_tts_normalize_audio`
- [ ] Add `test_chatterbox_tts_split_text_voice_cloning`
- [ ] Add `test_chatterbox_tts_speed_slower_voice_cloning`
- [ ] Add `test_chatterbox_tts_normalize_audio_voice_cloning`

**`test_chatterbox_turbo.py` updates + additions (single_speaker_tts + voice_cloning):**
- [ ] Read file before editing to find `norm_loudness` references
- [ ] Rename all `norm_loudness=` occurrences to `normalize_audio=`
- [ ] Add `_decode_b64_wav()` helper and `LONG_TEXT` constant
- [ ] Add `test_chatterbox_turbo_tts_split_text`
- [ ] Add `test_chatterbox_turbo_tts_speed_slower`
- [ ] Add `test_chatterbox_turbo_tts_normalize_audio`
- [ ] Confirm `EmbedSpeakerRequest`, `EmbedSpeakerResponse` already imported
- [ ] Add `test_chatterbox_turbo_tts_split_text_voice_cloning`
- [ ] Add `test_chatterbox_turbo_tts_speed_slower_voice_cloning`
- [ ] Add `test_chatterbox_turbo_tts_normalize_audio_voice_cloning`

**`test_chatterbox_multilingual.py` additions (single_speaker_tts + voice_cloning):**
- [ ] Read file before editing to confirm `TEXT_INPUT` and `SHORT_REFERENCE_AUDIO` defined
- [ ] Add `_decode_b64_wav()` helper and `LONG_TEXT` constant
- [ ] Add `test_chatterbox_multilingual_tts_split_text`
- [ ] Add `test_chatterbox_multilingual_tts_speed_slower`
- [ ] Add `test_chatterbox_multilingual_tts_normalize_audio`
- [ ] Add `test_chatterbox_multilingual_tts_split_text_voice_cloning`
- [ ] Add `test_chatterbox_multilingual_tts_speed_slower_voice_cloning`
- [ ] Add `test_chatterbox_multilingual_tts_normalize_audio_voice_cloning`

**Validation:**
- [ ] Run `pytest tests/e2e/tts/test_chatterbox.py -v -m "e2e"` — all tests pass (including VC variants)
- [ ] Run `pytest tests/e2e/tts/test_chatterbox_turbo.py -v -m "e2e"` — all tests pass including renamed and VC variants
- [ ] Run `pytest tests/e2e/tts/test_chatterbox_multilingual.py -v -m "e2e"` — all tests pass (including VC variants)
