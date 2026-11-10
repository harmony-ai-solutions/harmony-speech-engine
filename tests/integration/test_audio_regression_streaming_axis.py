"""Phase 14-5: the streamed-vs-buffered audio-regression axis, hermetically.

The script's ``streaming:off-vs-on`` axis drives the same serving seam these
tests use. Here the synthetic Chatterbox stub (deterministic, identity
watermarker, 8 kHz) renders a multi-chunk request both ways and the test
asserts the **pre-watermark** PCM is sample-identical:

* buffered: the real 12-2 ``_chunked`` path (``crossfade_stitch``);
* streamed: the real 14-3 frame source (``_stream_audio_segments`` with
  ``CrossfadeStitcher``).

The watermarker is the stub's identity, so the per-segment vs whole-clip
divergence cannot mask a generator bug. This is the gate the host axis runs at
family breadth; no GPU, no weights.
"""

import numpy as np
import pytest
import soundfile as sf

from harmonyspeech.common.audio_utils import split_text_into_chunks
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.endpoints.openai.serving_text_to_speech import StreamState
from tests.integration.test_tts_chunking import RAW, SAMPLE_RATE, _chatterbox_harness, _request, _speak
from tests.integration.test_tts_streaming import CHUNK_SIZE, TEXT_3, _decode


def _wav_pcm16(samples: np.ndarray) -> np.ndarray:
    """Quantize samples exactly the way the regression axis writes both sides."""
    import io

    buffer = io.BytesIO()
    sf.write(buffer, samples.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    pcm, _rate = sf.read(buffer, dtype="int16")
    return np.asarray(pcm)



@pytest.mark.integration
@pytest.mark.asyncio
async def test_streamed_matches_buffered_pre_watermark_pcm_on_the_synthetic_stub():
    harness, model = _chatterbox_harness()
    options = {
        "split_text": True,
        "chunk_size": CHUNK_SIZE,
        "crossfade_ms": 40,
        "sentence_pause_ms": 120,
    }
    async with harness:
        request = _request(TEXT_3, **options)
        buffered_response = await _speak(harness, request)
        buffered_waveform, buffered_rate = _decode(buffered_response)

        request_data = TextToSpeechRequestInput.from_openai("tts-stream-axis", request)
        chunks = split_text_into_chunks(
            request_data.input_text,
            chunk_size=CHUNK_SIZE,
            language=getattr(request_data, "language_id", None),
        )
        assert len(chunks) > 1
        state = StreamState()
        segments = [
            segment
            async for segment in harness.serving._stream_audio_segments(
                raw_request=RAW,
                request_id="tts-stream-axis",
                request_data=request_data,
                artifacts={},
                chunks=chunks,
                state=state,
                requested_rate=None,
            )
        ]

    assert buffered_rate == SAMPLE_RATE == state.source_rate
    assert state.segments == len(segments)
    streamed = (
        np.concatenate(segments).astype(np.float32)
        if segments
        else np.empty(0, dtype=np.float32)
    )
    assert streamed.size == buffered_waveform.size
    # Same chunk plan: the stub saw the same chunks in the same order on the
    # streamed side (the buffered side ran first).
    assert model.generate_calls[-len(chunks) :] == chunks
    # Pre-watermark PCM equality through the axis's own WAV container (both
    # sides quantized by libsndfile; the raw float is sample-identical).
    assert np.array_equal(_wav_pcm16(streamed), _wav_pcm16(buffered_waveform))
