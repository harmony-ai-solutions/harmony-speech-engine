"""Real-model parity gate: batched B=1 vs ``pipeline.transcribe`` (Phase 10-2).

This is the benchmark-host gate the phase file defers: it needs a real
CTranslate2 Whisper model directory (no weights are downloaded here) and is
skipped unless ``HSE_WHISPER_PARITY_MODEL`` points at one, e.g.::

    set HSE_WHISPER_PARITY_MODEL=C:\\models\\faster-whisper-tiny
    pytest tests/e2e/stt/test_whisper_window_parity.py -m e2e -q

The batched path is expected to reproduce the sequential path's text and
segment timings for the same audio (temperature is pinned to 0 in both). The
single-window clip is the strict parity gate; the tiled >60 s clip additionally
checks multi-window stitching. The one documented semantic difference —
language detection on the first window vs on the concatenation of the first
windows — is reported for the multi-window case rather than asserted, because
both are defensible readings of the same audio.

Owner decision 2026-11-08 (Phase 11-0): the gate passed substantively (text,
segments, timings and language identical on all three measured cases), so
``language_probability`` is bounded at ``1e-2`` and marked informational — the
batched path reuses the batch encoder output and the number moves 0.2-0.8 pp.
The shipped configs therefore enable ``batched_inference: true`` for Whisper.
"""

import json
import os
from dataclasses import asdict

import numpy as np
import pytest
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.task_handler.whisper_batching import WhisperWindowBatcher

MODEL_PATH = os.environ.get("HSE_WHISPER_PARITY_MODEL")
SAMPLE_RATE = 16000
CLIP_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "test-data", "samples", "wanda4.wav")
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL_PATH or not os.path.isdir(MODEL_PATH),
        reason="set HSE_WHISPER_PARITY_MODEL to a local CTranslate2 Whisper directory",
    ),
]


def _sequential_payload(model, audio):
    pipeline = BatchedInferencePipeline(model=model)
    pipeline.last_speech_timestamp = 0.0
    segments, info = pipeline.transcribe(audio, batch_size=16)
    text = ""
    segment_data = []
    for segment in segments:
        text += segment.text
        segment_data.append(asdict(segment))
    return {"text": text, "segments": segment_data, "info": asdict(info)}


def _batched_payload(model, audio):
    artifact = AudioArtifact.from_numpy(audio, SAMPLE_RATE)
    request = EngineRequest(
        request_id="parity",
        request_data=SpeechTranscribeRequestInput(
            request_id="parity", requested_model="whisper", model="whisper", input_audio=None
        ),
        arrival_time=0.0,
        artifacts={"input_audio": artifact},
    )
    pairs = WhisperWindowBatcher(model).transcribe_batch([artifact], [request])
    payload, error = pairs[0]
    assert error is None, f"batched path failed: {error}"
    return payload


@pytest.fixture(scope="module")
def whisper_model():
    # Device "auto" is the benchmark host's choice (CUDA when available).
    return WhisperModel(MODEL_PATH, device="auto", compute_type="default")


@pytest.fixture(scope="module")
def clip_audio():
    return decode_audio(CLIP_PATH, sampling_rate=SAMPLE_RATE)


def _assert_timings_match(sequential, batched, tolerance=0.02):
    assert len(sequential["segments"]) == len(batched["segments"]), (
        f"segment count differs: {len(sequential['segments'])} vs {len(batched['segments'])}"
    )
    for index, (left, right) in enumerate(zip(sequential["segments"], batched["segments"], strict=True)):
        assert left["text"] == right["text"], f"segment {index} text differs: {left['text']!r} vs {right['text']!r}"
        assert abs(left["start"] - right["start"]) <= tolerance, (
            f"segment {index} start differs: {left['start']} vs {right['start']}"
        )
        assert abs(left["end"] - right["end"]) <= tolerance, (
            f"segment {index} end differs: {left['end']} vs {right['end']}"
        )


def test_batched_b1_matches_sequential_for_a_single_window_clip(whisper_model, clip_audio):
    sequential = _sequential_payload(whisper_model, clip_audio)
    batched = _batched_payload(whisper_model, clip_audio)

    assert batched["text"] == sequential["text"]
    _assert_timings_match(sequential, batched)
    assert batched["info"]["language"] == sequential["info"]["language"]
    # Informational (owner decision 2026-11-08): the batched path reuses the
    # batch encoder output for language detection, so `language_probability`
    # moves 0.2-0.8 pp between the paths (measured 2026-11-08). Text, segments,
    # timings and language are identical; the probability is a soft signal, so
    # it is bounded at 1e-2 rather than the original 1e-3.
    probability_delta = abs(
        float(batched["info"]["language_probability"])
        - float(sequential["info"]["language_probability"])
    )
    assert probability_delta <= 1e-2, (
        "language_probability (informational) moved "
        f"{probability_delta:.4f}: batched={batched['info']['language_probability']} "
        f"sequential={sequential['info']['language_probability']}"
    )
    assert batched["info"]["duration"] == pytest.approx(sequential["info"]["duration"], abs=1e-3)
    assert batched["info"]["duration_after_vad"] == pytest.approx(
        sequential["info"]["duration_after_vad"], abs=1e-3
    )
    # The payload is wire-compatible with the sequential path.
    assert set(json.loads(json.dumps(batched))) == {"text", "segments", "info"}


def test_batched_multi_window_stitching_matches_sequential_text(whisper_model, clip_audio):
    # Tile the real clip past 60 s so the collector forms multiple 30 s windows.
    repeats = int(np.ceil(65.0 * SAMPLE_RATE / clip_audio.shape[0]))
    long_audio = np.tile(clip_audio, repeats)[: int(65.0 * SAMPLE_RATE)]

    sequential = _sequential_payload(whisper_model, long_audio)
    batched = _batched_payload(whisper_model, long_audio)

    assert batched["text"] == sequential["text"]
    _assert_timings_match(sequential, batched)
    # Documented edge: upstream detects language on the concatenation of the
    # first windows, the batcher on the first window alone. Record it; do not
    # fail the gate on a defensible difference.
    if batched["info"]["language"] != sequential["info"]["language"]:
        pytest.skip(
            "documented language-detection edge: "
            f"batched={batched['info']['language']} sequential={sequential['info']['language']}"
        )
