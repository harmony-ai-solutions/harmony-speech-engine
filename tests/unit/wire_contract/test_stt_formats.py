"""STT response formats and translations plumbing (Phase 10-4).

Covers the pure renderer (timestamp edge cases, empty input, cue-text rules),
the admission validation of ``response_format``, the byte-identical ``json``
response shape, and the ``task`` plumbing that reaches the executor for both
``/v1/audio/transcriptions`` and ``/v1/audio/translations``.
"""

import pytest

from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.endpoints.openai.stt_formats import (
    RESPONSE_FORMATS,
    normalize_response_format,
    segments_to_srt,
    segments_to_vtt,
    validate_response_format,
)
from harmonyspeech.task_handler.inputs import prepare_faster_whisper_inputs
from tests.wire_contract_support import TRANSCRIPTION_PAYLOAD, sample_b64

STT_MODEL = "test-stt"
AUDIO_B64 = sample_b64("codec_tone.wav")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", RESPONSE_FORMATS)
def test_validate_supported_formats(fmt):
    assert validate_response_format(fmt) is None


def test_normalize_defaults_to_json():
    assert normalize_response_format(None) == "json"
    assert normalize_response_format("") == "json"
    assert normalize_response_format(" SRT ") == "srt"


def test_validate_unknown_format_rejected():
    message = validate_response_format("ass")
    assert message is not None
    assert "Unsupported response_format" in message
    for fmt in RESPONSE_FORMATS:
        assert fmt in message


# ---------------------------------------------------------------------------
# Timestamp rendering
# ---------------------------------------------------------------------------


def _srt(start, end, text="hello"):
    return segments_to_srt([{"start": start, "end": end, "text": text}])


def _vtt(start, end, text="hello"):
    return segments_to_vtt([{"start": start, "end": end, "text": text}])


def test_srt_renders_over_one_hour_and_milliseconds():
    assert _srt(3661.004, 3662.5) == "1\n01:01:01,004 --> 01:01:02,500\nhello\n"


def test_vtt_renders_over_one_hour_and_milliseconds():
    assert _vtt(3661.004, 3662.5) == "WEBVTT\n\n01:01:01.004 --> 01:01:02.500\nhello\n"


@pytest.mark.parametrize(
    "seconds,expected_ms",
    [
        (1.2345, 1235),  # exact half a millisecond rounds up
        (1.2344, 1234),
        (0.0005, 1),
        (0.0004, 0),
        (59.9996, 60000),  # rounds into the next second/minute
    ],
)
def test_sub_millisecond_rounding_is_half_up(seconds, expected_ms):
    line = _srt(seconds, seconds).splitlines()[1]
    start = line.split(" --> ")[0]
    hours, minutes, rest = start.split(":")
    whole_seconds, milliseconds = rest.split(",")
    total_ms = ((int(hours) * 60 + int(minutes)) * 60 + int(whole_seconds)) * 1000 + int(milliseconds)
    assert total_ms == expected_ms


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "not-a-number"])
def test_invalid_timestamps_clamp_to_zero(value):
    assert _srt(value, value) == "1\n00:00:00,000 --> 00:00:00,000\nhello\n"


def test_hours_are_zero_padded_not_capped():
    assert _srt(360000.0, 360001.0).splitlines()[1].startswith("100:00:00,000")


# ---------------------------------------------------------------------------
# Empty input and multiple cues
# ---------------------------------------------------------------------------


def test_empty_segments_render_empty_documents():
    assert segments_to_srt([]) == ""
    assert segments_to_srt(None) == ""
    assert segments_to_vtt([]) == "WEBVTT\n"
    assert segments_to_vtt(None) == "WEBVTT\n"


def test_multiple_cues_are_blank_line_separated():
    segments = [{"start": 0.0, "end": 1.0, "text": "one"}, {"start": 1.0, "end": 2.0, "text": "two"}]
    assert segments_to_srt(segments) == (
        "1\n00:00:00,000 --> 00:00:01,000\none\n\n2\n00:00:01,000 --> 00:00:02,000\ntwo\n"
    )
    assert segments_to_vtt(segments) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\none\n\n00:00:01.000 --> 00:00:02.000\ntwo\n"
    )


# ---------------------------------------------------------------------------
# Cue text rules
# ---------------------------------------------------------------------------


def test_cue_text_keeps_newlines_and_escapes_the_timing_arrow():
    text = "line one\nline two --> three"
    assert _srt(0.0, 1.0, text) == ("1\n00:00:00,000 --> 00:00:01,000\nline one\nline two - -> three\n")
    assert _vtt(0.0, 1.0, text) == ("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nline one\nline two - -> three\n")


def test_cue_text_normalizes_crlf_and_strips_outer_blank_lines():
    text = "\r\nfirst\r\nsecond\r\n"
    assert _srt(0.0, 1.0, text).endswith("\nfirst\nsecond\n")


def test_non_mapping_segments_are_ignored():
    assert segments_to_srt([None, "junk", {"start": 0, "end": 1, "text": "ok"}]).startswith("1\n")


# ---------------------------------------------------------------------------
# Endpoint: json stays byte-identical; text/srt/vtt are text/plain
# ---------------------------------------------------------------------------


def test_json_response_shape_is_unchanged(wire_client):
    response = wire_client.client.post("/v1/audio/transcriptions", json={"model": STT_MODEL, "input_audio": AUDIO_B64})
    assert response.status_code == 200, response.text
    body = response.json()

    # Exact legacy shape and values (no response_format field added).
    assert list(body) == ["id", "model", "created", "text", "language", "timestamps"]
    assert set(body) == {"id", "model", "created", "text", "language", "timestamps"}
    assert body["model"] == STT_MODEL
    assert body["text"] == TRANSCRIPTION_PAYLOAD["text"]
    assert body["language"] is None
    assert body["timestamps"] is None
    assert body["id"].startswith("stt-")
    assert isinstance(body["created"], int)


def test_json_response_with_flags_keeps_the_legacy_values(wire_client):
    response = wire_client.client.post(
        "/v1/audio/transcriptions",
        json={"model": STT_MODEL, "input_audio": AUDIO_B64, "get_language": True, "get_timestamps": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "en"
    assert body["timestamps"] == TRANSCRIPTION_PAYLOAD["segments"]


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("text", "hello world"),
        ("srt", "1\n00:00:00,000 --> 00:00:01,000\nhello world\n"),
        ("vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello world\n"),
    ],
)
def test_non_json_formats_are_plain_text(wire_client, fmt, expected):
    response = wire_client.client.post(
        "/v1/audio/transcriptions", json={"model": STT_MODEL, "input_audio": AUDIO_B64, "response_format": fmt}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == expected


# ---------------------------------------------------------------------------
# Admission validation: unknown format never queues inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/v1/audio/transcriptions", "/v1/audio/translations"])
def test_unknown_response_format_is_rejected_before_generation(wire_client, endpoint):
    before = len(wire_client.engine.requests)
    response = wire_client.client.post(
        endpoint, json={"model": STT_MODEL, "input_audio": AUDIO_B64, "response_format": "ass"}
    )
    assert response.status_code == 400
    assert "Unsupported response_format" in response.json()["message"]
    assert len(wire_client.engine.requests) == before


# ---------------------------------------------------------------------------
# task plumbing
# ---------------------------------------------------------------------------


def test_transcription_defaults_to_the_transcribe_task(wire_client):
    response = wire_client.client.post("/v1/audio/transcriptions", json={"model": STT_MODEL, "input_audio": AUDIO_B64})
    assert response.status_code == 200, response.text
    assert wire_client.engine.requests[-1].task == "transcribe"


def test_translation_threads_the_translate_task_and_hints(wire_client):
    response = wire_client.client.post(
        "/v1/audio/translations",
        json={"model": STT_MODEL, "input_audio": AUDIO_B64, "language": "de", "prompt": "Harmony"},
    )
    assert response.status_code == 200, response.text
    request_data = wire_client.engine.requests[-1]
    assert isinstance(request_data, SpeechTranscribeRequestInput)
    assert request_data.task == "translate"
    assert request_data.language == "de"
    assert request_data.prompt == "Harmony"
    # The JSON body mirrors the transcription shape.
    assert response.json()["text"] == TRANSCRIPTION_PAYLOAD["text"]


def test_prepare_normalizes_task_and_rejects_unknown():
    request = SpeechTranscribeRequestInput(
        request_id="r", requested_model="m", model="m", input_audio=AUDIO_B64, task="TRANSLATE"
    )
    (artifact,) = prepare_faster_whisper_inputs([request])
    assert request.task == "translate"
    assert artifact.sample_rate == 16000

    bad = SpeechTranscribeRequestInput(
        request_id="bad", requested_model="m", model="m", input_audio=AUDIO_B64, task="summarize"
    )
    with pytest.raises(ValueError, match="task"):
        prepare_faster_whisper_inputs([bad])
