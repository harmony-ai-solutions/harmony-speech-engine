"""Unit tests for the VAD switch and entry rules (Phase 3-4).

Covers:

* Silero VAD producing Whisper-schema-compatible ``VadArtifact`` segments
  (seconds + empty text) and those segments surviving the OpenVoice encoder's
  segment filter (which is now backend aware);
* ``input_vad_mode`` preference + graceful fallback (also covered for
  ``next_step`` in ``test_pipeline_spec.py``);
* the ``EntryRule`` set replacing field-presence entry decisions, including
  ``single_speaker_tts`` starting at the synthesis step without mutating fields;
* forwarded requests never re-resolving entry.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from harmonyspeech.common.artifacts import AudioArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    SpeechEmbeddingRequestInput,
    TextToSpeechRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing import pipeline_registry as registry
from harmonyspeech.task_handler import inputs as prepare_module
from harmonyspeech.task_handler import model_runner_base as runner_module
from harmonyspeech.task_handler.inputs import (
    prepare_openvoice_tone_converter_encoder_inputs,
    prepare_silero_vad_inputs,
)
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

# ---------------------------------------------------------------------------
# Silero -> Whisper-schema artifact
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_silero_timestamps_are_normalized_to_seconds_with_empty_text():
    artifact = VadArtifact.from_silero_timestamps([{"start": 16000, "end": 48000}, {"start": 64000, "end": 80000}])

    assert artifact.source == "silero"
    assert artifact.text == ""
    assert artifact.segments == [{"start": 1.0, "end": 3.0, "text": ""}, {"start": 4.0, "end": 5.0, "text": ""}]


@pytest.mark.unit
def test_silero_timestamps_in_seconds_are_passed_through():
    artifact = VadArtifact.from_silero_timestamps(
        [{"start": 0.5, "end": 1.5}], sample_rate=16000, return_seconds=True
    )

    assert artifact.segments == [{"start": 0.5, "end": 1.5, "text": ""}]


@pytest.mark.unit
def test_silero_executor_always_emits_pipeline_segments(monkeypatch):
    """The VAD *response* stays flag-dependent; the artifact never is."""
    timestamps = [{"start": 16000, "end": 48000}]
    monkeypatch.setattr(runner_module, "get_speech_timestamps", lambda *a, **k: timestamps)

    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner._baseline_state = None
    runner.model_config = ModelConfig(
        name="silero",
        model="silero-vad",
        model_type="SileroVAD",
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )
    runner.model = object()

    request = SimpleNamespace(
        request_id="vad-1",
        requested_model="silero_vad",
        model="silero",
        input_audio=None,
        get_timestamps=False,  # standalone VAD client did not ask for segments
        metrics=SimpleNamespace(finished_time=None),
    )
    engine_request = SimpleNamespace(
        request_id="vad-1",
        request_data=request,
        artifacts={},
        pipeline_state=None,
        metrics=SimpleNamespace(finished_time=None),
    )
    audio = AudioArtifact.from_numpy(np.zeros(16000 * 4, dtype=np.float32), 16000)
    vad_params = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "speech_pad_ms": 30,
        "return_seconds": False,
        "get_timestamps": False,
    }

    (result,) = runner._execute_silero_vad([(audio, vad_params)], [engine_request])

    assert result.result_data.finish_reason == "stop"
    assert "segments" not in result.result_data.output  # response shape unchanged
    artifact = result.result_data.vad_artifact
    assert artifact.source == "silero"
    assert artifact.segments == [{"start": 1.0, "end": 3.0, "text": ""}]


# ---------------------------------------------------------------------------
# Encoder segment filter is backend aware
# ---------------------------------------------------------------------------


def _ov_config(sample_rate: int):
    data = SimpleNamespace(sampling_rate=sample_rate, filter_length=1024, hop_length=256, win_length=1024)
    return SimpleNamespace(data=data)


def _tone_audio(seconds: float, sample_rate: int) -> AudioArtifact:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return AudioArtifact.from_numpy((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)


def _encoder_segments(monkeypatch, vad_artifact, sample_rate=22050, seconds=3.0):
    audio_artifact = _tone_audio(seconds, sample_rate)
    request = SpeechEmbeddingRequestInput(request_id="emb-1", requested_model="openvoice_v2", model="ov2-enc")
    engine_request = SimpleNamespace(
        request_id="emb-1",
        request_data=request,
        artifacts={"input_audio": audio_artifact, "input_vad_data": vad_artifact},
    )
    monkeypatch.setattr(prepare_module, "get_model_flavour", lambda config: "default")
    monkeypatch.setattr(prepare_module, "get_model_config", lambda *args, **kwargs: _ov_config(sample_rate))
    return prepare_openvoice_tone_converter_encoder_inputs(
        MagicMock(model="ov2", model_type="OpenVoiceV2ToneConverterEncoder"), [engine_request]
    )[0]


@pytest.mark.unit
def test_silero_segments_survive_the_encoder_filter(monkeypatch):
    vad = VadArtifact.from_silero_timestamps([{"start": 0, "end": 48000}])

    segments = _encoder_segments(monkeypatch, vad)

    assert len(segments) == 1
    assert isinstance(segments[0], AudioArtifact)
    assert segments[0].duration_s == pytest.approx(3.0, rel=0.1)


@pytest.mark.unit
def test_whisper_segments_keep_the_text_length_filter(monkeypatch):
    """Text-less *whisper* segments are still dropped (old behaviour preserved)."""
    vad = VadArtifact(segments=[{"start": 0.0, "end": 3.0, "text": ""}], source="whisper")

    assert _encoder_segments(monkeypatch, vad) == []


@pytest.mark.unit
def test_whisper_segments_with_text_still_pass(monkeypatch):
    vad = VadArtifact(segments=[{"start": 0.0, "end": 3.0, "text": "hello world"}], source="whisper")

    segments = _encoder_segments(monkeypatch, vad)

    assert len(segments) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text_length", "expected_segments"),
    [
        # The pack clips transcribe to one whole-clip segment of ~206 characters
        # (large-v3-turbo), which the legacy OpenVoice heuristic used to drop
        # (26/28 pack voices failed the 2026-11-09 warm-up; Henry/Michael were
        # the only transcripts under 200 characters). The 2026-11-10 fix keeps
        # the anti-hallucination floor and drops the cap: the duration filter
        # already bounds the segment.
        (206, 1),
        (194, 1),
        (400, 1),
        (1, 0),
    ],
)
def test_whisper_segment_transcript_floor_without_the_legacy_length_cap(
    monkeypatch, text_length, expected_segments
):
    """Whisper segments are kept by duration, not transcript length.

    This incident was *not* a VAD detection failure: whisper emits one segment
    spanning the clip, and the OpenVoice encoder's historical
    ``2 <= len(text) < 200`` filter (upstream's proxy for the 1.5–20 s
    duration bound) then discarded it — 26/28 pack clips transcribe to 206
    characters. The cap is gone; only the non-empty-transcript floor remains.
    Silero segments carry no text and bypass the filter entirely.
    """
    vad = VadArtifact(
        segments=[{"start": 0.0, "end": 3.0, "text": "x" * text_length}], source="whisper"
    )

    assert len(_encoder_segments(monkeypatch, vad)) == expected_segments


@pytest.mark.unit
def test_silero_vad_step_receives_16khz_audio_from_a_44khz_reference():
    """The Silero seam must resample admission audio to its 16 kHz contract.

    A rate bug here would trade the whisper failure for a new flaky VAD; the
    pack clips are 44.1 kHz and Silero's ``get_speech_timestamps`` expects
    16 kHz.
    """
    reference = AudioArtifact.from_numpy(np.zeros(44100 * 3, dtype=np.float32), 44100)
    request = SpeechEmbeddingRequestInput(request_id="emb-1", requested_model="openvoice_v2", model="ov2-enc")
    engine_request = SimpleNamespace(
        request_id="emb-1", request_data=request, artifacts={"input_audio": reference}
    )

    (audio, _params) = prepare_silero_vad_inputs([engine_request])[0]

    assert audio.sample_rate == 16000
    assert audio.duration_s == pytest.approx(3.0, rel=0.01)


@pytest.mark.unit
def test_silero_segments_still_respect_the_duration_filter(monkeypatch):
    vad = VadArtifact.from_silero_timestamps([{"start": 0, "end": 16000}])  # 1.0 s < 1.5 s

    assert _encoder_segments(monkeypatch, vad) == []


# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------


def _cfg(name: str, model_type: str, language: str | None = None) -> ModelConfig:
    return ModelConfig(
        name=name,
        model=f"dummy/{name}",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        language=language,
    )


OPENVOICE_V2 = [
    _cfg("whisper", "FasterWhisper"),
    _cfg("ov2-enc", "OpenVoiceV2ToneConverterEncoder"),
    _cfg("ov2-syn", "MeloTTSSynthesizer", "EN"),
    _cfg("ov2-tc", "OpenVoiceV2ToneConverter"),
]
OPENVOICE_V2_WITH_SILERO = OPENVOICE_V2 + [_cfg("silero", "SileroVAD")]


def _engine(configs) -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(configs)
    engine.log_stats = False
    engine.request_timeout_s = 120.0
    engine.strict_pipelines = False
    engine._pending_error_outputs = []
    engine.available_families = []
    return engine


def _entry(requested_model: str, request, configs):
    spec = registry.PIPELINE_SPECS[requested_model]
    state = registry.resolve_entry(spec, request, {}, configs)
    return spec, state


def _tts(model="openvoice_v2", mode: str = "voice_cloning", **kwargs) -> TextToSpeechRequestInput:
    kwargs.setdefault("language_id", "EN")
    return TextToSpeechRequestInput(
        request_id="entry-1", requested_model=model, model=model, input_text="hello", mode=mode, **kwargs
    )


@pytest.mark.unit
def test_single_speaker_tts_starts_at_synthesis_without_mutating_fields():
    request = _tts(mode="single_speaker_tts")

    spec, state = _entry("openvoice_v2", request, OPENVOICE_V2)

    assert spec.steps[state.step_index].name == "synthesize"
    # No field coercion: the request is untouched.
    assert request.input_audio is None
    assert request.input_vad_data is None
    assert request.input_embedding is None
    assert request.mode == "single_speaker_tts"


@pytest.mark.unit
def test_single_speaker_tts_stops_after_synthesis():
    request = _tts(mode="single_speaker_tts")

    _spec, state = _entry("openvoice_v2", request, OPENVOICE_V2)

    assert state.stop_after == "synthesize"


@pytest.mark.unit
def test_embedding_only_request_starts_at_vad_and_stops_after_embed():
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="openvoice_v2", model="openvoice_v2", input_audio="legacy"
    )

    spec, state = _entry("openvoice_v2", request, OPENVOICE_V2)

    assert spec.steps[state.step_index].name == "vad"
    assert state.stop_after == "embed"


@pytest.mark.unit
def test_embedding_request_with_external_vad_data_starts_at_embed():
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1",
        requested_model="openvoice_v2",
        model="openvoice_v2",
        input_audio="legacy",
        input_vad_data='{"segments": [{"start": 0.0, "end": 3.0, "text": "hi"}]}',
    )

    spec, state = _entry("openvoice_v2", request, OPENVOICE_V2)

    assert spec.steps[state.step_index].name == "embed"
    assert state.stop_after == "embed"


@pytest.mark.unit
def test_voice_conversion_request_starts_and_stops_at_tone_conversion():
    request = VoiceConversionRequestInput(
        request_id="vc-1",
        requested_model="openvoice_v2",
        model="openvoice_v2",
        source_audio="legacy",
        target_audio=None,
        target_embedding="legacy",
        generation_options=None,
        output_options=None,
    )

    spec, state = _entry("openvoice_v2", request, OPENVOICE_V2)

    assert spec.steps[state.step_index].name == "tone_convert"
    assert state.stop_after == "tone_convert"


@pytest.mark.unit
def test_chatterbox_tts_entry_depends_on_the_embedding_executor():
    request = _tts(model="chatterbox", input_audio="legacy")
    spec = registry.PIPELINE_SPECS["chatterbox"]

    with_embed = registry.resolve_entry(
        spec, request, {}, [_cfg("cb", "ChatterboxTTS"), _cfg("cb-emb", "ChatterboxEmbedding")]
    )
    without_embed = registry.resolve_entry(spec, request, {}, [_cfg("cb", "ChatterboxTTS")])

    assert spec.steps[with_embed.step_index].name == "embed"
    assert spec.steps[without_embed.step_index].name == "synthesize"


# ---------------------------------------------------------------------------
# Forwarded requests never re-resolve entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forwarded_request_keeps_its_pipeline_state():
    engine = _engine(OPENVOICE_V2)
    request = _tts(input_vad_data='{"segments": []}', input_embedding="legacy")
    spec = registry.PIPELINE_SPECS["openvoice_v2"]
    # A state that no entry rule would produce for this request shape: only the
    # state machine can own it.
    state = registry.PipelineState(
        spec_id="openvoice_v2", step_index=spec.step_index("tone_convert"), stop_after="tone_convert"
    )

    resolved, error = engine._route_request(request, {}, state)

    assert error is None
    assert resolved is state
    assert request.model == "ov2-tc"


@pytest.mark.unit
def test_forwarded_request_is_not_rerouted_to_the_vad_step():
    engine = _engine(OPENVOICE_V2)
    request = _tts(input_audio="legacy")
    spec = registry.PIPELINE_SPECS["openvoice_v2"]
    state = registry.PipelineState(spec_id="openvoice_v2", step_index=spec.step_index("synthesize"))

    engine._route_request(request, {"input_audio": object(), "input_embedding": object()}, state)

    assert request.model == "ov2-syn"


# ---------------------------------------------------------------------------
# VAD mode at the engine boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_honours_silero_vad_mode(loguru_records):
    engine = _engine(OPENVOICE_V2_WITH_SILERO)
    request = _tts(input_audio="legacy", input_vad_mode="silero")

    state = engine._route_request(request, {}, None)[0]

    assert request.model == "silero"
    assert state.step_index == 0


@pytest.mark.unit
def test_engine_falls_back_when_silero_is_not_configured(loguru_records):
    engine = _engine(OPENVOICE_V2)
    request = _tts(input_audio="legacy", input_vad_mode="silero")

    state = engine._route_request(request, {}, None)[0]

    assert request.model == "whisper"
    assert state.step_index == 0
    assert any("silero" in record.lower() for record in loguru_records)
