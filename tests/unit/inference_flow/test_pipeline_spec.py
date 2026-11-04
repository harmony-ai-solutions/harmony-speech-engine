"""Unit tests for the declarative pipeline model (Phase 3-1).

Covers ``next_step`` resolution semantics (preference order, language variants,
artifact-first ``needs``, VAD-mode preference) and ``advance`` forwarding
semantics (declared ``produces`` mapping, field clearing, typed errors).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput, TextToSpeechRequestInput
from harmonyspeech.common.outputs import (
    SpeechEmbeddingRequestOutput,
    SpeechSynthesisRequestOutput,
    SpeechTranscriptionRequestOutput,
)
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.processing.pipeline_registry import (
    FAMILY_REQUIREMENTS,
    PIPELINE_SPECS,
    MissingStepArtifactError,
    MissingStepInputError,
    PipelineExecution,
    PipelineState,
    Step,
    advance,
    completed_step_state,
    get_spec,
    next_step,
    resolve_entry,
)


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


def _tts(**kwargs) -> TextToSpeechRequestInput:
    defaults = {
        "request_id": "spec-1",
        "requested_model": "openvoice_v2",
        "model": "openvoice_v2",
        "input_text": "hello",
        "mode": "voice_cloning",
    }
    defaults.update(kwargs)
    return TextToSpeechRequestInput(**defaults)


def _engine_request(request, artifacts=None) -> EngineRequest:
    return EngineRequest(request_id=request.request_id, request_data=request, arrival_time=0.0, artifacts=artifacts)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_contains_every_documented_pipeline():
    assert {
        "harmonyspeech",
        "openvoice_v1",
        "openvoice_v2",
        "voicefixer",
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "chatterbox_vc",
        "kitten_tts",
        "fasterwhisper",
        "silero_vad",
    } == set(PIPELINE_SPECS)


@pytest.mark.unit
def test_family_requirements_are_derived_from_the_registry():
    # Legacy Phase 0-4 surface: the eight client-facing toolchains, derived.
    assert set(FAMILY_REQUIREMENTS) == {
        "harmonyspeech",
        "openvoice_v1",
        "openvoice_v2",
        "voicefixer",
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "chatterbox_vc",
    }
    assert FAMILY_REQUIREMENTS["openvoice_v2"] == [
        "FasterWhisper",
        "OpenVoiceV2ToneConverterEncoder",
        "MeloTTSSynthesizer",
        "OpenVoiceV2ToneConverter",
    ]


@pytest.mark.unit
def test_openvoice_pipelines_have_four_steps_in_order():
    for spec_id in ("openvoice_v1", "openvoice_v2"):
        spec = PIPELINE_SPECS[spec_id]
        assert [step.name for step in spec.steps] == ["vad", "embed", "synthesize", "tone_convert"]
        assert spec.final_step == "tone_convert"
        assert spec.capabilities == frozenset({"tts", "embed", "vc"})


@pytest.mark.unit
def test_single_step_pipelines_are_flagged():
    for spec_id in ("kitten_tts", "fasterwhisper", "silero_vad"):
        spec = PIPELINE_SPECS[spec_id]
        assert spec.step_at(0) is not None
        assert spec.single_step
        assert spec.advertise_toolchain is False
    # ``chatterbox_vc`` is a single executor but still a client-facing alias.
    assert PIPELINE_SPECS["chatterbox_vc"].advertise_toolchain


# ---------------------------------------------------------------------------
# get_spec
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_spec_resolves_sentinels_only():
    assert get_spec("openvoice_v2").id == "openvoice_v2"
    assert get_spec("kitten_tts").id == "kitten_tts"
    assert get_spec("ov1-synthesizer-en") is None  # concrete model name
    assert get_spec("not_a_family") is None
    assert get_spec(None) is None


# ---------------------------------------------------------------------------
# next_step: preference order, language, needs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_next_step_prefers_the_first_required_type():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy")
    state = resolve_entry(spec, request, {})
    configs = [_cfg("silero", "SileroVAD"), _cfg("whisper", "FasterWhisper")]

    config = next_step(spec, state, request, configs, {})

    assert config.name == "whisper"  # requires order wins over config order


@pytest.mark.unit
def test_next_step_falls_back_to_alternative_backend():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy")
    state = resolve_entry(spec, request, {})

    config = next_step(spec, state, request, [_cfg("silero", "SileroVAD")], {})

    assert config.name == "silero"


@pytest.mark.unit
def test_next_step_language_variants_match_the_request():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy", language_id="ZH")
    state = resolve_entry(spec, request, {})
    state.step_index = spec.step_index("synthesize")
    configs = [_cfg("ov2-synth-en", "MeloTTSSynthesizer", "EN"), _cfg("ov2-synth-zh", "MeloTTSSynthesizer", "ZH")]

    config = next_step(spec, state, request, configs, {})

    assert config.name == "ov2-synth-zh"


@pytest.mark.unit
def test_next_step_returns_none_when_language_is_not_configured():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy", language_id="KR")
    state = resolve_entry(spec, request, {})
    state.step_index = spec.step_index("synthesize")

    assert next_step(spec, state, request, [_cfg("ov2-synth-en", "MeloTTSSynthesizer", "EN")], {}) is None


@pytest.mark.unit
def test_next_step_returns_none_without_a_matching_config():
    spec = PIPELINE_SPECS["voicefixer"]
    request = SimpleNamespace(
        request_id="vf",
        requested_model="voicefixer",
        model="voicefixer",
        source_audio="legacy",
        input_mel_spectrogram=None,
    )
    state = resolve_entry(spec, request, {})

    assert next_step(spec, state, request, [_cfg("other", "SomeOtherType")], {}) is None


@pytest.mark.unit
def test_needs_is_satisfied_by_the_artifact_store():
    """An artifact counts as a present field even when the wire field is unset."""
    spec = PIPELINE_SPECS["voicefixer"]
    request = SimpleNamespace(
        request_id="vf", requested_model="voicefixer", model="voicefixer", source_audio=None, input_mel_spectrogram=None
    )
    vad = MelArtifact.from_numpy(np.zeros((128, 4), dtype=np.float32), 44100, 441)
    state = PipelineState(spec_id="voicefixer", step_index=spec.step_index("vocode"), stop_after="vocode")

    config = next_step(spec, state, request, [_cfg("vf-voc", "VoiceFixerVocoder")], {"input_mel_spectrogram": vad})

    assert config.name == "vf-voc"


@pytest.mark.unit
def test_next_step_raises_typed_error_when_a_need_is_missing():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts()  # no audio, no vad data
    # Force the embed step, which needs audio + vad data.
    state = PipelineState(spec_id="openvoice_v2", step_index=spec.step_index("embed"))

    with pytest.raises(MissingStepInputError) as excinfo:
        next_step(spec, state, request, [_cfg("enc", "OpenVoiceV2ToneConverterEncoder")], {})

    assert excinfo.value.step_name == "embed"
    assert "input_audio" in excinfo.value.message
    assert "input_vad_data" in excinfo.value.message


@pytest.mark.unit
def test_vad_mode_silero_prefers_the_silero_config():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy", input_vad_mode="silero")
    state = resolve_entry(spec, request, {})
    configs = [_cfg("whisper", "FasterWhisper"), _cfg("silero", "SileroVAD")]

    config = next_step(spec, state, request, configs, {})

    assert config.name == "silero"


@pytest.mark.unit
def test_vad_mode_whisper_prefers_whisper_even_when_silero_is_first_in_config():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy", input_vad_mode="whisper")
    state = resolve_entry(spec, request, {})
    configs = [_cfg("silero", "SileroVAD"), _cfg("whisper", "FasterWhisper")]

    config = next_step(spec, state, request, configs, {})

    assert config.name == "whisper"


@pytest.mark.unit
def test_vad_mode_silero_falls_back_to_whisper_with_a_warning(loguru_records):
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = _tts(input_audio="legacy", input_vad_mode="silero")
    state = resolve_entry(spec, request, {})

    config = next_step(spec, state, request, [_cfg("whisper", "FasterWhisper")], {})

    assert config.name == "whisper"
    warnings = "\n".join(loguru_records)
    assert "silero" in warnings.lower()
    assert "FasterWhisper" in warnings


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_advance_maps_artifacts_and_stops_at_the_final_step():
    spec = PIPELINE_SPECS["chatterbox"]
    request = _tts(requested_model="chatterbox", input_audio="legacy")
    engine_request = _engine_request(request, {"input_audio": object()})
    engine_request.pipeline_state = PipelineState(
        spec_id="chatterbox", step_index=spec.step_index("embed"), stop_after="synthesize"
    )

    speaker = SpeakerArtifact(embedding=None, conditionals=SimpleNamespace(), backend="chatterbox")
    done = advance(
        spec, engine_request, SpeechEmbeddingRequestOutput(request_id="spec-1", output="", embedding_artifact=speaker)
    )

    assert done is False
    assert engine_request.pipeline_state.step_index == spec.step_index("synthesize")
    assert engine_request.artifacts["input_embedding"] is speaker
    # ``clears`` drops the superseded reference audio from both stores.
    assert "input_audio" not in engine_request.artifacts
    assert request.input_audio is None

    audio = AudioArtifact.from_numpy(np.zeros(8, dtype=np.float32), 24000)
    done = advance(
        spec, engine_request, SpeechSynthesisRequestOutput(request_id="spec-1", output="", audio_artifact=audio)
    )

    assert done is True
    assert engine_request.artifacts["input_audio"] is audio


@pytest.mark.unit
def test_advance_stops_at_the_entry_rule_stop_after():
    spec = PIPELINE_SPECS["openvoice_v2"]
    request = SpeechEmbeddingRequestInput(
        request_id="emb-1", requested_model="openvoice_v2", model="whisper", input_audio="legacy"
    )
    engine_request = _engine_request(request, {"input_audio": object()})
    state = resolve_entry(spec, request, engine_request.artifacts)
    assert state.stop_after == "embed"
    engine_request.pipeline_state = state

    vad = VadArtifact(segments=[{"start": 0.0, "end": 3.0, "text": "hi"}], source="whisper")

    assert (
        advance(spec, engine_request, SpeechTranscriptionRequestOutput(request_id="emb-1", output="", vad_artifact=vad))
        is False
    )
    engine_request.pipeline_state.step_index = spec.step_index("embed")

    speaker = SpeakerArtifact(embedding=torch.zeros(3), backend="openvoice")
    done = advance(
        spec, engine_request, SpeechEmbeddingRequestOutput(request_id="emb-1", output="", embedding_artifact=speaker)
    )

    assert done is True  # embedding-only request finishes after ``embed``


@pytest.mark.unit
def test_advance_without_state_is_a_typed_error():
    spec = PIPELINE_SPECS["kitten_tts"]
    context = PipelineExecution(request_data=_tts(requested_model="kitten_tts"), artifacts={})

    with pytest.raises(MissingStepArtifactError) as excinfo:
        advance(spec, context, SpeechSynthesisRequestOutput(request_id="x", output="", audio_artifact=object()))

    assert "kitten_tts" in excinfo.value.message


@pytest.mark.unit
def test_advance_missing_artifact_names_pipeline_and_step():
    spec = PIPELINE_SPECS["chatterbox"]
    request = _tts(requested_model="chatterbox", input_audio="legacy")
    engine_request = _engine_request(request, {"input_audio": object()})
    engine_request.pipeline_state = PipelineState(
        spec_id="chatterbox", step_index=spec.step_index("embed"), stop_after="synthesize"
    )

    with pytest.raises(MissingStepArtifactError) as excinfo:
        advance(spec, engine_request, SpeechEmbeddingRequestOutput(request_id="spec-1", output=""))

    message = excinfo.value.message
    assert "chatterbox" in message
    assert "'embed'" in message
    assert "embedding_artifact" in message


@pytest.mark.unit
def test_completed_step_state_finds_the_step_that_ran():
    spec = PIPELINE_SPECS["openvoice_v2"]

    state = completed_step_state(spec, "OpenVoiceV2ToneConverterEncoder")

    assert state.step_index == spec.step_index("embed")
    assert completed_step_state(spec, "UnknownType") is None


@pytest.mark.unit
def test_step_rejects_preference_keys_without_a_field():
    step = Step(name="x", requires=("A",), requires_by_preference=(("silero", ("B",)),))

    assert step.candidates_for(SimpleNamespace()) == ("A",)
    assert step.candidates_for(SimpleNamespace(input_vad_mode="silero")) == ("A",)
