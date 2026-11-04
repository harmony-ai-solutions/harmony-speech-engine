"""Declarative pipeline routing tests (Phase 3-5).

The **parity table** below was transcribed from the pre-Phase-3 ``if/elif``
ladders in ``harmonyspeech_engine.py`` (``reroute_request_*`` for routing and
``check_forward_processing`` for forwarding) *before* those ladders were
deleted. Every family × every entry shape keeps its exact model sequence and
final output type.

The tests drive the real engine loop (``add_request`` → ``step`` →
``check_forward_processing`` → ``add_request``) with stub executors, so routing,
forwarding and the artifact bus are exercised end to end without model weights.
"""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, MelArtifact, SpeakerArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import (
    AudioConversionRequestInput,
    DetectVoiceActivityRequestInput,
    SpeechEmbeddingRequestInput,
    SpeechTranscribeRequestInput,
    SynthesisRequestInput,
    TextToSpeechRequestInput,
    VocodeRequestInput,
    VoiceConversionRequestInput,
)
from harmonyspeech.common.outputs import (
    AudioConversionRequestOutput,
    DetectVoiceActivityRequestOutput,
    RequestOutput,
    SpeechEmbeddingRequestOutput,
    SpeechSynthesisRequestOutput,
    SpeechTranscriptionRequestOutput,
    TextToSpeechRequestOutput,
    VocodeRequestOutput,
    VoiceConversionRequestOutput,
)
from harmonyspeech.common.request import EngineRequest, ExecutorResult, RequestStatus
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.processing.pipeline_registry import PIPELINE_SPECS, EntryRule, PipelineSpec, Step, get_spec
from harmonyspeech.processing.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Artifact stubs
# ---------------------------------------------------------------------------


def _audio(sample_rate: int = 16000) -> AudioArtifact:
    return AudioArtifact.from_numpy(np.zeros(64, dtype=np.float32), sample_rate)


def _mel(sample_rate: int = 16000) -> MelArtifact:
    return MelArtifact.from_numpy(np.zeros((80, 6), dtype=np.float32), sample_rate, 200, breaks=[6])


def _vad() -> VadArtifact:
    return VadArtifact(segments=[{"start": 0.0, "end": 3.0, "text": "hello world"}], source="whisper")


def _speaker(backend: str = "openvoice") -> SpeakerArtifact:
    if backend == "chatterbox":
        return SpeakerArtifact(conditionals=SimpleNamespace(), backend="chatterbox")
    return SpeakerArtifact(embedding=torch.zeros(3), backend=backend)


def _default_output(model_type: str, request, step_name: str | None = None) -> RequestOutput:
    """Return the typed output a real executor of ``model_type`` produces."""
    if step_name == "embed" and model_type in ("ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS"):
        # The TTS executor also serves embedding requests (prepare_conditionals).
        return SpeechEmbeddingRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", embedding_artifact=_speaker("chatterbox")
        )
    if model_type == "FasterWhisper":
        if isinstance(request, DetectVoiceActivityRequestInput):
            return DetectVoiceActivityRequestOutput(
                request_id=request.request_id, output="{}", finish_reason="stop", vad_artifact=_vad()
            )
        return SpeechTranscriptionRequestOutput(
            request_id=request.request_id, output="{}", finish_reason="stop", vad_artifact=_vad()
        )
    if model_type == "SileroVAD":
        vad = VadArtifact.from_silero_timestamps([{"start": 0, "end": 48000}])
        return DetectVoiceActivityRequestOutput(
            request_id=request.request_id, output="{}", finish_reason="stop", vad_artifact=vad
        )
    if model_type in ("HarmonySpeechEncoder", "OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"):
        backend = "harmonyspeech" if model_type == "HarmonySpeechEncoder" else "openvoice"
        return SpeechEmbeddingRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", embedding_artifact=_speaker(backend)
        )
    if model_type == "HarmonySpeechSynthesizer":
        return SpeechSynthesisRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", mel_artifact=_mel()
        )
    if model_type == "HarmonySpeechVocoder":
        return VocodeRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio(16000)
        )
    if model_type in ("OpenVoiceV1Synthesizer", "MeloTTSSynthesizer"):
        return SpeechSynthesisRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio(22050)
        )
    if model_type in ("OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"):
        return VoiceConversionRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio(22050)
        )
    if model_type == "VoiceFixerRestorer":
        return AudioConversionRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", mel_artifact=_mel(44100)
        )
    if model_type == "VoiceFixerVocoder":
        return AudioConversionRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio(44100)
        )
    if model_type == "ChatterboxEmbedding":
        return SpeechEmbeddingRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", embedding_artifact=_speaker("chatterbox")
        )
    if model_type == "ChatterboxVC":
        return VoiceConversionRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio(24000)
        )
    if model_type in ("ChatterboxTTS", "ChatterboxTurboTTS", "ChatterboxMultilingualTTS", "KittenTTSSynthesizer"):
        return TextToSpeechRequestOutput(
            request_id=request.request_id,
            text=getattr(request, "input_text", ""),
            output="",
            finish_reason="stop",
            audio_artifact=_audio(24000),
        )
    # Test-only model types (the new-toolchain extension test).
    if model_type == "ToyFirst":
        return SpeechTranscriptionRequestOutput(
            request_id=request.request_id, output="{}", finish_reason="stop", vad_artifact=_vad()
        )
    if model_type == "ToySecond":
        return SpeechSynthesisRequestOutput(
            request_id=request.request_id, output="", finish_reason="stop", audio_artifact=_audio()
        )
    raise AssertionError(f"no stub output for model_type {model_type!r}")


@dataclass
class _Call:
    model: str
    model_type: str
    step_index: int | None
    artifacts: frozenset
    artifact_map: dict
    engine_request: EngineRequest
    request: object


class _StubExecutor:
    """Executes ``_default_output`` and records every invocation."""

    def __init__(self, config: ModelConfig, calls: list[_Call], failing: set[str] | None = None):
        self.model_config = config
        self.calls = calls
        self.failing = failing or set()

    def execute_model(self, requests):
        results = []
        for engine_request in requests:
            request = engine_request.request_data
            state = getattr(engine_request, "pipeline_state", None)
            step_index = getattr(state, "step_index", None)
            spec = get_spec(getattr(request, "requested_model", None))
            step = spec.step_at(step_index) if spec is not None and step_index is not None else None
            artifact_map = dict(getattr(engine_request, "artifacts", None) or {})
            self.calls.append(
                _Call(
                    model=request.model,
                    model_type=self.model_config.model_type,
                    step_index=step_index,
                    artifacts=frozenset(artifact_map),
                    artifact_map=artifact_map,
                    engine_request=engine_request,
                    request=request,
                )
            )
            if self.model_config.model_type in self.failing:
                output = RequestOutput(request_id=request.request_id, finish_reason="error", error="boom")
            else:
                output = _default_output(self.model_config.model_type, request, getattr(step, "name", None))
            results.append(ExecutorResult(request.request_id, request, output))
        return results

    def check_health(self):
        return None


def _cfg(name: str, model_type: str, language: str | None = None, voices=None) -> ModelConfig:
    return ModelConfig(
        name=name,
        model=f"dummy/{name}",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        language=language,
        voices=voices,
    )


def _engine(configs, failing=None) -> tuple[HarmonySpeechEngine, list[_Call]]:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = list(configs)
    engine.log_stats = False
    engine.request_timeout_s = 120.0
    engine.strict_pipelines = False
    engine._pending_error_outputs = []
    engine.available_families = []
    calls: list[_Call] = []
    engine.model_executors = {config.name: _StubExecutor(config, calls, set(failing or set())) for config in configs}
    engine.scheduler = Scheduler(model_configs=list(configs))
    return engine, calls


def _run(engine, request, *, artifacts=None, max_steps=12):
    """Run ``request`` to completion and return the outputs plus call recorder."""
    engine.add_request("req-1", request, artifacts=artifacts)
    outputs: list[RequestOutput] = []
    for _ in range(max_steps):
        step_outputs, forwarded = engine.step()
        outputs.extend(step_outputs)
        if outputs:
            return outputs
        if not forwarded:
            break
    return outputs


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

VAD_JSON = json.dumps(
    {"text": "hello world", "segments": [{"start": 0.0, "end": 3.0, "text": "hello world"}], "info": {"language": "en"}}
)
MEL_JSON = json.dumps(np.zeros((128, 9), dtype=np.float32).tolist())
AUDIO_B64 = "legacy-audio"


def _tts(model: str, mode: str = "voice_cloning", **kwargs) -> TextToSpeechRequestInput:
    kwargs.setdefault("language_id", "EN")
    return TextToSpeechRequestInput(
        request_id="req-1", requested_model=model, model=model, input_text="hello", mode=mode, **kwargs
    )


def _embed(model: str, **kwargs) -> SpeechEmbeddingRequestInput:
    return SpeechEmbeddingRequestInput(request_id="req-1", requested_model=model, model=model, **kwargs)


def _synth(model: str, language: str = "EN") -> SynthesisRequestInput:
    return SynthesisRequestInput(
        request_id="req-1",
        requested_model=model,
        model=model,
        input_text="hello",
        language_id=language,
        voice_id="default",
        input_embedding="legacy-embedding",
    )


def _vocode(model: str) -> VocodeRequestInput:
    return VocodeRequestInput(request_id="req-1", requested_model=model, model=model, input_audio="legacy-mel")


def _ac(model: str, **kwargs) -> AudioConversionRequestInput:
    return AudioConversionRequestInput(
        request_id="req-1", requested_model=model, model=model, source_audio="legacy-audio", **kwargs
    )


def _vc(model: str) -> VoiceConversionRequestInput:
    return VoiceConversionRequestInput(
        request_id="req-1",
        requested_model=model,
        model=model,
        source_audio="legacy-audio",
        target_audio=None,
        target_embedding="legacy-embedding",
        generation_options=None,
        output_options=None,
    )


def _transcribe(model: str) -> SpeechTranscribeRequestInput:
    return SpeechTranscribeRequestInput(request_id="req-1", requested_model=model, model=model, input_audio=AUDIO_B64)


# ---------------------------------------------------------------------------
# Parity table
# ---------------------------------------------------------------------------

HARMONYSPEECH_CONFIGS = lambda: [  # noqa: E731
    _cfg("hs-enc", "HarmonySpeechEncoder"),
    _cfg("hs-syn", "HarmonySpeechSynthesizer"),
    _cfg("hs-voc", "HarmonySpeechVocoder"),
]
OPENVOICE_V1_CONFIGS = lambda: [  # noqa: E731
    _cfg("whisper", "FasterWhisper"),
    _cfg("ov1-enc", "OpenVoiceV1ToneConverterEncoder"),
    _cfg("ov1-syn", "OpenVoiceV1Synthesizer", "EN"),
    _cfg("ov1-tc", "OpenVoiceV1ToneConverter"),
]
OPENVOICE_V2_CONFIGS = lambda: [  # noqa: E731
    _cfg("whisper", "FasterWhisper"),
    _cfg("ov2-enc", "OpenVoiceV2ToneConverterEncoder"),
    _cfg("ov2-syn", "MeloTTSSynthesizer", "EN"),
    _cfg("ov2-tc", "OpenVoiceV2ToneConverter"),
]
VOICEFIXER_CONFIGS = lambda: [_cfg("vf-r", "VoiceFixerRestorer"), _cfg("vf-v", "VoiceFixerVocoder")]  # noqa: E731
CHATTERBOX_CONFIGS = lambda: [_cfg("cb-embed", "ChatterboxEmbedding"), _cfg("cb", "ChatterboxTTS")]  # noqa: E731
CHATTERBOX_ONLY_CONFIGS = lambda: [_cfg("cb", "ChatterboxTTS")]  # noqa: E731


@dataclass
class ParityCase:
    name: str
    request: object
    configs: list
    expected_models: list[str]
    expected_output: type
    artifacts: dict | None = None
    field: dict = field(default_factory=dict)


def _harmonyspeech_cases():
    return [
        ParityCase(
            "harmonyspeech/tts_cloning_from_audio",
            _tts("harmonyspeech", input_audio=AUDIO_B64),
            HARMONYSPEECH_CONFIGS(),
            ["hs-enc", "hs-syn", "hs-voc"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "harmonyspeech/tts_with_embedding",
            _tts("harmonyspeech", input_embedding="legacy-embedding"),
            HARMONYSPEECH_CONFIGS(),
            ["hs-syn", "hs-voc"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "harmonyspeech/tts_vocode",
            _tts("harmonyspeech", input_audio=AUDIO_B64, input_embedding="legacy-embedding"),
            HARMONYSPEECH_CONFIGS(),
            ["hs-voc"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "harmonyspeech/embed_request",
            _embed("harmonyspeech", input_audio=AUDIO_B64),
            HARMONYSPEECH_CONFIGS(),
            ["hs-enc"],
            SpeechEmbeddingRequestOutput,
        ),
        ParityCase(
            "harmonyspeech/synthesis_request",
            _synth("harmonyspeech"),
            HARMONYSPEECH_CONFIGS(),
            ["hs-syn"],
            SpeechSynthesisRequestOutput,
        ),
        ParityCase(
            "harmonyspeech/vocode_request",
            _vocode("harmonyspeech"),
            HARMONYSPEECH_CONFIGS(),
            ["hs-voc"],
            VocodeRequestOutput,
        ),
    ]


def _openvoice_cases(family: str, configs, prefix: str) -> list[ParityCase]:
    enc, syn, tc = f"{prefix}-enc", f"{prefix}-syn", f"{prefix}-tc"
    return [
        ParityCase(
            f"{family}/tts_cloning_from_audio",
            _tts(family, input_audio=AUDIO_B64),
            configs,
            ["whisper", enc, syn, tc],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            f"{family}/tts_cloning_from_vad",
            _tts(family, input_audio=AUDIO_B64, input_vad_data=VAD_JSON),
            configs,
            [enc, syn, tc],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            f"{family}/tts_final_conversion",
            _tts(family, input_audio=AUDIO_B64, input_vad_data=VAD_JSON, input_embedding="legacy-embedding"),
            configs,
            [tc],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            f"{family}/tts_cloning_from_embedding",
            _tts(family, input_embedding="legacy-embedding"),
            configs,
            [syn, tc],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            f"{family}/tts_single_speaker",
            _tts(family, mode="single_speaker_tts"),
            configs,
            [syn],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            f"{family}/embed_from_audio",
            _embed(family, input_audio=AUDIO_B64),
            configs,
            ["whisper", enc],
            SpeechEmbeddingRequestOutput,
        ),
        ParityCase(
            f"{family}/embed_with_vad",
            _embed(family, input_audio=AUDIO_B64, input_vad_data=VAD_JSON),
            configs,
            [enc],
            SpeechEmbeddingRequestOutput,
        ),
        ParityCase(f"{family}/synthesis_request", _synth(family), configs, [syn], SpeechSynthesisRequestOutput),
        ParityCase(f"{family}/voice_conversion", _vc(family), configs, [tc], VoiceConversionRequestOutput),
        ParityCase(f"{family}/transcribe", _transcribe(family), configs, ["whisper"], SpeechTranscriptionRequestOutput),
        ParityCase(
            f"{family}/explicit_synthesizer_model",
            _tts(syn, mode="single_speaker_tts"),
            configs,
            [syn],
            SpeechSynthesisRequestOutput,
        ),
    ]


def _voicefixer_cases():
    return [
        ParityCase(
            "voicefixer/restore_then_vocode",
            _ac("voicefixer"),
            VOICEFIXER_CONFIGS(),
            ["vf-r", "vf-v"],
            AudioConversionRequestOutput,
        ),
        ParityCase(
            "voicefixer/vocode_only",
            _ac("voicefixer", input_mel_spectrogram=MEL_JSON),
            VOICEFIXER_CONFIGS(),
            ["vf-v"],
            AudioConversionRequestOutput,
        ),
    ]


def _chatterbox_cases():
    return [
        ParityCase(
            "chatterbox/tts_cloning_with_embed_executor",
            _tts("chatterbox", input_audio=AUDIO_B64),
            CHATTERBOX_CONFIGS(),
            ["cb-embed", "cb"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox/tts_cloning_without_embed_executor",
            _tts("chatterbox", input_audio=AUDIO_B64),
            CHATTERBOX_ONLY_CONFIGS(),
            ["cb"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox/tts_single_speaker",
            _tts("chatterbox", mode="single_speaker_tts"),
            CHATTERBOX_CONFIGS(),
            ["cb"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox/tts_with_embedding",
            _tts("chatterbox", input_embedding="legacy-embedding"),
            CHATTERBOX_CONFIGS(),
            ["cb"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox/embed_request",
            _embed("chatterbox", input_audio=AUDIO_B64),
            CHATTERBOX_CONFIGS(),
            ["cb-embed"],
            SpeechEmbeddingRequestOutput,
        ),
        ParityCase(
            "chatterbox/embed_falls_back_to_tts",
            _embed("chatterbox", input_audio=AUDIO_B64),
            CHATTERBOX_ONLY_CONFIGS(),
            ["cb"],
            SpeechEmbeddingRequestOutput,
        ),
        ParityCase(
            "chatterbox_turbo/tts_single_speaker",
            _tts("chatterbox_turbo", mode="single_speaker_tts"),
            [_cfg("cb-turbo", "ChatterboxTurboTTS")],
            ["cb-turbo"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox_multilingual/tts_single_speaker",
            _tts("chatterbox_multilingual", mode="single_speaker_tts"),
            [_cfg("cb-multi", "ChatterboxMultilingualTTS")],
            ["cb-multi"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "chatterbox_vc/voice_conversion",
            _vc("chatterbox_vc"),
            [_cfg("cb-vc", "ChatterboxVC")],
            ["cb-vc"],
            VoiceConversionRequestOutput,
        ),
        ParityCase(
            "kitten_tts/explicit_model",
            _tts("kitten-tts-mini", mode="single_speaker_tts"),
            [_cfg("kitten-tts-mini", "KittenTTSSynthesizer", voices=["Jasper"])],
            ["kitten-tts-mini"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "kitten_tts/sentinel_alias",
            _tts("kitten_tts", mode="single_speaker_tts"),
            [_cfg("kitten-tts-mini", "KittenTTSSynthesizer", voices=["Jasper"])],
            ["kitten-tts-mini"],
            TextToSpeechRequestOutput,
        ),
        ParityCase(
            "fasterwhisper/explicit_model",
            _transcribe("faster-whisper"),
            [_cfg("faster-whisper", "FasterWhisper")],
            ["faster-whisper"],
            SpeechTranscriptionRequestOutput,
        ),
        ParityCase(
            "silero_vad/explicit_model",
            DetectVoiceActivityRequestInput(
                request_id="req-1", requested_model="silero-vad", model="silero-vad", input_audio=AUDIO_B64
            ),
            [_cfg("silero-vad", "SileroVAD")],
            ["silero-vad"],
            DetectVoiceActivityRequestOutput,
        ),
    ]


PARITY_CASES = (
    _harmonyspeech_cases()
    + _openvoice_cases("openvoice_v1", OPENVOICE_V1_CONFIGS(), "ov1")
    + _openvoice_cases("openvoice_v2", OPENVOICE_V2_CONFIGS(), "ov2")
    + _voicefixer_cases()
    + _chatterbox_cases()
)


@pytest.mark.unit
@pytest.mark.parametrize("case", PARITY_CASES, ids=[case.name for case in PARITY_CASES])
def test_parity_model_sequence_and_final_output(case):
    engine, calls = _engine(case.configs)

    outputs = _run(engine, case.request, artifacts=case.artifacts)

    assert [call.model for call in calls] == case.expected_models
    assert len(outputs) == 1, outputs
    output = outputs[0]
    assert type(output) is case.expected_output, f"{type(output).__name__} != {case.expected_output.__name__}"
    assert output.finish_reason == "stop"
    assert not engine.scheduler.has_unfinished_requests()


@pytest.mark.unit
def test_parity_table_covers_every_pipeline_family():
    covered = {case.request.requested_model for case in PARITY_CASES}
    covered |= {case.name.split("/")[0] for case in PARITY_CASES}
    assert {"harmonyspeech", "openvoice_v1", "openvoice_v2", "voicefixer"} <= covered
    assert {"chatterbox", "chatterbox_turbo", "chatterbox_multilingual", "chatterbox_vc"} <= covered
    assert {"kitten_tts", "fasterwhisper", "silero_vad"} <= covered
    # Every registered pipeline appears in the table.
    assert set(PIPELINE_SPECS) <= covered


# ---------------------------------------------------------------------------
# Artifact flow
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openvoice_artifact_flow_declares_produces_and_never_writes_wire_fields():
    engine, calls = _engine(OPENVOICE_V2_CONFIGS())
    request = _tts("openvoice_v2", input_audio=AUDIO_B64)

    # The serving boundary decodes the reference clip once at admission (Phase 2).
    _run(engine, request, artifacts={"input_audio": _audio()})

    assert [call.model for call in calls] == ["whisper", "ov2-enc", "ov2-syn", "ov2-tc"]
    # Step indices walk forward one hop at a time.
    assert [call.step_index for call in calls] == [0, 1, 2, 3]
    # Each hop sees exactly the artifacts produced so far.
    assert calls[0].artifacts == frozenset({"input_audio"})
    assert calls[1].artifacts == frozenset({"input_audio", "input_vad_data"})
    assert calls[2].artifacts == frozenset({"input_vad_data", "input_embedding"})
    assert calls[3].artifacts == frozenset({"input_vad_data", "input_embedding", "input_audio"})
    # No wire field is written on an internal hop.
    assert request.input_vad_data is None
    assert request.input_embedding is None
    assert request.input_audio is None  # cleared by the embed step


@pytest.mark.unit
def test_harmonyspeech_mel_hop_travels_as_a_mel_artifact():
    engine, calls = _engine(HARMONYSPEECH_CONFIGS())
    request = _tts("harmonyspeech", input_audio=AUDIO_B64)

    _run(engine, request)

    vocoder_call = calls[2]
    artifacts = vocoder_call.artifact_map
    assert isinstance(artifacts["input_audio"], MelArtifact)
    assert isinstance(artifacts["input_embedding"], SpeakerArtifact)


@pytest.mark.unit
def test_forwarding_writes_only_declared_produces_keys():
    """Every hop's new artifact key set is exactly the step's ``produces``."""
    engine, calls = _engine(OPENVOICE_V2_CONFIGS())
    _run(engine, _tts("openvoice_v2", input_audio=AUDIO_B64))

    spec = PIPELINE_SPECS["openvoice_v2"]
    for previous_call, call in zip(calls, calls[1:], strict=False):
        produced = {field for _slot, field in spec.steps[previous_call.step_index].produces}
        new_keys = set(call.artifacts) - set(previous_call.artifacts)
        # Exactly the declared keys appear; nothing undeclared is written.
        assert new_keys == produced


# ---------------------------------------------------------------------------
# Silero VAD inside the pipeline (3-4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_input_vad_mode_silero_routes_the_vad_step_to_silero():
    configs = OPENVOICE_V2_CONFIGS() + [_cfg("silero", "SileroVAD")]
    engine, calls = _engine(configs)
    request = _tts("openvoice_v2", input_audio=AUDIO_B64, input_vad_mode="silero")

    outputs = _run(engine, request)

    assert [call.model for call in calls] == ["silero", "ov2-enc", "ov2-syn", "ov2-tc"]
    assert len(outputs) == 1 and outputs[0].finish_reason == "stop"
    # Silero's segments survive the encoder step: the pipeline got past embed.
    assert "input_vad_data" in calls[1].artifacts
    vad = calls[1].artifact_map["input_vad_data"]
    assert vad.source == "silero"
    assert vad.segments and vad.segments[0]["text"] == ""


@pytest.mark.unit
def test_input_vad_mode_whisper_defaults_to_whisper():
    configs = OPENVOICE_V2_CONFIGS() + [_cfg("silero", "SileroVAD")]
    engine, calls = _engine(configs)
    _run(engine, _tts("openvoice_v2", input_audio=AUDIO_B64, input_vad_mode="whisper"))

    assert calls[0].model == "whisper"


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "failing_type",
    ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder", "MeloTTSSynthesizer", "OpenVoiceV2ToneConverter"],
)
def test_mid_pipeline_error_at_every_step_finishes_with_error(failing_type):
    engine, calls = _engine(OPENVOICE_V2_CONFIGS(), failing={failing_type})

    outputs = _run(engine, _tts("openvoice_v2", input_audio=AUDIO_B64))

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert outputs[0].error == "boom"
    # The request is not requeued or re-forwarded after the failure.
    assert not engine.scheduler.has_unfinished_requests()
    assert [call.model_type for call in calls][-1] == failing_type


@pytest.mark.unit
def test_missing_step_artifact_fails_naming_the_step():
    class _BrokenEmbeddingExecutor(_StubExecutor):
        def execute_model(self, requests):
            results = []
            for engine_request in requests:
                request = engine_request.request_data
                self.calls.append(
                    _Call(
                        model=request.model,
                        model_type=self.model_config.model_type,
                        step_index=getattr(engine_request.pipeline_state, "step_index", None),
                        artifacts=frozenset(),
                        artifact_map={},
                        engine_request=engine_request,
                        request=request,
                    )
                )
                # Speaker artifact deliberately omitted.
                output = SpeechEmbeddingRequestOutput(request_id=request.request_id, output="", finish_reason="stop")
                results.append(ExecutorResult(request.request_id, request, output))
            return results

    configs = OPENVOICE_V2_CONFIGS()
    engine, _calls = _engine(configs)
    engine.model_executors["ov2-enc"] = _BrokenEmbeddingExecutor(configs[1], [])

    outputs = _run(engine, _tts("openvoice_v2", input_audio=AUDIO_B64, input_vad_data=VAD_JSON))

    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert "openvoice_v2" in outputs[0].error
    assert "'embed'" in outputs[0].error
    assert "embedding_artifact" in outputs[0].error
    assert not engine.scheduler.has_unfinished_requests()


@pytest.mark.unit
def test_client_disconnect_mid_pipeline_stops_forwarding():
    engine, calls = _engine(OPENVOICE_V2_CONFIGS())
    engine.add_request("req-1", _tts("openvoice_v2", input_audio=AUDIO_B64))

    step_outputs, forwarded = engine.step()
    assert step_outputs == []
    assert forwarded  # the vad hop is in flight for the next step
    assert len(calls) == 1

    engine.abort_request("req-1")
    outputs, forwarded = engine.step()

    assert outputs == []
    assert forwarded == []
    assert len(calls) == 1  # no further hop ran
    assert not engine.scheduler.has_unfinished_requests()


@pytest.mark.unit
def test_unsupported_request_shape_is_a_typed_error():
    engine, calls = _engine(OPENVOICE_V2_CONFIGS())
    # single-speaker TTS *with* reference audio is not a supported OpenVoice shape.
    request = _tts("openvoice_v2", mode="single_speaker_tts", input_audio=AUDIO_B64)

    outputs = _run(engine, request)

    assert calls == []
    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert "openvoice_v2" in outputs[0].error


@pytest.mark.unit
def test_unresolvable_family_never_enters_the_scheduler():
    engine, calls = _engine([_cfg("ov2-syn", "MeloTTSSynthesizer", "EN")])

    outputs = _run(engine, _tts("openvoice_v2", input_audio=AUDIO_B64))

    assert calls == []
    assert len(outputs) == 1
    assert outputs[0].finish_reason == "error"
    assert "openvoice_v2" in outputs[0].error
    assert "missing one of" in outputs[0].error
    assert "OpenVoiceV2ToneConverter" in outputs[0].error


# ---------------------------------------------------------------------------
# New toolchain = one registry entry (exit criterion 1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_two_step_toolchain_is_a_registry_only_extension(monkeypatch):
    """A brand-new toolchain works by registering one ``PipelineSpec``."""
    toy_spec = PipelineSpec(
        id="toy_chain",
        description="Test-only two-step toolchain (registry extension proof)",
        steps=(
            Step(
                name="analyse",
                requires=("ToyFirst",),
                needs=frozenset({"input_audio"}),
                produces=(("vad_artifact", "input_vad_data"),),
            ),
            Step(
                name="speak",
                requires=("ToySecond",),
                needs=frozenset({"input_vad_data"}),
                produces=(("audio_artifact", "input_audio"),),
                advertises=("tts",),
            ),
        ),
        final_step="speak",
        capabilities=frozenset({"tts"}),
        entry_rules=(EntryRule("start", lambda request, artifacts: True, "analyse"),),
    )
    monkeypatch.setitem(PIPELINE_SPECS, "toy_chain", toy_spec)

    configs = [_cfg("toy-first", "ToyFirst"), _cfg("toy-second", "ToySecond")]
    engine, calls = _engine(configs)
    assert get_spec("toy_chain") is toy_spec

    outputs = _run(engine, _tts("toy_chain", input_audio=AUDIO_B64))

    assert [call.model for call in calls] == ["toy-first", "toy-second"]
    assert len(outputs) == 1
    assert isinstance(outputs[0], TextToSpeechRequestOutput)
    assert outputs[0].audio_artifact is not None

    # ...and the registry alone is enough to advertise it.
    from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing

    card_ids = {card.id for card in OpenAIServing.model_cards_from_specs(configs, "tts")}
    assert "toy_chain" in card_ids


# ---------------------------------------------------------------------------
# Strict pipelines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strict_pipelines_boot_error_names_pipeline_and_step():
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = [_cfg("ov2-syn", "MeloTTSSynthesizer", "EN")]
    engine.strict_pipelines = True

    with pytest.raises(RuntimeError) as excinfo:
        engine._validate_family_configuration()

    message = str(excinfo.value)
    assert "openvoice_v2" in message
    assert "OpenVoiceV2ToneConverterEncoder" in message
    assert "embed" in message


@pytest.mark.unit
def test_incomplete_pipeline_warns_by_default(loguru_records):
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = [_cfg("ov2-syn", "MeloTTSSynthesizer", "EN")]
    engine.strict_pipelines = False

    available = engine._validate_family_configuration()

    assert "openvoice_v2" not in available
    assert "openvoice_v2" in "\n".join(loguru_records)


@pytest.mark.unit
def test_cli_exposes_strict_pipelines_flag():
    from harmonyspeech.endpoints.openai.args import make_arg_parser

    args = make_arg_parser().parse_args(["--strict-pipelines"])

    assert args.strict_pipelines is True


@pytest.mark.unit
def test_pipeline_state_is_released_with_the_request():
    """The finished request drops its artifact store (and thus pipeline state)."""
    engine, calls = _engine(HARMONYSPEECH_CONFIGS())

    _run(engine, _tts("harmonyspeech", input_audio=AUDIO_B64))

    first = calls[0].engine_request
    assert first.status == RequestStatus.FINISHED_FORWARDED or first.is_finished()
    assert first.artifacts is None
