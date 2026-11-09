"""Unit tests for spec-driven startup validation and model cards (Phase 3-3).

``validate_registry`` replaces the hand-maintained family requirements and the
serving modules no longer carry their own ``_*_MODEL_TYPES`` / ``_*_MODEL_GROUPS``
lists: everything advertised on ``/v1/models`` is generated from
``PIPELINE_SPECS``.
"""

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.endpoints.openai.serving_audio_conversion import OpenAIServingAudioConversion
from harmonyspeech.endpoints.openai.serving_speech_to_text import OpenAIServingSpeechToText
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.endpoints.openai.serving_voice_activity_detection import OpenAIServingVoiceActivityDetection
from harmonyspeech.endpoints.openai.serving_voice_conversion import OpenAIServingVoiceConversion
from harmonyspeech.endpoints.openai.serving_voice_embed import OpenAIServingVoiceEmbedding
from harmonyspeech.processing.pipeline_registry import (
    PIPELINE_SPECS,
    model_types_advertising,
    specs_for_capabilities,
    validate_registry,
)

SERVING_BY_CAPABILITY = {
    "tts": OpenAIServingTextToSpeech,
    "embed": OpenAIServingVoiceEmbedding,
    "stt": OpenAIServingSpeechToText,
    "vc": OpenAIServingVoiceConversion,
    "vad": OpenAIServingVoiceActivityDetection,
    "restore": OpenAIServingAudioConversion,
}


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


def _complete_openvoice_v2() -> list[ModelConfig]:
    return [
        _cfg("whisper", "FasterWhisper"),
        _cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer", "EN", ["EN-Newest"]),
        _cfg("ov2-synthesizer-zh", "MeloTTSSynthesizer", "ZH", ["ZH"]),
        _cfg("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]


# ---------------------------------------------------------------------------
# validate_registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_registry_names_the_missing_step_and_model_type():
    configs = [
        _cfg("whisper", "FasterWhisper"),
        _cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer", "EN"),
        # OpenVoiceV2ToneConverter deliberately missing
    ]

    incomplete = validate_registry(configs)

    assert incomplete["openvoice_v2"] == ["tone_convert (OpenVoiceV2ToneConverter)"]


@pytest.mark.unit
def test_validate_registry_reports_alternative_backends_as_one_step():
    configs = [_cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder")]

    incomplete = validate_registry(configs)

    assert "vad (SileroVAD, FasterWhisper)" in incomplete["openvoice_v2"]


@pytest.mark.unit
def test_optional_steps_never_make_a_pipeline_incomplete():
    # No ChatterboxEmbedding configured: still complete (optional step).
    incomplete = validate_registry([_cfg("cb", "ChatterboxTTS")])

    assert "chatterbox" not in incomplete


@pytest.mark.unit
def test_silero_vad_satisfies_the_openvoice_vad_step():
    configs = [
        _cfg("silero", "SileroVAD"),
        _cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer", "EN"),
        _cfg("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]

    assert "openvoice_v2" not in validate_registry(configs)


@pytest.mark.unit
def test_language_variants_only_need_one_configured_variant():
    configs = _complete_openvoice_v2()
    registry = validate_registry(configs)
    assert "openvoice_v2" not in registry

    # Adding a second language variant does not change completeness.
    configs.append(_cfg("ov2-synthesizer-fr", "MeloTTSSynthesizer", "FR"))
    assert "openvoice_v2" not in validate_registry(configs)


# ---------------------------------------------------------------------------
# Model cards from specs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cards_omit_incomplete_family_but_keep_individual_models():
    configs = [
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer", "EN"),
        _cfg("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]

    card_ids = {card.id for card in OpenAIServingTextToSpeech.models_from_config(configs)}

    assert "openvoice_v2" not in card_ids
    assert "ov2-synthesizer-en" in card_ids


@pytest.mark.unit
def test_incomplete_family_does_not_affect_other_families():
    configs = [_cfg("cb", "ChatterboxTTS"), _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer", "EN")]

    card_ids = {card.id for card in OpenAIServingTextToSpeech.models_from_config(configs)}

    assert "chatterbox" in card_ids
    assert "openvoice_v2" not in card_ids


@pytest.mark.unit
def test_toolchain_card_carries_capabilities_and_members():
    configs = _complete_openvoice_v2()

    cards = {card.id: card for card in OpenAIServingTextToSpeech.models_from_config(configs)}
    card = cards["openvoice_v2"]

    assert card.object == "toolchain"
    assert card.capabilities == ["embed", "tts", "vc"]
    assert set(card.models) == {
        "whisper",
        "ov2-encoder",
        "ov2-synthesizer-en",
        "ov2-synthesizer-zh",
        "ov2-tone-converter",
    }


@pytest.mark.unit
def test_toolchain_card_lists_languages_from_resolved_variants():
    configs = _complete_openvoice_v2()

    cards = {card.id: card for card in OpenAIServingTextToSpeech.models_from_config(configs)}
    languages = {option.language for option in cards["openvoice_v2"].languages}

    assert languages == {"EN", "ZH"}
    # Voices travel with their language.
    voices_en = next(option for option in cards["openvoice_v2"].languages if option.language == "EN")
    assert [voice.voice for voice in voices_en.voices] == ["EN-Newest"]


@pytest.mark.unit
def test_cards_are_filtered_by_endpoint_capability():
    configs = _complete_openvoice_v2() + [_cfg("vf-r", "VoiceFixerRestorer"), _cfg("vf-v", "VoiceFixerVocoder")]
    expected = {
        "tts": {"openvoice_v2"},
        "embed": {"openvoice_v2"},
        "vc": {"openvoice_v2"},
        "stt": set(),
        "vad": set(),
        "restore": {"voicefixer"},
    }
    for capability, serving in SERVING_BY_CAPABILITY.items():
        card_ids = {card.id for card in serving.models_from_config(configs)}
        toolchains = {card.id for card in serving.models_from_config(configs) if card.object == "toolchain"}
        assert toolchains == expected[capability], capability
        assert card_ids >= toolchains


@pytest.mark.unit
def test_single_step_executors_are_individual_cards_only():
    configs = [_cfg("kitten-tts-mini", "KittenTTSSynthesizer", voices=["Jasper"])]

    cards = OpenAIServingTextToSpeech.models_from_config(configs)

    assert [card.id for card in cards] == ["kitten-tts-mini"]
    assert cards[0].object == "model"


@pytest.mark.unit
def test_advertised_types_match_the_old_hand_maintained_lists():
    """The capability → model_type map reproduces the deleted per-module lists."""
    assert model_types_advertising("tts") == {
        "OpenVoiceV1Synthesizer",
        "MeloTTSSynthesizer",
        "KittenTTSSynthesizer",
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
    }
    assert model_types_advertising("embed") == {
        "HarmonySpeechEncoder",
        "ChatterboxEmbedding",
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
    }
    assert model_types_advertising("vc") == {"OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter", "ChatterboxVC"}
    assert model_types_advertising("stt") == {"FasterWhisper"}
    assert model_types_advertising("vad") == {"FasterWhisper", "SileroVAD"}
    assert model_types_advertising("restore") == set()


@pytest.mark.unit
def test_translation_capability_is_declared_and_advertised():
    """Phase 10-4: the Whisper spec advertises translation without hand lists."""
    from harmonyspeech.processing.pipeline_registry import CAPABILITIES, capabilities_for_model_type

    assert "translation" in CAPABILITIES
    assert model_types_advertising("translation") == {"FasterWhisper"}
    assert capabilities_for_model_type("FasterWhisper") == {"stt", "translation", "vad"}

    cards = OpenAIServingSpeechToText.models_from_config([_cfg("whisper", "FasterWhisper")])
    assert [card.id for card in cards] == ["whisper"]
    assert cards[0].capabilities == ["stt", "translation", "vad"]


@pytest.mark.unit
def test_no_serving_module_keeps_a_hand_maintained_model_list():
    import pathlib

    module_dir = pathlib.Path(__file__).resolve().parents[3] / "harmonyspeech" / "endpoints" / "openai"
    offenders = []
    for path in module_dir.glob("serving_*.py"):
        text = path.read_text(encoding="utf-8")
        if "_MODEL_TYPES = [" in text or "_MODEL_GROUPS = {" in text:
            offenders.append(path.name)
    assert offenders == []


@pytest.mark.unit
def test_model_cards_keep_the_legacy_schema_fields():
    configs = _complete_openvoice_v2()
    card = next(
        card for card in OpenAIServingTextToSpeech.models_from_config(configs) if card.id == "ov2-synthesizer-en"
    )

    payload = card.model_dump()

    assert payload["id"] == "ov2-synthesizer-en"
    assert payload["object"] == "model"
    assert payload["root"] == "ov2-synthesizer-en"
    for legacy_field in ("owned_by", "created", "languages"):
        assert legacy_field in payload
    # Additive fields only.
    assert "capabilities" in payload
    assert "models" in payload


@pytest.mark.unit
def test_specs_for_capabilities_covers_every_family():
    assert {spec.id for spec in specs_for_capabilities("tts")} == {
        "harmonyspeech",
        "openvoice_v1",
        "openvoice_v2",
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "kitten_tts",
    }
    assert {spec.id for spec in specs_for_capabilities("vc")} == {"openvoice_v1", "openvoice_v2", "chatterbox_vc"}


@pytest.mark.unit
def test_every_registered_spec_declares_known_capabilities():
    from harmonyspeech.processing.pipeline_registry import CAPABILITIES

    for spec in PIPELINE_SPECS.values():
        assert spec.capabilities
        assert spec.capabilities <= CAPABILITIES, spec.id


@pytest.mark.unit
def test_default_config_advertises_the_same_models_as_before_migration():
    """``/v1/models`` parity against the pre-migration hand-maintained lists.

    The ids below were recorded from the old per-serving-module lists evaluated
    against ``config.yml`` before they were deleted.
    """
    import pathlib

    from harmonyspeech.common.config import EngineConfig

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    config = EngineConfig.load_config_from_yaml(str(repo_root / "config.yml"))
    expected = {
        "tts": {
            "chatterbox",
            "chatterbox_multilingual",
            "chatterbox_turbo",
            "harmonyspeech",
            "kitten-tts-mini",
            "openvoice_v1",
            "openvoice_v2",
            "ov1-synthesizer-en",
            "ov1-synthesizer-zh",
            "ov2-synthesizer-en",
            "ov2-synthesizer-es",
            "ov2-synthesizer-fr",
            "ov2-synthesizer-jp",
            "ov2-synthesizer-kr",
            "ov2-synthesizer-zh",
        },
        "embed": {
            "chatterbox",
            "chatterbox_multilingual",
            "chatterbox_turbo",
            "harmonyspeech",
            "hs1-encoder",
            "openvoice_v1",
            "openvoice_v2",
        },
        "stt": {"faster-whisper-medium", "faster-whisper-tiny"},
        "vc": {"chatterbox_vc", "openvoice_v1", "openvoice_v2", "ov1-tone-converter", "ov2-tone-converter"},
        "vad": {"faster-whisper-medium", "faster-whisper-tiny", "silero-vad"},
        "restore": {"voicefixer"},
    }

    for capability, serving in SERVING_BY_CAPABILITY.items():
        cards = serving.models_from_config(config.model_configs)
        ids = [card.id for card in cards]
        assert len(ids) == len(set(ids)), f"duplicate cards for {capability}: {ids}"
        assert set(ids) == expected[capability], capability


@pytest.mark.unit
def test_model_id_matching_a_family_id_stays_a_single_card():
    """A model named after its family (the default ``chatterbox``) is one card."""
    configs = [_cfg("chatterbox", "ChatterboxTTS")]

    cards = OpenAIServingTextToSpeech.models_from_config(configs)

    assert [card.id for card in cards] == ["chatterbox"]
    # The toolchain metadata is still attached additively.
    assert cards[0].capabilities == ["embed", "tts"]
    assert cards[0].models == ["chatterbox"]


# ---------------------------------------------------------------------------
# Strict pipelines
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strict_pipelines_raises_the_same_message_as_the_warning():
    from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = _complete_openvoice_v2()[:-1]  # tone converter missing
    engine.strict_pipelines = True

    with pytest.raises(RuntimeError) as excinfo:
        engine._validate_family_configuration()

    message = str(excinfo.value)
    assert "pipeline 'openvoice_v2'" in message
    assert "tone_convert (OpenVoiceV2ToneConverter)" in message


@pytest.mark.unit
def test_complete_configuration_passes_even_in_strict_mode():
    from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    # One complete pipeline; the other registered specs are incomplete, so use a
    # config that satisfies every registered spec step.
    engine.model_configs = [
        _cfg("whisper", "FasterWhisper"),
        _cfg("silero", "SileroVAD"),
        _cfg("hs-enc", "HarmonySpeechEncoder"),
        _cfg("hs-syn", "HarmonySpeechSynthesizer"),
        _cfg("hs-voc", "HarmonySpeechVocoder"),
        _cfg("ov1-enc", "OpenVoiceV1ToneConverterEncoder"),
        _cfg("ov1-syn", "OpenVoiceV1Synthesizer", "EN"),
        _cfg("ov1-tc", "OpenVoiceV1ToneConverter"),
        _cfg("ov2-enc", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-syn", "MeloTTSSynthesizer", "EN"),
        _cfg("ov2-tc", "OpenVoiceV2ToneConverter"),
        _cfg("vf-r", "VoiceFixerRestorer"),
        _cfg("vf-v", "VoiceFixerVocoder"),
        _cfg("cb", "ChatterboxTTS"),
        _cfg("cbt", "ChatterboxTurboTTS"),
        _cfg("cbm", "ChatterboxMultilingualTTS"),
        _cfg("cbvc", "ChatterboxVC"),
        _cfg("kitten", "KittenTTSSynthesizer"),
    ]
    engine.strict_pipelines = True

    assert "openvoice_v2" in engine._validate_family_configuration()


@pytest.mark.unit
def test_strict_pipelines_reaches_the_engine_from_cli():
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    from harmonyspeech.engine.args_tools import AsyncEngineArgs

    args = make_arg_parser().parse_args(["--strict-pipelines"])
    engine_args = AsyncEngineArgs.from_cli_args(args)

    assert engine_args.strict_pipelines is True
