"""Unit tests for startup toolchain validation (Phase 0-4).

Verifies the declarative family registry, the engine's ``available_families``
exposure with a warning per incomplete family, and that uploaded model cards
only advertise complete toolchains.
"""

import pytest
from loguru import logger as loguru_logger

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.endpoints.openai.serving_engine import OpenAIServing
from harmonyspeech.endpoints.openai.serving_text_to_speech import OpenAIServingTextToSpeech
from harmonyspeech.processing.pipeline_registry import (
    FAMILY_REQUIREMENTS,
    is_family_available,
    missing_family_requirements,
    validate_families,
)


def _cfg(name: str, model_type: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model=f"dummy/{name}",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
    )


def _complete_openvoice_v2() -> list[ModelConfig]:
    return [
        _cfg("stt", "FasterWhisper"),
        _cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"),
        _cfg("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]


@pytest.fixture
def loguru_records():
    """Capture loguru messages (pytest's caplog only sees stdlib logging)."""
    records: list[str] = []
    handler_id = loguru_logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        yield records
    finally:
        loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_families_includes_complete_family():
    families = validate_families(_complete_openvoice_v2())

    assert "openvoice_v2" in families
    assert is_family_available("openvoice_v2", _complete_openvoice_v2())
    assert "openvoice_v2" not in missing_family_requirements(_complete_openvoice_v2())


@pytest.mark.unit
def test_validate_families_excludes_incomplete_family():
    configs = [
        _cfg("stt", "FasterWhisper"),
        _cfg("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"),
        # OpenVoiceV2ToneConverter deliberately missing
    ]

    families = validate_families(configs)

    assert "openvoice_v2" not in families
    assert missing_family_requirements(configs)["openvoice_v2"] == ["OpenVoiceV2ToneConverter"]
    assert not is_family_available("openvoice_v2", configs)


@pytest.mark.unit
def test_registry_declares_every_supported_family():
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


# ---------------------------------------------------------------------------
# Engine startup validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_exposes_available_families_and_warns(loguru_records):
    from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine

    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_configs = [
        _cfg("stt", "FasterWhisper"),
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"),
    ]

    available = engine._validate_family_configuration()

    assert available == validate_families(engine.model_configs)
    assert "openvoice_v2" not in available
    warnings = "\n".join(loguru_records)
    assert "openvoice_v2" in warnings
    assert "OpenVoiceV2ToneConverter" in warnings


# ---------------------------------------------------------------------------
# Model cards
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_model_cards_omit_incomplete_family():
    configs = [
        _cfg("ov2-synthesizer-en", "MeloTTSSynthesizer"),
        _cfg("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
    ]

    cards = OpenAIServingTextToSpeech.models_from_config(configs)
    card_ids = {card.id for card in cards}

    assert "openvoice_v2" not in card_ids
    # The individually configured synthesizer is still advertised.
    assert "ov2-synthesizer-en" in card_ids


@pytest.mark.unit
def test_model_cards_include_complete_family():
    cards = OpenAIServingTextToSpeech.models_from_config(_complete_openvoice_v2())
    card_ids = {card.id for card in cards}

    assert "openvoice_v2" in card_ids
    assert "ov2-synthesizer-en" in card_ids


@pytest.mark.unit
def test_model_cards_leave_undeclared_groups_untouched():
    """Groups not declared in the family registry keep the old behaviour."""
    configs = [_cfg("custom-model", "CustomModelType")]

    cards = OpenAIServing.model_cards_from_config_groups(
        configs, individual_model_types=[], model_groups={"custom_toolchain": ["CustomModelType"]}
    )

    assert [card.id for card in cards] == ["custom_toolchain"]
