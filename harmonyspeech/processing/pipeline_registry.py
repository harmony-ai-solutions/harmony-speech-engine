"""Declarative model-toolchain (family) requirements.

This module is the seed of the Phase 3 ``PipelineSpec`` registry: a single place
that declares which ``model_type``\\ s a pipeline family needs in order to serve
requests end to end. Phase 0 only uses it for startup validation and to decide
which families may be advertised on ``/v1/models``; Phase 3 will grow the same
structure into full pipeline descriptors.

A family is *complete* when every required ``model_type`` is present in the
engine configuration. Incomplete families are still loaded (operators may
deliberately run subsets, e.g. a CPU-only deployment without Whisper), but they
are logged at startup and must not be advertised to clients.
"""

from harmonyspeech.common.config import ModelConfig

FAMILY_REQUIREMENTS: dict[str, list[str]] = {
    "harmonyspeech": ["HarmonySpeechEncoder", "HarmonySpeechSynthesizer", "HarmonySpeechVocoder"],
    "openvoice_v1": ["FasterWhisper", "OpenVoiceV1ToneConverterEncoder", "OpenVoiceV1Synthesizer", "OpenVoiceV1ToneConverter"],
    "openvoice_v2": ["FasterWhisper", "OpenVoiceV2ToneConverterEncoder", "MeloTTSSynthesizer", "OpenVoiceV2ToneConverter"],
    "voicefixer": ["VoiceFixerRestorer", "VoiceFixerVocoder"],
    "chatterbox": ["ChatterboxTTS"],
    "chatterbox_turbo": ["ChatterboxTurboTTS"],
    "chatterbox_multilingual": ["ChatterboxMultilingualTTS"],
    "chatterbox_vc": ["ChatterboxVC"],
}


def available_model_types(model_configs: list[ModelConfig] | None) -> set[str]:
    """Return the set of ``model_type`` values present in the configuration."""
    return {config.model_type for config in (model_configs or [])}


def validate_families(model_configs: list[ModelConfig] | None) -> list[str]:
    """Return the family ids fully satisfied by the given configs.

    Order follows :data:`FAMILY_REQUIREMENTS`, so callers get a stable result.
    """
    available_types = available_model_types(model_configs)
    return [
        family
        for family, required_types in FAMILY_REQUIREMENTS.items()
        if all(model_type in available_types for model_type in required_types)
    ]


def missing_family_requirements(model_configs: list[ModelConfig] | None) -> dict[str, list[str]]:
    """Map each *incomplete* family to the ``model_type``\\ s it is missing.

    Families with no configured members at all are included — an engine that
    does not host a family should not advertise it either.
    """
    available_types = available_model_types(model_configs)
    missing: dict[str, list[str]] = {}
    for family, required_types in FAMILY_REQUIREMENTS.items():
        absent = [model_type for model_type in required_types if model_type not in available_types]
        if absent:
            missing[family] = absent
    return missing


def is_family_available(family: str, model_configs: list[ModelConfig] | None) -> bool:
    """Return True if ``family`` is declared and complete for these configs."""
    if family not in FAMILY_REQUIREMENTS:
        return False
    available_types = available_model_types(model_configs)
    return all(model_type in available_types for model_type in FAMILY_REQUIREMENTS[family])
