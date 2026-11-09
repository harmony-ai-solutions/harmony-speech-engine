"""Unit tests for the 13-3 voices listing (individual models + toolchain ids).

The listing is what ``GET /v1/audio/voices`` serves. It mirrors the model
endpoints' presentation — one entry per configured model advertising ``tts`` or
``vc``, then one entry per complete voice-capable toolchain id — and it is
honest about capability: a model whose embed path is only partially configured
does not advertise pack names its request path would reject.
"""

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling import voice_catalog as catalog

pytestmark = pytest.mark.unit

PACK = catalog.standard_voice_names()


def _config(name, model_type, **extra):
    return ModelConfig(
        name=name,
        model=f"stub/{name}",
        model_type=model_type,
        max_batch_size=1,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **extra,
    )


def _by_model(listing):
    return {entry.model: entry for entry in listing}


def _sources(entry):
    return [(voice.name, voice.source) for voice in entry.voices]


def _standard_names(entry):
    return [voice.name for voice in entry.voices if voice.source == "standard"]


def _native_names(entry):
    return [voice.name for voice in entry.voices if voice.source == "native"]


def _openvoice_v1_configs():
    return [
        _config("faster-whisper", "FasterWhisper"),
        _config("ov1-encoder", "OpenVoiceV1ToneConverterEncoder", language="EN"),
        _config(
            "ov1-synth-en",
            "OpenVoiceV1Synthesizer",
            language="EN",
            voices=["default", "friendly"],
        ),
        _config(
            "ov1-synth-zh",
            "OpenVoiceV1Synthesizer",
            language="ZH",
            voices=["default"],
        ),
        _config("ov1-tone-converter", "OpenVoiceV1ToneConverter"),
    ]


def _shipped_like_configs():
    return [
        _config("hs1-encoder", "HarmonySpeechEncoder"),
        _config("hs1-synthesizer", "HarmonySpeechSynthesizer"),
        _config("hs1-vocoder", "HarmonySpeechVocoder"),
        *_openvoice_v1_configs(),
        _config("ov2-synth-en", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]),
        _config("ov2-tone-converter", "OpenVoiceV2ToneConverter"),
        _config("ov2-encoder", "OpenVoiceV2ToneConverterEncoder"),
        _config("kitten", "KittenTTSSynthesizer", voices=["Bella", "Jasper"]),
        _config("chatterbox", "ChatterboxTTS"),
        _config("chatterbox_vc", "ChatterboxVC"),
    ]


def test_listing_mirrors_the_model_endpoints_voice_relevant_ids():
    listing = catalog.voice_catalog_listing(_shipped_like_configs())

    # Individual cards first (config order), then toolchain ids (registry
    # order). ``chatterbox``/``chatterbox_vc`` are configured model names and
    # toolchain ids at once, so they stay single entries.
    assert [entry.model for entry in listing] == [
        "ov1-synth-en",
        "ov1-synth-zh",
        "ov1-tone-converter",
        "ov2-synth-en",
        "ov2-tone-converter",
        "kitten",
        "chatterbox",
        "chatterbox_vc",
        "harmonyspeech",
        "openvoice_v1",
        "openvoice_v2",
    ]
    entries = _by_model(listing)
    assert entries["harmonyspeech"].kind == "toolchain"
    assert entries["openvoice_v1"].kind == "toolchain"
    assert entries["openvoice_v2"].kind == "toolchain"
    assert entries["chatterbox"].kind == "model"
    assert entries["chatterbox_vc"].kind == "model"
    assert entries["openvoice_v1"].model_type is None
    assert entries["chatterbox"].model_type == "ChatterboxTTS"
    # Encoder/synthesizer members that only advertise "embed" (or nothing) are
    # not individual voice entries; the toolchain id covers them.
    assert "hs1-encoder" not in entries
    assert "faster-whisper" not in entries
    assert "ov1-encoder" not in entries


def test_single_speaker_models_list_native_voices_only():
    entries = _by_model(catalog.voice_catalog_listing(_shipped_like_configs()))

    melo = entries["ov2-synth-en"]
    assert _sources(melo) == [("EN-Newest", "native")]
    assert melo.voices[0].languages == ("EN",)
    assert melo.cloning_capable is False
    assert melo.vc_target_kind is None

    kitten = entries["kitten"]
    assert _native_names(kitten) == ["Bella", "Jasper"]
    assert _standard_names(kitten) == []
    assert kitten.cloning_capable is False


def test_cloning_capable_entries_list_native_then_standard():
    entries = _by_model(catalog.voice_catalog_listing(_shipped_like_configs()))

    chatterbox = entries["chatterbox"]
    assert chatterbox.cloning_capable is True
    assert chatterbox.vc_target_kind is None
    assert _native_names(chatterbox) == []
    assert _standard_names(chatterbox) == list(PACK)
    assert all(voice.languages == () for voice in chatterbox.voices)

    converter = entries["ov1-tone-converter"]
    assert converter.cloning_capable is True
    assert converter.vc_target_kind == catalog.VC_TARGET_EMBEDDING
    assert _native_names(converter) == []
    assert _standard_names(converter) == list(PACK)


def test_pipeline_entry_merges_native_voices_across_languages():
    entry = _by_model(catalog.voice_catalog_listing(_shipped_like_configs()))["openvoice_v1"]

    native = {voice.name: voice.languages for voice in entry.voices if voice.source == "native"}
    # ``default`` exists in both configured language variants: one entry, both
    # languages. Styles stay language-tagged.
    assert native == {"default": ("EN", "ZH"), "friendly": ("EN",)}
    assert _standard_names(entry) == list(PACK)
    assert entry.cloning_capable is True
    assert entry.vc_target_kind == catalog.VC_TARGET_EMBEDDING


def test_pipeline_id_matching_a_model_name_is_one_merged_entry():
    listing = catalog.voice_catalog_listing([_config("chatterbox", "ChatterboxTTS")])

    assert [entry.model for entry in listing] == ["chatterbox"]
    entry = listing[0]
    # Mirrors /v1/audio/speech/models: the configured card is reused.
    assert entry.kind == "model"
    assert entry.model_type == "ChatterboxTTS"
    assert entry.cloning_capable is True
    assert _standard_names(entry) == list(PACK)


def test_vc_target_kinds_are_listed_distinctly():
    entries = _by_model(catalog.voice_catalog_listing(_shipped_like_configs()))

    assert entries["chatterbox_vc"].vc_target_kind == catalog.VC_TARGET_AUDIO
    assert entries["ov1-tone-converter"].vc_target_kind == catalog.VC_TARGET_EMBEDDING
    assert entries["ov2-tone-converter"].vc_target_kind == catalog.VC_TARGET_EMBEDDING
    assert entries["openvoice_v1"].vc_target_kind == catalog.VC_TARGET_EMBEDDING
    assert entries["openvoice_v2"].vc_target_kind == catalog.VC_TARGET_EMBEDDING
    for model in ("chatterbox", "ov1-synth-en", "kitten", "harmonyspeech"):
        assert entries[model].vc_target_kind is None


def test_disabled_pack_hides_standard_entries_but_keeps_capability_flags():
    configs = _shipped_like_configs()
    enabled = _by_model(catalog.voice_catalog_listing(configs))
    disabled = _by_model(
        catalog.voice_catalog_listing(configs, standard_pack_enabled=False)
    )

    assert set(enabled) == set(disabled)
    for model, entry in disabled.items():
        assert _standard_names(entry) == [], model
        # Capability flags describe the model, not the deployment switch:
        # explicit reference clips keep working with the pack off.
        assert entry.cloning_capable == enabled[model].cloning_capable, model
        assert entry.vc_target_kind == enabled[model].vc_target_kind, model
    assert disabled["chatterbox"].voices == ()
    # Native voices are unaffected.
    assert _native_names(disabled["ov1-synth-en"]) == ["default", "friendly"]


def test_tone_converter_without_its_encoder_is_honestly_not_capable():
    # A lone OpenVoice tone converter cannot embed a pack clip: no VAD, no
    # encoder, so no target_voice. The listing must not advertise pack names
    # the request path rejects.
    listing = catalog.voice_catalog_listing(
        [_config("ov1-tone-converter", "OpenVoiceV1ToneConverter")]
    )

    assert [entry.model for entry in listing] == ["ov1-tone-converter"]
    entry = listing[0]
    assert entry.cloning_capable is False
    assert entry.vc_target_kind is None
    assert _standard_names(entry) == []
    assert entry.voices == ()


def test_incomplete_pipelines_are_not_listed_as_toolchains():
    # The Chatterbox VC family is complete on its own (single step), but the
    # OpenVoice pipeline needs VAD + encoder + synthesizer + converter.
    listing = catalog.voice_catalog_listing([_config("chatterbox_vc", "ChatterboxVC")])

    assert [entry.model for entry in listing] == ["chatterbox_vc"]
    assert listing[0].kind == "model"
    assert listing[0].vc_target_kind == catalog.VC_TARGET_AUDIO
    assert _standard_names(listing[0]) == list(PACK)


def test_listing_is_deterministic():
    configs = _shipped_like_configs()
    assert catalog.voice_catalog_listing(configs) == catalog.voice_catalog_listing(configs)
