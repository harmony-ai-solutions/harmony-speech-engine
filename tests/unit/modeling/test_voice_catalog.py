"""Unit tests for the standard voice pack catalog (Phase 13-1).

Pins the public voice-name set, the per-model-type listing (native vs standard),
and the request-level resolution rules. The pack is vendored package data, so
these tests are hermetic and read the shipped WAVs.
"""

import base64

import pytest

from harmonyspeech.common.config import DeviceConfig, EngineConfig, ModelConfig
from harmonyspeech.endpoints.openai.standard_voice import standard_voice_wire_payload
from harmonyspeech.modeling import voice_catalog as catalog

pytestmark = pytest.mark.unit

#: The vendored pack names, sorted. Renaming a clip changes the public voice
#: identifier, so this list is deliberately explicit (not derived at test time).
EXPECTED_PACK = (
    "Abigail",
    "Adrian",
    "Alexander",
    "Alice",
    "Austin",
    "Axel",
    "Connor",
    "Cora",
    "Elena",
    "Eli",
    "Emily",
    "Everett",
    "Gabriel",
    "Gianna",
    "Henry",
    "Ian",
    "Jade",
    "Jeremiah",
    "Jordan",
    "Julian",
    "Layla",
    "Leonardo",
    "Michael",
    "Miles",
    "Olivia",
    "Ryan",
    "Taylor",
    "Thomas",
)


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


def _sources(entries):
    return [(entry.name, entry.source) for entry in entries]


def test_pack_name_set_is_stable():
    assert catalog.standard_voice_names() == EXPECTED_PACK
    assert len(EXPECTED_PACK) == 28


def test_every_pack_clip_is_a_readable_wav_with_a_wire_payload():
    for name in EXPECTED_PACK:
        path = catalog.standard_voice_path(name)
        assert path is not None and path.is_file(), name
        raw = path.read_bytes()
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", name
        payload = standard_voice_wire_payload(name)
        assert payload is not None
        assert base64.b64decode(payload) == raw


def test_unknown_and_unsafe_names_are_not_standard():
    assert catalog.is_standard_voice("Abigail") is True
    assert catalog.is_standard_voice("abigail") is False
    assert catalog.is_standard_voice("NotAVoice") is False
    assert catalog.is_standard_voice(None) is False
    assert catalog.standard_voice_path("../Abigail") is None
    assert standard_voice_wire_payload("nope") is None


@pytest.mark.parametrize(
    "model_type",
    [
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
        "ChatterboxVC",
        "OpenVoiceV1ToneConverter",
        "OpenVoiceV2ToneConverter",
    ],
)
def test_cloning_capable_model_types_list_native_then_standard(model_type):
    config = _config("model", model_type)
    entries = catalog.voices_for_model(config)
    assert [entry.name for entry in entries] == list(EXPECTED_PACK)
    assert all(entry.source == "standard" for entry in entries)
    assert catalog.voice_catalog(config).cloning_capable is True


@pytest.mark.parametrize(
    "model_type",
    [
        "MeloTTSSynthesizer",
        "KittenTTSSynthesizer",
        "OpenVoiceV1Synthesizer",
        "HarmonySpeechSynthesizer",
        "HarmonySpeechEncoder",
    ],
)
def test_non_direct_models_have_no_standard_entries(model_type):
    config = _config("model", model_type, voices=["NativeA", "NativeB"])
    entries = catalog.voices_for_model(config)
    assert entries and all(entry.source == "native" for entry in entries)
    assert catalog.voice_catalog(config).cloning_capable is False
    assert catalog.voice_catalog(config).vc_target_kind is None


def test_native_entries_carry_the_config_language():
    melo = _config("melo", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"])
    kitten = _config("kitten", "KittenTTSSynthesizer", voices=["Bella", "Jasper"])
    assert _sources(catalog.voices_for_model(melo)) == [("EN-Newest", "native")]
    assert catalog.voices_for_model(melo)[0].languages == ("EN",)
    assert _sources(catalog.voices_for_model(kitten)) == [
        ("Bella", "native"),
        ("Jasper", "native"),
    ]
    assert catalog.voices_for_model(kitten)[0].languages == ("default",)


def test_native_name_wins_over_a_pack_name_on_collision():
    # No shipped collision exists today; the precedence is still pinned because
    # a future pack could reuse a native id.
    config = _config("cb", "ChatterboxTTS", voices=["Abigail"])
    resolution = catalog.resolve_voice(config, "Abigail")
    assert resolution is not None and resolution.kind == "native"
    assert resolution.native_id == "Abigail"
    standard = catalog.resolve_voice(config, "Adrian")
    assert standard is not None and standard.kind == "standard"
    assert standard.wav_path.name == "Adrian.wav"


def test_resolve_voice_rejects_pack_names_on_single_speaker_models():
    melo = _config("melo", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"])
    assert catalog.resolve_voice(melo, "Abigail") is None
    assert catalog.resolve_voice(melo, "EN-Newest").kind == "native"


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


def test_pipeline_capability_is_structural_not_a_model_type_blanket():
    configs = _openvoice_v1_configs()
    assert catalog.request_cloning_capable(configs, "openvoice_v1")
    # A concrete synthesizer is a pipeline member but never reads the reference.
    assert not catalog.request_cloning_capable(configs, "ov1-synth-en")
    assert not catalog.request_cloning_capable(
        [_config("melo", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"])],
        "melo",
    )
    assert not catalog.request_cloning_capable(
        [_config("kitten", "KittenTTSSynthesizer", voices=["Jasper"])], "kitten"
    )
    # Chatterbox pipelines and directly named executors are capable.
    assert catalog.request_cloning_capable(
        [_config("cb", "ChatterboxTTS")], "chatterbox"
    )
    assert catalog.request_cloning_capable([_config("cb", "ChatterboxTTS")], "cb")


def test_request_native_voices_filters_by_language():
    configs = _openvoice_v1_configs()
    assert catalog.request_native_voices(
        configs, "openvoice_v1", language_id="EN"
    ) == ("default", "friendly")
    assert catalog.request_native_voices(
        configs, "openvoice_v1", language_id="ZH"
    ) == ("default",)


def test_vc_target_audio_capability_is_chatterbox_only():
    assert catalog.request_vc_target_audio_capable(
        [_config("chatterbox_vc", "ChatterboxVC")], "chatterbox_vc"
    )
    # OpenVoice VC consumes target_embedding, not target_audio.
    assert not catalog.request_vc_target_audio_capable(
        _openvoice_v1_configs(), "openvoice_v1"
    )


def test_vc_target_voice_kind_keeps_the_two_kinds_distinct():
    # Target-audio: the executor consumes the clip itself.
    assert (
        catalog.vc_target_voice_kind_for_model_type("ChatterboxVC")
        == catalog.VC_TARGET_AUDIO
    )
    assert (
        catalog.voice_catalog(_config("vc", "ChatterboxVC")).vc_target_kind
        == catalog.VC_TARGET_AUDIO
    )
    # Target-embedding: the serving layer must embed the clip first.
    for model_type in ("OpenVoiceV1ToneConverter", "OpenVoiceV2ToneConverter"):
        assert (
            catalog.vc_target_voice_kind_for_model_type(model_type)
            == catalog.VC_TARGET_EMBEDDING
        )
        assert (
            catalog.voice_catalog(_config("ov", model_type)).vc_target_kind
            == catalog.VC_TARGET_EMBEDDING
        )
    # Everything else is not a VC target.
    for model_type in ("ChatterboxTTS", "MeloTTSSynthesizer", None):
        assert catalog.vc_target_voice_kind_for_model_type(model_type) is None


def test_request_vc_target_voice_kind_requires_a_configured_embed_path():
    configs = _openvoice_v1_configs()
    assert (
        catalog.request_vc_target_voice_kind(configs, "openvoice_v1")
        == catalog.VC_TARGET_EMBEDDING
    )
    # A concrete tone converter resolves through its sibling pipeline encoder.
    assert (
        catalog.request_vc_target_voice_kind(configs, "ov1-tone-converter")
        == catalog.VC_TARGET_EMBEDDING
    )
    # Without the encoder the target cannot be embedded: honestly not capable.
    no_encoder = [c for c in configs if c.name != "ov1-encoder"]
    assert catalog.request_vc_target_voice_kind(no_encoder, "openvoice_v1") is None
    assert (
        catalog.request_vc_target_voice_kind(no_encoder, "ov1-tone-converter") is None
    )
    # Without the mandatory VAD step the embed path is incomplete too.
    no_vad = [c for c in configs if c.name != "faster-whisper"]
    assert catalog.request_vc_target_voice_kind(no_vad, "openvoice_v1") is None
    # A target-audio model is unaffected.
    assert (
        catalog.request_vc_target_voice_kind(
            [_config("chatterbox_vc", "ChatterboxVC")], "chatterbox_vc"
        )
        == catalog.VC_TARGET_AUDIO
    )
    assert (
        catalog.request_vc_target_voice_kind(
            [_config("melo", "MeloTTSSynthesizer", language="EN")], "melo"
        )
        is None
    )


def test_resolve_target_embedding_route_names_the_pipeline_and_encoder():
    configs = _openvoice_v1_configs()
    route = catalog.resolve_target_embedding_route(configs, "openvoice_v1")
    assert route is not None
    assert route.pipeline_id == "openvoice_v1"
    assert route.encoder_model_name == "ov1-encoder"

    # A concrete tone converter routes through its pipeline's embed step.
    route = catalog.resolve_target_embedding_route(configs, "ov1-tone-converter")
    assert route is not None and route.pipeline_id == "openvoice_v1"
    assert route.encoder_model_name == "ov1-encoder"

    # A non-target-embedding model has no route.
    assert (
        catalog.resolve_target_embedding_route(
            [_config("chatterbox_vc", "ChatterboxVC")], "chatterbox_vc"
        )
        is None
    )


def test_lookup_resolves_a_pack_name_on_a_capable_model():
    configs = [_config("cb", "ChatterboxTTS")]
    lookup = catalog.lookup_standard_voice(configs, "cb", "Abigail")
    assert lookup.is_standard_name and lookup.error is None
    assert lookup.resolution is not None
    assert lookup.resolution.kind == "standard"
    assert lookup.resolution.wav_path.name == "Abigail.wav"
    assert lookup.native_default is None


def test_lookup_substitutes_the_native_default_source_voice():
    configs = _openvoice_v1_configs()
    lookup = catalog.lookup_standard_voice(
        configs, "openvoice_v1", "Abigail", language_id="EN"
    )
    assert lookup.error is None and lookup.resolution.kind == "standard"
    # OpenVoice needs a valid source voice id for its synthesizer step.
    assert lookup.native_default == "default"


def test_lookup_is_a_noop_for_native_and_unknown_names():
    configs = [_config("cb", "ChatterboxTTS", voices=["Bella"])]
    for name in ("Bella", "NotAVoice", None):
        lookup = catalog.lookup_standard_voice(configs, "cb", name)
        assert lookup.is_standard_name is False
        assert lookup.resolution is None and lookup.error is None


def test_lookup_honest_errors():
    melo = [_config("melo", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"])]
    not_capable = catalog.lookup_standard_voice(melo, "melo", "Abigail")
    assert not_capable.error is not None
    assert "standard voice pack name" in not_capable.error
    assert "does not support voice cloning" in not_capable.error
    assert "EN-Newest" in not_capable.error

    disabled = catalog.lookup_standard_voice(
        [_config("cb", "ChatterboxTTS")], "cb", "Abigail", enabled=False
    )
    assert "standard voice pack is disabled" in disabled.error

    conflict = catalog.lookup_standard_voice(
        [_config("cb", "ChatterboxTTS")], "cb", "Abigail", has_explicit_reference=True
    )
    assert "not both" in conflict.error

    unknown_model = catalog.lookup_standard_voice(
        [_config("cb", "ChatterboxTTS")], "missing", "Abigail", model_exists=False
    )
    assert unknown_model.is_standard_name is True
    assert unknown_model.resolution is None and unknown_model.error is None


def test_lookup_target_voice_errors_and_success():
    vc = [_config("chatterbox_vc", "ChatterboxVC")]
    ok = catalog.lookup_standard_voice(vc, "chatterbox_vc", "Abigail", target=True)
    assert ok.error is None and ok.resolution.kind == "standard"
    assert ok.vc_target_kind == catalog.VC_TARGET_AUDIO

    # OpenVoice VC now resolves a pack target through its encoder (13-1
    # addendum); the kind tells the serving layer to embed it first.
    openvoice = catalog.lookup_standard_voice(
        _openvoice_v1_configs(), "openvoice_v1", "Abigail", target=True
    )
    assert openvoice.error is None
    assert openvoice.resolution.kind == "standard"
    assert openvoice.vc_target_kind == catalog.VC_TARGET_EMBEDDING

    conflict = catalog.lookup_standard_voice(
        vc,
        "chatterbox_vc",
        "Abigail",
        target=True,
        has_explicit_reference=True,
    )
    assert "not both" in conflict.error

    # A lone tone converter with no configured encoder cannot embed the pack.
    unsupported = catalog.lookup_standard_voice(
        [_config("ov1-tone-converter", "OpenVoiceV1ToneConverter")],
        "ov1-tone-converter",
        "Abigail",
        target=True,
    )
    assert "does not accept a standard target voice" in unsupported.error
    assert unsupported.vc_target_kind is None


def test_standard_voice_pack_knob_defaults_on_and_parses(tmp_path):
    assert EngineConfig(model_configs=[]).standard_voice_pack is True

    yaml_file = tmp_path / "config.yml"
    yaml_file.write_text(
        """
model_configs: []
standard_voice_pack: false
"""
    )
    parsed = EngineConfig.load_config_from_yaml(str(yaml_file))
    assert parsed.standard_voice_pack is False
