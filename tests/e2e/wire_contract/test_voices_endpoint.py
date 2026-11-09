"""Wire-contract tests for ``GET /v1/audio/voices`` (Phase 13-3).

Runs against the real FastAPI app with the serving modules wired to a
``FakeEngine`` that carries a compact model configuration, so the tests are
hermetic (no weights) while still exercising the real route, response model and
catalog.

The payload is additive to the model-list family: ``object: "list"`` plus
``data`` entries shaped ``{model, object, model_type, cloning_capable,
vc_target_kind, voices: [{name, source, languages}]}``.
"""

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from tests.wire_contract_support import WireApp

pytestmark = pytest.mark.e2e

#: The vendored pack names, sorted. Renaming a clip changes the public voice
#: identifier, so the set is pinned explicitly (Phase 13-1 contract).
EXPECTED_PACK = [
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
]

ENTRY_FIELDS = {
    "model",
    "object",
    "model_type",
    "cloning_capable",
    "vc_target_kind",
    "voices",
}
VOICE_FIELDS = {"name", "source", "languages"}


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


def _configs():
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
        _config("melo", "MeloTTSSynthesizer", language="EN", voices=["EN-Newest"]),
        _config("kitten", "KittenTTSSynthesizer", voices=["Bella", "Jasper"]),
        _config("chatterbox", "ChatterboxTTS"),
        _config("chatterbox_turbo", "ChatterboxTurboTTS"),
        _config("chatterbox_multilingual", "ChatterboxMultilingualTTS"),
        _config("chatterbox_vc", "ChatterboxVC"),
    ]


@pytest.fixture(scope="module")
def wire():
    with WireApp(model_configs=_configs()) as app:
        yield app


@pytest.fixture(scope="module")
def wire_no_pack():
    with WireApp(model_configs=_configs(), standard_voice_pack=False) as app:
        yield app


def _entries(payload):
    return {entry["model"]: entry for entry in payload["data"]}


def _standard(entry):
    return [voice["name"] for voice in entry["voices"] if voice["source"] == "standard"]


def _native(entry):
    return [voice["name"] for voice in entry["voices"] if voice["source"] == "native"]


def test_full_list_shape_is_model_list_consistent(wire):
    response = wire.client.get("/v1/audio/voices")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list) and payload["data"]

    ids = [entry["model"] for entry in payload["data"]]
    assert len(ids) == len(set(ids)), ids
    assert set(ids) == {
        "ov1-synth-en",
        "ov1-synth-zh",
        "ov1-tone-converter",
        "melo",
        "kitten",
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "chatterbox_vc",
        "openvoice_v1",
    }

    for entry in payload["data"]:
        assert set(entry) == ENTRY_FIELDS, entry["model"]
        assert entry["object"] in {"model", "toolchain"}
        assert isinstance(entry["cloning_capable"], bool)
        assert entry["vc_target_kind"] in {
            None,
            "target_audio",
            "target_embedding",
        }
        for voice in entry["voices"]:
            assert set(voice) == VOICE_FIELDS, entry["model"]
            assert voice["source"] in {"native", "standard"}
            assert isinstance(voice["name"], str) and voice["name"]


def test_model_query_narrows_to_one_toolchain_entry(wire):
    response = wire.client.get("/v1/audio/voices", params={"model": "openvoice_v1"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    entry = payload["data"][0]
    assert entry["model"] == "openvoice_v1"
    # A pipeline id that is not a configured model name is a toolchain card.
    assert entry["object"] == "toolchain"
    assert entry["model_type"] is None
    assert entry["cloning_capable"] is True
    assert entry["vc_target_kind"] == "target_embedding"
    assert _standard(entry) == EXPECTED_PACK
    assert _native(entry) == ["default", "friendly"]
    # ``default`` is available in both configured languages (EN + ZH).
    default = next(voice for voice in entry["voices"] if voice["name"] == "default")
    assert default["languages"] == ["EN", "ZH"]


def test_model_query_returns_a_configured_model_as_a_model_entry(wire):
    response = wire.client.get("/v1/audio/voices", params={"model": "chatterbox"})

    assert response.status_code == 200, response.text
    entry = response.json()["data"][0]
    assert entry["model"] == "chatterbox"
    assert entry["object"] == "model"
    assert entry["model_type"] == "ChatterboxTTS"
    assert entry["cloning_capable"] is True
    assert entry["vc_target_kind"] is None
    assert _standard(entry) == EXPECTED_PACK


def test_unknown_model_query_is_an_honest_400(wire):
    response = wire.client.get("/v1/audio/voices", params={"model": "does-not-exist"})

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["object"] == "error"
    assert body["type"] == "BadRequestError"
    assert body["code"] == 400
    assert body["param"] == "model"
    assert "has no voice catalog entry" in body["message"]
    assert "does-not-exist" in body["message"]
    assert "/v1/audio/voices" in body["message"]


def test_source_labels_are_honest_per_model_type(wire):
    entries = _entries(wire.client.get("/v1/audio/voices").json())

    # Single-speaker models: native voices only, no pack names.
    for model in ("melo", "kitten"):
        entry = entries[model]
        assert entry["cloning_capable"] is False
        assert entry["vc_target_kind"] is None
        assert _standard(entry) == []
        assert _native(entry)
    assert _native(entries["melo"]) == ["EN-Newest"]
    assert entries["melo"]["voices"][0]["languages"] == ["EN"]
    assert _native(entries["kitten"]) == ["Bella", "Jasper"]
    assert entries["kitten"]["voices"][0]["languages"] == ["default"]

    # Cloning families: pack names, no native presets of their own.
    for model in (
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_multilingual",
        "chatterbox_vc",
    ):
        entry = entries[model]
        assert entry["cloning_capable"] is True
        assert _native(entry) == []
        assert _standard(entry) == EXPECTED_PACK
        assert all(voice["languages"] is None for voice in entry["voices"])

    # A cloning pipeline exposes both: native source styles + pack targets.
    openvoice = entries["openvoice_v1"]
    assert _native(openvoice) == ["default", "friendly"]
    assert _standard(openvoice) == EXPECTED_PACK


def test_vc_target_kinds_are_listed_distinctly(wire):
    entries = _entries(wire.client.get("/v1/audio/voices").json())

    # Target-audio: the executor consumes the pack clip itself.
    assert entries["chatterbox_vc"]["vc_target_kind"] == "target_audio"
    # Target-embedding: the serving layer embeds the clip through the pipeline.
    assert entries["ov1-tone-converter"]["vc_target_kind"] == "target_embedding"
    assert entries["openvoice_v1"]["vc_target_kind"] == "target_embedding"
    # Everything else is not a VC target.
    for model in ("chatterbox", "chatterbox_turbo", "melo", "kitten", "ov1-synth-en"):
        assert entries[model]["vc_target_kind"] is None


def test_pack_name_set_is_stable(wire):
    entries = _entries(wire.client.get("/v1/audio/voices").json())

    for model in ("chatterbox", "openvoice_v1"):
        assert _standard(entries[model]) == EXPECTED_PACK
    assert len(EXPECTED_PACK) == 28


def test_disabled_pack_hides_standard_entries(wire_no_pack):
    response = wire_no_pack.client.get("/v1/audio/voices")

    assert response.status_code == 200, response.text
    entries = _entries(response.json())
    for entry in entries.values():
        assert _standard(entry) == [], entry["model"]
    # Native voices and capability flags are unaffected by the switch.
    assert _native(entries["melo"]) == ["EN-Newest"]
    assert _native(entries["kitten"]) == ["Bella", "Jasper"]
    assert entries["chatterbox"]["cloning_capable"] is True
    assert entries["chatterbox"]["voices"] == []
    assert entries["chatterbox_vc"]["vc_target_kind"] == "target_audio"
    assert entries["ov1-tone-converter"]["vc_target_kind"] == "target_embedding"
