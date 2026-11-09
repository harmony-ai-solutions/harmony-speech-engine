"""Catalog ↔ routing consistency for ``GET /v1/audio/voices`` (Phase 13-3).

The endpoint is only useful if what it lists is exactly what the request path
accepts. These tests run the shipped configuration through the catalog and the
real request-level resolvers the serving layer calls:

* every listed standard voice resolves through ``lookup_standard_voice`` in the
  direction the entry advertises (TTS ``voice`` and/or VC ``target_voice``);
* every routable standard voice is listed (no capable model silently missing);
* both VC target kinds appear and stay distinct.

The FastAPI route itself is exercised through the real app with a FakeEngine
that carries the shipped model configs, so no weights are needed.
"""

import pathlib

import pytest

from harmonyspeech.common.config import EngineConfig
from harmonyspeech.modeling import voice_catalog as catalog
from harmonyspeech.processing.pipeline_registry import PIPELINE_SPECS
from tests.wire_contract_support import WireApp

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACK = list(catalog.standard_voice_names())

#: The ids ``/v1/audio/speech/models`` and ``/v1/voice/convert/models``
#: advertise for ``config.yml`` (union of the "tts" and "vc" card sets pinned in
#: ``tests/unit/initialization/test_pipeline_validation_cards.py``). The voices
#: listing mirrors that presentation.
EXPECTED_VOICE_IDS = {
    # Individual TTS executors
    "ov1-synthesizer-en",
    "ov1-synthesizer-zh",
    "ov2-synthesizer-en",
    "ov2-synthesizer-zh",
    "ov2-synthesizer-es",
    "ov2-synthesizer-fr",
    "ov2-synthesizer-jp",
    "ov2-synthesizer-kr",
    "kitten-tts-mini",
    # Individual VC executors
    "ov1-tone-converter",
    "ov2-tone-converter",
    # Configured models that are also toolchain ids (single, merged entry)
    "chatterbox",
    "chatterbox_turbo",
    "chatterbox_multilingual",
    "chatterbox_vc",
    # Toolchain ids without a configured model of the same name
    "harmonyspeech",
    "openvoice_v1",
    "openvoice_v2",
}


@pytest.fixture(scope="module")
def shipped_config():
    return EngineConfig.load_config_from_yaml(str(REPO_ROOT / "config.yml"))


@pytest.fixture(scope="module")
def configs(shipped_config):
    return shipped_config.model_configs


@pytest.fixture(scope="module")
def listing(configs):
    return catalog.voice_catalog_listing(configs)


@pytest.fixture(scope="module")
def by_model(listing):
    return {entry.model: entry for entry in listing}


def _standard(entry):
    return [voice.name for voice in entry.voices if voice.source == "standard"]


def test_listing_ids_match_the_speech_and_voice_conversion_model_endpoints(listing):
    ids = [entry.model for entry in listing]
    assert len(ids) == len(set(ids)), ids
    assert set(ids) == EXPECTED_VOICE_IDS


def test_endpoint_serves_the_same_catalog_as_the_routing_layer(configs, listing):
    with WireApp(model_configs=configs) as wire:
        response = wire.client.get("/v1/audio/voices")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    served = {entry["model"]: entry for entry in payload["data"]}

    assert set(served) == {entry.model for entry in listing}
    for entry in listing:
        wire_entry = served[entry.model]
        assert wire_entry["object"] == entry.kind
        assert wire_entry["model_type"] == entry.model_type
        assert wire_entry["cloning_capable"] is entry.cloning_capable
        assert wire_entry["vc_target_kind"] == entry.vc_target_kind
        assert [
            voice["name"] for voice in wire_entry["voices"] if voice["source"] == "standard"
        ] == _standard(entry)
        assert [
            voice["name"] for voice in wire_entry["voices"] if voice["source"] == "native"
        ] == [voice.name for voice in entry.voices if voice.source == "native"]


@pytest.mark.parametrize("model", sorted(EXPECTED_VOICE_IDS))
def test_model_query_narrows_to_the_listed_entry(configs, model):
    with WireApp(model_configs=configs) as wire:
        response = wire.client.get("/v1/audio/voices", params={"model": model})

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert len(data) == 1
    assert data[0]["model"] == model


def test_every_listed_standard_voice_resolves_through_the_routing_path(configs, listing):
    checked = 0
    for entry in listing:
        standard = _standard(entry)
        if not standard:
            continue
        checked += 1

        tts_capable = catalog.request_cloning_capable(configs, entry.model)
        vc_kind = catalog.request_vc_target_voice_kind(configs, entry.model)
        assert tts_capable or vc_kind is not None, entry.model
        # The advertised kind is the request-level one (embed path configured).
        assert entry.vc_target_kind == vc_kind, entry.model

        for name in standard:
            if tts_capable:
                lookup = catalog.lookup_standard_voice(configs, entry.model, name)
                assert lookup.error is None, (entry.model, name, lookup.error)
                assert lookup.resolution is not None
                assert lookup.resolution.kind == "standard"
                assert lookup.resolution.wav_path is not None
            if vc_kind is not None:
                target = catalog.lookup_standard_voice(
                    configs, entry.model, name, target=True
                )
                assert target.error is None, (entry.model, name, target.error)
                assert target.resolution is not None
                assert target.resolution.kind == "standard"
                assert target.vc_target_kind == vc_kind

    # The shipped config exercises more than one cloning family.
    assert checked >= 6


def test_every_routable_standard_voice_is_listed(configs, by_model):
    candidates = {config.name for config in configs if config.name} | set(PIPELINE_SPECS)
    capable_ids = set()
    for model in candidates:
        tts_capable = catalog.request_cloning_capable(configs, model)
        vc_kind = catalog.request_vc_target_voice_kind(configs, model)
        if not (tts_capable or vc_kind):
            continue
        capable_ids.add(model)
        assert model in by_model, f"capable model missing from the listing: {model}"
        assert _standard(by_model[model]) == PACK, model
        assert by_model[model].cloning_capable is True, model
        assert by_model[model].vc_target_kind == vc_kind, model

    # And the reverse: nothing lists pack names without being routable.
    listing_capable = {entry.model for entry in by_model.values() if _standard(entry)}
    assert listing_capable == capable_ids


def test_both_vc_target_kinds_appear_and_stay_distinct(by_model):
    kinds = {entry.vc_target_kind for entry in by_model.values()}
    assert kinds == {"target_audio", "target_embedding", None}

    assert by_model["chatterbox_vc"].vc_target_kind == "target_audio"
    for model in ("ov1-tone-converter", "ov2-tone-converter", "openvoice_v1", "openvoice_v2"):
        assert by_model[model].vc_target_kind == "target_embedding"
    for model in ("chatterbox", "chatterbox_turbo", "harmonyspeech", "kitten-tts-mini"):
        assert by_model[model].vc_target_kind is None


def test_single_speaker_models_list_native_voices_only(by_model):
    for model in (
        "ov1-synthesizer-en",
        "ov1-synthesizer-zh",
        "ov2-synthesizer-en",
        "ov2-synthesizer-zh",
        "ov2-synthesizer-es",
        "ov2-synthesizer-fr",
        "ov2-synthesizer-jp",
        "ov2-synthesizer-kr",
        "kitten-tts-mini",
    ):
        entry = by_model[model]
        assert entry.cloning_capable is False, model
        assert entry.vc_target_kind is None, model
        assert _standard(entry) == [], model
        assert entry.voices, model
