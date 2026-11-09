"""Wire-contract tests for the Phase 13-3 voices catalog response models.

The endpoint's field names are part of the public surface: clients select a
voice from this payload and send it back as ``voice`` / ``target_voice``. The
models are ``ModelList``-consistent (``object: "list"`` + ``data``) and keep the
two VC target kinds distinct.
"""

import pytest
from pydantic import ValidationError

from harmonyspeech.endpoints.openai.protocol import (
    ModelVoiceCatalogEntry,
    VoiceCatalogList,
    VoiceDescriptor,
)

pytestmark = pytest.mark.unit


def test_voice_catalog_list_defaults_to_an_empty_model_list():
    assert VoiceCatalogList().model_dump() == {"object": "list", "data": []}


def test_voice_descriptor_field_names_and_source_vocabulary():
    assert set(VoiceDescriptor.model_fields) == {"name", "source", "languages"}

    native = VoiceDescriptor(name="Bella", source="native", languages=["default"])
    assert native.model_dump() == {
        "name": "Bella",
        "source": "native",
        "languages": ["default"],
    }

    # Pack clips are language-agnostic references: no language label.
    standard = VoiceDescriptor(name="Abigail", source="standard")
    assert standard.model_dump() == {
        "name": "Abigail",
        "source": "standard",
        "languages": None,
    }

    with pytest.raises(ValidationError):
        VoiceDescriptor(name="Abigail", source="pack")


def test_model_voice_catalog_entry_field_names_and_defaults():
    assert set(ModelVoiceCatalogEntry.model_fields) == {
        "model",
        "object",
        "model_type",
        "cloning_capable",
        "vc_target_kind",
        "voices",
    }

    entry = ModelVoiceCatalogEntry(model="chatterbox")
    assert entry.model_dump() == {
        "model": "chatterbox",
        "object": "model",
        "model_type": None,
        "cloning_capable": False,
        "vc_target_kind": None,
        "voices": [],
    }

    toolchain = ModelVoiceCatalogEntry(model="openvoice_v1", object="toolchain")
    assert toolchain.model_dump()["object"] == "toolchain"

    with pytest.raises(ValidationError):
        ModelVoiceCatalogEntry(model="m", object="family")


def test_vc_target_kind_keeps_the_two_kinds_distinct():
    for kind in ("target_audio", "target_embedding"):
        assert ModelVoiceCatalogEntry(model="m", vc_target_kind=kind).vc_target_kind == kind
    assert ModelVoiceCatalogEntry(model="m").vc_target_kind is None
    with pytest.raises(ValidationError):
        ModelVoiceCatalogEntry(model="m", vc_target_kind="target_clip")
