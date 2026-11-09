"""Wire-contract tests for the Phase 13-1 standard voice fields.

The pack is resolved in the serving layer, so the wire surface stays additive:
``voice`` is still a plain string and ``target_voice`` is a new optional VC
field. Existing requests (no pack name) are unchanged.
"""

import pytest

from harmonyspeech.common.inputs import TextToSpeechRequestInput, VoiceConversionRequestInput
from harmonyspeech.endpoints.openai.protocol import TextToSpeechRequest, VoiceConversionRequest
from tests.wire_contract_support import WireApp

pytestmark = pytest.mark.unit


def test_target_voice_is_an_optional_vc_wire_field():
    fields = VoiceConversionRequest.model_fields
    assert "target_voice" in fields
    request = VoiceConversionRequest(model="vc", source_audio="Zm9v")
    assert request.target_voice is None
    # The pre-13 fields keep their defaults.
    assert request.target_audio is None
    assert request.target_embedding is None


def test_target_voice_is_resolved_at_the_boundary_not_in_the_engine_input():
    request = VoiceConversionRequest(
        model="vc", source_audio="Zm9v", target_voice="Abigail"
    )
    request_input = VoiceConversionRequestInput.from_openai("vc-1", request)
    # The engine input keeps the existing target fields; the serving layer
    # turns a pack name into `target_audio` (target-audio VC) or
    # `target_embedding` (target-embedding VC) before admission.
    assert request_input.target_audio is None
    assert request_input.target_embedding is None
    assert not hasattr(request_input, "target_voice")


def test_tts_voice_field_is_still_a_plain_string():
    request = TextToSpeechRequest(
        model="tts", input="hello", mode="voice_cloning", voice="Abigail"
    )
    request_input = TextToSpeechRequestInput.from_openai("tts-1", request)
    # Unchanged mapping: the serving layer rewrites a pack name later.
    assert request_input.voice_id == "Abigail"
    assert request_input.input_audio is None
    assert request_input.mode == "voice_cloning"


def test_pack_voice_on_a_model_without_a_pipeline_is_an_honest_400(wire_client):
    response = wire_client.client.post(
        "/v1/audio/speech",
        json={
            "model": WireApp.MODEL_IDS["tts"],
            "input": "hello",
            "mode": "voice_cloning",
            "voice": "Abigail",
        },
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert "standard voice pack name" in body["message"]
    assert "does not support voice cloning" in body["message"]
    # Nothing reached the engine.
    assert wire_client.engine.requests == []


def test_pack_target_voice_conflict_is_an_honest_400(wire_client):
    response = wire_client.client.post(
        "/v1/voice/convert",
        json={
            "model": WireApp.MODEL_IDS["vc"],
            "source_audio": "Zm9v",
            "target_voice": "Abigail",
            "target_audio": "YmFy",
        },
    )
    assert response.status_code == 400, response.text
    assert "not both" in response.json()["message"]
    assert wire_client.engine.requests == []


def test_pack_target_voice_on_a_non_target_voice_model_is_an_honest_400(wire_client):
    response = wire_client.client.post(
        "/v1/voice/convert",
        json={
            "model": WireApp.MODEL_IDS["vc"],
            "source_audio": "Zm9v",
            "target_voice": "Abigail",
        },
    )
    assert response.status_code == 400, response.text
    assert "does not accept a standard target voice" in response.json()["message"]
    assert wire_client.engine.requests == []
