"""Headroom-aware padded-batch governance: decision maths + observability (Phase 14-6).

TDD contract (written before the implementation): these tests pin the **monitored
values** of the padded-batch governor, not its internals.

* the activation cost per padded unit is derived from the recorded 12-9 peaks
  (mel-frame tone converter: 2.6 GiB at 7500 frames; phone-axis synthesizers:
  1.8 GiB at 1024 phones) — the ``~2.5 GiB ⇒ TC width 2`` / ``~0.9 GiB ⇒ width 1``
  states from the task file are asserted as arithmetic + resulting plan width;
* the static 12-9 caps remain the ceiling: headroom can only shrink an area
  budget, never widen it, and the width floor is 1;
* the two Prometheus families (``hse_padded_batch_shrink_total{model,axis}``
  counter, ``hse_padded_batch_headroom_bytes{model,axis,device}`` gauge) are
  rendered with an explicit ``{model="none",axis="none"}`` zero state;
* the engine aggregates the runners and ``/metrics`` + ``/health`` surface the
  block next to the Whisper governance block.

Everything here is CPU-only: the headroom read is never performed at this layer
(the runner seam is covered by ``tests/unit/inference_flow/
test_padded_batch_headroom_governance.py``).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine
from harmonyspeech.task_handler.batching import (
    BATCHED_INFERENCE_FAMILIES,
    DEFAULT_MEL_FRAME_BATCH_BUDGET,
    DEFAULT_PHONE_BATCH_BUDGET,
    plan_batches,
)

GIB = 2**30
#: Headroom states from the Phase 14 verification record (16 GiB card).
FIXTURE_FREE = int(5.6 * GIB)  # config-phase12.yml, 8 resident models
SHIPPED_FREE = int(2.5 * GIB)  # shipped catalogue (low end of 2.5-3.5 GiB)
TIGHT_FREE = int(0.9 * GIB)  # shipped + whisper on GPU (0.87 GiB observed)

#: Derived costs (2.6 GiB / 7500 frames, 1.8 GiB / 1024 phones).
MEL_FRAME_BYTES = int(2.6 * GIB) // DEFAULT_MEL_FRAME_BATCH_BUDGET
PHONE_BYTES = int(1.8 * GIB) // DEFAULT_PHONE_BATCH_BUDGET

#: 3400-frame rows (≈39 s at OpenVoice's 86.13 fps) are the length at which the
#: task file's "~2.5 GiB ⇒ TC width 2" state holds: 2 x 3400 = 6800 <= 7211 and
#: 3 x 3400 = 10200 > 7211. The recorded 12-9 paragraph is 3660 frames, where the
#: same headroom already forces width 1 (asserted below as well).
LONG_ROW_FRAMES = 3400
RECORDED_PARAGRAPH_FRAMES = 3660


def _padded_batch():
    """Import the (initially absent) pure decision module inside the test."""
    from harmonyspeech.engine import padded_batch

    return padded_batch


def _render():
    from harmonyspeech.engine.padded_batch_metrics import render_padded_batch_metrics

    return render_padded_batch_metrics


def _plan_width(lengths, area_budget, max_batch_size):
    groups = plan_batches(lengths, max_batch_size, area_budget)
    return max(len(group) for group in groups)


def _engine_shell() -> HarmonySpeechEngine:
    engine = HarmonySpeechEngine.__new__(HarmonySpeechEngine)
    engine.model_executors = {}
    engine.model_configs = []
    engine.whisper_vram_budget_bytes = None
    engine.whisper_vram_report = None
    return engine


def _padded_executor(
    name: str = "ov2-tone-converter",
    model_type: str = "OpenVoiceV2ToneConverter",
    runner=None,
):
    config = ModelConfig(
        name=name,
        model="dummy/openvoice",
        model_type=model_type,
        max_batch_size=8,
        device_config=DeviceConfig("cuda"),
        dtype="float16",
    )
    return SimpleNamespace(
        model_config=config,
        driver_worker=SimpleNamespace(model_runner=runner or SimpleNamespace()),
        check_health=lambda: None,
    )


# ---------------------------------------------------------------------------
# Activation-cost derivation and the recorded headroom states
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_activation_costs_are_derived_from_the_recorded_12_9_peaks():
    padded_batch = _padded_batch()

    # TC: 2.6 GiB peak reserved at the 7500-frame static cap (Phase 12-9).
    assert padded_batch.PADDED_AXIS_BYTES_PER_UNIT["mel_frame"] == MEL_FRAME_BYTES
    assert MEL_FRAME_BYTES == 372_230
    # Synth: <= 1.8 GiB peak reserved at the 1024-phone static cap (Phase 12-9).
    assert padded_batch.PADDED_AXIS_BYTES_PER_UNIT["phone"] == PHONE_BYTES
    assert PHONE_BYTES == 1_887_436
    # The unmeasured speech-token axis is deliberately NOT given a guessed cost.
    assert "speech_token" not in padded_batch.PADDED_AXIS_BYTES_PER_UNIT


@pytest.mark.unit
def test_tone_converter_states_map_to_the_task_files_expected_widths():
    padded_batch = _padded_batch()
    lengths = [LONG_ROW_FRAMES] * 8

    # 5.6 GiB free (12-9 fixture): the static cap fits, no shrink, width 2.
    decision = padded_batch.allowed_padded_area(
        DEFAULT_MEL_FRAME_BATCH_BUDGET, FIXTURE_FREE, "mel_frame"
    )
    assert decision["allowed_area"] == DEFAULT_MEL_FRAME_BATCH_BUDGET
    assert decision["shrunk"] is False
    assert _plan_width(lengths, decision["allowed_area"], 8) == 2

    # ~2.5 GiB free (shipped catalogue): the cap shrinks below 7500 but the
    # plan keeps width 2 for ~39 s rows.
    decision = padded_batch.allowed_padded_area(
        DEFAULT_MEL_FRAME_BATCH_BUDGET, SHIPPED_FREE, "mel_frame"
    )
    assert decision["allowed_area"] == 7211
    assert decision["shrunk"] is True
    assert _plan_width(lengths, decision["allowed_area"], 8) == 2

    # ~0.9 GiB free: width floor, every long row alone.
    decision = padded_batch.allowed_padded_area(
        DEFAULT_MEL_FRAME_BATCH_BUDGET, TIGHT_FREE, "mel_frame"
    )
    assert decision["allowed_area"] == 2596
    assert decision["shrunk"] is True
    assert _plan_width(lengths, decision["allowed_area"], 8) == 1

    # The recorded 3660-frame paragraph is already width 1 at ~2.5 GiB.
    decision = padded_batch.allowed_padded_area(
        DEFAULT_MEL_FRAME_BATCH_BUDGET, SHIPPED_FREE, "mel_frame"
    )
    assert _plan_width([RECORDED_PARAGRAPH_FRAMES] * 8, decision["allowed_area"], 8) == 1


@pytest.mark.unit
def test_phone_axis_states_keep_the_static_cap_until_headroom_tightens():
    padded_batch = _padded_batch()

    # ~2.5 GiB still fits the 1024-phone cap (headroom area 1422).
    decision = padded_batch.allowed_padded_area(
        DEFAULT_PHONE_BATCH_BUDGET, SHIPPED_FREE, "phone"
    )
    assert decision["allowed_area"] == DEFAULT_PHONE_BATCH_BUDGET
    assert decision["shrunk"] is False
    assert _plan_width([500] * 8, decision["allowed_area"], 8) == 2

    # ~0.9 GiB shrinks to 512 phones: 500-phone rows alone, 128-phone chunks 4-wide.
    decision = padded_batch.allowed_padded_area(
        DEFAULT_PHONE_BATCH_BUDGET, TIGHT_FREE, "phone"
    )
    assert decision["allowed_area"] == 512
    assert decision["shrunk"] is True
    assert _plan_width([500] * 8, decision["allowed_area"], 8) == 1
    assert _plan_width([128] * 8, decision["allowed_area"], 8) == 4


@pytest.mark.unit
def test_headroom_can_only_shrink_and_never_widen_the_static_cap():
    padded_batch = _padded_batch()

    wide = padded_batch.allowed_padded_area(7500, 100 * GIB, "mel_frame")
    assert wide["allowed_area"] == 7500
    assert wide["shrunk"] is False

    below_cap = padded_batch.allowed_padded_area(6000, 100 * GIB, "mel_frame")
    assert below_cap["allowed_area"] == 6000

    # An explicit per-model budget is also a ceiling, not a floor.
    assert padded_batch.allowed_padded_area(1234, 100 * GIB, "phone")["allowed_area"] == 1234


@pytest.mark.unit
def test_width_one_floor_under_exhausted_headroom():
    padded_batch = _padded_batch()

    for free in (0, -1):
        decision = padded_batch.allowed_padded_area(7500, free, "mel_frame")
        assert decision["allowed_area"] == 1
        assert decision["shrunk"] is True

    groups = plan_batches([3400] * 4, 4, 1)
    assert groups == [[0], [1], [2], [3]]


@pytest.mark.unit
def test_ungoverned_axes_report_the_reason_instead_of_guessing():
    padded_batch = _padded_batch()

    decision = padded_batch.allowed_padded_area(500, TIGHT_FREE, "speech_token")
    assert decision["governed"] is False
    assert decision["allowed_area"] == 500
    assert decision["shrunk"] is False
    assert "no recorded activation cost" in decision["reason"]


@pytest.mark.unit
def test_model_axis_map_covers_every_governed_padded_family():
    padded_batch = _padded_batch()

    # Every mapped family really has a padded forward.
    assert set(padded_batch.PADDED_MODEL_AXES) <= BATCHED_INFERENCE_FAMILIES
    # Both tone converters (V1 and V2) get the same mel-frame treatment.
    assert padded_batch.PADDED_MODEL_AXES["OpenVoiceV1ToneConverter"] == ("mel_frame",)
    assert padded_batch.PADDED_MODEL_AXES["OpenVoiceV2ToneConverter"] == ("mel_frame",)
    # Both synthesizer axes are on the phone axis.
    assert padded_batch.PADDED_MODEL_AXES["OpenVoiceV1Synthesizer"] == ("phone",)
    assert padded_batch.PADDED_MODEL_AXES["MeloTTSSynthesizer"] == ("phone",)
    # VC's only default_token_budget consumer is the speech-token S3Gen vocode,
    # which has no recorded cost model (documented in the task log).
    assert "ChatterboxVC" not in padded_batch.PADDED_MODEL_AXES


@pytest.mark.unit
def test_encoder_sample_plan_converts_the_frame_cost():
    padded_batch = _padded_batch()

    # The OpenVoice encoder plans on segment sample counts while padding frames,
    # so one sample is one frame / hop-length worth of activation.
    assert padded_batch.bytes_per_unit_for("mel_frame", units_per_frame=441) == MEL_FRAME_BYTES // 441
    assert padded_batch.bytes_per_unit_for("mel_frame") == MEL_FRAME_BYTES
    # An axis with no recorded cost stays ungoverned (None), never a guess.
    assert padded_batch.bytes_per_unit_for("speech_token") is None


# ---------------------------------------------------------------------------
# Prometheus rendering (zero state is part of the contract)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_metrics_zero_state_when_nothing_was_governed():
    body = _render()(None)

    assert "# TYPE hse_padded_batch_shrink_total counter" in body
    assert "# TYPE hse_padded_batch_headroom_bytes gauge" in body
    assert 'hse_padded_batch_shrink_total{model="none",axis="none"} 0' in body
    assert (
        'hse_padded_batch_headroom_bytes{model="none",axis="none",device="none"} 0'
        in body
    )


@pytest.mark.unit
def test_render_metrics_emits_the_recorded_model_axis_values():
    stats = {
        "enabled": True,
        "models": {
            "ov2-tone-converter": {
                "model_type": "OpenVoiceV2ToneConverter",
                "device": "cuda:0",
                "headroom_bytes": TIGHT_FREE,
                "axes": {
                    "mel_frame": {
                        "shrinks_total": 3,
                        "headroom_bytes": TIGHT_FREE,
                        "requested_area": 7500,
                        "allowed_area": 2596,
                        "width": 1,
                        "max_width": 8,
                        "median_length": 3400,
                        "state": "floor",
                    }
                },
            }
        },
    }

    body = _render()(stats)

    assert (
        'hse_padded_batch_shrink_total{model="ov2-tone-converter",axis="mel_frame"} 3'
        in body
    )
    assert (
        'hse_padded_batch_headroom_bytes{model="ov2-tone-converter",axis="mel_frame",'
        f'device="cuda:0"}} {TIGHT_FREE}'
        in body
    )
    assert 'model="none"' not in body


# ---------------------------------------------------------------------------
# Engine aggregation + /metrics + /health wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_engine_stats_zero_fill_configured_padded_models():
    engine = _engine_shell()
    engine.model_executors = {
        "ov2-tone-converter": _padded_executor(),
        # Not a padded family: stays out of the governance block entirely.
        "whisper": SimpleNamespace(
            model_config=ModelConfig(
                name="whisper",
                model="dummy/whisper",
                model_type="FasterWhisper",
                max_batch_size=16,
                device_config=DeviceConfig("cuda"),
                dtype="float16",
            ),
            driver_worker=SimpleNamespace(model_runner=SimpleNamespace()),
            check_health=lambda: None,
        ),
    }

    stats = engine.get_padded_batch_stats()

    assert stats["enabled"] is True
    assert set(stats["models"]) == {"ov2-tone-converter"}
    entry = stats["models"]["ov2-tone-converter"]
    assert entry["device"] == "cuda"
    assert set(entry["axes"]) == {"mel_frame"}
    axis = entry["axes"]["mel_frame"]
    assert axis["shrinks_total"] == 0
    assert axis["headroom_bytes"] == 0
    assert axis["width"] == 0


@pytest.mark.unit
def test_engine_stats_merge_the_runner_recorded_state():
    recorded = {
        "enabled": True,
        "device": "cuda:0",
        "headroom_bytes": TIGHT_FREE,
        "axes": {
            "mel_frame": {
                "shrinks_total": 2,
                "headroom_bytes": TIGHT_FREE,
                "requested_area": 7500,
                "allowed_area": 2596,
                "width": 1,
                "max_width": 8,
                "median_length": 3400,
                "state": "floor",
            }
        },
    }
    runner = SimpleNamespace(padded_batch_stats=lambda: recorded)
    engine = _engine_shell()
    engine.model_executors = {"ov2-tone-converter": _padded_executor(runner=runner)}

    stats = engine.get_padded_batch_stats()

    entry = stats["models"]["ov2-tone-converter"]
    assert entry["device"] == "cuda:0"
    assert entry["headroom_bytes"] == TIGHT_FREE
    assert entry["axes"]["mel_frame"]["shrinks_total"] == 2
    assert entry["axes"]["mel_frame"]["state"] == "floor"
    assert entry["axes"]["mel_frame"]["allowed_area"] == 2596


@pytest.mark.unit
def test_engine_stats_are_empty_and_quiet_without_padded_models():
    stats = _engine_shell().get_padded_batch_stats()

    assert stats["enabled"] is False
    assert stats["models"] == {}


@pytest.mark.unit
def test_metrics_endpoint_renders_the_padded_batch_series(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.model_executors = {"ov2-tone-converter": _padded_executor()}
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", None)
    monkeypatch.setattr(api_server, "openai_serving_vc", None)
    monkeypatch.setattr(api_server, "openai_serving_embedding", None)

    response = asyncio.run(api_server.metrics())
    body = response.body.decode()

    assert (
        'hse_padded_batch_shrink_total{model="ov2-tone-converter",axis="mel_frame"} 0'
        in body
    )
    assert (
        'hse_padded_batch_headroom_bytes{model="ov2-tone-converter",axis="mel_frame"'
        in body
    )


@pytest.mark.unit
def test_metrics_endpoint_renders_zero_state_without_an_engine(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    for attribute in (
        "openai_serving_tts",
        "openai_serving_stt",
        "openai_serving_vc",
        "openai_serving_embedding",
    ):
        monkeypatch.setattr(api_server, attribute, None)

    body = asyncio.run(api_server.metrics()).body.decode()

    assert 'hse_padded_batch_shrink_total{model="none",axis="none"} 0' in body


@pytest.mark.unit
def test_health_endpoint_carries_the_padded_batch_block(monkeypatch):
    from harmonyspeech.endpoints.openai import api_server

    engine = _engine_shell()
    engine.model_executors = {"ov2-tone-converter": _padded_executor()}
    tts_engine = SimpleNamespace(check_health=AsyncMock(return_value=engine.check_health()))
    healthy_engine = SimpleNamespace(check_health=AsyncMock(return_value=None))
    monkeypatch.setattr(api_server, "openai_serving_tts", SimpleNamespace(engine=tts_engine))
    monkeypatch.setattr(api_server, "openai_serving_stt", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vc", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_embedding", SimpleNamespace(engine=healthy_engine))
    monkeypatch.setattr(api_server, "openai_serving_vad", SimpleNamespace(engine=healthy_engine))

    payload = json.loads(asyncio.run(api_server.health()).body)

    assert "padded_batch" in payload, "the block must sit next to whisper_vram/whisper_admission"
    block = payload["padded_batch"]
    assert block["enabled"] is True
    assert block["models"]["ov2-tone-converter"]["axes"]["mel_frame"]["state"] == "idle"
