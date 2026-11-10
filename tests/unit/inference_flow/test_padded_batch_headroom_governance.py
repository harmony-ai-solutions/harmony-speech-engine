"""Runner-level headroom governance for padded batches (Phase 14-6).

TDD contract: the headroom read, the per-axis shrink arithmetic, the warning
text, the no-spam transition rule, the width-1 floor, the static-cap ceiling,
the CPU no-go path and the per-runner metric state are pinned here **before**
the implementation. The headroom provider is the injectable seam
(``ModelRunnerBase.cuda_free_bytes_provider``), so no test touches a GPU.

Handler-level tests at the bottom prove the two headline call sites actually go
through the governor: the OpenVoice V1/V2 mel-frame tone converter and the
phone-axis synthesizers (OpenVoice V1, MeloTTS).
"""

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from loguru import logger as loguru_logger

import harmonyspeech.task_handler.model_runner_base as model_runner_base
from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import TextToSpeechRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.models.openvoice.openvoice import SynthesizerTrn
from harmonyspeech.task_handler.batching import (
    DEFAULT_MEL_FRAME_BATCH_BUDGET,
    DEFAULT_PHONE_BATCH_BUDGET,
    default_token_budget,
    plan_batches,
)
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

GIB = 2**30
FIXTURE_FREE = int(5.6 * GIB)
SHIPPED_FREE = int(2.5 * GIB)
TIGHT_FREE = int(0.9 * GIB)
LONG_ROW_FRAMES = 3400

#: Model geometry copied from the Phase 8-3 handler tests (hop 4 => 4 samples
#: per frame; 14640 samples render as 3660 mel frames).
SAMPLE_RATE = 16000
FILTER_LENGTH = 16
HOP_LENGTH = 4
WIN_LENGTH = 16
GIN_CHANNELS = 4


@pytest.fixture
def loguru_all():
    """Capture loguru at INFO+ (the recovery line is informational)."""
    records: list[str] = []
    handler_id = loguru_logger.add(lambda message: records.append(str(message)), level="INFO")
    try:
        yield records
    finally:
        loguru_logger.remove(handler_id)


def _runner(
    model_type: str = "OpenVoiceV2ToneConverter",
    *,
    device: str = "cuda:0",
    free: int | None = None,
    max_batch_size: int = 8,
    name: str = "ov2-tone-converter",
    **config,
) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = device
    runner.device_config = DeviceConfig(device)
    runner._baseline_state = None
    runner.model = None
    runner.embedding_cache = None
    runner.model_identity = None
    runner.model_config = ModelConfig(
        name=name,
        model="dummy/openvoice",
        model_type=model_type,
        max_batch_size=max_batch_size,
        device_config=DeviceConfig(device),
        dtype="float16",
        **config,
    )
    if free is not None:
        runner.cuda_free_bytes_provider = lambda: free
    return runner


def _plan(runner, lengths, *, axis="mel_frame", max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET, max_batch_size=8):
    return runner._plan_padded_batches(
        lengths,
        max_batch_size,
        axis=axis,
        max_budget=max_budget,
    )


def _width(groups):
    return max(len(group) for group in groups)


# ---------------------------------------------------------------------------
# The headroom read and the per-axis arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plan_reads_the_headroom_provider_at_plan_time():
    free = {"value": FIXTURE_FREE}
    runner = _runner()
    runner.cuda_free_bytes_provider = lambda: free["value"]

    groups = _plan(runner, [LONG_ROW_FRAMES] * 8)

    assert _width(groups) == 2
    stats = runner.padded_batch_stats()
    assert stats["headroom_bytes"] == FIXTURE_FREE
    assert stats["axes"]["mel_frame"]["state"] == "ok"

    free["value"] = TIGHT_FREE
    groups = _plan(runner, [LONG_ROW_FRAMES] * 8)

    assert _width(groups) == 1
    assert runner.padded_batch_stats()["headroom_bytes"] == TIGHT_FREE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("free", "allowed_area", "shrunk", "width"),
    [
        (FIXTURE_FREE, DEFAULT_MEL_FRAME_BATCH_BUDGET, False, 2),
        (SHIPPED_FREE, 7211, True, 2),
        (TIGHT_FREE, 2596, True, 1),
    ],
)
def test_mel_frame_shrink_arithmetic_and_chosen_width(free, allowed_area, shrunk, width):
    runner = _runner(free=free)

    groups = _plan(runner, [LONG_ROW_FRAMES] * 8)

    entry = runner.padded_batch_stats()["axes"]["mel_frame"]
    assert entry["requested_area"] == DEFAULT_MEL_FRAME_BATCH_BUDGET
    assert entry["allowed_area"] == allowed_area
    assert entry["headroom_bytes"] == free
    assert entry["width"] == width
    assert _width(groups) == width
    assert (entry["shrinks_total"] == 1) is shrunk


@pytest.mark.unit
def test_phone_axis_shrinks_only_when_headroom_tightens():
    shipped = _runner(model_type="MeloTTSSynthesizer", name="ov2-synthesizer-en", free=SHIPPED_FREE)
    groups = _plan(
        shipped,
        [500] * 8,
        axis="phone",
        max_budget=DEFAULT_PHONE_BATCH_BUDGET,
    )
    assert _width(groups) == 2
    assert shipped.padded_batch_stats()["axes"]["phone"]["shrinks_total"] == 0

    tight = _runner(model_type="MeloTTSSynthesizer", name="ov2-synthesizer-en", free=TIGHT_FREE)
    groups = _plan(
        tight,
        [500] * 8,
        axis="phone",
        max_budget=DEFAULT_PHONE_BATCH_BUDGET,
    )
    assert _width(groups) == 1
    entry = tight.padded_batch_stats()["axes"]["phone"]
    assert entry["allowed_area"] == 512
    assert entry["shrinks_total"] == 1

    # Short chunked rows still keep width under the same headroom.
    groups = _plan(
        tight,
        [128] * 8,
        axis="phone",
        max_budget=DEFAULT_PHONE_BATCH_BUDGET,
    )
    assert _width(groups) == 4


@pytest.mark.unit
def test_static_cap_remains_the_ceiling_even_with_huge_headroom():
    runner = _runner(free=100 * GIB)

    _plan(runner, [LONG_ROW_FRAMES] * 8)

    assert runner.padded_batch_stats()["axes"]["mel_frame"]["allowed_area"] == (
        DEFAULT_MEL_FRAME_BATCH_BUDGET
    )

    explicit = _runner(free=100 * GIB, batch_token_budget=1234)
    _plan(explicit, [LONG_ROW_FRAMES] * 8)
    entry = explicit.padded_batch_stats()["axes"]["mel_frame"]
    assert entry["requested_area"] == 1234
    assert entry["allowed_area"] == 1234
    assert entry["shrinks_total"] == 0


@pytest.mark.unit
def test_zero_headroom_hits_the_width_one_floor(loguru_records):
    runner = _runner(free=0)

    groups = _plan(runner, [LONG_ROW_FRAMES] * 4, max_batch_size=4)

    assert groups == [[0], [1], [2], [3]]
    entry = runner.padded_batch_stats()["axes"]["mel_frame"]
    assert entry["allowed_area"] == 1
    assert entry["state"] == "floor"
    assert any("batch width 1 of 4" in message for message in loguru_records)


@pytest.mark.unit
def test_ample_headroom_plan_is_identical_to_the_static_cap_plan():
    lengths = [120, 120, 300, 300, 3400, 3400, 3400, 3400]
    expected_budget = default_token_budget(
        lengths, 8, max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET
    )
    expected = plan_batches(lengths, 8, expected_budget)

    runner = _runner(free=FIXTURE_FREE)
    groups = _plan(runner, lengths)

    assert groups == expected
    assert runner.padded_batch_stats()["axes"]["mel_frame"]["shrinks_total"] == 0


# ---------------------------------------------------------------------------
# Warning text + transition discipline (no log spam)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_warning_names_model_axis_areas_headroom_and_chosen_width(loguru_records):
    runner = _runner(free=TIGHT_FREE)

    _plan(runner, [LONG_ROW_FRAMES] * 8)

    warnings = [message for message in loguru_records if "Padded batch shrunk" in message]
    assert len(warnings) == 1
    message = warnings[0]
    assert "ov2-tone-converter" in message
    assert "mel_frame axis" in message
    assert "7500 -> 2596 units" in message
    assert "0.90 GiB free device headroom" in message
    assert "batch width 1 of 8" in message
    assert "Phase 14-6" in message


@pytest.mark.unit
def test_warning_is_emitted_once_per_state_change_and_never_spams(loguru_records, loguru_all):
    runner = _runner(free=TIGHT_FREE)

    _plan(runner, [LONG_ROW_FRAMES] * 8)
    _plan(runner, [LONG_ROW_FRAMES] * 8)
    # A few MiB of jitter leaves the width unchanged: no second warning.
    runner.cuda_free_bytes_provider = lambda: TIGHT_FREE - 3 * 2**20
    _plan(runner, [LONG_ROW_FRAMES] * 8)

    warnings = [message for message in loguru_records if "Padded batch shrunk" in message]
    assert len(warnings) == 1

    # Recovery is a state change (info, not a warning), and re-entering the
    # shrunk state warns exactly once more.
    runner.cuda_free_bytes_provider = lambda: FIXTURE_FREE
    _plan(runner, [LONG_ROW_FRAMES] * 8)
    runner.cuda_free_bytes_provider = lambda: TIGHT_FREE
    _plan(runner, [LONG_ROW_FRAMES] * 8)
    _plan(runner, [LONG_ROW_FRAMES] * 8)

    warnings = [message for message in loguru_records if "Padded batch shrunk" in message]
    assert len(warnings) == 2
    restores = [message for message in loguru_all if "Padded batch width restored" in message]
    assert len(restores) == 1
    assert "batch width 2 of 8" in restores[0]


@pytest.mark.unit
def test_recovery_reports_the_restored_width_and_clears_the_state(loguru_all):
    runner = _runner(free=TIGHT_FREE)
    _plan(runner, [LONG_ROW_FRAMES] * 8)
    assert runner.padded_batch_stats()["axes"]["mel_frame"]["state"] == "floor"

    runner.cuda_free_bytes_provider = lambda: FIXTURE_FREE
    _plan(runner, [LONG_ROW_FRAMES] * 8)

    entry = runner.padded_batch_stats()["axes"]["mel_frame"]
    assert entry["state"] == "ok"
    assert entry["width"] == 2
    assert entry["shrinks_total"] == 1
    assert any("Padded batch width restored" in message for message in loguru_all)


# ---------------------------------------------------------------------------
# CPU path untouched + unavailable device memory fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cpu_runner_never_reads_headroom_and_plans_exactly_as_before(loguru_records):
    lengths = [120, 120, 300, 300, 3400, 3400]
    expected_budget = default_token_budget(
        lengths, 8, max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET
    )
    expected = plan_batches(lengths, 8, expected_budget)

    runner = _runner(device="cpu")

    def _explode():  # pragma: no cover - only runs on a regression
        raise AssertionError("the headroom provider must not be read on CPU")

    runner.cuda_free_bytes_provider = _explode

    groups = _plan(runner, lengths)

    assert groups == expected
    stats = runner.padded_batch_stats()
    assert stats["enabled"] is False
    assert stats["axes"] == {}
    assert loguru_records == []


@pytest.mark.unit
def test_unavailable_device_memory_falls_back_to_the_static_cap(monkeypatch, loguru_records):
    lengths = [LONG_ROW_FRAMES] * 8
    expected_budget = default_token_budget(
        lengths, 8, max_budget=DEFAULT_MEL_FRAME_BATCH_BUDGET
    )
    expected = plan_batches(lengths, 8, expected_budget)

    def _unavailable(*args, **kwargs):
        raise RuntimeError("no CUDA context")

    monkeypatch.setattr(torch.cuda, "mem_get_info", _unavailable)

    runner = _runner(device="cuda:0")  # no injected provider
    groups = _plan(runner, lengths)

    assert groups == expected
    assert runner.padded_batch_stats()["axes"] == {}
    assert loguru_records == []


# ---------------------------------------------------------------------------
# Per-runner state shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_padded_batch_stats_shape():
    runner = _runner(free=TIGHT_FREE)
    _plan(runner, [LONG_ROW_FRAMES] * 8)

    stats = runner.padded_batch_stats()

    assert stats["enabled"] is True
    assert stats["device"] == "cuda:0"
    assert stats["headroom_bytes"] == TIGHT_FREE
    assert set(stats["axes"]) == {"mel_frame"}
    entry = stats["axes"]["mel_frame"]
    assert entry == {
        "shrinks_total": 1,
        "headroom_bytes": TIGHT_FREE,
        "requested_area": DEFAULT_MEL_FRAME_BATCH_BUDGET,
        "allowed_area": 2596,
        "width": 1,
        "max_width": 8,
        "median_length": LONG_ROW_FRAMES,
        "state": "floor",
    }


@pytest.mark.unit
def test_every_budget_consumer_goes_through_the_governed_planner():
    import inspect

    source = inspect.getsource(model_runner_base)

    # One budget formula and one planner call: the shared resolver. (Substring
    # counting must not overlap: `_plan_padded_batches(` itself contains
    # `plan_batches(`, so the planner is counted by its assignment line.)
    assert source.count("default_token_budget(") == 1
    assert source.count("groups = plan_batches(") == 1
    # Every padded call site calls the resolver (10 consumers, Phase 14-6 log).
    assert source.count("self._plan_padded_batches(") == 10


# ---------------------------------------------------------------------------
# Handler wiring: the OpenVoice tone converter actually shrinks
# ---------------------------------------------------------------------------


def _fake_hf_config():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            filter_length=FILTER_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
    )


def _tiny_tone_converter() -> SynthesizerTrn:
    torch.manual_seed(0)
    return SynthesizerTrn(
        n_vocab=64,
        spec_channels=FILTER_LENGTH // 2 + 1,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=8,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=8,
        upsample_kernel_sizes=[4, 4],
        n_speakers=0,
        gin_channels=GIN_CHANNELS,
    ).eval()


def _clip(samples: int, freq: float = 220.0) -> AudioArtifact:
    t = np.arange(samples) / SAMPLE_RATE
    return AudioArtifact.from_numpy(
        (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE
    )


def _speaker(value: float) -> SpeakerArtifact:
    return SpeakerArtifact(
        embedding=torch.full((1, GIN_CHANNELS, 1), value), backend="openvoice"
    )


def _tc_request(request_id: str) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="ov2-tone-converter",
        model="ov2-tone-converter",
        input_text="hello",
        mode="voice_cloning",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _run_tone_converter(runner, samples, count=4):
    model = runner.model
    forwarded: list[tuple] = []
    original = model.voice_conversion_batched

    def spy(padded, lengths, **kwargs):
        forwarded.append(tuple(padded.shape))
        return original(padded, lengths, **kwargs)

    model.voice_conversion_batched = spy
    inputs = [
        (_clip(samples, 220.0 + 40.0 * index), _speaker(0.1), _speaker(0.2))
        for index in range(count)
    ]
    requests = [_tc_request(f"tc-{index}") for index in range(count)]
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config",
        return_value=_fake_hf_config(),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_flavour",
        return_value="default",
    ):
        results = runner._execute_openvoice_tone_converter(inputs, requests)
    return results, forwarded


@pytest.mark.unit
def test_tone_converter_batches_at_width_two_on_fixture_headroom():
    runner = _runner(free=FIXTURE_FREE)
    runner.model = _tiny_tone_converter()

    results, forwarded = _run_tone_converter(runner, samples=LONG_ROW_FRAMES * HOP_LENGTH)

    assert len(results) == 4
    assert len(forwarded) == 2, "4 x 3400-frame rows at 5.6 GiB free => two 2-row forwards"
    assert runner.padded_batch_stats()["axes"]["mel_frame"]["width"] == 2
    assert runner.padded_batch_stats()["axes"]["mel_frame"]["shrinks_total"] == 0


@pytest.mark.unit
def test_tone_converter_shrinks_under_the_shipped_headroom(loguru_records):
    runner = _runner(free=SHIPPED_FREE)
    runner.model = _tiny_tone_converter()

    results, forwarded = _run_tone_converter(runner, samples=3660 * HOP_LENGTH)

    assert len(results) == 4
    assert len(forwarded) == 4, "3660-frame rows do not fit two-wide in 2.5 GiB"
    entry = runner.padded_batch_stats()["axes"]["mel_frame"]
    assert entry["width"] == 1
    assert entry["allowed_area"] == 7211
    assert entry["shrinks_total"] == 1
    assert any("Padded batch shrunk" in message for message in loguru_records)


# ---------------------------------------------------------------------------
# Handler wiring: the phone-axis synthesizers shrink too (OV1 + MeloTTS)
# ---------------------------------------------------------------------------


class StubOpenVoiceModel(torch.nn.Module):
    """OpenVoice V1 stand-in: frames = tokens * frames_per_token."""

    def __init__(self, frames_per_token: int = 2):
        super().__init__()
        self.anchor = torch.nn.Linear(1, 1)
        self.frames_per_token = frames_per_token
        self.batched_calls: list[dict] = []

    def infer(self, x, x_lengths, sid=None, **kwargs):  # pragma: no cover - fallback only
        frames = int(x_lengths[0]) * self.frames_per_token
        return (torch.zeros(1, 1, frames * HOP_LENGTH), None, None, None)

    def infer_batched(self, x, x_lengths, sid=None, **kwargs):
        self.batched_calls.append({"batch": int(x.shape[0])})
        y_lengths = x_lengths * self.frames_per_token
        output = torch.zeros(
            int(x.shape[0]), 1, int(y_lengths.max()) * HOP_LENGTH
        )
        return output, y_lengths, None, None, None


class StubMeloModel(StubOpenVoiceModel):
    """MeloTTS stand-in with the extra padded inputs."""

    def infer(self, x, x_lengths, sid, tone, language, bert, ja_bert, **kwargs):  # pragma: no cover
        frames = int(x_lengths[0]) * self.frames_per_token
        return (torch.zeros(1, 1, frames * HOP_LENGTH), None, None, None)

    def infer_batched(self, x, x_lengths, sid, tone, language, bert, ja_bert, **kwargs):
        self.batched_calls.append({"batch": int(x.shape[0])})
        y_lengths = x_lengths * self.frames_per_token
        output = torch.zeros(
            int(x.shape[0]), 1, int(y_lengths.max()) * HOP_LENGTH
        )
        return output, y_lengths, None, None, None


SAMPLING = {"noise_scale": 0.667, "noise_scale_w": 0.6, "sdp_ratio": 0.2}


def _synth_request(request_id: str) -> EngineRequest:
    request_data = TextToSpeechRequestInput(
        request_id=request_id,
        requested_model="ov1-synthesizer-en",
        model="ov1-synthesizer-en",
        input_text="hello",
        mode="single_speaker_tts",
    )
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _run_openvoice_synth(runner, lengths):
    model = runner.model
    inputs = [
        ([torch.arange(1, length + 1, dtype=torch.long)], 0, 1.0, dict(SAMPLING))
        for length in lengths
    ]
    requests = [_synth_request(f"ov-{index}") for index in range(len(lengths))]
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config",
        return_value=_fake_hf_config(),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_flavour",
        return_value="default",
    ):
        runner._execute_openvoice_synthesizer(inputs, requests)
    return model.batched_calls


def _run_melo_synth(runner, lengths):
    model = runner.model
    inputs = []
    for length in lengths:
        item = (
            torch.zeros(1024, length),
            torch.zeros(768, length),
            torch.arange(1, length + 1, dtype=torch.long),
            torch.zeros(length, dtype=torch.long),
            torch.zeros(length, dtype=torch.long),
        )
        inputs.append(([item], 0, 1.0, dict(SAMPLING)))
    requests = [_synth_request(f"melo-{index}") for index in range(len(lengths))]
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config",
        return_value=_fake_hf_config(),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_flavour",
        return_value="default",
    ):
        runner._execute_melotts_synthesizer(inputs, requests)
    return model.batched_calls


@pytest.mark.unit
def test_openvoice_synthesizer_shrinks_on_the_phone_axis():
    ample = _runner(
        model_type="OpenVoiceV1Synthesizer",
        name="ov1-synthesizer-en",
        free=FIXTURE_FREE,
        max_batch_size=4,
    )
    ample.model = StubOpenVoiceModel()
    calls = _run_openvoice_synth(ample, [500] * 4)
    assert [call["batch"] for call in calls] == [2, 2]
    assert ample.padded_batch_stats()["axes"]["phone"]["shrinks_total"] == 0

    tight = _runner(
        model_type="OpenVoiceV1Synthesizer",
        name="ov1-synthesizer-en",
        free=TIGHT_FREE,
        max_batch_size=4,
    )
    tight.model = StubOpenVoiceModel()
    calls = _run_openvoice_synth(tight, [500] * 4)
    assert [call["batch"] for call in calls] == [1, 1, 1, 1]
    entry = tight.padded_batch_stats()["axes"]["phone"]
    assert entry["allowed_area"] == 512
    assert entry["shrinks_total"] == 1


@pytest.mark.unit
def test_melotts_synthesizer_shrinks_on_the_phone_axis():
    ample = _runner(
        model_type="MeloTTSSynthesizer",
        name="ov2-synthesizer-en",
        free=FIXTURE_FREE,
        max_batch_size=4,
    )
    ample.model = StubMeloModel()
    calls = _run_melo_synth(ample, [300] * 4)
    assert [call["batch"] for call in calls] == [3, 1]
    assert ample.padded_batch_stats()["axes"]["phone"]["shrinks_total"] == 0

    tight = _runner(
        model_type="MeloTTSSynthesizer",
        name="ov2-synthesizer-en",
        free=TIGHT_FREE,
        max_batch_size=4,
    )
    tight.model = StubMeloModel()
    calls = _run_melo_synth(tight, [300] * 4)
    assert [call["batch"] for call in calls] == [1, 1, 1, 1]
    assert tight.padded_batch_stats()["axes"]["phone"]["shrinks_total"] == 1
