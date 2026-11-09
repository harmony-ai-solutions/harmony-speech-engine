"""Unit tests for the batched embedder handlers (Phase 8-2).

The two embed handlers now flatten their work into rows — ``(request, VAD
segment)`` for the OpenVoice tone-converter encoder, ``(request, partial
utterance)`` for the HarmonySpeech encoder — pad them, forward one batch per
length-similar sub-batch, and reassemble per request. What is proven here:

* segment/partial rows from *different* requests share one forward;
* per-request aggregation matches the per-item loop (same segment order, same
  ``stack().mean()`` / ``np.mean`` arithmetic);
* cache hits (Phase 7-2) never enter the batch and outputs stay in request
  order;
* a failing batched forward degrades to the loop, a single request keeps the
  loop fast path, and a request with no surviving segment still gets its own
  error result without poisoning the batch.

Sub-batch grouping is the production rule (length similarity + per-step padded
area budget from :func:`default_token_budget`); the "one forward" assertions
pass a deliberately generous ``batch_token_budget`` so the expectation is about
row flattening, not about budget arithmetic (which has its own tests).
"""

import types
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harmonyspeech.common.artifacts import AudioArtifact, VadArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
from harmonyspeech.common.request import EngineRequest
from harmonyspeech.modeling.models.harmonyspeech.harmonyspeech import SpeakerEncoder
from harmonyspeech.modeling.models.openvoice.openvoice import ReferenceEncoder
from harmonyspeech.processing.embedding_cache import EmbeddingCache
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase

SAMPLE_RATE = 16000
FILTER_LENGTH = 16
HOP_LENGTH = 4
WIN_LENGTH = 16
SPEC_CHANNELS = FILTER_LENGTH // 2 + 1
GIN_CHANNELS = 6
MEL_CHANNELS = 6
PARTIAL_FRAMES = 20
#: Big enough that length-similarity, not the padded-area budget, decides groups.
ROOMY_BUDGET = 1_000_000


def _fake_hf_config():
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            filter_length=FILTER_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
    )


def _runner(model, model_type: str, cache=None, identity=None, **config) -> ModelRunnerBase:
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.device = "cpu"
    runner.device_config = DeviceConfig("cpu")
    runner._baseline_state = None
    runner.model = model
    runner.embedding_cache = cache
    runner.model_identity = identity
    runner.model_config = ModelConfig(
        name="stub",
        model="dummy/model",
        model_type=model_type,
        max_batch_size=config.pop("max_batch_size", 4),
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **config,
    )
    return runner


def _clip(samples: int, freq: float = 220.0) -> AudioArtifact:
    t = np.arange(samples) / SAMPLE_RATE
    return AudioArtifact.from_numpy((0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE)


def _embed_request(request_id: str, model: str) -> EngineRequest:
    request_data = SpeechEmbeddingRequestInput(request_id=request_id, requested_model=model, model=model)
    return EngineRequest(request_id=request_id, request_data=request_data, arrival_time=0.0)


def _ov_request(request_id: str, artifact: AudioArtifact, vad_source: str = "whisper") -> EngineRequest:
    """An OpenVoice embed request: the cache key hashes the decoded input audio."""
    request = _embed_request(request_id, "openvoice")
    request.artifacts = {
        "input_audio": artifact,
        "input_vad_data": VadArtifact(
            segments=[{"start": 0.0, "end": 0.1, "text": "hi"}], source=vad_source
        ),
    }
    return request


# ---------------------------------------------------------------------------
# Recording wrappers around the real vendored modules (no stub math)
# ---------------------------------------------------------------------------


def _recording_reference_encoder() -> tuple[ReferenceEncoder, list]:
    """A tiny real ``ReferenceEncoder`` whose batched calls are recorded."""
    torch.manual_seed(0)
    encoder = ReferenceEncoder(spec_channels=SPEC_CHANNELS, gin_channels=GIN_CHANNELS).eval()
    calls: list[list[int]] = []
    original = encoder.forward_batched

    def _record(inputs, frame_lengths=None):
        calls.append([int(value) for value in frame_lengths])
        return original(inputs, frame_lengths)

    encoder.forward_batched = _record
    return encoder, calls


def _recording_speaker_encoder() -> tuple[SpeakerEncoder, list]:
    """A tiny real ``SpeakerEncoder`` whose batched calls are recorded."""
    torch.manual_seed(0)
    encoder = SpeakerEncoder(
        mel_n_channels=MEL_CHANNELS,
        model_hidden_size=5,
        model_num_layers=1,
        model_embedding_size=4,
    ).eval()
    calls: list[list[int]] = []
    original = encoder.forward_batched

    def _record(utterances, frame_lengths=None, hidden_init=None):
        calls.append([int(value) for value in frame_lengths])
        return original(utterances, frame_lengths, hidden_init)

    encoder.forward_batched = _record
    return encoder, calls


class _FailingBatchedRefEncoder:
    """Real single-sample forward, but the batched entry point always fails."""

    def __init__(self, encoder: ReferenceEncoder):
        self.encoder = encoder
        self.batched_attempts = 0

    def __call__(self, inputs, mask=None):
        return self.encoder(inputs, mask)

    def forward_batched(self, inputs, frame_lengths=None):
        self.batched_attempts += 1
        raise RuntimeError("simulated batched-forward failure")


# ---------------------------------------------------------------------------
# OpenVoice tone-converter encoder
# ---------------------------------------------------------------------------


def _run_ov_encoder(runner, segments_per_request, requests):
    with patch(
        "harmonyspeech.task_handler.model_runner_base.get_model_config", return_value=_fake_hf_config()
    ), patch("harmonyspeech.task_handler.model_runner_base.get_model_flavour", return_value="default"):
        return runner._execute_openvoice_tone_converter_encoder(segments_per_request, requests)


def _ov_batch():
    """Three requests; the middle and last own several VAD segments.

    Frames at the STFT geometry used below (``filter_length=16``,
    ``hop_length=4``): 2000 samples → 500 frames, 2400 → 600, 3200 → 800.
    """
    segments = [
        [_clip(2000)],
        [_clip(2000, freq=330.0), _clip(3200, freq=440.0)],
        [_clip(3200, freq=550.0), _clip(2000, freq=660.0), _clip(2400, freq=770.0)],
    ]
    requests = [_ov_request(f"ov-{index}", batch[0]) for index, batch in enumerate(segments)]
    return segments, requests


@pytest.mark.unit
def test_openvoice_encoder_forwards_all_segments_in_one_batch():
    encoder, calls = _recording_reference_encoder()
    runner = _runner(
        types.SimpleNamespace(ref_enc=encoder),
        "OpenVoiceV2ToneConverterEncoder",
        max_batch_size=8,
        batch_token_budget=ROOMY_BUDGET,
    )
    segments, requests = _ov_batch()

    results = _run_ov_encoder(runner, segments, requests)

    assert [result.request_id for result in results] == ["ov-0", "ov-1", "ov-2"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    # Six segments from three requests → exactly one padded forward.
    assert calls == [[500, 500, 800, 800, 500, 600]]


@pytest.mark.unit
def test_openvoice_encoder_batched_matches_the_per_item_loop():
    encoder, _calls = _recording_reference_encoder()
    model = types.SimpleNamespace(ref_enc=encoder)
    batched_runner = _runner(model, "OpenVoiceV2ToneConverterEncoder", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(
        types.SimpleNamespace(ref_enc=encoder),
        "OpenVoiceV2ToneConverterEncoder",
        batched_inference=False,
    )
    segments, requests = _ov_batch()

    with torch.no_grad():
        batched = _run_ov_encoder(batched_runner, segments, requests)
        looped = _run_ov_encoder(
            loop_runner,
            segments,
            [_ov_request(f"ov-loop-{index}", batch[0]) for index, batch in enumerate(segments)],
        )

    assert len(batched) == len(looped) == 3
    for index in range(3):
        batched_embedding = batched[index].result_data.embedding_artifact.embedding
        loop_embedding = looped[index].result_data.embedding_artifact.embedding
        # Per-request segments are stacked and averaged in segment order, so the
        # only difference left is batch-size float accumulation (~1e-8).
        assert batched_embedding.shape == loop_embedding.shape == (1, GIN_CHANNELS, 1)
        assert torch.allclose(batched_embedding, loop_embedding, atol=1e-6)


@pytest.mark.unit
def test_openvoice_encoder_batch_of_four_with_two_hits_only_forwards_the_misses():
    encoder, calls = _recording_reference_encoder()
    cache = EmbeddingCache(ttl_s=300)
    identity = "ov2|OpenVoiceV2ToneConverterEncoder|float32|repo/ov@main|v1"
    runner = _runner(
        types.SimpleNamespace(ref_enc=encoder),
        "OpenVoiceV2ToneConverterEncoder",
        cache=cache,
        identity=identity,
        batch_token_budget=ROOMY_BUDGET,
    )
    segments = [
        [_clip(2000, freq=220.0)],
        [_clip(2400, freq=330.0)],
        [_clip(2800, freq=440.0)],
        [_clip(3200, freq=550.0)],
    ]
    artifacts = [batch[0] for batch in segments]

    # Warm requests 0 and 3 (one on each side of the batch).
    for index in (0, 3):
        _run_ov_encoder(runner, [segments[index]], [_ov_request(f"warm-{index}", artifacts[index])])
    assert cache.stats()["stores"] == 2
    calls.clear()

    requests = [_ov_request(f"ov-{index}", artifacts[index]) for index in range(4)]
    results = _run_ov_encoder(runner, segments, requests)

    assert [result.request_id for result in results] == ["ov-0", "ov-1", "ov-2", "ov-3"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    # Only the two misses reached the model, in one sub-batch of two segments
    # (2400 samples → 600 frames, 2800 → 700).
    assert calls == [[600, 700]]
    assert cache.stats()["hits"] == 2


@pytest.mark.unit
def test_openvoice_encoder_batched_failure_falls_back_to_the_loop():
    encoder, _calls = _recording_reference_encoder()
    failing = _FailingBatchedRefEncoder(encoder)
    batched_runner = _runner(types.SimpleNamespace(ref_enc=failing), "OpenVoiceV2ToneConverterEncoder")
    loop_runner = _runner(
        types.SimpleNamespace(ref_enc=encoder),
        "OpenVoiceV2ToneConverterEncoder",
        batched_inference=False,
    )
    segments, requests = _ov_batch()

    with torch.no_grad():
        results = _run_ov_encoder(batched_runner, segments, requests)
        reference = _run_ov_encoder(
            loop_runner,
            segments,
            [_ov_request(f"ov-ref-{index}", batch[0]) for index, batch in enumerate(segments)],
        )

    assert failing.batched_attempts >= 1
    assert all(result.result_data.finish_reason == "stop" for result in results)
    for index in range(len(segments)):
        assert torch.allclose(
            results[index].result_data.embedding_artifact.embedding,
            reference[index].result_data.embedding_artifact.embedding,
            atol=1e-6,
        )


@pytest.mark.unit
def test_openvoice_encoder_single_request_keeps_the_loop_path():
    encoder, calls = _recording_reference_encoder()
    runner = _runner(types.SimpleNamespace(ref_enc=encoder), "OpenVoiceV2ToneConverterEncoder")
    segments = [[_clip(2000), _clip(2400)]]

    results = _run_ov_encoder(runner, segments, [_ov_request("ov-solo", segments[0][0])])

    assert calls == []
    assert results[0].result_data.finish_reason == "stop"


@pytest.mark.unit
def test_openvoice_encoder_request_with_no_segment_errors_alone():
    encoder, calls = _recording_reference_encoder()
    runner = _runner(types.SimpleNamespace(ref_enc=encoder), "OpenVoiceV2ToneConverterEncoder")
    segments = [[], [_clip(2400)]]
    requests = [_ov_request("ov-empty", _clip(2000)), _ov_request("ov-full", segments[1][0])]

    results = _run_ov_encoder(runner, segments, requests)

    # A request with no surviving VAD segment cannot be embedded. The error is
    # the honest Phase 12-4 message, not ``torch.stack([])``'s opaque
    # "stack expects a non-empty TensorList".
    assert results[0].result_data.finish_reason == "error"
    assert "no detectable speech" in results[0].result_data.error
    assert "VAD segments empty" in results[0].result_data.error
    assert results[1].result_data.finish_reason == "stop"
    # The empty request contributed no row, so the batch held one segment.
    assert calls == [[600]]


@pytest.mark.unit
def test_openvoice_encoder_single_request_with_no_segment_raises_the_honest_error():
    """The ``len == 1`` loop path (no batching) reports the same message."""
    encoder, calls = _recording_reference_encoder()
    runner = _runner(
        types.SimpleNamespace(ref_enc=encoder),
        "OpenVoiceV2ToneConverterEncoder",
        batched_inference=False,
    )

    results = _run_ov_encoder(runner, [[]], [_ov_request("ov-empty-solo", _clip(2000))])

    assert results[0].result_data.finish_reason == "error"
    assert results[0].result_data.error == (
        "reference clip contains no detectable speech (VAD segments empty)"
    )
    assert calls == []


# ---------------------------------------------------------------------------
# HarmonySpeech encoder
# ---------------------------------------------------------------------------


def _partials(preprocessed) -> np.ndarray:
    """Deterministic fixed-length partials; the count follows the clip length.

    The vendored frontend always yields ``partial_utterance_n_frames`` frames per
    partial, so rows differ in count, not in length (the mixed-length row case is
    covered by ``tests/unit/modeling/test_batched_embedders.py``).
    """
    count = max(1, int(preprocessed.shape[-1]) // 400)
    base = np.linspace(0.0, 1.0, PARTIAL_FRAMES * MEL_CHANNELS, dtype=np.float32).reshape(
        PARTIAL_FRAMES, MEL_CHANNELS
    )
    return np.stack([base + 0.05 * index for index in range(count)])


def _run_hs_encoder(runner, artifacts, requests):
    with patch(
        "harmonyspeech.task_handler.model_runner_base.preprocess_wav",
        side_effect=lambda samples, source_sr: np.asarray(samples, dtype=np.float32),
    ), patch(
        "harmonyspeech.task_handler.model_runner_base.get_input_frames",
        side_effect=_partials,
    ):
        return runner._execute_harmonyspeech_encoder(artifacts, requests)


def _hs_batch():
    # 400-sample steps → 1, 2 and 3 partials for the three requests.
    artifacts = [_clip(400, freq=220.0), _clip(800, freq=330.0), _clip(1200, freq=440.0)]
    requests = [_embed_request(f"hs-{index}", "harmonyspeech") for index in range(3)]
    return artifacts, requests


@pytest.mark.unit
def test_harmonyspeech_encoder_forwards_all_partials_in_one_batch():
    encoder, calls = _recording_speaker_encoder()
    runner = _runner(
        encoder, "HarmonySpeechEncoder", max_batch_size=8, batch_token_budget=ROOMY_BUDGET
    )
    artifacts, requests = _hs_batch()

    results = _run_hs_encoder(runner, artifacts, requests)

    assert [result.request_id for result in results] == ["hs-0", "hs-1", "hs-2"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    # 1 + 2 + 3 partials → one forward of six rows.
    assert calls == [[PARTIAL_FRAMES] * 6]


@pytest.mark.unit
def test_harmonyspeech_encoder_batched_matches_the_per_item_loop():
    encoder, _calls = _recording_speaker_encoder()
    batched_runner = _runner(encoder, "HarmonySpeechEncoder", batch_token_budget=ROOMY_BUDGET)
    loop_runner = _runner(encoder, "HarmonySpeechEncoder", batched_inference=False)
    artifacts, requests = _hs_batch()

    with torch.no_grad():
        batched = _run_hs_encoder(batched_runner, artifacts, requests)
        looped = _run_hs_encoder(
            loop_runner,
            artifacts,
            [_embed_request(f"hs-loop-{index}", "harmonyspeech") for index in range(len(artifacts))],
        )

    for index in range(len(artifacts)):
        batched_embedding = batched[index].result_data.embedding_artifact.embedding
        loop_embedding = looped[index].result_data.embedding_artifact.embedding
        # Partials are averaged in order, so only batch-size float accumulation
        # can differ.
        assert torch.allclose(batched_embedding, loop_embedding, atol=1e-6)


@pytest.mark.unit
def test_harmonyspeech_encoder_batch_of_four_with_two_hits_only_forwards_the_misses():
    encoder, calls = _recording_speaker_encoder()
    cache = EmbeddingCache(ttl_s=300)
    identity = "hs|HarmonySpeechEncoder|float32|repo/hs@main|v1"
    runner = _runner(
        encoder,
        "HarmonySpeechEncoder",
        cache=cache,
        identity=identity,
        max_batch_size=8,
        batch_token_budget=ROOMY_BUDGET,
    )
    artifacts = [
        _clip(400, freq=220.0),
        _clip(800, freq=330.0),
        _clip(1200, freq=440.0),
        _clip(1600, freq=550.0),
    ]

    for index in (0, 2):
        _run_hs_encoder(
            runner, [artifacts[index]], [_embed_request(f"warm-{index}", "harmonyspeech")]
        )
    assert cache.stats()["stores"] == 2
    calls.clear()

    requests = [_embed_request(f"hs-{index}", "harmonyspeech") for index in range(4)]
    results = _run_hs_encoder(runner, artifacts, requests)

    assert [result.request_id for result in results] == ["hs-0", "hs-1", "hs-2", "hs-3"]
    assert all(result.result_data.finish_reason == "stop" for result in results)
    # Misses are requests 1 and 3 → 2 + 4 partials in one sub-batch.
    assert calls == [[PARTIAL_FRAMES] * 6]
    assert cache.stats()["hits"] == 2
