"""Unit tests for the clone-flood reference modes (Phase 7-3).

The measurement itself needs a GPU and a running server; these tests prove the
*load shape* the Phase 7 section depends on: ``--clone-reference same`` really
reuses one clip on every request (so the embedding cache can hit), while
``distinct`` really gives every request its own clip with a unique
``content_hash`` (so the cache must miss) — both at the original duration and
sample rate, i.e. still valid speech-shaped audio for VAD/embed.
"""

import base64

import numpy as np
import pytest

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact
from scripts import bench_throughput as bench


def _reference_b64(seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """A deterministic speech-shaped stand-in for a `tests/audio_fixtures` clip."""
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t) ** 2
    waveform = (0.2 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return base64.b64encode(audio_codec.encode_audio(waveform, sample_rate, "wav")).decode("ascii")


REFERENCE = _reference_b64()


def _artifact(reference_b64: str) -> AudioArtifact:
    waveform, sample_rate = audio_codec.decode_wav_b64(reference_b64)
    return AudioArtifact.from_numpy(waveform, sample_rate)


@pytest.mark.unit
def test_parser_defaults_to_the_same_reference_mode():
    args = bench.build_parser().parse_args([])

    assert args.clone_reference == "same"
    assert args.reference == "repo_voice_a"


@pytest.mark.unit
def test_parser_rejects_an_unknown_reference_mode():
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--clone-reference", "mixed"])


@pytest.mark.unit
def test_same_mode_sends_one_identical_payload_to_every_request():
    payloads = bench._reference_payloads("openvoice_v2", REFERENCE, "same", 5)

    assert len(payloads) == 5
    assert {payload["input_audio"] for payload in payloads} == {REFERENCE}
    assert len({_artifact(payload["input_audio"]).content_hash for payload in payloads}) == 1
    assert {payload["mode"] for payload in payloads} == {"voice_cloning"}


@pytest.mark.unit
def test_distinct_mode_produces_one_unique_payload_per_request():
    payloads = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 5)
    artifacts = [_artifact(payload["input_audio"]) for payload in payloads]

    assert len({payload["input_audio"] for payload in payloads}) == 5
    assert len({artifact.content_hash for artifact in artifacts}) == 5

    base = _artifact(REFERENCE)
    for artifact in artifacts:
        # Speech-length audio at the source rate: VAD/embedding still work.
        assert artifact.sample_rate == base.sample_rate
        assert artifact.num_samples == base.num_samples
    # Request 0 is the untouched clip, so `same` and `distinct` share a baseline.
    assert artifacts[0].content_hash == base.content_hash


@pytest.mark.unit
def test_distinct_mode_is_deterministic_across_runs():
    first = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 4)
    second = bench._reference_payloads("openvoice_v2", REFERENCE, "distinct", 4)

    assert first == second


@pytest.mark.unit
def test_families_without_reference_audio_ignore_the_mode():
    same = bench._reference_payloads("melo", REFERENCE, "same", 4)
    distinct = bench._reference_payloads("melo", REFERENCE, "distinct", 4)

    assert same == distinct
    assert "input_audio" not in same[0]


@pytest.mark.unit
def test_unknown_mode_is_rejected_by_the_payload_builder():
    with pytest.raises(ValueError):
        bench._reference_payloads("openvoice_v2", REFERENCE, "nope", 2)
