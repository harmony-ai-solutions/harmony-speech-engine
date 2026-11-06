"""Unit tests for the cross-request embedding cache policies (Phase 7-1).

Every eviction/copy rule from the plan is locked down here: TTL expiry (lazy on
get + the every-16-puts sweep), FIFO — not LRU — beyond the entry cap, the byte
ceiling, the disabled modes, deep-copy isolation in both directions, key
stability/sensitivity, per-model load versions and the counter contract.
"""

import time
from dataclasses import dataclass

import pytest
import torch

from harmonyspeech.common.artifacts import SpeakerArtifact
from harmonyspeech.processing.embedding_cache import EmbeddingCache

IDENTITY = "stub|OpenVoiceV2ToneConverterEncoder|torch.float32|repo/model@main|v1"
OTHER_IDENTITY = "stub2|OpenVoiceV1ToneConverterEncoder|torch.float32|repo/model2@main|v1"


class FakeClock:
    """Deterministic replacement for ``time.monotonic``."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeT3Cond:
    speaker_emb: torch.Tensor
    emotion_adv: torch.Tensor


@dataclass
class FakeConditionals:
    t3: FakeT3Cond
    gen: dict


def _embedding_artifact(fill: float = 1.0, samples: int = 4) -> SpeakerArtifact:
    return SpeakerArtifact(embedding=torch.full((samples,), fill), backend="openvoice")


def _conditionals_artifact(marker: float = 2.0) -> SpeakerArtifact:
    return SpeakerArtifact(
        conditionals=FakeConditionals(
            t3=FakeT3Cond(speaker_emb=torch.tensor([marker]), emotion_adv=torch.zeros(1)),
            gen={"ge_bes": torch.tensor([marker + 1.0])},
        ),
        backend="chatterbox",
    )


# ---------------------------------------------------------------------------
# Hit/miss basics + copy discipline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_put_then_get_serves_an_equal_deep_copy():
    cache = EmbeddingCache(ttl_s=300)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})

    cache.put(key, _embedding_artifact(), model_identity=IDENTITY)

    served = cache.get(key)
    assert served is not None
    assert torch.equal(served.embedding, torch.ones(4))
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["stores"] == 1


@pytest.mark.unit
def test_missing_key_is_a_miss():
    cache = EmbeddingCache(ttl_s=300)
    assert cache.get("nope") is None
    assert cache.stats()["misses"] == 1


@pytest.mark.unit
def test_stored_entry_is_unaffected_by_caller_mutations():
    cache = EmbeddingCache(ttl_s=300)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})
    artifact = _embedding_artifact()

    cache.put(key, artifact, model_identity=IDENTITY)
    artifact.embedding.add_(99.0)  # caller mutates the object it handed over

    assert torch.equal(cache.get(key).embedding, torch.ones(4))


@pytest.mark.unit
def test_served_copy_mutations_do_not_leak_into_the_cache():
    cache = EmbeddingCache(ttl_s=300)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})
    cache.put(key, _embedding_artifact(), model_identity=IDENTITY)

    first = cache.get(key)
    first.embedding.mul_(10.0)

    second = cache.get(key)
    assert first is not second
    assert torch.equal(second.embedding, torch.ones(4))


@pytest.mark.unit
def test_conditionals_are_deep_copied_on_serve():
    cache = EmbeddingCache(ttl_s=300)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"exaggeration": 0.5})
    original = _conditionals_artifact()
    cache.put(key, original, model_identity=IDENTITY)

    first = cache.get(key)
    assert first.conditionals is not original.conditionals
    assert first.conditionals.t3 is not original.conditionals.t3
    first.conditionals.t3.speaker_emb.add_(1.0)
    first.conditionals.gen["ge_bes"].add_(1.0)

    second = cache.get(key)
    assert second.conditionals is not first.conditionals
    assert torch.equal(second.conditionals.t3.speaker_emb, torch.tensor([2.0]))
    assert torch.equal(second.conditionals.gen["ge_bes"], torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_expired_entry_is_a_miss_and_counts(monkeypatch):
    clock = FakeClock(0.0)
    monkeypatch.setattr(time, "monotonic", clock)
    cache = EmbeddingCache(ttl_s=300)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})
    cache.put(key, _embedding_artifact(), model_identity=IDENTITY)

    clock.advance(299.0)
    assert cache.get(key) is not None

    clock.advance(1.0)
    assert cache.get(key) is None
    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["bytes"] == 0
    assert stats["ttl_expirations"] == 1
    assert stats["misses"] == 1


@pytest.mark.unit
def test_ttl_sweep_runs_every_sixteen_puts(monkeypatch):
    clock = FakeClock(0.0)
    monkeypatch.setattr(time, "monotonic", clock)
    cache = EmbeddingCache(ttl_s=100)

    for index in range(15):
        key = cache.make_key(model_identity=IDENTITY, audio_hash=f"a{index}", params={})
        cache.put(key, _embedding_artifact(), model_identity=IDENTITY)
    assert cache.stats()["ttl_expirations"] == 0

    clock.advance(500.0)  # every stored entry is now expired
    sweep_key = cache.make_key(model_identity=IDENTITY, audio_hash="a15", params={})
    cache.put(sweep_key, _embedding_artifact(), model_identity=IDENTITY)

    stats = cache.stats()
    assert stats["ttl_expirations"] == 15
    assert stats["entries"] == 1
    assert stats["stores"] == 16


@pytest.mark.unit
def test_ttl_s_zero_never_stores():
    cache = EmbeddingCache(ttl_s=0)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})

    cache.put(key, _embedding_artifact(), model_identity=IDENTITY)

    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["stores"] == 0
    assert cache.get(key) is None


# ---------------------------------------------------------------------------
# FIFO (not LRU) + byte ceiling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fifo_evicts_the_oldest_insertion_beyond_max_entries():
    cache = EmbeddingCache(ttl_s=300, max_entries=2)
    keys = [cache.make_key(model_identity=IDENTITY, audio_hash=f"a{index}", params={}) for index in range(3)]

    for key in keys:
        cache.put(key, _embedding_artifact(), model_identity=IDENTITY)

    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["fifo_evictions"] == 1
    assert cache.get(keys[0]) is None  # oldest insertion evicted first
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None


@pytest.mark.unit
def test_get_does_not_reorder_the_fifo_queue():
    cache = EmbeddingCache(ttl_s=300, max_entries=2)
    keys = [cache.make_key(model_identity=IDENTITY, audio_hash=f"a{index}", params={}) for index in range(3)]
    cache.put(keys[0], _embedding_artifact(), model_identity=IDENTITY)
    cache.put(keys[1], _embedding_artifact(), model_identity=IDENTITY)

    assert cache.get(keys[0]) is not None  # a hit must NOT refresh insertion order

    cache.put(keys[2], _embedding_artifact(), model_identity=IDENTITY)
    # FIFO: keys[0] is still the oldest insertion and is evicted; an LRU
    # implementation would have evicted keys[1] instead.
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None


@pytest.mark.unit
def test_byte_ceiling_evicts_the_oldest_insertion_first():
    # Each artifact below is exactly 32 bytes of tensor data (8 float32).
    cache = EmbeddingCache(ttl_s=300, max_entries=100, max_bytes=64)
    keys = [cache.make_key(model_identity=IDENTITY, audio_hash=f"a{index}", params={}) for index in range(3)]

    for key in keys:
        cache.put(key, _embedding_artifact(samples=8), model_identity=IDENTITY)

    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["byte_evictions"] == 1
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) is not None
    assert cache.get(keys[2]) is not None


@pytest.mark.unit
def test_byte_ceiling_is_a_hard_bound_even_for_an_oversized_entry():
    cache = EmbeddingCache(ttl_s=300, max_entries=100, max_bytes=16)
    key = cache.make_key(model_identity=IDENTITY, audio_hash="a0", params={})

    cache.put(key, _embedding_artifact(samples=8), model_identity=IDENTITY)  # 32 bytes

    assert cache.get(key) is None
    assert cache.stats()["entries"] == 0


# ---------------------------------------------------------------------------
# Keys + load versions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_make_key_is_stable_and_order_insensitive():
    cache = EmbeddingCache(ttl_s=300)
    base = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"a": 1, "b": 2})

    assert cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"a": 1, "b": 2}) == base
    assert cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"b": 2, "a": 1}) == base


@pytest.mark.unit
def test_make_key_differs_on_every_key_component():
    cache = EmbeddingCache(ttl_s=300)
    base = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"vad_source": "whisper"})

    assert cache.make_key(model_identity=OTHER_IDENTITY, audio_hash="abc", params={"vad_source": "whisper"}) != base
    assert cache.make_key(model_identity=IDENTITY, audio_hash="different", params={"vad_source": "whisper"}) != base
    assert cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={"vad_source": "silero"}) != base
    assert cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={}) != base


@pytest.mark.unit
def test_make_key_differs_on_load_version():
    cache = EmbeddingCache(ttl_s=300)
    version_one = cache.make_key(model_identity="m|v1", audio_hash="abc", params={})
    version_two = cache.make_key(model_identity="m|v2", audio_hash="abc", params={})
    assert version_one != version_two


@pytest.mark.unit
def test_bump_load_version_counts_per_model_name():
    cache = EmbeddingCache(ttl_s=300)

    assert cache.load_version("model-a") == 0
    assert cache.bump_load_version("model-a") == 1
    assert cache.bump_load_version("model-a") == 2
    assert cache.load_version("model-a") == 2
    assert cache.load_version("model-b") == 0


@pytest.mark.unit
def test_invalidated_model_entries_miss():
    cache = EmbeddingCache(ttl_s=300)
    identity_key = cache.make_key(model_identity=IDENTITY, audio_hash="abc", params={})
    other_key = cache.make_key(model_identity=OTHER_IDENTITY, audio_hash="abc", params={})
    cache.put(identity_key, _embedding_artifact(), model_identity=IDENTITY)
    cache.put(other_key, _embedding_artifact(), model_identity=OTHER_IDENTITY)

    removed = cache.invalidate_model(IDENTITY)

    assert removed == 1
    assert cache.get(identity_key) is None
    assert cache.get(other_key) is not None
    assert cache.stats()["entries"] == 1


@pytest.mark.unit
def test_version_bump_makes_old_identity_unreachable():
    cache = EmbeddingCache(ttl_s=300)
    v1_identity = "m|OpenVoiceV2ToneConverterEncoder|f32|repo@main|v1"
    v2_identity = "m|OpenVoiceV2ToneConverterEncoder|f32|repo@main|v2"
    v1_key = cache.make_key(model_identity=v1_identity, audio_hash="abc", params={})
    cache.put(v1_key, _embedding_artifact(), model_identity=v1_identity)

    # A reload bumps the version and invalidates the previous identity...
    cache.bump_load_version("m")
    assert cache.invalidate_model(v1_identity) == 1

    # ...so the old key misses and the new identity produces a different key.
    assert cache.get(v1_key) is None
    assert cache.make_key(model_identity=v2_identity, audio_hash="abc", params={}) != v1_key


# ---------------------------------------------------------------------------
# Counter contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stats_counters_are_consistent_after_a_scripted_sequence():
    cache = EmbeddingCache(ttl_s=300, max_entries=2, max_bytes=None)
    key_a = cache.make_key(model_identity=IDENTITY, audio_hash="a0", params={})
    key_b = cache.make_key(model_identity=IDENTITY, audio_hash="a1", params={})
    key_c = cache.make_key(model_identity=OTHER_IDENTITY, audio_hash="a2", params={})

    cache.put(key_a, _embedding_artifact(samples=8), model_identity=IDENTITY)  # store 1
    assert cache.get(key_a) is not None  # hit 1
    assert cache.get("missing") is None  # miss 1
    cache.put(key_b, _embedding_artifact(samples=8), model_identity=IDENTITY)  # store 2
    cache.put(key_c, _embedding_artifact(samples=8), model_identity=OTHER_IDENTITY)  # store 3, evicts a
    assert cache.get(key_b) is not None  # hit 2
    assert cache.get(key_a) is None  # miss 2 (FIFO-evicted, not TTL)
    assert cache.invalidate_model(IDENTITY) == 1  # drops b

    stats = cache.stats()
    assert stats == {
        "entries": 1,
        "bytes": 32,
        "hits": 2,
        "misses": 2,
        "stores": 3,
        "ttl_expirations": 0,
        "fifo_evictions": 1,
        "byte_evictions": 0,
    }
