"""Voice-pack persistent cache unit gates (Phase 13-2).

Hermetic, CPU-only: no engine, no model weights. Covers the pieces that are pure
data plumbing — the cache directory default, the content key (including wav-hash
and params invalidation), the atomic ``weights_only=True`` disk round-trip, corrupt
entry tolerance, pruning, warm-up planning, and disk -> 7-2 hydration.
"""

from __future__ import annotations

import torch

from harmonyspeech.common.artifacts import SpeakerArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.modeling.voice_catalog import standard_voice_names
from harmonyspeech.modeling.voice_pack_cache import (
    VoicePackDiskCache,
    build_warmup_targets,
    default_voice_pack_cache_dir,
    disk_key,
    hydrate_embedding_cache,
)
from harmonyspeech.processing.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Fakes (no engine / no weights)
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, identity):
        self.model_identity = identity


class _FakeExecutor:
    def __init__(self, identity):
        self.runner = _FakeRunner(identity)


class _FakeEngine:
    def __init__(self, configs, executors, cache):
        self.model_configs = configs
        self.model_executors = executors
        self.embedding_cache = cache
        self.voice_pack_disk_cache = None


def _config(name, model_type, *, model="stub/model", revision=None):
    return ModelConfig(
        name=name,
        model=model,
        model_type=model_type,
        max_batch_size=4,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        revision=revision,
    )


def _meta(**overrides):
    meta = {
        "model_type": "ChatterboxEmbedding",
        "model_repo": "resemble-ai/chatterbox",
        "revision": "main",
        "voice": "Abigail",
        "wav_hash": "h" * 64,
        "params": {"exaggeration": 0.5},
    }
    meta.update(overrides)
    return meta


def _key(**overrides):
    return disk_key(
        model_type=overrides.get("model_type", "ChatterboxEmbedding"),
        model_repo=overrides.get("model_repo", "resemble-ai/chatterbox"),
        revision=overrides.get("revision", "main"),
        voice=overrides.get("voice", "Abigail"),
        wav_hash=overrides.get("wav_hash", "h" * 64),
        params=overrides.get("params", {"exaggeration": 0.5}),
    )


def _conditionals_artifact():
    from chatterbox.tts import Conditionals, T3Cond

    return SpeakerArtifact(
        conditionals=Conditionals(
            T3Cond(
                speaker_emb=torch.tensor([[1.0, 2.0]]),
                emotion_adv=torch.tensor([[[0.5]]]),
            ),
            {"ge_bes": torch.tensor([0.25])},
        ),
        backend="chatterbox",
    )


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def test_cache_dir_default_is_stable_and_env_overridable(monkeypatch, tmp_path):
    monkeypatch.delenv("HSE_VOICE_PACK_CACHE_DIR", raising=False)
    first = default_voice_pack_cache_dir()
    second = default_voice_pack_cache_dir()
    assert first == second
    assert first.name == "voice-pack"

    monkeypatch.setenv("HSE_VOICE_PACK_CACHE_DIR", str(tmp_path / "custom"))
    assert default_voice_pack_cache_dir() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# Key / invalidation
# ---------------------------------------------------------------------------


def test_disk_key_changes_with_wav_hash_and_params():
    base = _key()
    assert _key() == base
    # Pack update ⇒ new wav content hash ⇒ a different key (old entry ignored).
    assert _key(wav_hash="b" * 64) != base
    # Params bake into the artifact (exaggeration / vad source) ⇒ key differs.
    assert _key(params={"exaggeration": 0.9}) != base
    assert _key(model_type="OpenVoiceV1ToneConverterEncoder") != base


# ---------------------------------------------------------------------------
# Disk store
# ---------------------------------------------------------------------------


def test_disk_store_round_trips_conditionals(tmp_path):
    disk = VoicePackDiskCache(tmp_path)
    key = _key()
    artifact = _conditionals_artifact()

    assert disk.write(key, _meta(), artifact) is True
    assert disk.has(key)

    loaded = disk.read(key)
    assert loaded is not None
    assert loaded.backend == "chatterbox"
    assert torch.equal(loaded.conditionals.t3.speaker_emb, artifact.conditionals.t3.speaker_emb)
    assert torch.equal(loaded.conditionals.gen["ge_bes"], artifact.conditionals.gen["ge_bes"])
    # Atomic write: no temp files left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_disk_store_round_trips_embedding(tmp_path):
    disk = VoicePackDiskCache(tmp_path)
    key = _key(model_type="OpenVoiceV1ToneConverterEncoder", params={"vad_source": "whisper"})
    artifact = SpeakerArtifact(embedding=torch.tensor([0.1, 0.2, 0.3]), backend="openvoice")

    assert disk.write(key, _meta(model_type="OpenVoiceV1ToneConverterEncoder"), artifact) is True
    loaded = disk.read(key)
    assert loaded is not None and loaded.backend == "openvoice"
    assert torch.equal(loaded.embedding, artifact.embedding)


def test_corrupt_and_unsupported_entries_are_ignored(tmp_path):
    disk = VoicePackDiskCache(tmp_path)
    key = _key()
    path = disk.path_for(key)

    path.write_bytes(b"this is not a torch checkpoint")
    assert disk.read(key) is None

    torch.save({"format": 999, "embedding": torch.zeros(1)}, path)
    assert disk.read(key) is None


def test_non_serializable_artifact_is_not_written(tmp_path):
    disk = VoicePackDiskCache(tmp_path)
    key = _key()
    # A test-double conditionals (no t3/gen) has nothing honest to persist.
    artifact = SpeakerArtifact(conditionals=object(), backend="chatterbox")
    assert disk.write(key, _meta(), artifact) is False
    assert not disk.has(key)


def test_prune_removes_entries_outside_the_keep_set(tmp_path):
    disk = VoicePackDiskCache(tmp_path)
    keep = _key(voice="Abigail")
    drop = _key(voice="Adrian")
    assert disk.write(keep, _meta(voice="Abigail"), _conditionals_artifact())
    assert disk.write(drop, _meta(voice="Adrian"), _conditionals_artifact())

    assert disk.prune({keep}) == 1
    assert disk.has(keep)
    assert not disk.has(drop)


# ---------------------------------------------------------------------------
# Warm-up planning
# ---------------------------------------------------------------------------


def test_build_warmup_targets_dedupes_shared_chatterbox_embed_executor():
    engine = _FakeEngine(
        [_config("chatterbox_embedding", "ChatterboxEmbedding")],
        {"chatterbox_embedding": _FakeExecutor("chatterbox_embedding|ChatterboxEmbedding|float32|stub/model@main|v1")},
        EmbeddingCache(),
    )
    targets = build_warmup_targets(engine)

    # One shared embedding executor × every pack voice, despite three TTS specs.
    assert len(targets) == len(standard_voice_names())
    assert {target.model_name for target in targets} == {"chatterbox_embedding"}
    assert {target.params["exaggeration"] for target in targets} == {0.5}
    assert all(target.cache_key and target.disk_key for target in targets)
    assert all(target.artifacts["input_audio"].content_hash for target in targets)
    # The request routes through the representative pipeline id, not a raw model.
    assert {target.request.requested_model for target in targets} == {"chatterbox"}
    assert all(target.request.input_audio is None for target in targets)


def test_build_warmup_targets_predicts_openvoice_and_harmonyspeech_params():
    engine = _FakeEngine(
        [
            _config("whisper", "FasterWhisper"),
            _config("ov1-encoder", "OpenVoiceV1ToneConverterEncoder"),
            _config("hs1-encoder", "HarmonySpeechEncoder"),
        ],
        {
            "ov1-encoder": _FakeExecutor("ov1|OpenVoiceV1ToneConverterEncoder|float32|stub/model@main|v1"),
            "hs1-encoder": _FakeExecutor("hs1|HarmonySpeechEncoder|float32|stub/model@main|v1"),
        },
        EmbeddingCache(),
    )
    by_model = {}
    for target in build_warmup_targets(engine):
        by_model.setdefault(target.model_name, target)

    assert by_model["ov1-encoder"].params == {"vad_source": "whisper"}
    assert by_model["ov1-encoder"].request.requested_model == "openvoice_v1"
    assert by_model["hs1-encoder"].params == {}
    assert by_model["hs1-encoder"].request.requested_model == "harmonyspeech"


def test_build_warmup_targets_predicts_silero_as_the_default_vad_source():
    """Regression (2026-11-10): the warm-up must key on the Silero default.

    Before the default flip the planner predicted ``"whisper"``; warm-up then
    warmed an artifact the default request would not use, and the whisper path
    dropped 26/28 pack voices in the OpenVoice encoder's text-length filter.
    """
    engine = _FakeEngine(
        [
            _config("whisper", "FasterWhisper"),
            _config("silero", "SileroVAD"),
            _config("ov1-encoder", "OpenVoiceV1ToneConverterEncoder"),
        ],
        {
            "ov1-encoder": _FakeExecutor("ov1|OpenVoiceV1ToneConverterEncoder|float32|stub/model@main|v1"),
        },
        EmbeddingCache(),
    )
    target = next(t for t in build_warmup_targets(engine) if t.model_name == "ov1-encoder")

    assert target.request.input_vad_mode is None
    assert target.params == {"vad_source": "silero"}


def test_build_warmup_targets_is_empty_without_cache_or_executors():
    engine = _FakeEngine([_config("chatterbox_embedding", "ChatterboxEmbedding")], {}, EmbeddingCache())
    assert build_warmup_targets(engine) == []
    engine.embedding_cache = None
    assert build_warmup_targets(engine) == []


# ---------------------------------------------------------------------------
# Hydration (disk -> 7-2 memory)
# ---------------------------------------------------------------------------


def test_hydrate_restores_disk_artifact_into_the_memory_cache(tmp_path):
    cache = EmbeddingCache()
    engine = _FakeEngine(
        [_config("chatterbox_embedding", "ChatterboxEmbedding")],
        {"chatterbox_embedding": _FakeExecutor("id")},
        cache,
    )
    targets = build_warmup_targets(engine)
    target = next(t for t in targets if t.voice == "Abigail")
    disk = VoicePackDiskCache(tmp_path)
    assert disk.write(target.disk_key, target.meta, _conditionals_artifact())

    assert hydrate_embedding_cache(engine, disk) == 1
    assert cache.contains(target.cache_key)
    # Hydration is idempotent: a second pass restores nothing.
    assert hydrate_embedding_cache(engine, disk) == 0
    # Presence checks must not look like cache misses in the 7-3 accounting.
    assert cache.stats()["misses"] == 0


def test_hydrate_ignores_a_missing_disk_entry(tmp_path):
    cache = EmbeddingCache()
    engine = _FakeEngine(
        [_config("chatterbox_embedding", "ChatterboxEmbedding")],
        {"chatterbox_embedding": _FakeExecutor("id")},
        cache,
    )
    assert hydrate_embedding_cache(engine, VoicePackDiskCache(tmp_path)) == 0
    assert cache.stats()["entries"] == 0


def test_pack_content_change_invalidates_the_disk_entry(tmp_path, monkeypatch):
    """A new reference-clip content hash is a different key: the old entry is ignored."""
    import numpy as np

    from harmonyspeech.common.audio_codec import encode_audio
    from harmonyspeech.modeling import voice_pack_cache

    cache = EmbeddingCache()
    engine = _FakeEngine(
        [_config("chatterbox_embedding", "ChatterboxEmbedding")],
        {"chatterbox_embedding": _FakeExecutor("id")},
        cache,
    )
    original = next(t for t in build_warmup_targets(engine) if t.voice == "Abigail")
    disk = VoicePackDiskCache(tmp_path)
    assert disk.write(original.disk_key, original.meta, _conditionals_artifact())

    # Simulate a pack update: the same voice now points at different samples.
    updated_wav = tmp_path / "Abigail.wav"
    updated_wav.write_bytes(encode_audio(np.full(16000, 0.05, dtype=np.float32), 16000, "wav"))
    real_path = voice_pack_cache.standard_voice_path
    monkeypatch.setattr(
        voice_pack_cache,
        "standard_voice_path",
        lambda name: updated_wav if name == "Abigail" else real_path(name),
    )

    updated = next(t for t in build_warmup_targets(engine) if t.voice == "Abigail")
    assert updated.wav_hash != original.wav_hash
    assert updated.disk_key != original.disk_key
    # The old entry is never read for the new content: hydration restores nothing.
    assert hydrate_embedding_cache(engine, disk) == 0
    assert not cache.contains(original.cache_key)
