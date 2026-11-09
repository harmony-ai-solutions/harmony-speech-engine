"""Persistent disk cache + warm-up for standard voice-pack embeddings (Phase 13-2).

Standard voice pack voices are **reference clips**, so a first request on each
voice-cloning family pays that family's embedding pass once. Phase 7-2 already
reuses that result in memory; this module makes it survive a restart:

* :class:`VoicePackDiskCache` stores one CPU ``SpeakerArtifact`` per
  ``(model_type, model repo, revision, voice, wav-content-hash, params)`` under a
  configurable root (default: the platform user-cache directory). Writes are
  atomic (temp file + ``os.replace``); an unreadable or corrupt entry is ignored
  and logged, never raised.
* :func:`build_warmup_targets` replays Phase 13-1's routing path for every
  (cloning-capable model × pack voice): it resolves the pipeline's embedding
  step to its configured executor and builds the same ``SpeechEmbeddingRequestInput``
  the serving layer's embed-once seam builds. **Warm what a first request would
  compute — nothing more** (Chatterbox conditionals via the embed step, OpenVoice
  speaker-encoder artifact, HarmonySpeech encoder output).
* :func:`hydrate_embedding_cache` restores the disk artifacts into the 7-2
  in-memory cache at model-init time, keyed with the *current* ``model_identity``
  (which carries the per-boot load version), so the first real request is a hit.
* :func:`warm_up_voice_packs` (async, on :class:`AsyncHarmonySpeech`) submits one
  embed request at a time through the normal engine scheduler, then persists what
  the handler cached. Because it goes through the scheduler it never races a
  real request on the model; an on-demand compute wins and warm-up skips a voice
  that is already in the 7-2 cache.

Serialization format
--------------------
The disk store is internal, so the honest minimal format is a ``torch.save`` of a
plain dict of tensors and primitives, loaded with ``weights_only=True`` (the same
safety rule as every other deserialization in the engine)::

    {
      "format": 1, "kind": "speaker_artifact",
      "model_type": ..., "model_repo": ..., "revision": ...,
      "voice": ..., "wav_hash": ..., "params": {...},
      "backend": "openvoice" | "harmonyspeech" | "chatterbox",
      "embedding": tensor | None,
      "conditionals": {"t3": {...}, "gen": {...}} | None,
    }

Chatterbox conditionals reuse ``common.serialization.conditionals_to_payload``
(the ``Conditionals.save()`` schema) and are rebuilt with the upstream
``chatterbox.tts.Conditionals``/``T3Cond`` classes; without that package the entry
is treated as unreadable and recomputed on demand. ``reference_audio`` is not
persisted — the pack WAV is already shipped, so it is never part of a cache value.

**Mutable-artifact precondition.** Persisting conditionals is safe only because
``task_handler.ar_runner.cast_conditionals`` clones the T3 half before the dtype
move and before applying ``exaggeration``; a cached object is never written to
per request. ``tests/unit/modeling/test_chatterbox_ar_runner.py`` pins that a
second request with a different exaggeration leaves the cached object
byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from harmonyspeech.common.artifacts import AudioArtifact, SpeakerArtifact
from harmonyspeech.common.audio_codec import decode_audio
from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput
from harmonyspeech.common.serialization import conditionals_to_payload
from harmonyspeech.modeling.voice_catalog import (
    spec_is_tts_cloning_capable,
    standard_voice_names,
    standard_voice_path,
)
from harmonyspeech.processing.pipeline_registry import PIPELINE_SPECS

__all__ = [
    "DEFAULT_EXAGGERATION",
    "WarmupTarget",
    "VoicePackDiskCache",
    "build_warmup_targets",
    "default_voice_pack_cache_dir",
    "disk_key",
    "hydrate_embedding_cache",
    "warm_up_voice_packs",
]

#: Bump when the on-disk payload shape changes; older files are ignored (and
#: recomputed) rather than misread.
_FORMAT_VERSION = 1

#: The Chatterbox embed step's default exaggeration (the handler's default when a
#: request carries no ``generation_options``). Warm-up computes exactly that.
DEFAULT_EXAGGERATION = 0.5

#: Model types whose embed-step artifact is Chatterbox conditionals.
_CHATTERBOX_EMBED_TYPES: frozenset[str] = frozenset(
    {
        "ChatterboxEmbedding",
        "ChatterboxTTS",
        "ChatterboxTurboTTS",
        "ChatterboxMultilingualTTS",
    }
)

#: Model types whose embed-step artifact is the OpenVoice speaker-encoder output.
_OPENVOICE_ENCODER_TYPES: frozenset[str] = frozenset(
    {"OpenVoiceV1ToneConverterEncoder", "OpenVoiceV2ToneConverterEncoder"}
)


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------


def default_voice_pack_cache_dir() -> Path:
    """Return the stable per-user cache directory for voice-pack artifacts.

    Zero new dependencies: on Windows ``%LOCALAPPDATA%\\harmony-speech\\voice-pack``;
    elsewhere ``$XDG_CACHE_HOME/harmony-speech/voice-pack`` (``~/.cache`` when
    unset). ``HSE_VOICE_PACK_CACHE_DIR`` overrides both. The directory is created
    lazily on the first write, so boot never touches the filesystem.
    """
    override = os.environ.get("HSE_VOICE_PACK_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "harmony-speech" / "voice-pack"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "harmony-speech" / "voice-pack"


def disk_key(
    *,
    model_type: str,
    model_repo: str,
    revision: str | None,
    voice: str,
    wav_hash: str,
    params: dict | None,
) -> str:
    """Stable content-addressed key for one persisted artifact.

    The wav-content hash is the invalidation story (pack update ⇒ new hash ⇒ the
    old entry is simply never read again). ``params`` is folded in because it
    changes the artifact (Chatterbox ``exaggeration`` bakes into the conditionals;
    OpenVoice ``vad_source`` selects the segment set).
    """
    digest = hashlib.sha256()
    for part in (
        str(model_type),
        str(model_repo),
        str(revision or "main"),
        str(voice),
        str(wav_hash),
        json.dumps(params or {}, sort_keys=True, separators=(",", ":")),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Artifact <-> payload
# ---------------------------------------------------------------------------


def _cpu_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    return value


def _cpuify(value):
    """Recursively detach every tensor onto the CPU (the disk store is device-free)."""
    if torch.is_tensor(value):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _cpuify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpuify(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpuify(item) for item in value)
    return value


def _conditionals_serializable(conditionals) -> bool:
    return conditionals is not None and hasattr(conditionals, "t3") and hasattr(conditionals, "gen")


def _artifact_to_payload(artifact: SpeakerArtifact, *, meta: dict) -> dict | None:
    """Build the ``weights_only=True`` payload, or ``None`` if nothing persists.

    An artifact with neither a tensor embedding nor a Chatterbox-style
    conditionals (e.g. a test double) has nothing worth storing; the caller
    treats ``None`` as "skip persistence" instead of writing an empty file.
    """
    embedding = getattr(artifact, "embedding", None)
    conditionals = getattr(artifact, "conditionals", None)
    has_embedding = torch.is_tensor(embedding)
    has_conditionals = _conditionals_serializable(conditionals)
    if not has_embedding and not has_conditionals:
        return None
    payload = {
        "format": _FORMAT_VERSION,
        "kind": "speaker_artifact",
        **meta,
        "backend": getattr(artifact, "backend", None),
        "embedding": _cpu_tensor(embedding) if has_embedding else None,
        "conditionals": _cpuify(conditionals_to_payload(conditionals)) if has_conditionals else None,
    }
    return payload


def _conditionals_from_payload(payload: dict):
    """Rebuild an upstream Chatterbox ``Conditionals`` from the payload.

    Imported lazily: the disk layer must stay importable without the Chatterbox
    dependency (CPU/OpenVoice deployments). A missing package makes the entry
    unreadable, which the store logs and ignores.
    """
    from chatterbox.tts import Conditionals, T3Cond

    return Conditionals(T3Cond(**payload["t3"]), dict(payload["gen"]))


def _payload_to_artifact(payload: dict) -> SpeakerArtifact | None:
    embedding = payload.get("embedding")
    conditionals_payload = payload.get("conditionals")
    conditionals = None
    if conditionals_payload is not None:
        conditionals = _conditionals_from_payload(conditionals_payload)
    if embedding is None and conditionals is None:
        return None
    return SpeakerArtifact(
        embedding=embedding,
        conditionals=conditionals,
        backend=payload.get("backend"),
    )


# ---------------------------------------------------------------------------
# Disk store
# ---------------------------------------------------------------------------


class VoicePackDiskCache:
    """One CPU artifact file per content key, written atomically."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()
        self._lock = threading.Lock()

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.pt"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def read(self, key: str) -> SpeakerArtifact | None:
        """Return the stored artifact, or ``None`` for missing/corrupt entries."""
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - corrupt entry must never fail boot
            logger.warning(f"Voice pack cache: ignoring unreadable entry {path.name}: {exc}")
            return None
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT_VERSION:
            logger.warning(
                f"Voice pack cache: ignoring entry {path.name} with unsupported format "
                f"{payload.get('format') if isinstance(payload, dict) else type(payload).__name__}"
            )
            return None
        try:
            return _payload_to_artifact(payload)
        except Exception as exc:  # noqa: BLE001 - schema drift is not a boot failure
            logger.warning(f"Voice pack cache: ignoring entry {path.name}: {exc}")
            return None

    def write(self, key: str, meta: dict, artifact: SpeakerArtifact) -> bool:
        """Persist ``artifact`` atomically; return False when not representable."""
        payload = _artifact_to_payload(artifact, meta=meta)
        if payload is None:
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path_for(key)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{key}.", suffix=".tmp", dir=str(self.root)
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    torch.save(payload, handle)
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
                raise
        except Exception as exc:  # noqa: BLE001 - the cache must never fail warm-up
            logger.warning(f"Voice pack cache: could not write entry {key[:12]}…: {exc}")
            return False
        return True

    def prune(self, keep_keys: set[str]) -> int:
        """Delete ``*.pt`` entries whose key is not in ``keep_keys``; return count.

        Never called on the boot path; exposed so an operator (or a future
        maintenance task) can reclaim entries invalidated by a pack or model
        revision change. Missing roots are a no-op.
        """
        if not self.root.is_dir():
            return 0
        removed = 0
        with self._lock:
            for path in self.root.glob("*.pt"):
                if path.stem in keep_keys:
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:  # pragma: no cover - best effort
                    logger.warning(f"Voice pack cache: could not remove {path.name}: {exc}")
        return removed


# ---------------------------------------------------------------------------
# Pack audio + per-family params
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _pack_audio(path_str: str) -> AudioArtifact:
    """Decode a pack WAV exactly like admission does (same content hash)."""
    waveform, sample_rate = decode_audio(Path(path_str).read_bytes())
    return AudioArtifact.from_numpy(waveform, sample_rate)


def _embed_params(model_type: str, vad_source: str | None) -> dict:
    """The 7-2 cache ``params`` a first request computes for this embed step."""
    if model_type in _CHATTERBOX_EMBED_TYPES:
        return {"exaggeration": DEFAULT_EXAGGERATION}
    if model_type in _OPENVOICE_ENCODER_TYPES:
        # Default VAD is Silero since the 2026-11-10 owner decision; an explicit
        # ``input_vad_mode: "whisper"`` request still keys on its actual source.
        return {"vad_source": vad_source or "silero"}
    return {}


def _predicted_vad_source(spec, by_type: dict[str, Any]) -> str | None:
    """The VAD backend a default (no ``input_vad_mode``) request would use.

    Mirrors ``Step.candidates_for``: the first ``requires`` entry backed by a
    configured model wins (Silero first since the 2026-11-10 owner decision,
    with Faster-Whisper as the fallback). Returns ``None`` when the pipeline has
    no resolvable VAD step (an incomplete OpenVoice toolchain, skipped by
    warm-up).
    """
    for step in getattr(spec, "steps", ()):
        if step.name != "vad":
            continue
        for model_type in step.requires:
            if model_type in by_type:
                return "whisper" if model_type == "FasterWhisper" else "silero"
        return None
    return None


def _embed_step(spec):
    for step in getattr(spec, "steps", ()):
        if any(artifact == "embedding_artifact" for artifact, _field in step.produces):
            return step
    return None


def _runner_of(executor):
    """Return the model runner behind an executor (real or test double)."""
    runner = getattr(executor, "runner", None)
    if runner is None:
        runner = getattr(getattr(executor, "driver_worker", None), "model_runner", None)
    return runner


# ---------------------------------------------------------------------------
# Warm-up planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupTarget:
    """One (embedding executor × pack voice) unit of work."""

    model_name: str
    pipeline_id: str
    voice: str
    model_type: str
    model_repo: str
    revision: str
    params: dict
    wav_hash: str
    disk_key: str
    model_identity: str | None
    cache_key: str | None
    request: SpeechEmbeddingRequestInput
    artifacts: dict = field(default_factory=dict)

    @property
    def meta(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_repo": self.model_repo,
            "revision": self.revision,
            "voice": self.voice,
            "wav_hash": self.wav_hash,
            "params": self.params,
        }

    @property
    def request_id(self) -> str:
        return f"voice-pack-warmup-{self.model_name}-{self.voice}"


def _resolve_embed_targets(engine) -> list[tuple[str, Any, Any, str | None]]:
    """Return ``(pipeline_id, embed_step, config, vad_source)`` per embed executor.

    Deduplicated by configured executor name: the three Chatterbox TTS pipelines
    share one ``ChatterboxEmbedding`` executor, so warm-up computes each pack
    voice once for it (13-1's cross-family cache hit). When no dedicated embed
    executor is configured, each TTS family resolves to its own executor and is
    warmed separately.
    """
    configs = list(getattr(engine, "model_configs", None) or ())
    by_type: dict[str, Any] = {}
    for config in configs:
        model_type = getattr(config, "model_type", None)
        if model_type and model_type not in by_type:
            by_type[model_type] = config

    resolved: dict[str, tuple[str, Any, Any, str | None]] = {}
    for spec in PIPELINE_SPECS.values():
        if not spec_is_tts_cloning_capable(spec):
            continue
        step = _embed_step(spec)
        if step is None:
            continue
        config = next((by_type[mt] for mt in step.requires if mt in by_type), None)
        if config is None:
            continue
        vad_source = _predicted_vad_source(spec, by_type)
        if config.model_type in _OPENVOICE_ENCODER_TYPES and vad_source is None:
            # OpenVoice needs its mandatory VAD step; an incomplete toolchain is
            # already reported at boot, so warm-up skips it honestly.
            continue
        if config.name in resolved:
            continue
        resolved[config.name] = (spec.id, step, config, vad_source)
    return list(resolved.values())


def build_warmup_targets(engine) -> list[WarmupTarget]:
    """Plan the warm-up for every (cloning-capable embed executor × pack voice).

    Safe on a shell engine (no executors/cache): returns ``[]``. The request is
    the *embed-step* request the serving layer builds (Phase 12-2 embed-once
    seam): ``requested_model`` is the representative pipeline id so the engine
    routes it to that pipeline's embedding step, and the pack WAV is carried as
    the ``input_audio`` artifact (exactly what admission produces).
    """
    cache = getattr(engine, "embedding_cache", None)
    executors = getattr(engine, "model_executors", None) or {}
    voices = standard_voice_names()
    if cache is None or not voices:
        return []

    targets: list[WarmupTarget] = []
    for pipeline_id, _step, config, vad_source in _resolve_embed_targets(engine):
        executor = executors.get(config.name)
        if executor is None:
            continue
        runner = _runner_of(executor)
        model_identity = getattr(runner, "model_identity", None)
        params = _embed_params(config.model_type, vad_source)
        revision = getattr(config, "revision", None) or "main"
        model_repo = str(getattr(config, "model", ""))
        for voice in voices:
            path = standard_voice_path(voice)
            if path is None:
                continue
            audio = _pack_audio(str(path))
            wav_hash = audio.content_hash
            key = disk_key(
                model_type=config.model_type,
                model_repo=model_repo,
                revision=revision,
                voice=voice,
                wav_hash=wav_hash,
                params=params,
            )
            cache_key = (
                cache.make_key(
                    model_identity=model_identity,
                    audio_hash=wav_hash,
                    params=params,
                )
                if model_identity
                else None
            )
            request = SpeechEmbeddingRequestInput(
                request_id=f"voice-pack-warmup-{config.name}-{voice}",
                requested_model=pipeline_id,
                model=pipeline_id,
                input_audio=None,
                input_vad_mode=None,
                input_vad_data=None,
                generation_options=None,
            )
            targets.append(
                WarmupTarget(
                    model_name=config.name,
                    pipeline_id=pipeline_id,
                    voice=voice,
                    model_type=config.model_type,
                    model_repo=model_repo,
                    revision=revision,
                    params=params,
                    wav_hash=wav_hash,
                    disk_key=key,
                    model_identity=model_identity,
                    cache_key=cache_key,
                    request=request,
                    artifacts={"input_audio": audio},
                )
            )
    return targets


# ---------------------------------------------------------------------------
# Hydration (disk -> 7-2 memory)
# ---------------------------------------------------------------------------


def hydrate_embedding_cache(engine, disk_cache: VoicePackDiskCache | None = None) -> int:
    """Restore persisted artifacts into the in-memory cache; return the count.

    Cheap I/O only (no model forwards). Runs at model-init time so the first real
    request after a restart is a cache hit. A voice already present in memory
    (e.g. hydration ran twice) is left alone.
    """
    cache = getattr(engine, "embedding_cache", None)
    disk = disk_cache if disk_cache is not None else getattr(engine, "voice_pack_disk_cache", None)
    if cache is None or disk is None or not disk.root.is_dir():
        # No cache directory yet (first boot): skip the 28 pack decodes entirely.
        return 0
    restored = 0
    for target in build_warmup_targets(engine):
        if target.cache_key is None or cache.contains(target.cache_key):
            continue
        artifact = disk.read(target.disk_key)
        if artifact is None:
            continue
        try:
            cache.put(target.cache_key, artifact, model_identity=target.model_identity)
            restored += 1
        except Exception as exc:  # noqa: BLE001 - hydration is an optimization
            logger.warning(
                f"Voice pack cache: could not hydrate voice {target.voice!r} on "
                f"{target.model_name}: {exc}"
            )
    if restored:
        logger.info(f"Voice pack cache: hydrated {restored} embedding(s) from {disk.root}")
    return restored


# ---------------------------------------------------------------------------
# Warm-up execution (async, through the engine scheduler)
# ---------------------------------------------------------------------------


async def warm_up_voice_packs(engine) -> dict:
    """Compute (or reuse) every pack voice's artifact and persist it to disk.

    ``engine`` is the async wrapper (``AsyncHarmonySpeech``) because the requests
    must go through its scheduler, never straight into a model: that is what keeps
    warm-up serialized with real traffic (one model at a time) and lets the 7-2
    cache deduplicate. Per-voice failures are logged and skipped; the on-demand
    path remains the fallback.
    """
    inner = getattr(engine, "engine", engine)
    disk = getattr(inner, "voice_pack_disk_cache", None)
    cache = getattr(inner, "embedding_cache", None)
    stats = {
        "enabled": bool(getattr(inner, "voice_pack_warmup", False)),
        "targets": 0,
        "computed": 0,
        "cache_hits": 0,
        "persisted": 0,
        "failed": 0,
        "skipped": 0,
    }
    if disk is None or cache is None:
        stats["skipped"] = 1
        return stats

    targets = build_warmup_targets(inner)
    stats["targets"] = len(targets)
    for target in targets:
        try:
            if target.cache_key is not None and cache.contains(target.cache_key):
                # An on-demand request (or a previous warm-up) already computed
                # this voice; warm-up only makes sure the disk copy exists.
                stats["cache_hits"] += 1
                if not disk.has(target.disk_key):
                    artifact = cache.get(target.cache_key)
                    if artifact is not None and disk.write(target.disk_key, target.meta, artifact):
                        stats["persisted"] += 1
                continue

            stream = await engine.add_request(
                target.request_id, target.request, artifacts=dict(target.artifacts)
            )
            final = None
            async for output in stream:
                final = output
            if final is None or getattr(final, "finish_reason", None) == "error":
                stats["failed"] += 1
                logger.warning(
                    f"Voice pack warm-up: voice {target.voice!r} on {target.model_name} "
                    f"failed: {getattr(final, 'error', None) or 'no output'}"
                )
                continue
            artifact = getattr(final, "embedding_artifact", None)
            if artifact is None:
                stats["failed"] += 1
                logger.warning(
                    f"Voice pack warm-up: voice {target.voice!r} on {target.model_name} "
                    "produced no embedding"
                )
                continue
            stats["computed"] += 1
            if disk.write(target.disk_key, target.meta, artifact):
                stats["persisted"] += 1
        except Exception as exc:  # noqa: BLE001 - one voice must never fail warm-up
            stats["failed"] += 1
            logger.warning(
                f"Voice pack warm-up: voice {target.voice!r} on {target.model_name} "
                f"failed: {exc}"
            )
    logger.info(
        "Voice pack warm-up: "
        f"{stats['computed']} computed, {stats['cache_hits']} cache hits, "
        f"{stats['persisted']} persisted, {stats['failed']} failed "
        f"({stats['targets']} targets)"
    )
    inner.voice_pack_warmup_stats = dict(stats)
    return stats
