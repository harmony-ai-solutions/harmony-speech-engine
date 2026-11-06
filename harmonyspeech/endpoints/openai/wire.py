"""The single serialization seam between the HTTP wire and the artifact bus.

Phase 2's rule is "serialization happens only in the serving layer". These
helpers make it code:

* :func:`from_wire_audio` decodes inbound audio fields (base64 on the wire) into
  typed :class:`~harmonyspeech.common.artifacts.AudioArtifact` objects **once**,
  at admission. The legacy base64 fields are left untouched so observability and
  the JSON contract do not change.
* :func:`to_wire_audio` encodes a result's ``audio_artifact`` (or falls back to
  the legacy base64 WAV string) honoring ``output_options``.
* :func:`to_wire_embedding` serializes ``embedding_artifact`` for external embed
  clients in exactly the pre-Phase-2 format (raw float32 for HarmonySpeech,
  ``torch.save`` tensor for OpenVoice, 0-1-safe conditionals payload for
  Chatterbox).

Engine core (``harmonyspeech_engine``, runners, scheduler) therefore contains no
wire encoding at all — enforced by
``tests/unit/wire_contract/test_serialization_boundary.py``.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.inputs import AUDIO_FIELDS, prefill_audio_artifacts
from harmonyspeech.common.serialization import conditionals_to_payload
from harmonyspeech.endpoints.openai.output_encoding import EncodedAudio, apply_output_options, encode_artifact

__all__ = ["from_wire_audio", "to_wire_audio", "to_wire_embedding"]


def _store_of(artifacts) -> dict:
    """Return the artifact dict to fill (dict, object with ``.artifacts``, or None)."""
    if artifacts is None:
        return {}
    if isinstance(artifacts, dict):
        return artifacts
    store = getattr(artifacts, "artifacts", None)
    if isinstance(store, dict):
        return store
    raise TypeError("from_wire_audio expected a dict or an object with an '.artifacts' dict")


def from_wire_audio(request_input, artifacts=None, fields: list[str] | None = None) -> dict:
    """Decode inbound ``fields`` once and cache ready-to-use artifacts.

    :param request_input: a ``RequestInput`` (or anything exposing its fields).
    :param artifacts: destination store — a dict, an ``EngineRequest``, or None.
    :param fields: restrict decoding to these field names (default: every known
        audio/VAD/mel field present on the request).
    """
    store = _store_of(artifacts)
    prefill_audio_artifacts(request_input, store)
    # Phase 7-2: warm each artifact's content hash at admission so every
    # downstream embedding-cache key computation is a dict lookup, never a
    # re-hash of the same waveform.
    for artifact in store.values():
        if isinstance(artifact, AudioArtifact):
            _ = artifact.content_hash
    if fields is not None:
        allowed = set(fields)
        for key in list(store):
            if key not in allowed and key in AUDIO_FIELDS:
                # Keep non-audio artifacts (VAD/mel) unless explicitly excluded.
                store.pop(key, None)
    return store


def to_wire_audio(result_output, output_options=None) -> EncodedAudio:
    """Encode a result for the wire, preferring its typed audio artifact."""
    artifact = getattr(result_output, "audio_artifact", None)
    waveform = getattr(artifact, "waveform", None)
    if artifact is not None and waveform is not None and waveform.numel() > 0:
        return encode_artifact(artifact, output_options)
    return apply_output_options(getattr(result_output, "output", "") or "", output_options)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def to_wire_embedding(result_output) -> str:
    """Serialize ``embedding_artifact`` exactly like the pre-Phase-2 executor.

    Falls back to the legacy ``output`` field when the producer did not attach an
    artifact (fake engines in tests, forward-compat consumers).
    """
    artifact = getattr(result_output, "embedding_artifact", None)
    legacy = getattr(result_output, "output", "") or ""
    if artifact is None:
        return legacy

    conditionals = getattr(artifact, "conditionals", None)
    if conditionals is not None:
        buffer = io.BytesIO()
        torch.save(conditionals_to_payload(conditionals), buffer)
        return _b64(buffer.getvalue())

    embedding = getattr(artifact, "embedding", None)
    if embedding is None:
        return legacy

    tensor = embedding.detach().cpu().float()
    if getattr(artifact, "backend", None) == "harmonyspeech":
        # Legacy HarmonySpeech encoder payload: raw float32 samples, no container.
        return _b64(np.ascontiguousarray(tensor.numpy(), dtype=np.float32).tobytes())

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return _b64(buffer.getvalue())
