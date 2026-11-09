"""Standard voice pack wire helpers (Phase 13-1, serving boundary).

The pack catalog lives in ``harmonyspeech.modeling.voice_catalog`` and exposes
filesystem paths; turning a pack WAV into the base64 ``input_audio`` /
``target_audio`` wire payload is a boundary concern, so it stays here (the
engine-core serialization-boundary test forbids wire encoding elsewhere).
"""

from __future__ import annotations

import base64
from functools import cache

from harmonyspeech.modeling.voice_catalog import standard_voice_path

__all__ = ["standard_voice_wire_payload"]


@cache
def standard_voice_wire_payload(name: str) -> str | None:
    """Return the pack WAV as the base64 string the wire audio fields use.

    Cached: the pack is immutable package data, and a pack voice may be
    requested thousands of times. The decoded artifact's content hash (which
    keys the 7-2 embedding cache) is derived from the WAV bytes, so the cache
    story is identical to a client-supplied clip.
    """
    path = standard_voice_path(name)
    if path is None:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")
