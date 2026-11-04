"""Safe (``weights_only=True``) serialization of client-supplied tensors/conditionals.

Phase 0-1 moved every client-controlled ``torch.load`` here so a crafted
``__reduce__`` gadget fails closed instead of executing code. Phase 2 keeps that
guarantee but moves the helpers out of the model runner: the request-input
getters (``harmonyspeech.common.inputs``) decode embedding/conditionals payloads
before any executor sees them, and the HTTP boundary
(``harmonyspeech.endpoints.openai.wire``) re-serializes them for external
clients.

This module deliberately contains **no wire encoding**: it operates on raw bytes so the
"serialization only at the boundary" rule stays greppable.
"""

from __future__ import annotations

import io
from typing import Any

import torch

__all__ = [
    "EMBEDDING_PAYLOAD_HELP",
    "PICKLE_PRIMITIVES",
    "conditionals_to_payload",
    "load_conditionals_payload",
    "load_embedding_tensor",
]

# Python primitives that are safe to carry inside a ``weights_only=True``
# torch.save payload (no arbitrary classes, so no __reduce__ gadgets).
PICKLE_PRIMITIVES = (str, int, float, bool, bytes, type(None))

EMBEDDING_PAYLOAD_HELP = (
    "only torch-saved tensors (weights_only=True compatible) are supported; "
    "supply the payload produced by the encoder / embedding step"
)


def load_embedding_tensor(payload: Any, field_name: str = "embedding") -> torch.Tensor:
    """Load a client-supplied embedding blob without executing pickle code.

    ``weights_only=True`` restricts the unpickler to tensors and primitives, so
    a crafted ``__reduce__`` gadget fails to load instead of running inside the
    server process.

    Raises:
        ValueError: if the payload is not a single torch tensor produced by
            ``torch.save``.
    """
    try:
        loaded = torch.load(payload, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"Unsupported {field_name} payload: {EMBEDDING_PAYLOAD_HELP}. Details: {e}") from e
    if not torch.is_tensor(loaded):
        raise ValueError(
            f"Unsupported {field_name} payload: expected a single tensor, got {type(loaded).__name__}. "
            f"{EMBEDDING_PAYLOAD_HELP}."
        )
    return loaded


def load_conditionals_payload(raw: bytes | io.BytesIO, field_name: str = "conditionals"):
    """Deserialize a Chatterbox conditionals blob without executing pickle code.

    Accepts the raw bytes stored in a request field (or a file-like object) and
    returns the upstream ``chatterbox.tts.Conditionals`` object.
    """
    from chatterbox.tts import Conditionals

    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = io.BytesIO(bytes(raw))
    try:
        return Conditionals.load(raw, map_location="cpu")
    except Exception as e:
        raise ValueError(
            f"Unsupported {field_name} payload: expected a Chatterbox conditionals blob "
            f"(a plain dict of tensors, loadable with weights_only=True). Details: {e}"
        ) from e


def conditionals_to_payload(conditionals) -> dict:
    """Convert a Chatterbox ``Conditionals`` object into a pickle-safe dict.

    Schema (mirrors ``chatterbox.tts.Conditionals.save()``)::

        {"t3":  {<T3Cond field>: tensor | primitive},   # speaker_emb, clap_emb, ...
         "gen": {<S3Gen ref field>: tensor | primitive}}  # prompt_token, prompt_feat, ...

    Only tensors and Python primitives are kept, so the payload round-trips
    through ``torch.save`` / ``Conditionals.load`` with ``weights_only=True``.
    """
    t3_state = {
        key: value
        for key, value in vars(conditionals.t3).items()
        if torch.is_tensor(value) or isinstance(value, PICKLE_PRIMITIVES)
    }
    gen_state = {
        key: value
        for key, value in dict(conditionals.gen).items()
        if torch.is_tensor(value) or isinstance(value, PICKLE_PRIMITIVES)
    }
    return {"t3": t3_state, "gen": gen_state}
