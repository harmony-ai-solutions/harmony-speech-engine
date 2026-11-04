import numpy as np

from harmonyspeech.modeling.models.harmonyspeech.synthesizer.text import text_to_sequence

__all__ = ["prepare_synthesis_inputs"]


def prepare_synthesis_inputs(text: str, embedding):
    """Tokenize ``text`` and shape the speaker embedding for the synthesizer.

    ``embedding`` is the decoded embedding array (``[dim]``) taken from a
    :class:`~harmonyspeech.common.artifacts.SpeakerArtifact`; Phase 2 removed the
    wire decoding from this layer so serialization only happens at the HTTP
    boundary.
    """
    input_text = np.array([text_to_sequence(text.strip(), ["english_cleaners"])])
    if embedding is None:
        return input_text, None
    input_embedding = np.asarray(embedding, dtype=np.float32)
    if input_embedding.ndim == 1:
        input_embedding = input_embedding[None, :]
    return input_text, input_embedding
