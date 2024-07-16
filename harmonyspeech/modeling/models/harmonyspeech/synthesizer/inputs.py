import base64

import numpy as np

from harmonyspeech.modeling.models.harmonyspeech.synthesizer.text import text_to_sequence


def prepare_synthesis_inputs(text: str, embedding: str):
    # Decode Embedding
    input_embedding = base64.b64decode(embedding.encode('utf-8'))
    input_embedding = np.frombuffer(input_embedding, dtype=np.float32)
    # Convert into NP Arrays
    input_text = np.array([text_to_sequence(text.strip(), ["english_cleaners"])])
    input_embedding = np.array([input_embedding])
    return input_text, input_embedding

