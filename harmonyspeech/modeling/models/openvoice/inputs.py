import re
import torch

from harmonyspeech.modeling.models.openvoice import commons
from harmonyspeech.modeling.models.openvoice.text import text_to_sequence
from harmonyspeech.modeling.models.openvoice.utils import split_sentence


_supported_languages = ['EN', 'ZH']


def split_sentences_into_pieces(text, language_str):
    texts = split_sentence(text, language_str=language_str)
    return texts


def get_text(text, hf_config, is_symbol):
    text_norm = text_to_sequence(text, hf_config.symbols, [] if is_symbol else hf_config.data.text_cleaners)
    if hf_config.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def normalize_text_inputs(text, hf_config, language='EN'):
    global _supported_languages
    assert language in _supported_languages, f"language {language} is not supported"

    texts = split_sentences_into_pieces(text, language)
    inputs_list = []

    for t in texts:
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = f'[{language}]{t}[{language}]'
        text_normalized = get_text(t, hf_config, False)
        inputs_list.append(text_normalized)

    return inputs_list
