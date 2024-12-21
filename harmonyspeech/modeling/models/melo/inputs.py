import re

from harmonyspeech.modeling.models.melo.utils import get_text_for_tts_infer
from harmonyspeech.modeling.models.melo.split_utils import split_sentence

_supported_languages = ['EN', 'ZH', 'ZH_MIX_EN', 'FR', 'ES', 'SP', 'JP', 'KR']


def split_sentences_into_pieces(text, language_str):
    texts = split_sentence(text, language_str=language_str)
    return texts


def normalize_text_inputs(text, hf_config, language='EN'):
    global _supported_languages
    assert language in _supported_languages, f"language {language} is not supported"

    texts = split_sentences_into_pieces(text, language)
    inputs_list = []

    # Symbol-to-ID mapping differs between weights of different languages
    symbol_to_id = {s: i for i, s in enumerate(hf_config.symbols)}

    for t in texts:
        if language in ['EN', 'ZH_MIX_EN']:
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)

        # Prepare Inputs using bert
        # FIXME: This is a bunch of super complicated, imperformant and underoptimized code from MeloTTS repo
        # FIXME: Seems like Bert is being allocated for each normalized part of the text separately
        # FIXME: Honestly, everything that happens here needs to be completely rewritten when there is time
        input_params = get_text_for_tts_infer(t, language, hf_config, 'cpu', symbol_to_id)
        inputs_list.append(input_params)

    return inputs_list
