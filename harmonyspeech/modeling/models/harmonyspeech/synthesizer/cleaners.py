"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You"ll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from typing import Any, Dict

from unidecode import unidecode

from harmonyspeech.modeling.models.harmonyspeech.synthesizer.numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
  ("mrs", "misess"),
  ("mr", "mister"),
  ("dr", "doctor"),
  ("st", "saint"),
  ("co", "company"),
  ("jr", "junior"),
  ("maj", "major"),
  ("gen", "general"),
  ("drs", "doctors"),
  ("rev", "reverend"),
  ("lt", "lieutenant"),
  ("hon", "honorable"),
  ("sgt", "sergeant"),
  ("capt", "captain"),
  ("esq", "esquire"),
  ("ltd", "limited"),
  ("col", "colonel"),
  ("ft", "fort"),
  ("mk", "mark"),
  ("jan", "january"),
  ("feb", "february"),
  ("mar", "march"),
  ("apr", "april"),
  ("aug", "august"),
  ("sept", "september"),
  ("oct", "october"),
  ("nov", "november"),
  ("dec", "december")
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  """lowercase input tokens."""
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, " ", text)

def no_cleaners(text):
  return text

def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  """Basic pipeline that lowercases and collapses whitespace without transliteration."""
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  """Pipeline for non-English text that transliterates to ASCII."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  """Pipeline for English text, including number and abbreviation expansion."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

# def to_phonemes(text: str, lang: str) -> str:
#   phonemes = phonemize(text,
#                        language=lang,
#                        backend='espeak',
#                        strip=True,
#                        preserve_punctuation=True,
#                        with_stress=False,
#                        njobs=1,
#                        punctuation_marks=';:,.!?¡¿—…"«»“”()',
#                        language_switch='remove-flags')
#   phonemes = ''.join([p for p in phonemes if p in phonemes_set])
#   return phonemes

class Cleaner:

  def __init__(self,
               cleaner_name: str,
               use_phonemes: bool,
               lang: str) -> None:
    if cleaner_name == 'english_cleaners':
      self.clean_func = english_cleaners
    elif cleaner_name == 'no_cleaners':
      self.clean_func = no_cleaners
    else:
      raise ValueError(f'Cleaner not supported: {cleaner_name}! '
                       f'Currently supported: [\'english_cleaners\', \'no_cleaners\']')
    self.use_phonemes = use_phonemes
    self.lang = lang

  def __call__(self, text: str) -> str:
    text = self.clean_func(text)
    # if self.use_phonemes:
    #   text = to_phonemes(text, self.lang)
    text = collapse_whitespace(text)
    text = text.strip()
    return text

  @classmethod
  def from_config(cls, config: Dict[str, Any]) -> 'Cleaner':
    return Cleaner(
      cleaner_name=config['preprocessing']['cleaner_name'],
      use_phonemes=config['preprocessing']['use_phonemes'],
      lang=config['preprocessing']['language']
    )