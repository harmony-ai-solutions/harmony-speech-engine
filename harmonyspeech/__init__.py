import warnings

# resemble-perth 1.0.1 (the audio watermark library, also imported directly by
# upstream chatterbox-tts) loads its model via pkg_resources — a deprecated
# API whose import-time UserWarning fires on every fresh process (server,
# benchmark scripts, tests) no matter which module pulls perth in first. No
# fixed release exists (2026-11-15); silence exactly this message, globally,
# before any engine import can transitively reach perth. Remove when a
# pkg_resources-free resemble-perth ships.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

from harmonyspeech.engine.harmonyspeech_engine import HarmonySpeechEngine  # noqa: E402 - after the filter

from .version import __commit__, __short_commit__, __version__  # noqa: E402

__all__ = ["__commit__", "__short_commit__", "__version__", "HarmonySpeechEngine"]
