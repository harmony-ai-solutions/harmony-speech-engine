"""
Internal logging utility.
"""

import logging
import os
import sys

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

RICH_CONSOLE = Console()
LOG_LEVEL = os.getenv("HARMONYSPEECH_LOG_LEVEL", "INFO").upper()


def _log_console_width() -> int | None:
    """Width for the engine-log console (Phase 11-0).

    An interactive terminal keeps rich's automatic (terminal) width. A
    redirected stream — the benchmark harness's ``--engine-log`` — has no
    visible width, and rich's 80-column fallback wrapped long DEBUG lines
    mid-sentence, so a line-oriented parser saw e.g. ``padded\\nsub-batch``.
    For a redirected stream use ``HARMONYSPEECH_LOG_WIDTH`` (or ``COLUMNS``)
    when set, else a width no engine line reaches: that disables wrapping
    without cropping, which rich's ``soft_wrap`` would do.
    """
    try:
        if sys.stdout.isatty():
            return None
    except Exception:  # pragma: no cover - exotic stdout replacements
        pass
    for name in ("HARMONYSPEECH_LOG_WIDTH", "COLUMNS"):
        raw = os.getenv(name)
        if not raw:
            continue
        try:
            width = int(raw)
        except (TypeError, ValueError):
            continue
        if width > 0:
            return width
    return 4096


#: Dedicated sink console: log lines never wrap when the stream is redirected,
#: while the progress bars keep ``RICH_CONSOLE``'s normal terminal sizing.
LOG_CONSOLE = Console(width=_log_console_width())


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default
    return wrapped


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
    )


def _log_formatter(record: dict):
    """Log message formatter."""

    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }
    level = record.get("level")
    level_color = color_map.get(level.name, "cyan")
    colored_level = f"[{level_color}]{level.name}[/{level_color}]:"

    separator = " " * (9 - len(level.name))

    message = unwrap(record.get("message"), "")

    # Replace once loguru allows for turning off str.format
    message = message.replace("{", "{{").replace("}", "}}").replace("<", "\\<")

    # Escape markup tags from Rich
    message = escape(message)
    lines = message.splitlines()

    fmt = ""
    if len(lines) > 1:
        fmt = "\n".join([f"{colored_level}{separator}{line}" for line in lines])
    else:
        fmt = f"{colored_level}{separator}{message}"

    return fmt


_logged_messages = set()


def log_once(level, message, *args, **kwargs):
    if message not in _logged_messages:
        _logged_messages.add(message)
        logger.log(level, message, *args, **kwargs)


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(record.levelname, self.format(record).rstrip())


# Uvicorn config for logging. Passed into run when creating all loggers in
# server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class": f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}"  # noqa
        }
    },
    "root": {"handlers": ["uvicorn"], "propagate": False, "level": LOG_LEVEL},
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(LOG_CONSOLE.print, level=LOG_LEVEL, format=_log_formatter, colorize=True)

    # Silence noisy HTTP request logs from httpx/httpcore (used by
    # huggingface_hub).  These emit an INFO line for every HEAD/GET request,
    # which floods the console during model loading and inference.
    # for noisy_logger in ("httpx", "httpcore"):
    #     logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.log_once = log_once


setup_logger()
