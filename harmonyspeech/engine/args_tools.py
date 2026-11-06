import argparse
import dataclasses
import os
from dataclasses import dataclass

from harmonyspeech.common.config import EngineConfig


def _strict_pipelines_default() -> bool:
    """``--strict-pipelines`` default from ``HSE_STRICT_PIPELINES``."""
    return os.environ.get("HSE_STRICT_PIPELINES", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EngineArgs:
    """Arguments For HarmonySpeechEngine"""

    disable_log_stats: bool = False
    # Overrides EngineConfig.request_timeout_s when set.
    request_timeout_s: float | None = None
    # Turn incomplete-pipeline startup warnings into a boot error.
    strict_pipelines: bool = False
    # Embedding-cache overrides (Phase 7-1); None keeps the config-file value.
    embedding_cache_ttl_s: float | None = None
    embedding_cache_max_entries: int | None = None
    no_embedding_cache: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for the Aphrodite engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument("--disable-log-stats", action="store_true", help="disable logging statistics")
        parser.add_argument(
            "--request-timeout-s",
            type=float,
            default=None,
            help="Per-request deadline in seconds, overriding the config file. 0 disables the deadline.",
        )
        parser.add_argument(
            "--strict-pipelines",
            action="store_true",
            default=_strict_pipelines_default(),
            help=(
                "Fail startup when a declared model pipeline is incomplete. Without this flag "
                "incomplete pipelines are logged as warnings and hidden from /v1/models. "
                "Can also be enabled with HSE_STRICT_PIPELINES=1."
            ),
        )
        parser.add_argument(
            "--embedding-cache-ttl-s",
            type=float,
            default=None,
            help=(
                "Cross-request embedding cache TTL in seconds, overriding the config file. "
                "0 disables the cache entirely."
            ),
        )
        parser.add_argument(
            "--embedding-cache-max-entries",
            type=int,
            default=None,
            help="Cross-request embedding cache FIFO cap, overriding the config file.",
        )
        parser.add_argument(
            "--no-embedding-cache",
            action="store_true",
            help="Disable the cross-request embedding cache entirely.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_config(self, base_config: EngineConfig) -> EngineConfig:
        # Use this method to inject cli args to override the base engine config if needed
        overrides = {}
        if self.request_timeout_s is not None:
            overrides["request_timeout_s"] = self.request_timeout_s
        if self.strict_pipelines:
            overrides["strict_pipelines"] = True
        if self.embedding_cache_ttl_s is not None:
            overrides["embedding_cache_ttl_s"] = self.embedding_cache_ttl_s
        if self.embedding_cache_max_entries is not None:
            overrides["embedding_cache_max_entries"] = self.embedding_cache_max_entries
        if self.no_embedding_cache:
            overrides["embedding_cache_enabled"] = False
        if overrides:
            return dataclasses.replace(base_config, **overrides)
        return base_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous Aphrodite engine."""

    disable_log_requests: bool = False
    max_log_len: int = 0

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument("--disable-log-requests", action="store_true", help="disable logging requests")
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=0,
            help="max number of prompt characters or prompt ID numbers being printed in log. Default: unlimited.",
        )
        return parser
