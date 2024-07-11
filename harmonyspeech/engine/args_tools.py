import argparse
import dataclasses
from dataclasses import dataclass

from harmonyspeech.common.config import EngineConfig


@dataclass
class EngineArgs:
    """Arguments For HarmonySpeechEngine"""

    disable_log_stats: bool = False

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for the Aphrodite engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="disable logging statistics",
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
        return base_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous Aphrodite engine."""

    disable_log_requests: bool = False
    max_log_len: int = 0

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--disable-log-requests",
            action="store_true",
            help="disable logging requests",
        )
        parser.add_argument(
            "--max-log-len",
            type=int,
            default=0,
            help="max number of prompt characters or prompt "
            "ID numbers being printed in log. "
            "Default: unlimited.",
        )
        return parser
