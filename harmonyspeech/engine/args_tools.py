import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class EngineArgs:
    """Arguments For HarmonySpeechEngine"""

    models: List[str]


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous Aphrodite engine."""

    disable_log_requests: bool = False
    max_log_len: int = 0

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
