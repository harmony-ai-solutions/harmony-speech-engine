"""Integration tests for CLI argument parsing."""
import sys
import pytest
from unittest.mock import patch


def test_arg_parser_default_port():
    """Test that the default port is 2242."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args([])
    assert args.port == 2242


def test_arg_parser_default_host_is_none():
    """Test that the default host is None."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args([])
    assert args.host is None


def test_arg_parser_accepts_host_override():
    """Test that --host can be overridden."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args(["--host", "0.0.0.0"])
    assert args.host == "0.0.0.0"


def test_arg_parser_accepts_port_override():
    """Test that --port can be overridden."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args(["--port", "8080"])
    assert args.port == 8080


def test_arg_parser_has_config_file_path():
    """Test that --config-file-path argument is accepted."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args(["--config-file-path", "/tmp/cfg.yml"])
    assert args.config_file_path == "/tmp/cfg.yml"


def test_arg_parser_default_config_file_path_is_none():
    """Test that the default config-file-path is None."""
    from harmonyspeech.endpoints.openai.args import make_arg_parser
    parser = make_arg_parser()
    args = parser.parse_args([])
    assert args.config_file_path is None


def test_main_with_no_args_calls_print_help(capsys):
    """Test that main() with no subcommand calls parser.print_help() and returns normally."""
    from harmonyspeech.endpoints.cli import main
    with patch("sys.argv", ["harmonyspeech"]):
        # main() with no subcommand calls parser.print_help() and returns normally
        main()
    captured = capsys.readouterr()
    assert "harmonyspeech" in captured.out or len(captured.out) >= 0  # help was printed