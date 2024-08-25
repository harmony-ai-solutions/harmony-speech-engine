import argparse
import json

from harmonyspeech.engine.args_tools import AsyncEngineArgs


def make_arg_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Harmony Speech Engine OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=2242, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        "--api.md-keys",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the "
        "header.")
    parser.add_argument(
        "--admin-key",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the "
        "header for admin operations.")
    parser.add_argument("--config-file-path",
                        type=str,
                        default=None,
                        help="Path to the config.yml file used to configure the engine and models")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, Aphrodite will add it to the server using "
        "@app.middleware('http'). "
        "If a class is provided, Aphrodite will add it to the server using "
        "app.add_middleware(). ")

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser