import argparse

from harmonyspeech.endpoints.openai.api_server import run_server
from harmonyspeech.endpoints.openai.args import make_arg_parser


def main():
    parser = argparse.ArgumentParser(description="Harmony Speech Engine CLI")
    subparsers = parser.add_subparsers()

    serve_parser = subparsers.add_parser(
        "run",
        help="Start the Harmony Speech Engine OpenAI Compatible API server",
        usage="harmonyspeech run [options]")
    make_arg_parser(serve_parser)
    serve_parser.set_defaults(func=run_server)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
