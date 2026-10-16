"""Print a greeting.

Example command showing the subcommand interface.
"""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", default="world", help="who to greet")
    parser.add_argument("-n", "--times", type=int, default=1, help="repeat count")


def run(args: argparse.Namespace) -> int:
    for _ in range(args.times):
        print(f"Hello, {args.name}!")
    return 0
