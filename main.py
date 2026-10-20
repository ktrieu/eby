"""eby CLI entry point: dispatches to the subcommands listed in COMMANDS.

Run with `uv run main.py <command> [args...]`.

A command module must define:

    def add_arguments(parser: argparse.ArgumentParser) -> None: ...
    def run(args: argparse.Namespace) -> int | None: ...

The module docstring's first line becomes the command's help text.
"""

import argparse
import sys

from commands import gen_dataset, process_edata, weight_ridings

COMMANDS = {
    "gen_dataset": gen_dataset,
    "process_edata": process_edata,
    "weight_ridings": weight_ridings,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eby")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    for name, module in COMMANDS.items():
        doc = (module.__doc__ or "").strip()
        sub = subparsers.add_parser(
            name,
            help=doc.splitlines()[0] if doc else None,
            description=doc or None,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(sub)
        sub.set_defaults(run=module.run)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
