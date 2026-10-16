"""Process the raw elections CSVs in raw_data/.

Takes no arguments. Run `download` first to fetch the input files.
"""

import argparse
import codecs
import csv
import io
import sys
from collections.abc import Iterator

from commands.download import VA_RESULTS_CSV, VP_RESULTS_CSV
from paths import RAW_DATA_DIR

# Both files start with a UTF-8 BOM, but only vp_results is actually UTF-8;
# va_results' body is cp1252 (e.g. "Fran\xe7oise").
ENCODINGS: dict[str, str] = {
    VP_RESULTS_CSV: "utf-8",
    VA_RESULTS_CSV: "cp1252",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def read_rows(filename: str) -> Iterator[dict[str, str]]:
    with (RAW_DATA_DIR / filename).open("rb") as raw:
        if raw.read(len(codecs.BOM_UTF8)) != codecs.BOM_UTF8:
            raw.seek(0)
        f = io.TextIOWrapper(raw, encoding=ENCODINGS[filename], newline="")
        yield from csv.DictReader(f)


def run(args: argparse.Namespace) -> int:
    missing = [f for f in (VP_RESULTS_CSV, VA_RESULTS_CSV) if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print(f"missing {', '.join(missing)} in {RAW_DATA_DIR}; run `download` first", file=sys.stderr)
        return 1

    for filename in (VP_RESULTS_CSV, VA_RESULTS_CSV):
        count = sum(1 for _ in read_rows(filename))
        print(f"{filename}: {count} rows")

    return 0
