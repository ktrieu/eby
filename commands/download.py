"""Download the project's data files into raw_data/.

Takes no arguments. Files are listed in `FILES`; raw_data/ is gitignored.
"""

import argparse

import requests

from paths import RAW_DATA_DIR

VP_RESULTS_CSV = "vp_results.csv"
VA_RESULTS_CSV = "va_results.csv"

# URL -> filename under raw_data/
FILES: dict[str, str] = {
    "https://catalogue.data.gov.bc.ca/dataset/44914a35-de9a-4830-ac48-870001ef8935/resource/4e7be8e3-0805-4f39-87f3-47c4c29e262d/download/provincial_voting_results_by_voting_place.csv": VP_RESULTS_CSV,
    "https://catalogue.data.gov.bc.ca/dataset/44914a35-de9a-4830-ac48-870001ef8935/resource/fb40239e-b718-4a79-b18f-7a62139d9792/download/provincial_voting_results_by_va.csv": VA_RESULTS_CSV
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(args: argparse.Namespace) -> int:
    RAW_DATA_DIR.mkdir(exist_ok=True)

    for url, filename in FILES.items():
        dest = RAW_DATA_DIR / filename
        if dest.exists():
            print(f"skip {dest} (exists)")
            continue

        print(f"{url} -> {dest}")
        # Write to a temp file first so an interrupted download isn't
        # mistaken for a complete one on the next run.
        tmp = dest.with_name(dest.name + ".part")
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        tmp.replace(dest)

    return 0
