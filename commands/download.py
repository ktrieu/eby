"""Download the project's data files into raw_data/.

Takes no arguments. Files are listed in `FILES` and `SHAPEFILES`; raw_data/
is gitignored.
"""

import argparse
import tempfile
import zipfile
from pathlib import Path

import requests

from paths import RAW_DATA_DIR

VP_RESULTS_CSV = "vp_results.csv"
VA_RESULTS_CSV = "va_results.csv"
VA_2017_SHP = "va_2017.shp"

# URL -> filename under raw_data/
FILES: dict[str, str] = {
    "https://catalogue.data.gov.bc.ca/dataset/44914a35-de9a-4830-ac48-870001ef8935/resource/4e7be8e3-0805-4f39-87f3-47c4c29e262d/download/provincial_voting_results_by_voting_place.csv": VP_RESULTS_CSV,
    "https://catalogue.data.gov.bc.ca/dataset/44914a35-de9a-4830-ac48-870001ef8935/resource/fb40239e-b718-4a79-b18f-7a62139d9792/download/provincial_voting_results_by_va.csv": VA_RESULTS_CSV
}

# URL of a zipped shapefile -> .shp filename under raw_data/. The sidecar
# files (.dbf, .shx, .prj, ...) are renamed to match.
SHAPEFILES: dict[str, str] = {
    "https://catalogue.data.gov.bc.ca/dataset/f309df17-0462-40c0-bb11-b76c1cb83024/resource/712d1b73-24b8-4fb3-b491-50227e1cd720/download/edvas02232017.zip": VA_2017_SHP,
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def download(url: str, dest: Path) -> None:
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


def extract_shapefile(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        # Extract the .shp last so its presence means the set is complete.
        members.sort(key=lambda m: m.filename.lower().endswith(".shp"))
        for member in members:
            # "edvas02232017.shp.xml" -> "va_2017.shp.xml"
            suffix = Path(member.filename).name.partition(".")[2]
            target = dest.with_name(f"{dest.stem}.{suffix}")
            tmp = target.with_name(target.name + ".part")
            with zf.open(member) as src, tmp.open("wb") as out:
                while chunk := src.read(1 << 16):
                    out.write(chunk)
            tmp.replace(target)


def run(args: argparse.Namespace) -> int:
    RAW_DATA_DIR.mkdir(exist_ok=True)

    for url, filename in FILES.items():
        dest = RAW_DATA_DIR / filename
        if dest.exists():
            print(f"skip {dest} (exists)")
            continue
        download(url, dest)

    for url, filename in SHAPEFILES.items():
        dest = RAW_DATA_DIR / filename
        if dest.exists():
            print(f"skip {dest} (exists)")
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "archive.zip"
            download(url, archive)
            extract_shapefile(archive, dest)

    return 0
