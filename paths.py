"""Filesystem locations shared across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"


def processed_path(name: str) -> Path:
    """Resolve a file name inside PROCESSED_DATA_DIR, for use as an argparse type."""
    return PROCESSED_DATA_DIR / name
