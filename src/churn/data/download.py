"""Download the Telco churn dataset with MD5 integrity verification.

The URL and expected hash are pinned constants so every clone fetches the same
bytes. The hash is checked over the streamed download; on mismatch the partial
file is deleted and an ``IntegrityError`` is raised — we never leave a corrupt
dataset on disk.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

from churn.config import get_settings
from churn.logging_setup import configure_logging, get_logger

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
TELCO_MD5 = "3b0bfab28a8101b4e4fdd08025a5c235"
TELCO_FILENAME = "telco.csv"

_log = get_logger(__name__)


class IntegrityError(Exception):
    """Raised when a downloaded file's checksum does not match the expected value."""


def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Return the hex MD5 of a file, streamed in fixed-size chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_md5(
    url: str,
    dest: Path,
    expected_md5: str,
    chunk_size: int = 8192,
    timeout: float = 60.0,
) -> Path:
    """Stream ``url`` to ``dest``, verifying its MD5 against ``expected_md5``.

    Writes to ``<dest>.part`` first and renames atomically on success. On
    checksum mismatch the partial file is removed and ``IntegrityError`` is
    raised — callers always see either a verified file or no file at all.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.md5()

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                hasher.update(chunk)

    actual = hasher.hexdigest()
    if actual != expected_md5:
        tmp.unlink(missing_ok=True)
        raise IntegrityError(f"MD5 mismatch for {url}: expected {expected_md5}, got {actual}")
    tmp.replace(dest)
    return dest


def download_telco(dest_dir: Path | None = None) -> Path:
    """Download the Telco CSV into ``dest_dir`` (or ``data/raw`` from settings).

    If a verified copy already exists at the target path, returns it unchanged.
    """
    settings = get_settings()
    target_dir = dest_dir if dest_dir is not None else settings.data_dir / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / TELCO_FILENAME

    if dest.exists() and compute_md5(dest) == TELCO_MD5:
        _log.info("dataset_already_present", path=str(dest), md5=TELCO_MD5)
        return dest

    _log.info("dataset_download_start", url=TELCO_URL, dest=str(dest))
    path = download_with_md5(TELCO_URL, dest, TELCO_MD5)
    _log.info("dataset_download_ok", path=str(path), md5=TELCO_MD5)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Telco churn dataset.")
    parser.add_argument("--dest-dir", type=Path, default=None, help="Override target directory.")
    args = parser.parse_args()

    configure_logging()
    download_telco(args.dest_dir)


if __name__ == "__main__":
    main()
