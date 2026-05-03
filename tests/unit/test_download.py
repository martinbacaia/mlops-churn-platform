from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from churn.data import download
from churn.data.download import (
    IntegrityError,
    compute_md5,
    download_with_md5,
)


def test_compute_md5_matches_known_hash(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert compute_md5(p) == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_compute_md5_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert compute_md5(p) == "d41d8cd98f00b204e9800998ecf8427e"


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` supporting context-manager + streaming."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def _patch_requests_get(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    # String path keeps mypy --strict happy: it does not require ``requests``
    # to be re-exported from ``churn.data.download``'s namespace.
    monkeypatch.setattr(
        "churn.data.download.requests.get",
        lambda *a, **k: _FakeResponse(payload),
    )


def test_download_with_md5_writes_file_when_hash_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"deterministic-content-for-test"
    expected = hashlib.md5(payload).hexdigest()
    _patch_requests_get(monkeypatch, payload)

    dest = tmp_path / "nested" / "x.csv"
    result = download_with_md5("http://x", dest, expected, chunk_size=4)

    assert result == dest
    assert dest.read_bytes() == payload
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_with_md5_raises_on_mismatch_and_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"some-bytes"
    _patch_requests_get(monkeypatch, payload)

    dest = tmp_path / "x.csv"
    with pytest.raises(IntegrityError, match="MD5 mismatch"):
        download_with_md5("http://x", dest, "0" * 32)

    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_telco_skips_when_existing_file_is_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the dataset is already present and matches, no HTTP call is made."""
    dest_dir = tmp_path / "raw"
    dest_dir.mkdir()
    dest = dest_dir / download.TELCO_FILENAME

    # Write any bytes and patch the expected MD5 so the existing file looks valid.
    dest.write_bytes(b"already-here")
    monkeypatch.setattr(download, "TELCO_MD5", hashlib.md5(b"already-here").hexdigest())

    def _explode(*_a: Any, **_k: Any) -> None:
        raise AssertionError("requests.get must not be called when the file is valid")

    monkeypatch.setattr("churn.data.download.requests.get", _explode)

    result = download.download_telco(dest_dir=dest_dir)
    assert result == dest
