from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from config import INCLUDE_ABSOLUTE_PATHS


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def format_file_size(size_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{size_bytes} B"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def artifact_record(path: Path, relative_path: str) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    return {
        "path": relative_path,
        "size_bytes": size_bytes,
        "size_human_readable": format_file_size(size_bytes),
        "sha256": calculate_sha256(path),
    }


def format_duration(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    return f"{float(seconds):.2f} seconds"


def format_bit_rate(bits_per_second: int | float | None) -> str | None:
    if bits_per_second is None:
        return None
    value = float(bits_per_second)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} Mbps"
    if value >= 1_000:
        return f"{value / 1_000:.2f} Kbps"
    return f"{value:.0f} bps"


def safe_relative_path(path: Path, base_dir: Path | None = None) -> str:
    base = base_dir or Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def get_file_evidence(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    size_bytes = path.stat().st_size
    report_path = (
        str(resolved_path)
        if INCLUDE_ABSOLUTE_PATHS
        else safe_relative_path(path)
    )
    return {
        "filename": path.name,
        "path": report_path,
        "absolute_path_included": INCLUDE_ABSOLUTE_PATHS,
        "extension": path.suffix.lower(),
        "size_bytes": size_bytes,
        "size_human_readable": format_file_size(size_bytes),
        "sha256": "",
    }
