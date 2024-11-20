import os
from pathlib import Path
from typing import Optional

from sportslabkit.logger import logger

from .downloader import KaggleDownloader


__all__ = ["available", "get_path", "KaggleDownloader"]

_module_path = Path(__file__).parent
_available_dir = {
    "GNSS": _module_path / "GNSS",
    "top_view": _module_path / "top_view",
    "wide_view": _module_path / "wide_view",
}

for d, path in _available_dir.copy().items():
    if not path.exists():
        _available_dir.pop(d)

_available_files = {
    "drone_keypoints": _module_path / "drone_keypoints.json",
    "fisheye_keypoints": _module_path / "fisheye_keypoints.json",
    "gnss_keypoints": _module_path / "gnss_keypoints.json",
}

for d, path in _available_files.items():
    if not path.exists:
        _available_files.pop(d), print(f"Dataset {d} not available")

available = list(_available_dir.keys()) + list(_available_files.keys())

