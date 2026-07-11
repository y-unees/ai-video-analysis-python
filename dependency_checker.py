from __future__ import annotations

import importlib
import shutil
import subprocess
from typing import Any


REQUIRED_PYTHON_PACKAGES = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "PIL": "Pillow",
    "imagehash": "ImageHash",
}


def missing_python_dependencies() -> list[str]:
    missing: list[str] = []
    for import_name, package_name in REQUIRED_PYTHON_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)
    return missing


def tool_is_available(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def get_tool_version(tool_name: str) -> str | None:
    if not tool_is_available(tool_name):
        return None

    try:
        result = subprocess.run(
            [tool_name, "-version"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line or None


def get_package_version(import_name: str) -> str | None:
    try:
        module: Any = importlib.import_module(import_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)
