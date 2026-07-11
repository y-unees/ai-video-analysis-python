from __future__ import annotations

import platform
import sys
from datetime import datetime
from typing import Any

from config import APP_VERSION, SCHEMA_VERSION
from dependency_checker import get_package_version, get_tool_version


def collect_environment_info(
    analysis_started_at: datetime,
    analysis_completed_at: datetime,
) -> dict[str, Any]:
    return {
        "application_version": APP_VERSION,
        "report_schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "opencv_version": get_package_version("cv2"),
        "numpy_version": get_package_version("numpy"),
        "pillow_version": get_package_version("PIL"),
        "imagehash_version": get_package_version("imagehash"),
        "ffmpeg_version": get_tool_version("ffmpeg"),
        "ffprobe_version": get_tool_version("ffprobe"),
        "analysis_start_time": analysis_started_at.isoformat(timespec="milliseconds"),
        "analysis_completion_time": analysis_completed_at.isoformat(timespec="milliseconds"),
        "total_processing_duration_seconds": round(
            (analysis_completed_at - analysis_started_at).total_seconds(),
            3,
        ),
    }
