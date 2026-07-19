from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


DatasetLabel = Literal["real", "ai_generated"]


@dataclass(frozen=True)
class DatasetManifestEntry:
    sample_id: str
    video_name: str
    video_sha256: str
    expected_label: DatasetLabel
    feature_schema_version: str
    application_version: str
    feature_file: str
    source: str | None
    generator_or_camera: str | None
    notes: str | None
    added_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSummary:
    total_samples: int
    real_sample_count: int
    ai_generated_sample_count: int
    duplicate_count: int
    valid_sample_count: int
    invalid_sample_count: int
    feature_schema_versions_present: list[str]
    d3_status_counts: dict[str, int]
    missing_value_count_per_feature: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
