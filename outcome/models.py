from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ExpectedLabel = Literal["real", "ai_generated"] | None


@dataclass(frozen=True)
class OutcomeIdentity:
    feature_schema_version: str
    application_version: str
    analysis_id: str | None
    video_name: str | None
    video_sha256: str | None
    expected_label: ExpectedLabel = None


@dataclass(frozen=True)
class MediaFeatures:
    duration_seconds: float | None
    width: int | None
    height: int | None
    frame_rate: float | None
    frame_count: int | None
    audio_present: bool | None


@dataclass(frozen=True)
class MetadataIndicatorFeatures:
    creation_time_present: bool | None
    encoder_metadata_present: bool | None
    duration_metadata_consistent: bool | None


@dataclass(frozen=True)
class FrameSummaryFeatures:
    sampled_frame_count: int | None
    mean_brightness: float | None
    mean_contrast: float | None
    mean_sharpness: float | None
    dark_frame_ratio: float | None
    white_frame_ratio: float | None


@dataclass(frozen=True)
class TemporalSummaryFeatures:
    analyzed_transition_count: int | None
    notable_transition_count: int | None
    scene_boundary_count: int | None
    near_static_interval_count: int | None
    mean_normalized_frame_difference: float | None
    maximum_normalized_frame_difference: float | None
    mean_flow_warp_residual: float | None
    maximum_flow_warp_residual: float | None
    mean_motion_magnitude: float | None
    maximum_motion_magnitude: float | None


@dataclass(frozen=True)
class AudioSummaryFeatures:
    audio_window_count: int | None
    ranked_audio_transition_count: int | None
    silence_interval_count: int | None
    clipping_interval_count: int | None
    maximum_audio_energy_change: float | None
    mean_audio_rms: float | None


@dataclass(frozen=True)
class VisualConsistencySummaryFeatures:
    analyzed_region_record_count: int | None
    ranked_consistency_transition_count: int | None
    sustained_instability_interval_count: int | None
    maximum_unstable_region_count: int | None
    maximum_edge_instability: float | None
    maximum_texture_distance: float | None
    maximum_detail_residual: float | None


@dataclass(frozen=True)
class UnifiedEvidenceSummaryFeatures:
    timeline_event_count: int | None
    high_priority_event_count: int | None
    moderate_priority_event_count: int | None
    cross_modal_event_count: int | None
    maximum_independent_evidence_group_count: int | None


@dataclass(frozen=True)
class D3SummaryFeatures:
    d3_status: str | None
    d3_raw_score: float | None
    d3_encoder: str | None
    d3_distance_mode: str | None
    d3_selected_frame_count: int | None
    d3_first_order_value_count: int | None
    d3_second_order_value_count: int | None
    d3_second_order_mean: float | None
    d3_second_order_standard_deviation: float | None
    d3_calibration_status: str | None
    d3_score_direction: str | None


@dataclass(frozen=True)
class OutcomeFeatures:
    identity: OutcomeIdentity
    media: MediaFeatures
    metadata_indicators: MetadataIndicatorFeatures
    frame_summary: FrameSummaryFeatures
    temporal_summary: TemporalSummaryFeatures
    audio_summary: AudioSummaryFeatures
    visual_consistency_summary: VisualConsistencySummaryFeatures
    unified_evidence_summary: UnifiedEvidenceSummaryFeatures
    d3_summary: D3SummaryFeatures

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
