from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

APP_VERSION = "0.9.2"
SCHEMA_VERSION = "0.7"

SOURCE_DIR_NAME = "source_videos"
REPORTS_DIR_NAME = "reports"
INCLUDE_ABSOLUTE_PATHS = False

SHORT_VIDEO_MAX_SECONDS = 10.0
SHORT_VIDEO_SAMPLE_COUNT = 8
LONG_VIDEO_SAMPLE_COUNT = 12
END_SEEK_SAFETY_SECONDS = 0.05
JPEG_QUALITY = 88
COMPARISON_MAX_WIDTH = 480

# These are lightweight heuristics for triage only, not scientific thresholds.
DARK_PIXEL_THRESHOLD = 16
BRIGHT_PIXEL_THRESHOLD = 240
BLACK_FRAME_RATIO_THRESHOLD = 0.98
WHITE_FRAME_RATIO_THRESHOLD = 0.98

NEAR_DUPLICATE_HASH_DISTANCE = 4
LARGE_CHANGE_HASH_DISTANCE = 24
LOW_SHARPNESS_HEURISTIC = 50.0
BRIGHTNESS_JUMP_THRESHOLD = 50.0
LARGE_CHANGE_MEAN_ABS_DIFF = 0.35
NEAR_DUPLICATE_MEAN_ABS_DIFF = 0.03
SEEK_ERROR_WARNING_SECONDS = 0.25
PERCEPTUAL_HASH_SIZE = 8
HISTOGRAM_BINS = 64

TEMPORAL_REQUESTED_ANALYSIS_FPS = 5.0
TEMPORAL_MAX_ANALYZED_FRAMES = 1500
TEMPORAL_RESIZE_MAX_WIDTH = 320

FARNEBACK_PYR_SCALE = 0.5
FARNEBACK_LEVELS = 3
FARNEBACK_WINSIZE = 15
FARNEBACK_ITERATIONS = 3
FARNEBACK_POLY_N = 5
FARNEBACK_POLY_SIGMA = 1.2
FLOW_STATIONARY_MAGNITUDE_THRESHOLD = 0.5

SCENE_CUT_PIXEL_DIFFERENCE_THRESHOLD = 0.20
SCENE_CUT_HISTOGRAM_CORRELATION_THRESHOLD = 0.80
SCENE_CUT_PHASH_DISTANCE_THRESHOLD = 18
ELEVATED_MOTION_FLOW_MAGNITUDE_THRESHOLD = 8.0

STATIC_PHASH_DISTANCE_THRESHOLD = 2
STATIC_PIXEL_DIFFERENCE_THRESHOLD = 0.01
STATIC_MINIMUM_DURATION_SECONDS = 0.6

SCENE_REPRESENTATIVE_FRAMES_PER_SCENE = 3
MAX_SCENE_REPRESENTATIVE_FRAMES = 24

FLOW_WARP_HIGH_RESIDUAL_THRESHOLD = 0.10
MAX_NOTABLE_TRANSITIONS = 5
NOTABLE_TRANSITION_MINIMUM_METRIC_REASONS = 1

AUDIO_WINDOW_SECONDS = 0.5
AUDIO_HOP_SECONDS = 0.25
AUDIO_SILENCE_SAMPLE_AMPLITUDE_THRESHOLD = 0.01
AUDIO_SILENCE_WINDOW_RMS_THRESHOLD = 0.01
AUDIO_MINIMUM_SILENCE_INTERVAL_SECONDS = 0.5
AUDIO_CLIPPING_AMPLITUDE_THRESHOLD = 0.99
AUDIO_CLIPPING_RATIO_WARNING_THRESHOLD = 0.01
AUDIO_ENERGY_CHANGE_RATIO_THRESHOLD = 3.0
AUDIO_MAX_REVIEW_TRANSITIONS = 5
AUDIO_MAX_NOTABLE_TRANSITIONS = AUDIO_MAX_REVIEW_TRANSITIONS
AUDIO_EXTRACTION_TIMEOUT_SECONDS = 60

VISUAL_CONSISTENCY_GRID_ROWS = 4
VISUAL_CONSISTENCY_GRID_COLUMNS = 4
VISUAL_CONSISTENCY_MINIMUM_VALID_REGION_RATIO = 0.5
VISUAL_CONSISTENCY_MAX_RANKED_INTERVALS = 5
VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD1 = 80
VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD2 = 160
VISUAL_CONSISTENCY_TEXTURE_HISTOGRAM_BINS = 16
VISUAL_CONSISTENCY_HIGH_DETAIL_RESIDUAL_THRESHOLD = 0.18
VISUAL_CONSISTENCY_STATIONARY_FLOW_THRESHOLD = 0.5
VISUAL_CONSISTENCY_HIGH_MOTION_FLOW_THRESHOLD = 4.0
VISUAL_CONSISTENCY_UNSTABLE_EDGE_THRESHOLD = 0.35
VISUAL_CONSISTENCY_UNSTABLE_TEXTURE_THRESHOLD = 0.30
VISUAL_CONSISTENCY_UNSTABLE_DETAIL_THRESHOLD = 0.16
VISUAL_CONSISTENCY_UNSTABLE_BRIGHTNESS_THRESHOLD = 0.12
VISUAL_CONSISTENCY_MINIMUM_INTERVAL_TRANSITIONS = 3
VISUAL_CONSISTENCY_MINIMUM_INTERVAL_DURATION_SECONDS = 0.4

EVIDENCE_MERGE_TOLERANCE_SECONDS = 0.25
EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS = 1.25
REGIONAL_GROUP_MINIMUM_OVERLAP_RATIO = 0.5
AI_INPUT_MAX_TIMELINE_EVENTS = 10
AI_INPUT_MAX_FINDINGS_PER_EVENT = 8
AI_INPUT_MAX_ARTIFACTS_PER_EVENT = 4
AI_INPUT_MAX_GLOBAL_FINDINGS = 20
AI_INPUT_TARGET_CHARACTER_COUNT = 60000


def heuristic_configuration() -> dict[str, object]:
    return {
        "brightness_scale": "0-255 grayscale mean",
        "dark_pixel_intensity_threshold": DARK_PIXEL_THRESHOLD,
        "bright_pixel_intensity_threshold": BRIGHT_PIXEL_THRESHOLD,
        "near_black_frame_rules": {
            "dark_pixel_ratio_at_or_above": BLACK_FRAME_RATIO_THRESHOLD,
        },
        "near_white_frame_rules": {
            "bright_pixel_ratio_at_or_above": WHITE_FRAME_RATIO_THRESHOLD,
        },
        "near_duplicate_pair_rules": {
            "perceptual_hash_distance_at_or_below": NEAR_DUPLICATE_HASH_DISTANCE,
            "normalized_mean_absolute_difference_at_or_below": NEAR_DUPLICATE_MEAN_ABS_DIFF,
            "logic": "both thresholds must be met",
        },
        "large_change_pair_rules": {
            "perceptual_hash_distance_at_or_above": LARGE_CHANGE_HASH_DISTANCE,
            "normalized_mean_absolute_difference_at_or_above": LARGE_CHANGE_MEAN_ABS_DIFF,
            "logic": "either threshold may trigger",
        },
        "low_laplacian_variance_rule": {
            "laplacian_variance_below": LOW_SHARPNESS_HEURISTIC,
        },
        "brightness_jump_rule": {
            "absolute_brightness_difference_at_or_above": BRIGHTNESS_JUMP_THRESHOLD,
        },
        "seek_error_warning_rule": {
            "absolute_seek_error_seconds_above": SEEK_ERROR_WARNING_SECONDS,
            "interpretation": "decoder or sampling limitation, not video evidence",
        },
        "perceptual_hash_algorithm": "Pillow ImageHash phash",
        "perceptual_hash_size": PERCEPTUAL_HASH_SIZE,
        "pixel_difference_color_space": "grayscale",
        "pixel_difference_normalization": "mean absolute difference divided by 255",
        "histogram_color_space": "grayscale",
        "histogram_bins": HISTOGRAM_BINS,
        "histogram_comparison_method": "OpenCV HISTCMP_CORREL",
        "temporal_analysis": temporal_configuration(),
        "visual_consistency_analysis": visual_consistency_configuration(),
        "unified_evidence": unified_evidence_configuration(),
    }


def temporal_configuration() -> dict[str, object]:
    return {
        "requested_analysis_fps": TEMPORAL_REQUESTED_ANALYSIS_FPS,
        "maximum_analyzed_frames": TEMPORAL_MAX_ANALYZED_FRAMES,
        "resize_max_width": TEMPORAL_RESIZE_MAX_WIDTH,
        "scene_cut_rules": {
            "normalized_mean_absolute_difference_at_or_above": SCENE_CUT_PIXEL_DIFFERENCE_THRESHOLD,
            "histogram_correlation_at_or_below": SCENE_CUT_HISTOGRAM_CORRELATION_THRESHOLD,
            "perceptual_hash_distance_at_or_above": SCENE_CUT_PHASH_DISTANCE_THRESHOLD,
            "logic": "pixel difference threshold and either histogram or pHash threshold",
        },
        "near_static_rules": {
            "perceptual_hash_distance_at_or_below": STATIC_PHASH_DISTANCE_THRESHOLD,
            "normalized_mean_absolute_difference_at_or_below": STATIC_PIXEL_DIFFERENCE_THRESHOLD,
            "minimum_duration_seconds": STATIC_MINIMUM_DURATION_SECONDS,
            "logic": "consecutive transitions meeting both thresholds are merged",
        },
        "optical_flow": {
            "algorithm": "OpenCV Farneback dense optical flow",
            "pyr_scale": FARNEBACK_PYR_SCALE,
            "levels": FARNEBACK_LEVELS,
            "winsize": FARNEBACK_WINSIZE,
            "iterations": FARNEBACK_ITERATIONS,
            "poly_n": FARNEBACK_POLY_N,
            "poly_sigma": FARNEBACK_POLY_SIGMA,
            "stationary_magnitude_threshold": FLOW_STATIONARY_MAGNITUDE_THRESHOLD,
            "flow_warp_high_residual_threshold": FLOW_WARP_HIGH_RESIDUAL_THRESHOLD,
        },
        "notable_transition_ranking": {
            "maximum_notable_transitions": MAX_NOTABLE_TRANSITIONS,
            "minimum_metric_reasons": NOTABLE_TRANSITION_MINIMUM_METRIC_REASONS,
            "method": "rank transitions within the current video across the complete configured metric set, merge duplicate selections, then sort by combined percentile, reason count, start timestamp, and transition ID",
            "selection_basis": "relative_within_video",
            "absolute_significance_assessed": False,
            "combined_percentile_metric_count": 8,
            "missing_metric_behavior": "explicitly recorded as unavailable and excluded from available_metric_count",
            "metrics": [
                "normalized_mean_absolute_difference descending",
                "perceptual_hash_distance descending",
                "absolute_brightness_difference descending",
                "mean_optical_flow descending",
                "95th_percentile_optical_flow descending",
                "histogram_correlation ascending",
                "flow_warp_residual_mean descending",
                "flow_warp_residual_95th_percentile descending",
            ],
        },
        "scene_representative_frames_per_scene": SCENE_REPRESENTATIVE_FRAMES_PER_SCENE,
        "maximum_scene_representative_frames": MAX_SCENE_REPRESENTATIVE_FRAMES,
    }


def audio_configuration() -> dict[str, object]:
    return {
        "window_seconds": AUDIO_WINDOW_SECONDS,
        "hop_seconds": AUDIO_HOP_SECONDS,
        "silence_sample_amplitude_threshold": AUDIO_SILENCE_SAMPLE_AMPLITUDE_THRESHOLD,
        "silence_window_rms_threshold": AUDIO_SILENCE_WINDOW_RMS_THRESHOLD,
        "minimum_silence_interval_seconds": AUDIO_MINIMUM_SILENCE_INTERVAL_SECONDS,
        "clipping_amplitude_threshold": AUDIO_CLIPPING_AMPLITUDE_THRESHOLD,
        "clipping_ratio_warning_threshold": AUDIO_CLIPPING_RATIO_WARNING_THRESHOLD,
        "energy_change_ratio_threshold": AUDIO_ENERGY_CHANGE_RATIO_THRESHOLD,
        "maximum_notable_transitions": AUDIO_MAX_NOTABLE_TRANSITIONS,
        "extraction_timeout_seconds": AUDIO_EXTRACTION_TIMEOUT_SECONDS,
        "interpretation": "Starter thresholds for lightweight local signal observations, not forensic standards.",
    }


def visual_consistency_configuration() -> dict[str, object]:
    return {
        "analysis_fps": TEMPORAL_REQUESTED_ANALYSIS_FPS,
        "resize_max_width": TEMPORAL_RESIZE_MAX_WIDTH,
        "grid_rows": VISUAL_CONSISTENCY_GRID_ROWS,
        "grid_columns": VISUAL_CONSISTENCY_GRID_COLUMNS,
        "minimum_valid_region_ratio": VISUAL_CONSISTENCY_MINIMUM_VALID_REGION_RATIO,
        "maximum_ranked_intervals": VISUAL_CONSISTENCY_MAX_RANKED_INTERVALS,
        "edge_method": "OpenCV Canny on blurred grayscale analysis frames",
        "edge_canny_threshold1": VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD1,
        "edge_canny_threshold2": VISUAL_CONSISTENCY_EDGE_CANNY_THRESHOLD2,
        "edge_comparison": "previous edge map is warped into current-frame coordinates with backward Farneback optical flow when available",
        "texture_method": "normalized grayscale histogram plus local variance and Sobel gradient-energy comparison",
        "texture_histogram_bins": VISUAL_CONSISTENCY_TEXTURE_HISTOGRAM_BINS,
        "detail_method": "absolute difference between motion-compensated previous Laplacian magnitude and current Laplacian magnitude, normalized to 0-1",
        "motion_context": {
            "stationary_flow_magnitude_at_or_below": VISUAL_CONSISTENCY_STATIONARY_FLOW_THRESHOLD,
            "high_motion_mean_flow_at_or_above": VISUAL_CONSISTENCY_HIGH_MOTION_FLOW_THRESHOLD,
        },
        "unstable_region_rules": {
            "edge_instability_at_or_above": VISUAL_CONSISTENCY_UNSTABLE_EDGE_THRESHOLD,
            "texture_distance_at_or_above": VISUAL_CONSISTENCY_UNSTABLE_TEXTURE_THRESHOLD,
            "detail_residual_at_or_above": VISUAL_CONSISTENCY_UNSTABLE_DETAIL_THRESHOLD,
            "brightness_normalized_difference_at_or_above": VISUAL_CONSISTENCY_UNSTABLE_BRIGHTNESS_THRESHOLD,
            "logic": "any threshold may mark a region for review",
        },
        "sustained_interval_rules": {
            "minimum_consecutive_transitions": VISUAL_CONSISTENCY_MINIMUM_INTERVAL_TRANSITIONS,
            "minimum_duration_seconds": VISUAL_CONSISTENCY_MINIMUM_INTERVAL_DURATION_SECONDS,
            "overlap_behavior": "runs are detected per region, then simultaneous affected regions are summarized without claiming causation",
        },
        "ranking": {
            "selection_basis": "relative_within_video",
            "absolute_significance_assessed": False,
            "combined_percentile_metric_count": 7,
            "metrics": [
                "maximum regional detail residual descending",
                "average regional detail residual descending",
                "maximum edge instability descending",
                "average edge instability descending",
                "maximum texture distance descending",
                "unstable-region count descending",
                "regional brightness-change concentration descending",
            ],
        },
    }


def unified_evidence_configuration() -> dict[str, object]:
    return {
        "merge_tolerance_seconds": EVIDENCE_MERGE_TOLERANCE_SECONDS,
        "maximum_anchor_event_span_seconds": EVIDENCE_MAX_ANCHOR_EVENT_SPAN_SECONDS,
        "merging_strategy": "anchor_based_non_transitive",
        "maximum_anchor_event_span_purpose": "Timeline segmentation control only; not a forensic threshold.",
        "regional_group_minimum_overlap_ratio": REGIONAL_GROUP_MINIMUM_OVERLAP_RATIO,
        "timeline_basis": "selected_video_stream_normalized",
        "review_priority_levels": ["low", "moderate", "high"],
        "review_priority_rules": {
            "high": [
                "two_or_more_independent_evidence_groups",
                "visual_and_audio_temporal_overlap",
                "multiple_visual_methods_with_ranked_source_findings",
            ],
            "moderate": [
                "ranked_source_findings",
                "multiple_observations",
                "review_artifacts_available",
            ],
            "low": [
                "single_source_or_global_context_only",
            ],
            "meaning": "Review priority only; it is not an authenticity score, AI probability, or manipulation verdict.",
        },
        "ai_input_limits": {
            "maximum_timeline_events": AI_INPUT_MAX_TIMELINE_EVENTS,
            "maximum_findings_per_event": AI_INPUT_MAX_FINDINGS_PER_EVENT,
            "domain_finding_caps": {
                "visual_consistency": 3,
                "visual_temporal": 3,
                "audio_signal": 2,
                "frame_sampling": 1,
            },
            "maximum_artifacts_per_event": AI_INPUT_MAX_ARTIFACTS_PER_EVENT,
            "maximum_global_findings": AI_INPUT_MAX_GLOBAL_FINDINGS,
            "target_character_count": AI_INPUT_TARGET_CHARACTER_COUNT,
        },
    }


def learned_detector_configuration() -> dict[str, object]:
    import os

    return {
        "learned_detectors_enabled": _env_bool("LEARNED_DETECTORS_ENABLED", False),
        "d3": {
            "enabled": _env_bool("D3_ENABLED", False),
            "device": os.getenv("D3_DEVICE", "auto"),
            "encoder": os.getenv("D3_ENCODER", "XCLIP-16"),
            "distance_mode": os.getenv("D3_DISTANCE", "l2"),
            "random_seed": _env_int("D3_RANDOM_SEED", 42),
            "timeout_seconds": _env_positive_int("D3_TIMEOUT_SECONDS", 300),
            "allow_model_download": _env_bool("D3_ALLOW_MODEL_DOWNLOAD", False),
            "model_cache_directory": os.getenv("D3_MODEL_CACHE_DIRECTORY", ""),
            "preprocessing_mode": os.getenv("D3_PREPROCESSING_MODE", "upstream_compatible"),
            "preserve_temporary_frames": _env_bool("D3_PRESERVE_TEMPORARY_FRAMES", False),
        },
    }


def gemini_compact_report_configuration() -> dict[str, object]:
    return {
        "enabled": _env_bool("GEMINI_COMPACT_REPORT_ENABLED", True),
        "maximum_key_events": _env_positive_int("GEMINI_COMPACT_MAX_KEY_EVENTS", 5),
        "maximum_findings_per_event": _env_positive_int("GEMINI_COMPACT_MAX_FINDINGS_PER_EVENT", 5),
        "maximum_artifacts_per_event": _env_positive_int("GEMINI_COMPACT_MAX_ARTIFACTS_PER_EVENT", 4),
        "preferred_size_bytes": _env_positive_int("GEMINI_COMPACT_PREFERRED_SIZE_BYTES", 8000),
        "acceptable_size_bytes": _env_positive_int("GEMINI_COMPACT_ACCEPTABLE_SIZE_BYTES", 12000),
        "hard_size_limit_bytes": _env_positive_int("GEMINI_COMPACT_HARD_SIZE_BYTES", 16000),
    }


def _env_bool(name: str, default: bool) -> bool:
    import os

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default
