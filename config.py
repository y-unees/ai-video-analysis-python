from __future__ import annotations

APP_VERSION = "0.4.1"
SCHEMA_VERSION = "0.4"

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
            "method": "rank transitions within the current video across temporal-change metrics, merge duplicate selections, then sort by reason count and combined percentile",
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
