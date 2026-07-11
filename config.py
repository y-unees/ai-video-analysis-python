from __future__ import annotations

APP_VERSION = "0.3.0"
SCHEMA_VERSION = "0.3"

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
    }
