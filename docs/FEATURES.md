# Feature Registry and Standardized Schema

Project version: `0.9.4`
Feature schema version: `0.9.4`
Registry version: `0.9.4`
Schema fingerprint: `8f06d41a32ef09f3565011dbf754263ab4d661b454368fa6f52e40e39305d228`

This document is generated from `schemas/feature_registry.json` and `schemas/feature_schema.json`.

No fitted normalization, target-aware selection, probabilities, thresholds, or model training are performed in v0.9.4.

## Feature Categories
- `analysis_metadata`
- `deprecated`
- `derived_numeric`
- `identifier`
- `leakage`
- `path`
- `raw_boolean`
- `raw_categorical`
- `raw_numeric`
- `source_metadata`
- `target`
- `timestamp`
- `unsupported`

## Null Policy

JSON uses `null`; CSV uses empty fields. Missing values are not imputed or replaced with fitted values.

## Standardized Model Features

### `audio_summary.ranked_audio_transition_count`

- ID: `audio_summary.ranked_audio_transition_count`
- Display name: Audio Summary Ranked Audio Transition Count
- Description: Exported field `audio_summary.ranked_audio_transition_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `audio_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `d3_summary.d3_raw_score`

- ID: `d3_summary.d3_raw_score`
- Display name: D3 Summary D3 Raw Score
- Description: Exported field `d3_summary.d3_raw_score` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `d3_summary.d3_second_order_mean`

- ID: `d3_summary.d3_second_order_mean`
- Display name: D3 Summary D3 Second Order Mean
- Description: Exported field `d3_summary.d3_second_order_mean` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `d3_summary.d3_second_order_standard_deviation`

- ID: `d3_summary.d3_second_order_standard_deviation`
- Display name: D3 Summary D3 Second Order Standard Deviation
- Description: Exported field `d3_summary.d3_second_order_standard_deviation` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `frame_summary.mean_brightness`

- ID: `frame_summary.mean_brightness`
- Display name: Frame Summary Mean Brightness
- Description: Exported field `frame_summary.mean_brightness` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `pixel_intensity`
- Nullable: `False`
- Range: `0.0` to `255.0`
- Source module: `frame_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `frame_summary.mean_contrast`

- ID: `frame_summary.mean_contrast`
- Display name: Frame Summary Mean Contrast
- Description: Exported field `frame_summary.mean_contrast` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `frame_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `frame_summary.mean_sharpness`

- ID: `frame_summary.mean_sharpness`
- Display name: Frame Summary Mean Sharpness
- Description: Exported field `frame_summary.mean_sharpness` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `frame_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `frame_summary.sampled_frame_count`

- ID: `frame_summary.sampled_frame_count`
- Display name: Frame Summary Sampled Frame Count
- Description: Exported field `frame_summary.sampled_frame_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `frame_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `media.duration_seconds`

- ID: `media.duration_seconds`
- Display name: Media Duration Seconds
- Description: Exported field `media.duration_seconds` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `seconds`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `metadata_extractor`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `media.frame_count`

- ID: `media.frame_count`
- Display name: Media Frame Count
- Description: Exported field `media.frame_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `metadata_extractor`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `media.frame_rate`

- ID: `media.frame_rate`
- Display name: Media Frame Rate
- Description: Exported field `media.frame_rate` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `frames_per_second`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `metadata_extractor`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `media.height`

- ID: `media.height`
- Display name: Media Height
- Description: Exported field `media.height` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `pixels`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `metadata_extractor`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `media.width`

- ID: `media.width`
- Display name: Media Width
- Description: Exported field `media.width` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `pixels`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `metadata_extractor`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.analyzed_transition_count`

- ID: `temporal_summary.analyzed_transition_count`
- Display name: Temporal Summary Analyzed Transition Count
- Description: Exported field `temporal_summary.analyzed_transition_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.maximum_flow_warp_residual`

- ID: `temporal_summary.maximum_flow_warp_residual`
- Display name: Temporal Summary Maximum Flow Warp Residual
- Description: Exported field `temporal_summary.maximum_flow_warp_residual` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.maximum_motion_magnitude`

- ID: `temporal_summary.maximum_motion_magnitude`
- Display name: Temporal Summary Maximum Motion Magnitude
- Description: Exported field `temporal_summary.maximum_motion_magnitude` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.mean_flow_warp_residual`

- ID: `temporal_summary.mean_flow_warp_residual`
- Display name: Temporal Summary Mean Flow Warp Residual
- Description: Exported field `temporal_summary.mean_flow_warp_residual` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.mean_motion_magnitude`

- ID: `temporal_summary.mean_motion_magnitude`
- Display name: Temporal Summary Mean Motion Magnitude
- Description: Exported field `temporal_summary.mean_motion_magnitude` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `temporal_summary.near_static_interval_count`

- ID: `temporal_summary.near_static_interval_count`
- Display name: Temporal Summary Near Static Interval Count
- Description: Exported field `temporal_summary.near_static_interval_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `temporal_analysis`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `unified_evidence_summary.cross_modal_event_count`

- ID: `unified_evidence_summary.cross_modal_event_count`
- Display name: Unified Evidence Summary Cross Modal Event Count
- Description: Exported field `unified_evidence_summary.cross_modal_event_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `unified_evidence`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `unified_evidence_summary.high_priority_event_count`

- ID: `unified_evidence_summary.high_priority_event_count`
- Display name: Unified Evidence Summary High Priority Event Count
- Description: Exported field `unified_evidence_summary.high_priority_event_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `unified_evidence`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `unified_evidence_summary.maximum_independent_evidence_group_count`

- ID: `unified_evidence_summary.maximum_independent_evidence_group_count`
- Display name: Unified Evidence Summary Maximum Independent Evidence Group Count
- Description: Exported field `unified_evidence_summary.maximum_independent_evidence_group_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `unified_evidence`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `unified_evidence_summary.moderate_priority_event_count`

- ID: `unified_evidence_summary.moderate_priority_event_count`
- Display name: Unified Evidence Summary Moderate Priority Event Count
- Description: Exported field `unified_evidence_summary.moderate_priority_event_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `unified_evidence`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `unified_evidence_summary.timeline_event_count`

- ID: `unified_evidence_summary.timeline_event_count`
- Display name: Unified Evidence Summary Timeline Event Count
- Description: Exported field `unified_evidence_summary.timeline_event_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `unified_evidence`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.analyzed_region_record_count`

- ID: `visual_consistency_summary.analyzed_region_record_count`
- Display name: Visual Consistency Summary Analyzed Region Record Count
- Description: Exported field `visual_consistency_summary.analyzed_region_record_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.maximum_detail_residual`

- ID: `visual_consistency_summary.maximum_detail_residual`
- Display name: Visual Consistency Summary Maximum Detail Residual
- Description: Exported field `visual_consistency_summary.maximum_detail_residual` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.maximum_edge_instability`

- ID: `visual_consistency_summary.maximum_edge_instability`
- Display name: Visual Consistency Summary Maximum Edge Instability
- Description: Exported field `visual_consistency_summary.maximum_edge_instability` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.maximum_texture_distance`

- ID: `visual_consistency_summary.maximum_texture_distance`
- Display name: Visual Consistency Summary Maximum Texture Distance
- Description: Exported field `visual_consistency_summary.maximum_texture_distance` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.maximum_unstable_region_count`

- ID: `visual_consistency_summary.maximum_unstable_region_count`
- Display name: Visual Consistency Summary Maximum Unstable Region Count
- Description: Exported field `visual_consistency_summary.maximum_unstable_region_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `visual_consistency_summary.sustained_instability_interval_count`

- ID: `visual_consistency_summary.sustained_instability_interval_count`
- Display name: Visual Consistency Summary Sustained Instability Interval Count
- Description: Exported field `visual_consistency_summary.sustained_instability_interval_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `visual_consistency`
- Model candidate: `True`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: None

### `derived.exposure_extreme_ratio`

- ID: `derived.exposure_extreme_ratio`
- Display name: Exposure Extreme Ratio
- Description: Dark-frame ratio plus white-frame ratio for the same sample.
- Category: `derived_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `2.0`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['frame_summary.dark_frame_ratio', 'frame_summary.white_frame_ratio']`
- Transformation: `{'inputs': ['frame_summary.dark_frame_ratio', 'frame_summary.white_frame_ratio'], 'null_behavior': 'null_if_any_missing', 'type': 'sum'}`
- Aliases: `[]`
- Notes: None

### `derived.sharpness_contrast_ratio`

- ID: `derived.sharpness_contrast_ratio`
- Display name: Sharpness To Contrast Ratio
- Description: Mean sharpness divided by mean contrast for the same sample.
- Category: `derived_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['frame_summary.mean_sharpness', 'frame_summary.mean_contrast']`
- Transformation: `{'denominator': 'frame_summary.mean_contrast', 'numerator': 'frame_summary.mean_sharpness', 'type': 'safe_ratio', 'zero_denominator': 'null'}`
- Aliases: `[]`
- Notes: None

### `derived.motion_spike_ratio`

- ID: `derived.motion_spike_ratio`
- Display name: Motion Spike Ratio
- Description: Maximum motion magnitude divided by mean motion magnitude for the same sample.
- Category: `derived_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['temporal_summary.maximum_motion_magnitude', 'temporal_summary.mean_motion_magnitude']`
- Transformation: `{'denominator': 'temporal_summary.mean_motion_magnitude', 'numerator': 'temporal_summary.maximum_motion_magnitude', 'type': 'safe_ratio', 'zero_denominator': 'null'}`
- Aliases: `[]`
- Notes: None

### `derived.flow_residual_ratio`

- ID: `derived.flow_residual_ratio`
- Display name: Flow Residual Ratio
- Description: Maximum flow-warp residual divided by mean flow-warp residual for the same sample.
- Category: `derived_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['temporal_summary.maximum_flow_warp_residual', 'temporal_summary.mean_flow_warp_residual']`
- Transformation: `{'denominator': 'temporal_summary.mean_flow_warp_residual', 'numerator': 'temporal_summary.maximum_flow_warp_residual', 'type': 'safe_ratio', 'zero_denominator': 'null'}`
- Aliases: `[]`
- Notes: None

### `derived.audio_features_available`

- ID: `derived.audio_features_available`
- Display name: Audio Features Available
- Description: Indicator set to 1 when audio window metrics are present for the same sample, otherwise 0.
- Category: `derived_numeric`
- Type: `integer`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `1.0`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['audio_summary.audio_window_count']`
- Transformation: `{'input': 'audio_summary.audio_window_count', 'null_behavior': '0', 'type': 'availability_indicator'}`
- Aliases: `[]`
- Notes: None

### `derived.evidence_module_coverage`

- ID: `derived.evidence_module_coverage`
- Display name: Evidence Module Coverage
- Description: Fraction of core evidence modules with available same-sample evidence.
- Category: `derived_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `True`
- Range: `0.0` to `1.0`
- Source module: `feature_engineering`
- Model candidate: `True`
- Introduced: `0.9.4`
- Deprecated: `None`
- Dependencies: `['frame_summary.sampled_frame_count', 'temporal_summary.analyzed_transition_count', 'audio_summary.audio_window_count', 'visual_consistency_summary.analyzed_region_record_count', 'd3_summary.d3_status']`
- Transformation: `{'denominator': 5, 'inputs': ['frame_summary.sampled_frame_count', 'temporal_summary.analyzed_transition_count', 'audio_summary.audio_window_count', 'visual_consistency_summary.analyzed_region_record_count', 'd3_summary.d3_status'], 'type': 'availability_count'}`
- Aliases: `[]`
- Notes: None

## Excluded Fields

### `added_at`

- ID: `added_at`
- Display name: Added At
- Description: Exported field `added_at` from the v0.9 feature dataset.
- Category: `timestamp`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is timestamp
- Exclusion reason: category is timestamp

### `audio_summary.audio_window_count`

- ID: `audio_summary.audio_window_count`
- Display name: Audio Summary Audio Window Count
- Description: Exported field `audio_summary.audio_window_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `True`
- Range: `0.0` to `None`
- Source module: `audio_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: missing value ratio > 0.4
- Exclusion reason: not a model candidate

### `audio_summary.clipping_interval_count`

- ID: `audio_summary.clipping_interval_count`
- Display name: Audio Summary Clipping Interval Count
- Description: Exported field `audio_summary.clipping_interval_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `audio_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `audio_summary.maximum_audio_energy_change`

- ID: `audio_summary.maximum_audio_energy_change`
- Display name: Audio Summary Maximum Audio Energy Change
- Description: Exported field `audio_summary.maximum_audio_energy_change` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `audio_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: missing value ratio > 0.4
- Exclusion reason: not a model candidate

### `audio_summary.mean_audio_rms`

- ID: `audio_summary.mean_audio_rms`
- Display name: Audio Summary Mean Audio Rms
- Description: Exported field `audio_summary.mean_audio_rms` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `dimensionless`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `audio_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: missing value ratio > 0.4
- Exclusion reason: not a model candidate

### `audio_summary.silence_interval_count`

- ID: `audio_summary.silence_interval_count`
- Display name: Audio Summary Silence Interval Count
- Description: Exported field `audio_summary.silence_interval_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `audio_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `d3_summary.d3_calibration_status`

- ID: `d3_summary.d3_calibration_status`
- Display name: D3 Summary D3 Calibration Status
- Description: Exported field `d3_summary.d3_calibration_status` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `ratio`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `d3_summary.d3_distance_mode`

- ID: `d3_summary.d3_distance_mode`
- Display name: D3 Summary D3 Distance Mode
- Description: Exported field `d3_summary.d3_distance_mode` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `d3_summary.d3_encoder`

- ID: `d3_summary.d3_encoder`
- Display name: D3 Summary D3 Encoder
- Description: Exported field `d3_summary.d3_encoder` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `d3_summary.d3_first_order_value_count`

- ID: `d3_summary.d3_first_order_value_count`
- Display name: D3 Summary D3 First Order Value Count
- Description: Exported field `d3_summary.d3_first_order_value_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `d3_summary.d3_score_direction`

- ID: `d3_summary.d3_score_direction`
- Display name: D3 Summary D3 Score Direction
- Description: Exported field `d3_summary.d3_score_direction` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `d3_summary.d3_second_order_value_count`

- ID: `d3_summary.d3_second_order_value_count`
- Display name: D3 Summary D3 Second Order Value Count
- Description: Exported field `d3_summary.d3_second_order_value_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `d3_summary.d3_selected_frame_count`

- ID: `d3_summary.d3_selected_frame_count`
- Display name: D3 Summary D3 Selected Frame Count
- Description: Exported field `d3_summary.d3_selected_frame_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `d3_summary.d3_status`

- ID: `d3_summary.d3_status`
- Display name: D3 Summary D3 Status
- Description: Exported field `d3_summary.d3_status` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `learned_detectors.d3`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `feature_file`

- ID: `feature_file`
- Display name: Feature File
- Description: Exported field `feature_file` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `frame_summary.dark_frame_ratio`

- ID: `frame_summary.dark_frame_ratio`
- Display name: Frame Summary Dark Frame Ratio
- Description: Exported field `frame_summary.dark_frame_ratio` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `False`
- Range: `0.0` to `1.0`
- Source module: `frame_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `frame_summary.white_frame_ratio`

- ID: `frame_summary.white_frame_ratio`
- Display name: Frame Summary White Frame Ratio
- Description: Exported field `frame_summary.white_frame_ratio` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `float`
- Units: `ratio`
- Nullable: `False`
- Range: `0.0` to `1.0`
- Source module: `frame_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `generator_or_camera`

- ID: `generator_or_camera`
- Display name: Generator Or Camera
- Description: Exported field `generator_or_camera` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `string`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `identity.analysis_id`

- ID: `identity.analysis_id`
- Display name: Analysis Id
- Description: Exported field `identity.analysis_id` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `identity.application_version`

- ID: `identity.application_version`
- Display name: Application Version
- Description: Exported field `identity.application_version` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `identity.expected_label`

- ID: `identity.expected_label`
- Display name: Expected Label
- Description: Exported field `identity.expected_label` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `identity.feature_schema_version`

- ID: `identity.feature_schema_version`
- Display name: Feature Schema Version
- Description: Exported field `identity.feature_schema_version` from the v0.9 feature dataset.
- Category: `analysis_metadata`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is analysis_metadata
- Exclusion reason: category is analysis_metadata

### `identity.video_name`

- ID: `identity.video_name`
- Display name: Video Name
- Description: Exported field `identity.video_name` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `identity.video_sha256`

- ID: `identity.video_sha256`
- Display name: Video Sha256
- Description: Exported field `identity.video_sha256` from the v0.9 feature dataset.
- Category: `identifier`
- Type: `categorical`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `outcome_identity`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is identifier
- Exclusion reason: category is identifier

### `media.audio_present`

- ID: `media.audio_present`
- Display name: Media Audio Present
- Description: Exported field `media.audio_present` from the v0.9 feature dataset.
- Category: `raw_boolean`
- Type: `boolean`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `metadata_extractor`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: categorical fields are not included until encoding is designed
- Exclusion reason: not a model candidate

### `metadata_indicators.creation_time_present`

- ID: `metadata_indicators.creation_time_present`
- Display name: Metadata Indicators Creation Time Present
- Description: Exported field `metadata_indicators.creation_time_present` from the v0.9 feature dataset.
- Category: `timestamp`
- Type: `boolean`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `metadata_extractor`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is timestamp
- Exclusion reason: category is timestamp

### `metadata_indicators.duration_metadata_consistent`

- ID: `metadata_indicators.duration_metadata_consistent`
- Display name: Metadata Indicators Duration Metadata Consistent
- Description: Exported field `metadata_indicators.duration_metadata_consistent` from the v0.9 feature dataset.
- Category: `raw_boolean`
- Type: `boolean`
- Units: `ratio`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `metadata_extractor`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `metadata_indicators.encoder_metadata_present`

- ID: `metadata_indicators.encoder_metadata_present`
- Display name: Metadata Indicators Encoder Metadata Present
- Description: Exported field `metadata_indicators.encoder_metadata_present` from the v0.9 feature dataset.
- Category: `raw_boolean`
- Type: `boolean`
- Units: `none`
- Nullable: `False`
- Range: `None` to `None`
- Source module: `metadata_extractor`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: categorical fields are not included until encoding is designed
- Exclusion reason: not a model candidate

### `notes`

- ID: `notes`
- Display name: Notes
- Description: Exported field `notes` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `string`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `sample_id`

- ID: `sample_id`
- Display name: Sample Id
- Description: Exported field `sample_id` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `source`

- ID: `source`
- Display name: Source
- Description: Exported field `source` from the v0.9 feature dataset.
- Category: `leakage`
- Type: `categorical`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `dataset_tools`
- Model candidate: `False`
- Introduced: `0.9.1`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is possible_leakage
- Exclusion reason: leakage risk

### `temporal_summary.maximum_normalized_frame_difference`

- ID: `temporal_summary.maximum_normalized_frame_difference`
- Display name: Temporal Summary Maximum Normalized Frame Difference
- Description: Exported field `temporal_summary.maximum_normalized_frame_difference` from the v0.9 feature dataset.
- Category: `unsupported`
- Type: `string`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is fully_empty
- Exclusion reason: category is unsupported

### `temporal_summary.mean_normalized_frame_difference`

- ID: `temporal_summary.mean_normalized_frame_difference`
- Display name: Temporal Summary Mean Normalized Frame Difference
- Description: Exported field `temporal_summary.mean_normalized_frame_difference` from the v0.9 feature dataset.
- Category: `unsupported`
- Type: `string`
- Units: `none`
- Nullable: `True`
- Range: `None` to `None`
- Source module: `temporal_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: semantic category is fully_empty
- Exclusion reason: category is unsupported

### `temporal_summary.notable_transition_count`

- ID: `temporal_summary.notable_transition_count`
- Display name: Temporal Summary Notable Transition Count
- Description: Exported field `temporal_summary.notable_transition_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `temporal_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `temporal_summary.scene_boundary_count`

- ID: `temporal_summary.scene_boundary_count`
- Display name: Temporal Summary Scene Boundary Count
- Description: Exported field `temporal_summary.scene_boundary_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `temporal_analysis`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

### `visual_consistency_summary.ranked_consistency_transition_count`

- ID: `visual_consistency_summary.ranked_consistency_transition_count`
- Display name: Visual Consistency Summary Ranked Consistency Transition Count
- Description: Exported field `visual_consistency_summary.ranked_consistency_transition_count` from the v0.9 feature dataset.
- Category: `raw_numeric`
- Type: `integer`
- Units: `count`
- Nullable: `False`
- Range: `0.0` to `None`
- Source module: `visual_consistency`
- Model candidate: `False`
- Introduced: `0.9.0`
- Deprecated: `None`
- Dependencies: `[]`
- Transformation: `None`
- Aliases: `[]`
- Notes: column has one unique non-missing value
- Exclusion reason: not a model candidate

## Lineage

All v0.9.4 derived features are same-sample, non-fitted, and do not use the target label.

## Compatibility

Description-only changes are non-breaking. Type, unit, derivation, removed-feature, or feature-order changes can require regeneration or be breaking.
