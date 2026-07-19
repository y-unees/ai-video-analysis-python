# Outcome Features

v0.9.0 adds `outcome_features.json`, a stable JSON feature record for future outcome testing.

The feature record exists to support later labeled-dataset work, D3 score-behavior inspection, classifier training experiments, and calibration testing. It is not a user-facing result and it does not contain a probability, threshold, confidence score, real/fake label, manipulation label, or authenticity verdict.

## Feature Groups

- Identity: feature schema version, application version, analysis ID, video name, source SHA-256, and optional manual `expected_label`.
- Media: duration, dimensions, frame rate, frame count, and audio presence from normalized metadata.
- Metadata indicators: creation-time tag presence, encoder metadata presence, and exact duration-metadata consistency from normalized metadata.
- Frame summary: sampled-frame count and aggregate brightness, contrast, sharpness, dark-frame ratio, and white-frame ratio from representative-frame analysis.
- Temporal summary: transition counts, scene-boundary count, near-static interval count, motion magnitude, and flow-warp residual values from temporal analysis summaries.
- Audio summary: audio-window count, ranked transition count, silence interval count, clipping interval count, maximum energy change, and mean RMS from audio analysis.
- Visual-consistency summary: region-record count, ranked transition count, sustained interval count, and maximum regional instability values from visual-consistency summaries.
- Unified evidence summary: timeline-event count, highlighted priority counts, cross-modal highlighted event count, and maximum independent evidence group count from unified evidence review highlights.
- D3 summary: execution status, raw score, encoder, distance mode, selected-frame count, first/second-order counts, second-order summary values, calibration status, and score direction from the D3 result.

Unavailable optional measurements are `null`. Missing values are not replaced with zero unless zero is the actual measured value.

## Manual Labels

`expected_label` defaults to `null`. Later dataset-building tools may assign it manually as one of:

- `real`
- `ai_generated`

The project does not infer labels from folder names, filenames, metadata, D3 output, or analysis findings.

## Feature vs Conclusion

A feature is a measured or summarized value that may later be useful for dataset analysis. A conclusion is an interpreted outcome such as "real", "AI-generated", "manipulated", or "authentic".

v0.9.0 creates features only. It does not create conclusions.

## D3 Semantics

D3 raw score remains an uncalibrated temporal statistic. Its score direction is not verified in this project, and no validated operating threshold exists. The D3 fields in `outcome_features.json` are included for later empirical study, not for immediate classification.

## Next Step

The next appropriate step is collecting manually labeled real and AI-generated videos, generating `outcome_features.json` for each sample, and studying feature behavior before any probability model or calibration policy is introduced.
