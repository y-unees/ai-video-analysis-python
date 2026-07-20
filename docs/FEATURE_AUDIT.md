# Dataset Statistics and Feature Audit

v0.9.3 adds a local-only audit layer for exported dataset features.

The audit exists to answer a narrow question before any model work begins: which exported columns are structurally usable, which are risky or invalid, and what must be fixed before future probability or classifier experiments?

It does not train a classifier, make predictions, calculate probabilities, choose thresholds, calibrate D3, call Gemini, or evaluate model performance.

## Commands

Run the full workflow:

```powershell
python tools\dataset_tool.py audit-features
```

Run with an explicit input:

```powershell
python tools\dataset_tool.py audit-features --input dataset\exports\dataset_features.csv
```

Reusable command aliases:

```powershell
python tools\dataset_tool.py statistics
python tools\dataset_tool.py feature-audit
python tools\dataset_tool.py model-schema
```

All four commands share the same internal audit logic and write the same output set.

## Defaults

Default input:

```text
dataset/exports/dataset_features.csv
```

If the CSV is missing but the dataset manifest exists, the command attempts to regenerate the CSV export using the existing dataset toolkit.

Default output directory:

```text
dataset/statistics/
```

Default thresholds:

- Missing-heavy: more than `40%` missing.
- Near-constant: dominant value is at least `95%` of non-missing values.
- High correlation: absolute Pearson correlation at least `0.95`.

## Outputs

```text
dataset/statistics/
├── dataset_profile.json
├── column_profile.csv
├── feature_statistics.csv
├── class_comparison.csv
├── missing_values.csv
├── invalid_values.csv
├── constant_features.csv
├── correlation_matrix.csv
├── high_correlation_pairs.csv
├── leakage_report.json
├── feature_quality.json
├── model_feature_schema.json
└── statistics_report.txt
```

JSON outputs are UTF-8, human-readable, and written without non-standard `NaN` or infinity tokens. CSV outputs use deterministic columns and row ordering where practical.

## Column Classification

Every column receives one semantic category:

- `target`
- `numeric_feature`
- `categorical_feature`
- `identifier`
- `source_metadata`
- `analysis_metadata`
- `path`
- `timestamp`
- `possible_leakage`
- `unsupported`
- `fully_empty`

The audit uses both column names and observed values. It does not include a column in the model schema simply because it is numeric.

## Leakage Prevention

The future model schema excludes target labels, duplicated label fields, sample IDs, hashes, filenames, paths, feature-file references, source fields, generator/camera notes, and values that expose class-bearing conventions such as `real__` or `ai__`.

The target column is expected and is classified as `target`; duplicated labels or class-bearing metadata are treated as leakage risks.

## Statistics

For eligible numeric features, the audit computes:

- descriptive statistics across all samples
- descriptive statistics by class
- exploratory real-versus-AI differences
- standardized effect-size estimates where valid
- point-biserial target correlation where valid
- Mann-Whitney U normal approximation where valid
- numeric-feature correlation matrix
- high-correlation pair warnings

These statistics are exploratory. They are not model performance metrics and must not be interpreted as proof that any feature is predictive.

## Model Readiness

Readiness values include:

- `not_ready`
- `insufficient_dataset_size`
- `requires_cleanup`
- `audit_passed_for_experimentation`
- `ready_for_baseline_experiment`

The current 12-video pilot dataset is expected to receive `insufficient_dataset_size`. It can validate the workflow, but it is not large enough for reliable generalization.

## Optional Plots

The CLI accepts `--plots`, but plot generation is not part of the v0.9.3 core audit. The command still generates textual and tabular outputs when plots are unavailable.
