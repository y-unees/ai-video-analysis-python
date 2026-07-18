# Gemini-Ready Compact Evidence Report

v0.8.2 adds `gemini_evidence_report.json`, a compact derivative artifact prepared for a future Gemini-assisted interpretation stage.

No Gemini API call is made in v0.8.2. No SDK, API key handling, upload, network request, evidence fusion, probability, threshold, real/fake classification, or authenticity verdict is added.

## Purpose

The compact report is the recommended single JSON input for a future external-model interpretation pass. It avoids sending the complete verbose forensic report by default.

The local sources of truth remain:

- `report.json`
- `report.txt`
- raw JSONL metric artifacts
- review frame artifacts
- `unified_evidence.json`
- `evidence_timeline.jsonl`
- optional D3 artifacts

`gemini_evidence_report.json` is a derivative summary, not the authoritative forensic record.

## Structure

Top-level sections:

- `schema_version`
- `analysis_identity`
- `media_summary`
- `deterministic_summary`
- `key_review_events`
- `d3_summary`
- `limitations`
- `gemini_instructions`
- `future_gemini_output_contract`
- `source_artifacts`

The compact report excludes raw ffprobe JSON, complete frame arrays, complete transition arrays, audio-window arrays, regional grids, environment dumps, and absolute paths.

## Event Selection

Key review events default to a maximum of 5. Selection is deterministic and prioritizes:

- higher review priority
- more independent evidence groups
- audio/visual temporal overlap
- multiple visual methods
- ranked source findings
- stronger within-video ranking
- timestamp and event ID tie-breakers

Ranking is for human review only. It is not proof of manipulation, authenticity, or AI generation.

## Finding Deduplication

Findings are collapsed when they share the same source record, evidence domain, observation type, summary, and metric keys. Independent evidence from different domains is preserved.

Each selected event is capped by `GEMINI_COMPACT_MAX_FINDINGS_PER_EVENT`.

## Artifact References

Each selected event includes only a small list of useful relative artifact references. Duplicate paths are removed. Preferred review artifacts include before/after frames, absolute-difference images, flow-warp residual images, combined heatmaps, and detail-residual heatmaps when available.

## D3 Restrictions

The D3 section remains standalone and uncalibrated:

- `score_direction` remains `not_verified`
- `calibration_status` remains `uncalibrated`
- `classification` remains `not_assigned`
- no probability is created
- no threshold is created
- no real/fake verdict is created

The D3 raw score is described only as an uncalibrated temporal statistic.

## Size Limits

Default limits:

- preferred size: 8,000 bytes
- acceptable size: 12,000 bytes
- hard size limit: 16,000 bytes

If the preferred size is exceeded, the builder reduces content in this order:

1. artifact references
2. findings per event
3. selected event count
4. repeated summaries
5. optional metrics

JSON is never truncated as a string.

## Configuration

Environment variables:

- `GEMINI_COMPACT_REPORT_ENABLED`: default `true`
- `GEMINI_COMPACT_MAX_KEY_EVENTS`: default `5`
- `GEMINI_COMPACT_MAX_FINDINGS_PER_EVENT`: default `5`
- `GEMINI_COMPACT_MAX_ARTIFACTS_PER_EVENT`: default `4`
- `GEMINI_COMPACT_PREFERRED_SIZE_BYTES`: default `8000`
- `GEMINI_COMPACT_ACCEPTABLE_SIZE_BYTES`: default `12000`
- `GEMINI_COMPACT_HARD_SIZE_BYTES`: default `16000`

Invalid positive-integer values fall back through the existing environment configuration helpers.

## Failure Behavior

Compact-report generation failure does not fail the main forensic analysis. The main report records a structured failed artifact reference, removes any temporary partial compact file, prints a concise warning, and continues writing `report.json` and `report.txt`.

## Future Gemini Output

The future response contract is prepared at `schemas/gemini_interpretation_response.schema.json`. It supports concise terminal display fields such as summary, major findings, important timestamps, uncertainty, limitations, human-review prompts, and a model disclaimer.

## Privacy

v0.8.2 remains local-only. A later Gemini integration would send report content to an external model, so users should treat compact reports and referenced artifacts as potentially sensitive.
