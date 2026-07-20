from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_tools import (
    check_feature_compatibility,
    plan_faceforensics_samples,
    export_dataset_csv,
    export_dataset_jsonl,
    generate_feature_docs,
    generate_feature_schema,
    load_feature_registry,
    print_audit_summary,
    print_preparation_summary,
    register_sample,
    run_feature_audit,
    run_feature_preparation,
    summarize_dataset,
    validate_feature_registry,
    validate_plan,
    validate_dataset,
    write_faceforensics_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset preparation toolkit for outcome feature artifacts.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root directory. Defaults to dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="Register one outcome_features.json sample.")
    register.add_argument("feature_path", help="Path to an outcome_features.json artifact.")
    register.add_argument("expected_label", choices=["real", "ai_generated"], help="Manually assigned label.")
    register.add_argument("--source", default=None, help="Optional source description.")
    register.add_argument("--generator-or-camera", default=None, help="Optional generator or camera description.")
    register.add_argument("--notes", default=None, help="Optional notes.")
    register.add_argument("--allow-duplicate", action="store_true", help="Allow duplicate source video SHA-256 values.")

    subparsers.add_parser("validate", help="Validate dataset manifest and feature files.")

    export_jsonl = subparsers.add_parser("export-jsonl", help="Export flattened features to JSONL.")
    export_jsonl.add_argument("--output", default=None, help="Optional output path.")

    export_csv = subparsers.add_parser("export-csv", help="Export flattened features to CSV.")
    export_csv.add_argument("--output", default=None, help="Optional output path.")

    subparsers.add_parser("summary", help="Print dataset summary counts.")

    audit_source = subparsers.add_parser("audit-source", help="Audit a supported source metadata CSV.")
    audit_source.add_argument("--dataset", required=True, choices=["faceforensics_pp"])
    audit_source.add_argument("--metadata", required=True)

    plan_source = subparsers.add_parser("plan-source", help="Create a balanced pilot plan from source metadata.")
    plan_source.add_argument("--dataset", required=True, choices=["faceforensics_pp"])
    plan_source.add_argument("--metadata", required=True)
    plan_source.add_argument("--real-count", type=int, default=10)
    plan_source.add_argument("--ai-count", type=int, default=10)
    plan_source.add_argument("--seed", type=int, default=42)

    validate_plan_parser = subparsers.add_parser("validate-plan", help="Validate a source pilot plan JSONL.")
    validate_plan_parser.add_argument("plan_path")
    validate_plan_parser.add_argument("--real-count", type=int, default=None)
    validate_plan_parser.add_argument("--ai-count", type=int, default=None)

    def add_audit_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--input", default=None, help="Feature export path. Defaults to dataset/exports/dataset_features.csv.")
        command.add_argument("--output-dir", default="dataset/statistics", help="Directory for audit outputs.")
        command.add_argument("--missing-threshold", type=float, default=0.40, help="Missing-heavy threshold. Defaults to 0.40.")
        command.add_argument("--near-constant-threshold", type=float, default=0.95, help="Near-constant dominant-value threshold.")
        command.add_argument("--correlation-threshold", type=float, default=0.95, help="High-correlation absolute threshold.")
        command.add_argument("--plots", action="store_true", help="Request optional plots. Core audit still succeeds if plots are unavailable.")
        command.add_argument("--strict", action="store_true", help="Reserved for stricter future fatal checks.")
        command.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing audit output files.")

    statistics_parser = subparsers.add_parser("statistics", help="Generate dataset statistics outputs.")
    add_audit_options(statistics_parser)

    feature_audit_parser = subparsers.add_parser("feature-audit", help="Generate feature quality and leakage audit outputs.")
    add_audit_options(feature_audit_parser)

    model_schema_parser = subparsers.add_parser("model-schema", help="Generate the future model feature schema.")
    add_audit_options(model_schema_parser)

    audit_features_parser = subparsers.add_parser("audit-features", help="Run the full dataset statistics and feature audit workflow.")
    add_audit_options(audit_features_parser)

    def add_prepare_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--input", default=None, help="Raw feature export path. Defaults to dataset/exports/dataset_features.csv.")
        command.add_argument("--output-dir", default="dataset/standardized", help="Directory for standardized outputs.")
        command.add_argument("--registry", default="schemas/feature_registry.json", help="Feature registry path.")
        command.add_argument("--schema", default="schemas/feature_schema.json", help="Feature schema path.")
        command.add_argument("--strict", action="store_true", help="Fail on strict validation findings.")
        command.add_argument("--overwrite", action="store_true", default=True, help="Overwrite generated outputs.")
        command.add_argument("--allow-unregistered", action="store_true", help="Allow unregistered exported fields as warnings.")
        command.add_argument("--report-only", action="store_true", help="Run validation/preparation logic without writing standardized outputs.")

    feature_registry_parser = subparsers.add_parser("feature-registry", help="Validate the canonical feature registry.")
    feature_registry_parser.add_argument("--registry", default="schemas/feature_registry.json")

    validate_features_parser = subparsers.add_parser("validate-features", help="Validate raw exported feature values against the registry.")
    add_prepare_options(validate_features_parser)

    engineer_features_parser = subparsers.add_parser("engineer-features", help="Calculate approved same-sample derived features.")
    add_prepare_options(engineer_features_parser)

    standardize_features_parser = subparsers.add_parser("standardize-features", help="Write standardized feature CSV and JSONL outputs.")
    add_prepare_options(standardize_features_parser)

    compatibility_parser = subparsers.add_parser("check-feature-compatibility", help="Check standardized dataset or schema compatibility.")
    compatibility_parser.add_argument("--dataset", default=None)
    compatibility_parser.add_argument("--old-schema", default=None)
    compatibility_parser.add_argument("--new-schema", default="schemas/feature_schema.json")

    generate_docs_parser = subparsers.add_parser("generate-feature-docs", help="Generate feature documentation from registry and schema.")
    generate_docs_parser.add_argument("--registry", default="schemas/feature_registry.json")
    generate_docs_parser.add_argument("--schema", default="schemas/feature_schema.json")
    generate_docs_parser.add_argument("--output", default="docs/FEATURES.md")

    prepare_features_parser = subparsers.add_parser("prepare-features", help="Run the full v0.9.4 feature preparation workflow.")
    add_prepare_options(prepare_features_parser)

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    try:
        if args.command == "register":
            entry = register_sample(
                feature_path=Path(args.feature_path),
                expected_label=args.expected_label,
                dataset_root=dataset_root,
                source=args.source,
                generator_or_camera=args.generator_or_camera,
                notes=args.notes,
                allow_duplicate=args.allow_duplicate,
            )
            print(f"Registered sample: {entry.sample_id} ({entry.expected_label})")
            return 0
        if args.command == "validate":
            result = validate_dataset(dataset_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "valid" else 1
        if args.command == "export-jsonl":
            path = export_dataset_jsonl(dataset_root, Path(args.output) if args.output else None)
            print(f"JSONL export written: {path}")
            return 0
        if args.command == "export-csv":
            path = export_dataset_csv(dataset_root, Path(args.output) if args.output else None)
            print(f"CSV export written: {path}")
            return 0
        if args.command == "summary":
            summary = summarize_dataset(dataset_root)
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "audit-source":
            audit = write_faceforensics_audit(Path(args.metadata))
            print(
                f"Audit written: {audit['audit_report_path']} "
                f"(rows={audit['row_count']}, columns={len(audit['column_names'])})"
            )
            print(f"Video availability: {audit['video_availability']}")
            return 0
        if args.command == "plan-source":
            result = plan_faceforensics_samples(
                metadata_path=Path(args.metadata),
                real_count=args.real_count,
                ai_count=args.ai_count,
                seed=args.seed,
            )
            print(
                f"Plan written: {result['plan_path']} "
                f"(real={result['selected_real_count']}/{result['requested_real_count']}, "
                f"ai_generated={result['selected_ai_generated_count']}/{result['requested_ai_generated_count']})"
            )
            return 0
        if args.command == "validate-plan":
            result = validate_plan(Path(args.plan_path), requested_real_count=args.real_count, requested_ai_count=args.ai_count)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "valid" else 1
        if args.command in {"statistics", "feature-audit", "model-schema", "audit-features"}:
            audit = run_feature_audit(
                input_path=Path(args.input) if args.input else None,
                output_dir=Path(args.output_dir),
                dataset_root=dataset_root,
                missing_threshold=args.missing_threshold,
                near_constant_threshold=args.near_constant_threshold,
                correlation_threshold=args.correlation_threshold,
                include_plots=args.plots,
                strict=args.strict,
                overwrite=args.overwrite,
            )
            if args.command == "model-schema":
                print(f"Model schema written: {Path(args.output_dir) / 'model_feature_schema.json'}")
            elif args.command == "statistics":
                print(f"Statistics written: {Path(args.output_dir)}")
            elif args.command == "feature-audit":
                print(f"Feature audit written: {Path(args.output_dir)}")
            else:
                print_audit_summary(audit)
            return 0
        if args.command == "feature-registry":
            registry = load_feature_registry(Path(args.registry))
            result = validate_feature_registry(registry)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "valid" else 1
        if args.command in {"validate-features", "engineer-features", "standardize-features", "prepare-features"}:
            result = run_feature_preparation(
                input_path=Path(args.input) if args.input else None,
                output_dir=Path(args.output_dir),
                registry_path=Path(args.registry),
                schema_path=Path(args.schema),
                strict=args.strict,
                overwrite=args.overwrite,
                allow_unregistered=args.allow_unregistered,
                report_only=args.report_only,
            )
            if args.command == "validate-features":
                print(
                    f"Feature validation complete: errors={result['validation_error_count']}, "
                    f"warnings={result['validation_warning_count']}, unregistered={result['unregistered_field_count']}"
                )
            elif args.command == "engineer-features":
                print(
                    f"Feature engineering complete: derived={result['derived_candidate_count']}, "
                    f"missing_dependencies={result['missing_dependency_count']}"
                )
            elif args.command == "standardize-features":
                print(f"Standardized features written: {result['output_paths'].get('standardized_csv')}")
            else:
                print_preparation_summary(result)
            return 0 if result["validation_error_count"] == 0 or not args.strict else 1
        if args.command == "check-feature-compatibility":
            result = check_feature_compatibility(
                old_schema_path=Path(args.old_schema) if args.old_schema else None,
                new_schema_path=Path(args.new_schema) if args.new_schema else None,
                dataset_path=Path(args.dataset) if args.dataset else None,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] in {"compatible", "compatible_with_warnings", "unknown"} else 1
        if args.command == "generate-feature-docs":
            registry = load_feature_registry(Path(args.registry))
            schema = generate_feature_schema(registry, Path(args.schema))
            path = generate_feature_docs(registry, schema, Path(args.output))
            print(f"Feature documentation written: {path}")
            return 0
    except Exception as error:
        print(f"Dataset tool error: {error}")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
