from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_tools.feature_preparation import (
    check_feature_compatibility,
    feature_lineage,
    generate_feature_schema,
    load_feature_registry,
    run_feature_preparation,
    schema_fingerprint,
    validate_feature_registry,
    validate_feature_values,
)
from tools import dataset_tool


class FeaturePreparationTests(unittest.TestCase):
    def test_valid_registry_and_duplicate_failures(self) -> None:
        registry = _registry()
        self.assertEqual(validate_feature_registry(registry)["status"], "valid")
        duplicate_id = _registry()
        duplicate_id["features"].append(dict(duplicate_id["features"][0]))
        self.assertEqual(validate_feature_registry(duplicate_id)["status"], "invalid")
        duplicate_alias = _registry()
        duplicate_alias["features"][0]["aliases"] = ["old_name"]
        duplicate_alias["features"][1]["aliases"] = ["old_name"]
        self.assertEqual(validate_feature_registry(duplicate_alias)["status"], "invalid")

    def test_registry_validation_rules(self) -> None:
        cases = []
        missing = _registry()
        del missing["features"][0]["data_type"]
        cases.append(missing)
        bad_type = _registry()
        bad_type["features"][0]["data_type"] = "decimal"
        cases.append(bad_type)
        bad_range = _registry()
        bad_range["features"][0]["minimum"] = 10
        bad_range["features"][0]["maximum"] = 1
        cases.append(bad_range)
        leakage_candidate = _registry()
        leakage_candidate["features"][1]["model_candidate"] = True
        cases.append(leakage_candidate)
        unknown_dependency = _registry()
        unknown_dependency["features"][-1]["dependencies"] = ["missing.feature"]
        cases.append(unknown_dependency)
        circular = _registry()
        circular["features"][-1]["dependencies"] = ["derived.ratio"]
        cases.append(circular)
        for registry in cases:
            self.assertEqual(validate_feature_registry(registry)["status"], "invalid")

    def test_value_validation_strict_and_default(self) -> None:
        registry = _registry()
        rows = [{"sample_id": "a", "expected_label": "real", "raw.a": "2", "raw.b": "0", "unknown": "x"}]
        validation = validate_feature_values(rows, ["sample_id", "expected_label", "raw.a", "raw.b", "unknown"], registry, strict=False)
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(validation["unregistered_fields"], ["unknown"])
        strict = validate_feature_values(rows, ["sample_id", "expected_label", "raw.a", "raw.b", "unknown"], registry, strict=True)
        self.assertEqual(strict["status"], "invalid")
        invalid = validate_feature_values([{"sample_id": "a", "expected_label": "real", "raw.a": "nan", "raw.b": "1"}], ["sample_id", "expected_label", "raw.a", "raw.b"], registry)
        self.assertEqual(invalid["invalid_type_count"], 1)
        out_of_range = validate_feature_values([{"sample_id": "a", "expected_label": "real", "raw.a": "-1", "raw.b": "1"}], ["sample_id", "expected_label", "raw.a", "raw.b"], registry)
        self.assertEqual(out_of_range["out_of_range_count"], 1)

    def test_prepare_features_outputs_standardized_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(_registry(), indent=2, sort_keys=True), encoding="utf-8")
            input_path = _dataset(root / "features.csv")
            output_dir = root / "standardized"
            schema_path = root / "schema.json"
            result = run_feature_preparation(input_path=input_path, output_dir=output_dir, registry_path=registry_path, schema_path=schema_path)

            self.assertEqual(result["validation_error_count"], 0)
            self.assertEqual(result["raw_candidate_count"], 2)
            self.assertEqual(result["derived_candidate_count"], 1)
            self.assertEqual(result["standardized_feature_count"], 3)
            self.assertTrue((output_dir / "standardized_features.csv").exists())
            self.assertTrue((output_dir / "standardized_features.jsonl").exists())
            with (output_dir / "standardized_features.csv").open("r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(list(rows[0])[:4], ["sample_id", "expected_label", "feature_schema_version", "schema_fingerprint"])
            self.assertIn("derived.ratio", rows[0])
            payload = json.loads((output_dir / "standardized_features.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["features"]["derived.ratio"], 2.0)
            self.assertNotIn("expected_label", payload["features"])
            self.assertNotIn("sample_id", payload["features"])
            self.assertNotIn("NaN", (output_dir / "standardized_features.jsonl").read_text(encoding="utf-8"))

    def test_zero_denominator_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(_registry(), indent=2, sort_keys=True), encoding="utf-8")
            input_path = _dataset(root / "features.csv", denominator="0")
            result = run_feature_preparation(input_path=input_path, output_dir=root / "out", registry_path=registry_path, schema_path=root / "schema.json")
            row = json.loads((root / "out" / "standardized_features.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["features"]["derived.ratio"])
            lineage = feature_lineage(load_feature_registry(registry_path))
            self.assertFalse(lineage["features"][0]["uses_target"])
            self.assertFalse(lineage["features"][0]["fitted"])
            self.assertEqual(result["engineering_report"]["missing_dependency_count"], 0)

    def test_fingerprint_and_compatibility(self) -> None:
        registry = _registry()
        schema = generate_feature_schema(registry)
        self.assertEqual(schema["schema_fingerprint"], schema_fingerprint(schema))
        same = check_feature_compatibility(new_schema=schema)
        self.assertEqual(same["status"], "compatible")
        changed = json.loads(json.dumps(schema))
        changed["feature_order"] = list(reversed(changed["feature_order"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            old.write_text(json.dumps(schema), encoding="utf-8")
            new = root / "new.json"
            new.write_text(json.dumps(changed), encoding="utf-8")
            result = check_feature_compatibility(old_schema_path=old, new_schema_path=new)
            self.assertIn(result["status"], {"requires_regeneration", "incompatible"})

    def test_cli_commands_and_strict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(_registry(), indent=2, sort_keys=True), encoding="utf-8")
            input_path = _dataset(root / "features.csv")
            output_dir = root / "out"
            schema_path = root / "schema.json"
            for command in ("feature-registry", "validate-features", "engineer-features", "standardize-features", "check-feature-compatibility", "generate-feature-docs", "prepare-features"):
                argv = ["dataset_tool.py", command]
                if command == "feature-registry":
                    argv.extend(["--registry", str(registry_path)])
                elif command == "check-feature-compatibility":
                    argv.extend(["--new-schema", str(schema_path)])
                elif command == "generate-feature-docs":
                    argv.extend(["--registry", str(registry_path), "--schema", str(schema_path), "--output", str(root / "FEATURES.md")])
                else:
                    argv.extend(["--input", str(input_path), "--output-dir", str(output_dir), "--registry", str(registry_path), "--schema", str(schema_path)])
                with patch("sys.argv", argv):
                    self.assertEqual(dataset_tool.main(), 0, command)
            with patch("sys.argv", ["dataset_tool.py", "validate-features", "--input", str(input_path), "--output-dir", str(output_dir), "--registry", str(registry_path), "--schema", str(schema_path), "--strict"]):
                self.assertEqual(dataset_tool.main(), 0)
            broken_registry = _registry()
            broken_registry["features"][1]["model_candidate"] = True
            registry_path.write_text(json.dumps(broken_registry), encoding="utf-8")
            with patch("sys.argv", ["dataset_tool.py", "feature-registry", "--registry", str(registry_path)]):
                self.assertEqual(dataset_tool.main(), 1)


def _feature(feature_id: str, category: str, data_type: str, model_candidate: bool, **extra):
    base = {
        "id": feature_id,
        "export_name": feature_id,
        "display_name": feature_id,
        "description": feature_id,
        "category": category,
        "source_module": "test",
        "data_type": data_type,
        "units": "dimensionless",
        "nullable": False,
        "required": False,
        "minimum": None,
        "maximum": None,
        "allowed_values": None,
        "model_candidate": model_candidate,
        "leakage_risk": category == "leakage",
        "introduced_version": "0.9.4",
        "deprecated_version": None,
        "derived": False,
        "dependencies": [],
        "transformation": None,
        "aliases": [],
        "notes": None,
    }
    base.update(extra)
    return base


def _registry():
    return {
        "project_version": "0.9.4",
        "registry_version": "0.9.4",
        "features": [
            _feature("raw.a", "raw_numeric", "float", True, minimum=0.0, maximum=10.0),
            _feature("expected_label", "target", "categorical", False, allowed_values=["real", "ai_generated"], required=True),
            _feature("sample_id", "identifier", "string", False, required=True),
            _feature("raw.b", "raw_numeric", "float", True, minimum=0.0),
            _feature(
                "derived.ratio",
                "derived_numeric",
                "float",
                True,
                nullable=True,
                minimum=0.0,
                derived=True,
                dependencies=["raw.a", "raw.b"],
                transformation={"type": "safe_ratio", "numerator": "raw.a", "denominator": "raw.b", "zero_denominator": "null"},
            ),
        ],
    }


def _dataset(path: Path, denominator: str = "1") -> Path:
    rows = [
        {"sample_id": "real-1", "expected_label": "real", "raw.a": "2", "raw.b": denominator},
        {"sample_id": "ai-1", "expected_label": "ai_generated", "raw.a": "4", "raw.b": "2"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["expected_label", "raw.a", "raw.b", "sample_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    unittest.main()
