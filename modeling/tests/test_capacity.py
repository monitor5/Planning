from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from elder_guardian.capacity import (  # noqa: E402
    MEDIAN_SCENARIO,
    compare_capacity_models,
    evaluate_capacity_relationship,
    load_capacity_artifact,
    prepare_capacity_training_data,
    save_capacity_artifact,
)


def _strong_relationship_frame() -> pd.DataFrame:
    rows = []
    for group_number in range(6):
        for step in range(18):
            area = 60.0 + step * 22.0 + group_number * 0.15
            capacity = 8.0 + area * 0.31 + np.sin(step + group_number) * 0.35
            rows.append(
                {
                    "borough": f"borough-{group_number}",
                    "area_m2": area,
                    "capacity": capacity,
                }
            )
    return pd.DataFrame(rows)


class CapacityModelTests(unittest.TestCase):
    def test_strong_relationship_selects_and_serializes_guarded_challenger(self) -> None:
        frame = _strong_relationship_frame()
        result = compare_capacity_models(
            frame,
            trusted_for_deployment=True,
            min_samples=40,
            min_groups=4,
        )

        self.assertNotEqual(result.selected_model, MEDIAN_SCENARIO)
        self.assertTrue(result.deployable)
        self.assertEqual(result.metrics["cv"]["scheme"], "GroupKFold")
        self.assertTrue(
            result.metrics["cv"]["all_preprocessing_fit_within_fold"]
        )
        midpoint = float(frame["area_m2"].median())
        self.assertGreaterEqual(result.artifact.predict(midpoint), 0.0)
        with self.assertRaisesRegex(ValueError, "extrapolation is not allowed"):
            result.artifact.predict(float(frame["area_m2"].max()) + 1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "capacity.joblib"
            save_capacity_artifact(result, path, require_deployable=True)
            loaded = load_capacity_artifact(path)
            self.assertEqual(loaded.model_name, result.selected_model)
            self.assertAlmostEqual(
                loaded.predict(midpoint),
                result.artifact.predict(midpoint),
            )

    def test_low_correlation_explicitly_selects_median(self) -> None:
        random = np.random.default_rng(42)
        count = 120
        frame = pd.DataFrame(
            {
                "borough": [f"borough-{index % 6}" for index in range(count)],
                "area_m2": np.arange(1, count + 1, dtype=float),
                "capacity": random.permutation(
                    np.linspace(10.0, 100.0, count)
                ),
            }
        )
        result = compare_capacity_models(
            frame,
            trusted_for_deployment=True,
            min_samples=40,
            min_spearman=0.90,
        )

        self.assertEqual(result.selected_model, MEDIAN_SCENARIO)
        self.assertFalse(result.deployable)
        self.assertIn("insufficient_positive_correlation", result.reason)
        self.assertEqual(result.metrics["cv"]["status"], "completed")

    def test_insufficient_sample_selects_median_even_for_strong_pattern(self) -> None:
        frame = pd.DataFrame(
            {
                "borough": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "area_m2": np.arange(10.0, 90.0, 10.0),
                "capacity": np.arange(5.0, 45.0, 5.0),
            }
        )
        result = compare_capacity_models(
            frame,
            trusted_for_deployment=True,
            min_samples=20,
        )

        self.assertEqual(result.selected_model, MEDIAN_SCENARIO)
        self.assertFalse(result.deployable)
        self.assertIn("insufficient_samples", result.reason)

    def test_generated_or_proxy_target_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "borough": ["a", "b"],
                "area_m2": [100.0, 200.0],
                "estimated_capacity": [30.0, 60.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "generated/proxy label"):
            compare_capacity_models(
                frame,
                target_col="estimated_capacity",
                min_samples=2,
                min_groups=2,
            )

    def test_private_schema_join_uses_historical_target_not_legacy_proxy(self) -> None:
        master = pd.DataFrame(
            {
                "source_record_id": ["a", "b", "c", "d", "e"],
                "borough": ["x", "x", "y", "y", "z"],
                "legacy_usable_area_m2": [100.0, 200.0, np.nan, -1.0, 500.0],
                "p0_service_eligible": [True, True, True, True, False],
                "legacy_capacity_proxy": [9999, 9999, 9999, 9999, 9999],
                "estimated_capacity": [8888, 8888, 8888, 8888, 8888],
            }
        )
        historical = pd.DataFrame(
            {
                "source_record_id": ["a", "b", "c", "d", "e"],
                "borough": ["x", "x", "y", "y", "z"],
                "capacity_registered_historical_candidate": [
                    10.0,
                    0.0,
                    30.0,
                    40.0,
                    50.0,
                ],
                "requires_current_register_verification": [True] * 5,
                "may_use_as_operational_capacity": [False] * 5,
            }
        )

        prepared = prepare_capacity_training_data(master, historical)

        self.assertEqual(prepared["source_record_id"].tolist(), ["a"])
        self.assertEqual(prepared["capacity"].tolist(), [10.0])
        self.assertNotIn("legacy_capacity_proxy", prepared.columns)
        self.assertNotIn("estimated_capacity", prepared.columns)
        self.assertFalse(
            prepared.attrs["preparation"]["generated_capacity_labels_used"]
        )

    def test_csv_workflow_remains_non_deployable_for_unverified_lineage(self) -> None:
        strong = _strong_relationship_frame()
        identifiers = [f"id-{index}" for index in range(len(strong))]
        master = pd.DataFrame(
            {
                "source_record_id": identifiers,
                "borough": strong["borough"],
                "legacy_usable_area_m2": strong["area_m2"],
                "p0_service_eligible": True,
                "legacy_capacity_proxy": 9999.0,
            }
        )
        historical = pd.DataFrame(
            {
                "source_record_id": identifiers,
                "borough": strong["borough"],
                "capacity_registered_historical_candidate": strong["capacity"],
                "requires_current_register_verification": True,
                "may_use_as_operational_capacity": False,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            master_path = Path(temporary_directory) / "master.csv"
            historical_path = Path(temporary_directory) / "historical.csv"
            master.to_csv(master_path, index=False)
            historical.to_csv(historical_path, index=False)
            result = evaluate_capacity_relationship(
                master_path,
                historical_path,
                min_samples=40,
            )

        self.assertNotEqual(result.selected_model, MEDIAN_SCENARIO)
        self.assertFalse(result.deployable)
        self.assertIn("diagnostic_only", result.reason)
        self.assertFalse(result.metadata["source_workflow_deployable"])

    def test_prepared_historical_lineage_cannot_be_marked_trusted(self) -> None:
        strong = _strong_relationship_frame()
        identifiers = [f"id-{index}" for index in range(len(strong))]
        master = pd.DataFrame(
            {
                "source_record_id": identifiers,
                "borough": strong["borough"],
                "legacy_usable_area_m2": strong["area_m2"],
                "p0_service_eligible": True,
            }
        )
        historical = pd.DataFrame(
            {
                "source_record_id": identifiers,
                "borough": strong["borough"],
                "capacity_registered_historical_candidate": strong["capacity"],
                "requires_current_register_verification": True,
                "may_use_as_operational_capacity": False,
            }
        )
        prepared = prepare_capacity_training_data(master, historical)

        result = compare_capacity_models(
            prepared,
            trusted_for_deployment=True,
            min_samples=40,
        )

        self.assertNotEqual(result.selected_model, MEDIAN_SCENARIO)
        self.assertFalse(result.deployable)
        self.assertTrue(result.metadata["historical_legacy_lineage"])
        self.assertFalse(result.metadata["trusted_for_deployment"])


if __name__ == "__main__":
    unittest.main()
