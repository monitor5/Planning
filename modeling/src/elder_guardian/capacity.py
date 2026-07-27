"""Leakage-resistant capacity/area relationship diagnostics.

This module deliberately treats historical registered capacity and legacy usable
area as diagnostic inputs.  It never derives a label from area and explicitly
rejects known proxy/estimated-capacity label columns.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MEDIAN_SCENARIO = "median_scenario"
NONNEGATIVE_LINEAR = "nonnegative_linear"
HUBER = "huber"
RANDOM_FOREST = "random_forest"

# These are generated estimates or area-derived proxies, not observed labels.
PROHIBITED_TARGET_COLUMNS = frozenset(
    {
        "estimated_capacity",
        "legacy_capacity_proxy",
        "capacity_area_fire_proxy",
        "capacity_legal_nominal",
        "capacity_strict_unknown",
    }
)


@dataclass
class CapacityModelArtifact:
    """Serializable fitted model with a strict no-extrapolation boundary."""

    estimator: Any
    model_name: str
    feature_name: str
    area_min_m2: float
    area_max_m2: float
    deployable: bool
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    def predict(self, area_m2: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        """Predict only within the observed training-area interval.

        The method raises instead of clipping or extrapolating.  Predictions are
        floored at zero, but are intentionally not rounded to integer people so
        downstream scenario code must make its rounding policy explicit.
        """

        values = np.asarray(area_m2, dtype=float)
        scalar_input = values.ndim == 0
        flat = values.reshape(-1)
        if flat.size == 0:
            return np.asarray([], dtype=float)
        if not np.isfinite(flat).all():
            raise ValueError("area_m2 must contain only finite numeric values")
        if (flat <= 0).any():
            raise ValueError("area_m2 must be strictly positive")

        tolerance = max(
            1e-9,
            abs(self.area_max_m2 - self.area_min_m2) * 1e-12,
        )
        below = flat < self.area_min_m2 - tolerance
        above = flat > self.area_max_m2 + tolerance
        if below.any() or above.any():
            raise ValueError(
                "area_m2 is outside the observed training range "
                f"[{self.area_min_m2:.6g}, {self.area_max_m2:.6g}]; "
                "capacity extrapolation is not allowed"
            )

        predictions = np.asarray(
            self.estimator.predict(flat.reshape(-1, 1)),
            dtype=float,
        ).reshape(-1)
        predictions = np.maximum(predictions, 0.0)
        if scalar_input:
            return float(predictions[0])
        return predictions


@dataclass
class CapacityComparisonResult:
    """Model comparison output with an artifact and auditable dictionaries."""

    selected_model: str
    deployable: bool
    reason: str
    artifact: CapacityModelArtifact
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result summary without embedding the estimator."""

        return _json_compatible(
            {
                "selected_model": self.selected_model,
                "deployable": self.deployable,
                "reason": self.reason,
                "metrics": self.metrics,
                "metadata": self.metadata,
            }
        )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    return mapped.astype("boolean")


def _validate_target_name(target_col: str) -> None:
    normalized = target_col.strip().lower()
    if (
        normalized in PROHIBITED_TARGET_COLUMNS
        or "estimated_capacity" in normalized
        or normalized.endswith("_proxy")
    ):
        raise ValueError(
            f"{target_col!r} is a generated/proxy label and cannot be used as "
            "a capacity target"
        )


def prepare_capacity_training_data(
    facility_master: pd.DataFrame,
    historical_candidates: pd.DataFrame,
    *,
    require_p0_eligible: bool = True,
) -> pd.DataFrame:
    """Join the private processed schemas without constructing a synthetic label.

    Only positive, finite ``legacy_usable_area_m2`` and positive, finite
    ``capacity_registered_historical_candidate`` rows are returned.  The
    generated ``legacy_capacity_proxy`` column is not selected or consulted.
    """

    master_required = {
        "source_record_id",
        "borough",
        "legacy_usable_area_m2",
        "p0_service_eligible",
    }
    historical_required = {
        "source_record_id",
        "borough",
        "capacity_registered_historical_candidate",
        "requires_current_register_verification",
        "may_use_as_operational_capacity",
    }
    _require_columns(facility_master, master_required, "facility_master")
    _require_columns(
        historical_candidates,
        historical_required,
        "historical_candidates",
    )

    for name, frame in (
        ("facility_master", facility_master),
        ("historical_candidates", historical_candidates),
    ):
        duplicates = frame["source_record_id"].duplicated(keep=False)
        if duplicates.any():
            raise ValueError(
                f"{name}.source_record_id must be unique; "
                f"found {int(duplicates.sum())} duplicate rows"
            )

    optional_history = [
        column
        for column in ("match_tier", "candidate_status", "capacity_semantics")
        if column in historical_candidates.columns
    ]
    master_columns = [
        "source_record_id",
        "borough",
        "legacy_usable_area_m2",
        "p0_service_eligible",
    ]
    history_columns = [
        "source_record_id",
        "borough",
        "capacity_registered_historical_candidate",
        "requires_current_register_verification",
        "may_use_as_operational_capacity",
        *optional_history,
    ]
    joined = facility_master[master_columns].merge(
        historical_candidates[history_columns],
        on="source_record_id",
        how="inner",
        suffixes=("_master", "_historical"),
        validate="one_to_one",
    )

    borough_master = joined["borough_master"].astype("string").str.strip()
    borough_history = joined["borough_historical"].astype("string").str.strip()
    borough_mismatch = (
        borough_master.notna()
        & borough_history.notna()
        & borough_master.ne(borough_history)
    )
    if borough_mismatch.any():
        raise ValueError(
            "borough differs between master and historical candidates for "
            f"{int(borough_mismatch.sum())} joined rows"
        )

    area = pd.to_numeric(joined["legacy_usable_area_m2"], errors="coerce")
    capacity = pd.to_numeric(
        joined["capacity_registered_historical_candidate"],
        errors="coerce",
    )
    p0_eligible = _coerce_boolean(joined["p0_service_eligible"])
    valid_area = area.notna() & np.isfinite(area) & area.gt(0)
    valid_capacity = capacity.notna() & np.isfinite(capacity) & capacity.gt(0)
    valid_borough = borough_master.notna() & borough_master.ne("")
    eligible = p0_eligible.eq(True).fillna(False)
    keep = valid_area & valid_capacity & valid_borough
    if require_p0_eligible:
        keep &= eligible

    output = pd.DataFrame(
        {
            "source_record_id": joined.loc[keep, "source_record_id"].astype(
                "string"
            ),
            "borough": borough_master.loc[keep],
            "area_m2": area.loc[keep].astype(float),
            "capacity": capacity.loc[keep].astype(float),
            "requires_current_register_verification": _coerce_boolean(
                joined.loc[keep, "requires_current_register_verification"]
            ),
            "may_use_as_operational_capacity": _coerce_boolean(
                joined.loc[keep, "may_use_as_operational_capacity"]
            ),
        }
    ).reset_index(drop=True)
    for column in optional_history:
        output[column] = joined.loc[keep, column].reset_index(drop=True)

    verification_flags = _coerce_boolean(
        joined.loc[keep, "requires_current_register_verification"]
    )
    operational_flags = _coerce_boolean(
        joined.loc[keep, "may_use_as_operational_capacity"]
    )
    output.attrs["preparation"] = {
        "master_rows": int(len(facility_master)),
        "historical_rows": int(len(historical_candidates)),
        "joined_rows": int(len(joined)),
        "positive_finite_rows": int(keep.sum()),
        "dropped_missing_or_nonpositive_area": int((~valid_area).sum()),
        "dropped_missing_or_nonpositive_capacity": int((~valid_capacity).sum()),
        "dropped_ineligible_p0": int((~eligible).sum())
        if require_p0_eligible
        else 0,
        "all_rows_require_current_register_verification": bool(
            len(output) > 0 and verification_flags.eq(True).all()
        ),
        "any_row_marked_operationally_usable": bool(
            operational_flags.eq(True).any()
        ),
        "label_source": "capacity_registered_historical_candidate",
        "feature_source": "legacy_usable_area_m2",
        "generated_capacity_labels_used": False,
    }
    return output


def _model_candidates(random_state: int) -> dict[str, Any]:
    return {
        MEDIAN_SCENARIO: DummyRegressor(strategy="median"),
        NONNEGATIVE_LINEAR: LinearRegression(positive=True),
        HUBER: Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    HuberRegressor(
                        epsilon=1.35,
                        alpha=0.0001,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        RANDOM_FOREST: RandomForestRegressor(
            n_estimators=96,
            max_depth=8,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _score_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | None]:
    if observed.size == 0:
        return {
            "mae": None,
            "rmse": None,
            "median_absolute_error": None,
            "r2": None,
        }
    mae = float(mean_absolute_error(observed, predicted))
    rmse = float(math.sqrt(mean_squared_error(observed, predicted)))
    medae = float(median_absolute_error(observed, predicted))
    r2 = float(r2_score(observed, predicted)) if observed.size >= 2 else None
    if r2 is not None and not math.isfinite(r2):
        r2 = None
    return {
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": medae,
        "r2": r2,
    }


def _correlation(values_a: pd.Series, values_b: pd.Series, method: str) -> float | None:
    if values_a.nunique() < 2 or values_b.nunique() < 2:
        return None
    value = values_a.corr(values_b, method=method)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _negative_huber_slope(estimator: Any) -> bool:
    if not isinstance(estimator, Pipeline):
        return False
    model = estimator.named_steps.get("model")
    coefficients = getattr(model, "coef_", None)
    return bool(coefficients is not None and np.asarray(coefficients)[0] < 0)


def compare_capacity_models(
    training_frame: pd.DataFrame,
    *,
    area_col: str = "area_m2",
    target_col: str = "capacity",
    group_col: str = "borough",
    trusted_for_deployment: bool = False,
    min_samples: int = 60,
    min_groups: int = 4,
    min_spearman: float = 0.20,
    min_relative_mae_improvement: float = 0.05,
    min_fold_win_rate: float = 0.60,
    max_splits: int = 5,
    random_state: int = 20260727,
) -> CapacityComparisonResult:
    """Compare fixed candidates with borough-held-out GroupKFold validation.

    The median of each training fold is the baseline.  A challenger is selected
    only if the sample and positive correlation are adequate, its pooled
    out-of-fold MAE improves by the configured margin, its RMSE also improves,
    and it wins on enough held-out borough folds.  Otherwise the returned model
    is explicitly ``median_scenario`` and is never deployable.

    ``trusted_for_deployment`` is intentionally false by default.  It should
    only be true for current, verified observed labels and verified areas.
    """

    _validate_target_name(target_col)
    _require_columns(
        training_frame,
        {area_col, target_col, group_col},
        "training_frame",
    )
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if min_groups < 2:
        raise ValueError("min_groups must be at least 2")
    if max_splits < 2:
        raise ValueError("max_splits must be at least 2")
    if not 0 <= min_fold_win_rate <= 1:
        raise ValueError("min_fold_win_rate must be between 0 and 1")
    if min_relative_mae_improvement < 0:
        raise ValueError("min_relative_mae_improvement must be nonnegative")

    work = training_frame[[area_col, target_col, group_col]].copy()
    input_rows = len(work)
    work[area_col] = pd.to_numeric(work[area_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    group_present = work[group_col].notna()
    work[group_col] = work[group_col].astype("string").str.strip()
    valid = (
        work[area_col].notna()
        & work[target_col].notna()
        & np.isfinite(work[area_col])
        & np.isfinite(work[target_col])
        & work[area_col].gt(0)
        & work[target_col].gt(0)
        & group_present
        & work[group_col].ne("")
    )
    work = work.loc[valid].reset_index(drop=True)
    if work.empty:
        raise ValueError("no finite, positive area/capacity observations remain")

    area_series = work[area_col].astype(float)
    target_series = work[target_col].astype(float)
    group_series = work[group_col].astype(str)
    x = area_series.to_numpy().reshape(-1, 1)
    y = target_series.to_numpy()
    groups = group_series.to_numpy()
    n_samples = len(work)
    n_groups = int(group_series.nunique())
    pearson = _correlation(area_series, target_series, "pearson")
    spearman = _correlation(area_series, target_series, "spearman")
    sample_sufficient = n_samples >= min_samples
    groups_sufficient = n_groups >= min_groups
    correlation_sufficient = (
        spearman is not None and spearman >= min_spearman
    )

    preparation = training_frame.attrs.get("preparation")
    historical_legacy_lineage = bool(
        isinstance(preparation, Mapping)
        and (
            preparation.get("label_source")
            == "capacity_registered_historical_candidate"
            or preparation.get("feature_source") == "legacy_usable_area_m2"
            or preparation.get(
                "all_rows_require_current_register_verification",
                False,
            )
        )
    )
    effective_deployment_trust = bool(
        trusted_for_deployment and not historical_legacy_lineage
    )

    metrics: dict[str, Any] = {
        "sample": {
            "input_rows": int(input_rows),
            "valid_positive_rows": int(n_samples),
            "dropped_rows": int(input_rows - n_samples),
            "borough_groups": n_groups,
            "unique_areas": int(area_series.nunique()),
            "unique_capacities": int(target_series.nunique()),
        },
        "correlation": {
            "pearson": pearson,
            "spearman": spearman,
            "minimum_required_spearman": float(min_spearman),
            "sufficient_positive_correlation": correlation_sufficient,
        },
        "cv": {
            "scheme": "GroupKFold",
            "group_column": group_col,
            "n_splits": 0,
            "status": "not_run",
        },
        "models": {},
        "selection": {
            "minimum_relative_mae_improvement": float(
                min_relative_mae_improvement
            ),
            "minimum_fold_win_rate": float(min_fold_win_rate),
        },
    }
    metadata: dict[str, Any] = {
        "feature_column": area_col,
        "target_column": target_col,
        "group_column": group_col,
        "target_policy": "positive_observed_values_only",
        "generated_capacity_labels_used": False,
        "prohibited_target_columns": sorted(PROHIBITED_TARGET_COLUMNS),
        "deployment_trust_requested": bool(trusted_for_deployment),
        "trusted_for_deployment": effective_deployment_trust,
        "historical_legacy_lineage": historical_legacy_lineage,
        "training_area_min_m2": float(area_series.min()),
        "training_area_max_m2": float(area_series.max()),
        "random_state": int(random_state),
    }
    if isinstance(preparation, Mapping):
        metadata["preparation"] = dict(preparation)

    candidates = _model_candidates(random_state)
    can_cross_validate = n_groups >= 2 and n_samples >= 2
    if can_cross_validate:
        n_splits = min(max_splits, n_groups)
        splitter = GroupKFold(n_splits=n_splits)
        predictions = {
            name: np.full(n_samples, np.nan, dtype=float)
            for name in candidates
        }
        fold_metrics: dict[str, list[dict[str, Any]]] = {
            name: [] for name in candidates
        }
        errors: dict[str, list[str]] = {name: [] for name in candidates}

        for fold_number, (train_index, test_index) in enumerate(
            splitter.split(x, y, groups),
            start=1,
        ):
            for name, candidate in candidates.items():
                fitted = clone(candidate)
                try:
                    fitted.fit(x[train_index], y[train_index])
                    fold_prediction = np.maximum(
                        np.asarray(
                            fitted.predict(x[test_index]),
                            dtype=float,
                        ),
                        0.0,
                    )
                    if not np.isfinite(fold_prediction).all():
                        raise ValueError("model produced non-finite predictions")
                    predictions[name][test_index] = fold_prediction
                    fold_score = _score_predictions(
                        y[test_index],
                        fold_prediction,
                    )
                    fold_metrics[name].append(
                        {
                            "fold": fold_number,
                            "train_rows": int(len(train_index)),
                            "test_rows": int(len(test_index)),
                            **fold_score,
                        }
                    )
                except Exception as exc:  # candidate failure must not hide baseline
                    errors[name].append(
                        f"fold {fold_number}: {type(exc).__name__}: {exc}"
                    )

        metrics["cv"].update(
            {
                "n_splits": int(n_splits),
                "status": "completed",
                "all_preprocessing_fit_within_fold": True,
            }
        )
        for name in candidates:
            complete = np.isfinite(predictions[name]).all()
            if complete:
                model_score = _score_predictions(y, predictions[name])
                metrics["models"][name] = {
                    "status": "ok",
                    **model_score,
                    "folds": fold_metrics[name],
                }
            else:
                metrics["models"][name] = {
                    "status": "failed",
                    "errors": errors[name],
                    "folds": fold_metrics[name],
                }

        baseline_metrics = metrics["models"].get(MEDIAN_SCENARIO, {})
        if baseline_metrics.get("status") != "ok":
            raise RuntimeError("median baseline failed during GroupKFold")
        baseline_mae = float(baseline_metrics["mae"])
        baseline_rmse = float(baseline_metrics["rmse"])
        baseline_folds = baseline_metrics["folds"]
        for name in (NONNEGATIVE_LINEAR, HUBER, RANDOM_FOREST):
            model_metrics = metrics["models"].get(name, {})
            if model_metrics.get("status") != "ok":
                continue
            model_mae = float(model_metrics["mae"])
            if baseline_mae > 0:
                relative_improvement = (baseline_mae - model_mae) / baseline_mae
            else:
                relative_improvement = 0.0
            wins = sum(
                float(model_fold["mae"]) < float(baseline_fold["mae"])
                for model_fold, baseline_fold in zip(
                    model_metrics["folds"],
                    baseline_folds,
                )
            )
            model_metrics["relative_mae_improvement_vs_median"] = float(
                relative_improvement
            )
            model_metrics["fold_win_rate_vs_median"] = float(wins / n_splits)
            model_metrics["rmse_improved_vs_median"] = bool(
                float(model_metrics["rmse"]) < baseline_rmse
            )

    selected_model = MEDIAN_SCENARIO
    selection_reason = ""
    deployable = False

    if not sample_sufficient:
        selection_reason = f"insufficient_samples:{n_samples}<{min_samples}"
    elif not groups_sufficient:
        selection_reason = f"insufficient_groups:{n_groups}<{min_groups}"
    elif not correlation_sufficient:
        shown = "undefined" if spearman is None else f"{spearman:.6f}"
        selection_reason = (
            "insufficient_positive_correlation:"
            f"spearman={shown}<{min_spearman:.6f}"
        )
    elif metrics["cv"]["status"] != "completed":
        selection_reason = "group_cross_validation_unavailable"
    else:
        eligible_models: list[tuple[str, dict[str, Any]]] = []
        for name in (NONNEGATIVE_LINEAR, HUBER, RANDOM_FOREST):
            model_metrics = metrics["models"].get(name, {})
            if model_metrics.get("status") == "ok":
                eligible_models.append((name, model_metrics))
        if not eligible_models:
            selection_reason = "all_challenger_models_failed"
        else:
            best_name, best_metrics = min(
                eligible_models,
                key=lambda item: float(item[1]["mae"]),
            )
            improvement = float(
                best_metrics["relative_mae_improvement_vs_median"]
            )
            win_rate = float(best_metrics["fold_win_rate_vs_median"])
            rmse_improved = bool(best_metrics["rmse_improved_vs_median"])
            metrics["selection"]["best_challenger"] = best_name
            metrics["selection"]["best_relative_mae_improvement"] = improvement
            metrics["selection"]["best_fold_win_rate"] = win_rate
            if improvement < min_relative_mae_improvement:
                selection_reason = (
                    "no_material_improvement:"
                    f"{improvement:.6f}<{min_relative_mae_improvement:.6f}"
                )
            elif win_rate < min_fold_win_rate:
                selection_reason = (
                    "inconsistent_borough_fold_improvement:"
                    f"{win_rate:.6f}<{min_fold_win_rate:.6f}"
                )
            elif not rmse_improved:
                selection_reason = "challenger_rmse_not_better_than_median"
            else:
                selected_model = best_name
                if effective_deployment_trust:
                    deployable = True
                    selection_reason = "validated_challenger_selected"
                else:
                    selection_reason = (
                        "validated_challenger_selected_for_diagnostic_only:"
                        "inputs_not_trusted_for_deployment"
                    )

    final_estimator = clone(candidates[selected_model])
    try:
        final_estimator.fit(x, y)
        if selected_model == HUBER and _negative_huber_slope(final_estimator):
            raise ValueError("Huber fitted a negative area coefficient")
    except Exception as exc:
        selected_model = MEDIAN_SCENARIO
        deployable = False
        selection_reason = (
            "selected_challenger_failed_full_fit:"
            f"{type(exc).__name__}:{exc}"
        )
        final_estimator = clone(candidates[MEDIAN_SCENARIO]).fit(x, y)

    metrics["selection"].update(
        {
            "selected_model": selected_model,
            "deployable": deployable,
            "reason": selection_reason,
        }
    )
    metadata["selection_reason"] = selection_reason
    artifact = CapacityModelArtifact(
        estimator=final_estimator,
        model_name=selected_model,
        feature_name=area_col,
        area_min_m2=float(area_series.min()),
        area_max_m2=float(area_series.max()),
        deployable=deployable,
        metrics=metrics,
        metadata=metadata,
    )
    return CapacityComparisonResult(
        selected_model=selected_model,
        deployable=deployable,
        reason=selection_reason,
        artifact=artifact,
        metrics=metrics,
        metadata=metadata,
    )


def evaluate_capacity_relationship(
    facility_master_csv: str | Path,
    historical_candidates_csv: str | Path,
    *,
    artifact_path: str | Path | None = None,
    min_samples: int = 60,
    min_groups: int = 4,
    min_spearman: float = 0.20,
    min_relative_mae_improvement: float = 0.05,
    min_fold_win_rate: float = 0.60,
    max_splits: int = 5,
    random_state: int = 20260727,
) -> CapacityComparisonResult:
    """Load the processed private CSVs and run a diagnostic-only comparison."""

    master_path = Path(facility_master_csv)
    historical_path = Path(historical_candidates_csv)
    facility_master = pd.read_csv(master_path, low_memory=False)
    historical_candidates = pd.read_csv(historical_path, low_memory=False)
    training = prepare_capacity_training_data(
        facility_master,
        historical_candidates,
    )
    result = compare_capacity_models(
        training,
        trusted_for_deployment=False,
        min_samples=min_samples,
        min_groups=min_groups,
        min_spearman=min_spearman,
        min_relative_mae_improvement=min_relative_mae_improvement,
        min_fold_win_rate=min_fold_win_rate,
        max_splits=max_splits,
        random_state=random_state,
    )
    result.metadata.update(
        {
            "facility_master_csv": str(master_path),
            "historical_candidates_csv": str(historical_path),
            "lineage_policy": (
                "historical capacity and legacy area are diagnostic-only until "
                "both are verified against current authoritative records"
            ),
            "source_workflow_deployable": False,
        }
    )
    if artifact_path is not None:
        saved = save_capacity_artifact(result.artifact, artifact_path)
        result.metadata["artifact_path"] = str(saved)
    return result


# A concise alias for callers that organize jobs around CSV inputs.
evaluate_capacity_csvs = evaluate_capacity_relationship


def save_capacity_artifact(
    artifact_or_result: CapacityModelArtifact | CapacityComparisonResult,
    path: str | Path,
    *,
    require_deployable: bool = False,
) -> Path:
    """Persist the guarded artifact with joblib."""

    artifact = (
        artifact_or_result.artifact
        if isinstance(artifact_or_result, CapacityComparisonResult)
        else artifact_or_result
    )
    if not isinstance(artifact, CapacityModelArtifact):
        raise TypeError("expected CapacityModelArtifact or CapacityComparisonResult")
    if require_deployable and not artifact.deployable:
        raise ValueError("refusing to save a non-deployable capacity artifact")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A plain, versioned payload remains loadable even when this module is
    # executed through ``python -m`` (where local classes otherwise pickle as
    # ``__main__``).
    payload = {
        "artifact_type": "elder_guardian.capacity.CapacityModelArtifact",
        "artifact_version": 1,
        "estimator": artifact.estimator,
        "model_name": artifact.model_name,
        "feature_name": artifact.feature_name,
        "area_min_m2": artifact.area_min_m2,
        "area_max_m2": artifact.area_max_m2,
        "deployable": artifact.deployable,
        "metrics": artifact.metrics,
        "metadata": artifact.metadata,
    }
    joblib.dump(payload, destination, compress=3)
    return destination


def load_capacity_artifact(path: str | Path) -> CapacityModelArtifact:
    """Load and type-check a previously saved artifact."""

    payload = joblib.load(Path(path))
    if isinstance(payload, CapacityModelArtifact):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("joblib payload is not a capacity artifact mapping")
    if payload.get("artifact_type") != (
        "elder_guardian.capacity.CapacityModelArtifact"
    ):
        raise TypeError("joblib payload has an unknown artifact type")
    if payload.get("artifact_version") != 1:
        raise ValueError(
            f"unsupported capacity artifact version: "
            f"{payload.get('artifact_version')!r}"
        )
    required = {
        "estimator",
        "model_name",
        "feature_name",
        "area_min_m2",
        "area_max_m2",
        "deployable",
        "metrics",
        "metadata",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"capacity artifact is missing fields: {missing}")
    return CapacityModelArtifact(
        estimator=payload["estimator"],
        model_name=str(payload["model_name"]),
        feature_name=str(payload["feature_name"]),
        area_min_m2=float(payload["area_min_m2"]),
        area_max_m2=float(payload["area_max_m2"]),
        deployable=bool(payload["deployable"]),
        metrics=dict(payload["metrics"]),
        metadata=dict(payload["metadata"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare historical positive capacity with legacy usable area using "
            "borough GroupKFold; output is diagnostic-only for these CSVs."
        )
    )
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--min-groups", type=int, default=4)
    parser.add_argument("--min-spearman", type=float, default=0.20)
    parser.add_argument("--min-relative-mae-improvement", type=float, default=0.05)
    parser.add_argument("--min-fold-win-rate", type=float, default=0.60)
    parser.add_argument("--max-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260727)
    arguments = parser.parse_args(argv)

    result = evaluate_capacity_relationship(
        arguments.master,
        arguments.historical,
        artifact_path=arguments.artifact,
        min_samples=arguments.min_samples,
        min_groups=arguments.min_groups,
        min_spearman=arguments.min_spearman,
        min_relative_mae_improvement=arguments.min_relative_mae_improvement,
        min_fold_win_rate=arguments.min_fold_win_rate,
        max_splits=arguments.max_splits,
        random_state=arguments.random_state,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
