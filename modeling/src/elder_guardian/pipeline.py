"""End-to-end training, exact validation, artifact, and report pipeline."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .capacity import evaluate_capacity_relationship
from .data import (
    apply_1km_residual_sensitivity,
    load_facilities,
    load_population_100m,
    sha256_file,
)
from .exact import evaluate_candidates_exact
from .features import FEATURE_COLUMNS, FeatureResult, build_candidate_features
from .reporting import (
    add_wgs84,
    plot_base_candidates,
    plot_surrogate_validation,
    write_run_report,
)
from .surrogate import (
    fit_final_gcn,
    ranking_curve,
    spatial_cross_validate_gcn,
    spatial_cross_validate_tabular,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _git_state(repository: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"path": str(repository), "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(repository), "commit": None, "dirty": None}


def _select_one_km_archive(population_root: Path) -> Path:
    matches = sorted(population_root.glob("서울전체*/*.zip"))
    if len(matches) != 1:
        raise ValueError(f"expected one Seoul 1km archive, found {len(matches)}")
    return matches[0]


def _scenario_population(
    lower_bound: pd.DataFrame, sensitivity: pd.DataFrame, scenario: dict
) -> pd.DataFrame:
    mode = scenario.get("population_mode", "PUBLISHED_LOWER_BOUND")
    if mode == "PUBLISHED_LOWER_BOUND":
        return lower_bound
    if mode == "ONE_KM_RESIDUAL_SENSITIVITY":
        return sensitivity
    raise ValueError(f"unknown population mode: {mode}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _build_scenario_contexts(
    *,
    config: dict,
    lower_population: pd.DataFrame,
    sensitivity_population: pd.DataFrame,
    facilities_by_mode: dict[str, pd.DataFrame],
) -> list[dict]:
    contexts: list[dict] = []
    speed = float(config["walking_speed_m_per_min"])
    for scenario in config["scenarios"]:
        population = _scenario_population(
            lower_population, sensitivity_population, scenario
        )
        facilities = facilities_by_mode[scenario["existing_capacity_mode"]]
        feature_result = build_candidate_features(
            population, facilities, scenario, speed_m_per_min=speed
        )
        contexts.append(
            {
                "scenario": scenario,
                "population": population,
                "facilities": facilities,
                "feature_result": feature_result,
            }
        )
    return contexts


def _candidate_mask(
    contexts: list[dict], *, minimum_local_population: float, stride_m: float
) -> np.ndarray:
    masks = []
    for context in contexts:
        local = np.expm1(
            context["feature_result"].features[
                "log_local_population_1000m"
            ].to_numpy()
        )
        masks.append(local >= minimum_local_population)
    # A common candidate universe avoids scenario-specific missing target rows.
    valid = np.logical_and.reduce(masks)
    if stride_m > 100.0:
        population = contexts[0]["population"]
        x0 = float(population["x"].min())
        y0 = float(population["y"].min())
        on_stride = (
            np.isclose(np.mod(population["x"] - x0, stride_m), 0.0)
            & np.isclose(np.mod(population["y"] - y0, stride_m), 0.0)
        )
        valid &= on_stride.to_numpy()
    return valid


def _evaluate_all_scenarios(
    contexts: list[dict],
    preliminary_indices: np.ndarray,
    *,
    speed_m_per_min: float,
) -> tuple[pd.DataFrame, np.ndarray, list[dict], np.ndarray]:
    long_frames: list[pd.DataFrame] = []
    target_rows: list[np.ndarray] = []
    summaries: list[dict] = []
    candidate_validity: list[np.ndarray] = []
    for context in contexts:
        scenario = context["scenario"]
        population: pd.DataFrame = context["population"]
        feature_result: FeatureResult = context["feature_result"]
        candidate = population.iloc[preliminary_indices]
        demand = population.iloc[feature_result.demand_index]
        exact = evaluate_candidates_exact(
            feature_result.baseline_demand_accessibility,
            feature_result.demand_population,
            demand[["x", "y"]].to_numpy(dtype=float),
            candidate[["x", "y"]].to_numpy(dtype=float),
            new_capacity=float(scenario["new_capacity"]),
            speed_m_per_min=speed_m_per_min,
            tau_minutes=float(scenario["tau_minutes"]),
            cutoff_minutes=float(scenario["cutoff_minutes"]),
        )
        valid = np.isfinite(exact.delta_gini)
        candidate_validity.append(valid)
        target_rows.append(exact.delta_gini)
        rows = candidate[
            [
                "gid",
                "borough",
                "x",
                "y",
                "population_lower_bound",
                "population",
                "is_suppressed",
            ]
        ].copy()
        rows["scenario"] = scenario["name"]
        rows["existing_capacity_mode"] = scenario["existing_capacity_mode"]
        rows["population_mode"] = scenario["population_mode"]
        rows["new_capacity"] = float(scenario["new_capacity"])
        rows["tau_minutes"] = float(scenario["tau_minutes"])
        rows["cutoff_minutes"] = float(scenario["cutoff_minutes"])
        rows["baseline_gini"] = exact.baseline_gini
        rows["candidate_gini"] = exact.candidate_gini
        rows["delta_gini"] = exact.delta_gini
        rows["candidate_demand_denominator"] = exact.denominators
        rows["valid"] = valid
        rows["exact_runtime_seconds"] = exact.runtime_seconds
        long_frames.append(rows)
        if not valid.any():
            raise RuntimeError(f"scenario {scenario['name']} has no valid candidate")
        best_position = int(np.nanargmax(exact.delta_gini))
        summaries.append(
            {
                "name": scenario["name"],
                "existing_capacity_mode": scenario["existing_capacity_mode"],
                "population_mode": scenario["population_mode"],
                "baseline_gini": exact.baseline_gini,
                "best_candidate_gini": float(exact.candidate_gini[best_position]),
                "best_delta_gini": float(exact.delta_gini[best_position]),
                "best_gid": str(candidate.iloc[best_position]["gid"]),
                "valid_candidates": int(valid.sum()),
                "invalid_zero_demand_candidates": int((~valid).sum()),
                "exact_runtime_seconds": exact.runtime_seconds,
                "conservation_error": feature_result.conservation_error,
            }
        )
    return (
        pd.concat(long_frames, ignore_index=True),
        np.vstack(target_rows),
        summaries,
        np.logical_and.reduce(candidate_validity),
    )


def _robust_ranking(
    exact_results: pd.DataFrame, scenario_names: list[str]
) -> pd.DataFrame:
    valid = exact_results.loc[
        exact_results["scenario"].isin(scenario_names) & exact_results["valid"]
    ].copy()
    valid["percentile"] = valid.groupby("scenario")["delta_gini"].rank(
        method="average", pct=True
    )
    robust = (
        valid.groupby(["gid", "borough", "x", "y"], as_index=False)
        .agg(
            mean_percentile=("percentile", "mean"),
            worst_percentile=("percentile", "min"),
            mean_delta_gini=("delta_gini", "mean"),
            scenario_count=("scenario", "nunique"),
        )
        .loc[lambda frame: frame["scenario_count"] == len(scenario_names)]
        .sort_values(
            ["mean_percentile", "worst_percentile"],
            ascending=False,
            kind="mergesort",
        )
    )
    return add_wgs84(robust)


def _artifact_hashes(run_dir: Path) -> dict:
    output: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            output[str(path.relative_to(run_dir))] = sha256_file(path)
    return output


def model_code_hashes(project_root: Path) -> dict:
    """Hash the executable model source and fixed configuration."""

    modeling = project_root / "modeling"
    paths = [
        modeling / "pyproject.toml",
        modeling / "README.md",
        modeling / "GNN_MODEL_CARD.md",
        modeling / "SECURITY.md",
        *sorted((modeling / "config").rglob("*")),
        *sorted((modeling / "src").rglob("*.py")),
        *sorted((modeling / "scripts").rglob("*.py")),
        *sorted((modeling / "tests").rglob("*.py")),
    ]
    return {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def run_pipeline(
    *,
    project_root: str | Path,
    config_path: str | Path,
    run_id: str,
    population_root: str | Path | None = None,
    facility_master_csv: str | Path | None = None,
    historical_capacity_csv: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Run data checks, exact model, spatial validation, and packaging."""

    started = time.perf_counter()
    project = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    population_path = Path(
        population_root
        or project / "Data" / "서울_격자별노인인구(100M)"
    ).resolve()
    master_path = Path(
        facility_master_csv
        or project
        / "tmp"
        / "planning_data"
        / "data"
        / "facilities"
        / "processed"
        / "facility-master.csv"
    ).resolve()
    history_path = Path(
        historical_capacity_csv
        or project
        / "tmp"
        / "planning_data"
        / "data"
        / "facilities"
        / "processed"
        / "historical-capacity-candidates.csv"
    ).resolve()
    artifacts_root = Path(output_root or project / "modeling" / "artifacts").resolve()
    run_dir = artifacts_root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    lower_population, population_audit = load_population_100m(population_path)
    one_km_path = _select_one_km_archive(population_path)
    sensitivity_population, population_sensitivity = apply_1km_residual_sensitivity(
        lower_population, one_km_path
    )
    lower_population.to_csv(run_dir / "population_model_input.csv", index=False)
    sensitivity_population[
        ["gid", "population", "coarse_1km_population"]
    ].to_csv(run_dir / "population_1km_residual_sensitivity.csv", index=False)

    capacity_modes = sorted(
        {scenario["existing_capacity_mode"] for scenario in config["scenarios"]}
        | {"STRICT_UNKNOWN"}
    )
    facilities_by_mode: dict[str, pd.DataFrame] = {}
    facility_audit: dict[str, dict] = {}
    for mode in capacity_modes:
        facilities, audit = load_facilities(
            master_path,
            historical_csv=history_path,
            capacity_mode=mode,
        )
        facilities_by_mode[mode] = facilities
        facility_audit[mode] = audit
        facilities[
            [
                "source_record_id",
                "facility_name",
                "borough",
                "latitude",
                "longitude",
                "x",
                "y",
                "model_capacity",
                "capacity_source",
                "capacity_confidence",
            ]
        ].to_csv(run_dir / f"facilities_{mode.lower()}.csv", index=False)

    strict_unknown_gate = {
        "capacity_sum": facility_audit["STRICT_UNKNOWN"]["capacity_sum"],
        "baseline_gini_defined": False,
        "status": "REJECTED_UNDEFINED_ZERO_MEAN",
        "reason": "no verified operational concurrent-capacity observations",
    }

    capacity_result = evaluate_capacity_relationship(
        master_path,
        history_path,
        artifact_path=run_dir / "capacity_diagnostic.joblib",
        random_state=int(config["random_seed"]),
    )
    capacity_validation = capacity_result.to_dict()
    _write_json(run_dir / "capacity_validation.json", capacity_validation)

    contexts = _build_scenario_contexts(
        config=config,
        lower_population=lower_population,
        sensitivity_population=sensitivity_population,
        facilities_by_mode=facilities_by_mode,
    )
    preliminary_mask = _candidate_mask(
        contexts,
        minimum_local_population=float(
            config["candidate_min_local_population_1000m"]
        ),
        stride_m=float(config["candidate_stride_m"]),
    )
    preliminary_indices = np.flatnonzero(preliminary_mask)
    exact_results, targets_preliminary, scenario_summary, common_valid = (
        _evaluate_all_scenarios(
            contexts,
            preliminary_indices,
            speed_m_per_min=float(config["walking_speed_m_per_min"]),
        )
    )
    exact_results = add_wgs84(exact_results)
    exact_results.sort_values(
        ["scenario", "delta_gini"],
        ascending=[True, False],
        kind="mergesort",
        inplace=True,
    )
    exact_results.to_csv(run_dir / "exact_candidate_results.csv", index=False)
    _write_json(run_dir / "scenario_summary.json", scenario_summary)

    final_indices = preliminary_indices[common_valid]
    candidate_nodes = lower_population.iloc[final_indices].reset_index(drop=True)
    feature_tensor = np.stack(
        [
            context["feature_result"].features.iloc[final_indices][
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float32)
            for context in contexts
        ]
    )
    target_tensor = targets_preliminary[:, common_valid]
    scenario_names = [context["scenario"]["name"] for context in contexts]
    np.savez_compressed(
        run_dir / "surrogate_training_tensor.npz",
        features=feature_tensor,
        targets=target_tensor,
        xy=candidate_nodes[["x", "y"]].to_numpy(dtype=np.float64),
        groups=candidate_nodes["borough"].astype(str).to_numpy(dtype="U16"),
        gids=candidate_nodes["gid"].astype(str).to_numpy(dtype="U16"),
        feature_names=np.asarray(FEATURE_COLUMNS, dtype="U64"),
        scenario_names=np.asarray(scenario_names, dtype="U64"),
    )

    surrogate_config = config["surrogate"]
    gcn_metrics, gcn_predictions = spatial_cross_validate_gcn(
        feature_tensor,
        target_tensor,
        candidate_nodes[["x", "y"]].to_numpy(dtype=float),
        candidate_nodes["borough"].astype(str).to_numpy(),
        hidden_dim=int(surrogate_config["hidden_dim"]),
        dropout=float(surrogate_config["dropout"]),
        epochs=int(surrogate_config["epochs"]),
        patience=int(surrogate_config["patience"]),
        learning_rate=float(surrogate_config["learning_rate"]),
        weight_decay=float(surrogate_config["weight_decay"]),
        top_loss_weight=float(surrogate_config["top_loss_weight"]),
        folds=int(surrogate_config["folds"]),
        top_k=int(surrogate_config["top_k"]),
        random_seed=int(config["random_seed"]),
    )
    tabular_metrics, tabular_predictions = spatial_cross_validate_tabular(
        feature_tensor,
        target_tensor,
        candidate_nodes["borough"].astype(str).to_numpy(),
        folds=int(surrogate_config["folds"]),
        top_k=int(surrogate_config["top_k"]),
        random_seed=int(config["random_seed"]),
    )
    metrics = [asdict(gcn_metrics), asdict(tabular_metrics)]
    _write_json(run_dir / "surrogate_metrics.json", metrics)
    ranking_curves = {
        "SpatialGCN": ranking_curve(
            target_tensor,
            gcn_predictions,
            candidate_nodes["borough"].astype(str).to_numpy(),
            random_seed=int(config["random_seed"]),
        ),
        "HistGradientBoosting": ranking_curve(
            target_tensor,
            tabular_predictions,
            candidate_nodes["borough"].astype(str).to_numpy(),
            random_seed=int(config["random_seed"]),
        ),
    }
    _write_json(run_dir / "surrogate_ranking_curves.json", ranking_curves)
    np.savez_compressed(
        run_dir / "surrogate_oof_predictions.npz",
        truth=target_tensor,
        gcn=gcn_predictions,
        hist_gradient_boosting=tabular_predictions,
    )

    gates_config = config["gates"]
    gcn_approved = bool(
        gcn_metrics.spearman >= float(gates_config["min_spearman"])
        and gcn_metrics.recall_at_k >= float(gates_config["min_recall_at_20"])
        and gcn_metrics.best_in_predicted_top_k
        >= float(gates_config["min_recall_best_at_20"])
        and gcn_metrics.relative_policy_regret
        <= float(gates_config["max_relative_policy_regret_at_20"])
    )
    gates = {
        "gcn_approved": gcn_approved,
        "thresholds": gates_config,
        "strict_unknown": strict_unknown_gate,
        "exact_approved_for_screening": bool(
            max(item["conservation_error"] for item in scenario_summary) < 1e-7
        ),
        "final_authority": "exact_e2sfca_weighted_gini",
        "deployment_scope": "WALK_EUCLIDEAN_FEASIBILITY_UNKNOWN_SCREENING",
    }
    _write_json(run_dir / "validation_gates.json", gates)

    fit_final_gcn(
        feature_tensor,
        target_tensor,
        candidate_nodes[["x", "y"]].to_numpy(dtype=float),
        candidate_nodes["borough"].astype(str).to_numpy(),
        feature_names=FEATURE_COLUMNS,
        scenario_names=scenario_names,
        output_path=run_dir / "gnn_surrogate.pt",
        hidden_dim=int(surrogate_config["hidden_dim"]),
        dropout=float(surrogate_config["dropout"]),
        epochs=int(surrogate_config["epochs"]),
        learning_rate=float(surrogate_config["learning_rate"]),
        weight_decay=float(surrogate_config["weight_decay"]),
        top_loss_weight=float(surrogate_config["top_loss_weight"]),
        random_seed=int(config["random_seed"]),
    )

    base_name = config["base_scenario"]
    base_top = (
        exact_results.loc[
            (exact_results["scenario"] == base_name) & exact_results["valid"]
        ]
        .nlargest(20, "delta_gini")
        .copy()
    )
    base_top.to_csv(run_dir / "recommendations_base_top20.csv", index=False)
    robust = _robust_ranking(exact_results, scenario_names)
    robust.head(20).to_csv(
        run_dir / "recommendations_robust_top20.csv", index=False
    )
    plot_base_candidates(
        exact_results,
        lower_population,
        scenario_name=base_name,
        output_path=run_dir / "map_base_exact.png",
    )
    plot_surrogate_validation(
        target_tensor,
        {
            "SpatialGCN": gcn_predictions,
            "HistGradientBoosting": tabular_predictions,
        },
        output_path=run_dir / "surrogate_validation.png",
        random_seed=int(config["random_seed"]),
    )

    input_hashes = {
        "config": sha256_file(config_file),
        "facility_master": sha256_file(master_path),
        "historical_capacity": sha256_file(history_path),
        "population_archives": {
            str(path.relative_to(project)): sha256_file(path)
            for path in sorted(population_path.rglob("*.zip"))
        },
    }
    manifest = {
        "run_id": run_id,
        "model_version": "0.1.0",
        "objective": "minimize_65plus_population_weighted_e2sfca_gini",
        "status": "SCREENING_ONLY_NOT_POLICY_DEPLOYABLE",
        "config": config,
        "population_audit": asdict(population_audit),
        "population_sensitivity": population_sensitivity,
        "facility_audit": facility_audit,
        "capacity_validation": capacity_validation,
        "validation_gates": gates,
        "scenario_summary": scenario_summary,
        "candidate_counts": {
            "all_physical_grids": int(len(lower_population)),
            "preliminary_local_demand": int(len(preliminary_indices)),
            "valid_in_every_scenario": int(common_valid.sum()),
        },
        "sources": {
            "working_repository": _git_state(project),
            "planning_data_repository": _git_state(master_path.parents[3]),
            "input_hashes": input_hashes,
            "modeling_code_hashes": model_code_hashes(project),
        },
        "runtime_seconds_before_manifest": time.perf_counter() - started,
        "artifact_hashes": {},
    }
    write_run_report(
        output_path=run_dir / "MODEL_REPORT.md",
        run_id=run_id,
        manifest=manifest,
        scenario_summary=scenario_summary,
        base_top=base_top.head(5),
        robust_top=robust.head(5),
        metrics=metrics,
        gates=gates,
        capacity_validation=capacity_validation,
    )
    manifest["artifact_hashes"] = _artifact_hashes(run_dir)
    manifest["runtime_seconds_total"] = time.perf_counter() - started
    _write_json(run_dir / "run_manifest.json", manifest)
    return run_dir
