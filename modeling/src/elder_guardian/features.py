"""Pre-install candidate features for leakage-safe surrogate training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .accessibility import e2sfca, gaussian_weight_matrix


FEATURE_COLUMNS = [
    "log_population",
    "log_local_population_500m",
    "log_local_population_1000m",
    "log_local_population_2000m",
    "log_baseline_accessibility",
    "log_local_supply_500m",
    "log_local_supply_1000m",
    "log_local_supply_2000m",
    "log_nearest_facility_distance",
    "log_demand_supply_ratio_1000m",
    "log_candidate_weighted_demand",
    "log_candidate_baseline_access_mean",
    "log_candidate_baseline_access_std",
    "candidate_low_access_share",
    "log_candidate_low_access_gap",
    "new_capacity",
    "tau_minutes",
    "cutoff_minutes",
    "historical_reference_mode",
]

FORBIDDEN_POST_INSTALL_FEATURES = {
    "candidate_accessibility",
    "post_install_accessibility",
    "post_install_gini",
    "delta_gini",
    "exact_rank",
}


@dataclass(frozen=True)
class FeatureResult:
    features: pd.DataFrame
    demand_index: np.ndarray
    demand_population: np.ndarray
    baseline_demand_accessibility: np.ndarray
    baseline_all_accessibility: np.ndarray
    baseline_gini: float
    conservation_error: float


def _radius_sum(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    source_values: np.ndarray,
    radius: float,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    tree = cKDTree(source_xy)
    values = np.asarray(source_values, dtype=np.float64)
    output = np.zeros(len(query_xy), dtype=np.float64)
    for start in range(0, len(query_xy), chunk_size):
        stop = min(start + chunk_size, len(query_xy))
        neighbors = tree.query_ball_point(query_xy[start:stop], radius)
        output[start:stop] = [
            float(values[np.asarray(index, dtype=int)].sum()) if len(index) else 0.0
            for index in neighbors
        ]
    return output


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = quantile * cumulative[-1]
    return float(sorted_values[min(np.searchsorted(cumulative, cutoff), len(order) - 1)])


def build_candidate_features(
    population: pd.DataFrame,
    facilities: pd.DataFrame,
    scenario: dict,
    *,
    speed_m_per_min: float,
) -> FeatureResult:
    """Build only information available before adding the candidate facility."""

    from .accessibility import weighted_gini

    all_xy = population[["x", "y"]].to_numpy(dtype=float)
    all_population = population["population"].to_numpy(dtype=float)
    demand_index = np.flatnonzero(all_population > 0)
    demand_xy = all_xy[demand_index]
    demand_population = all_population[demand_index]
    facility_xy = facilities[["x", "y"]].to_numpy(dtype=float)
    capacity = facilities["model_capacity"].to_numpy(dtype=float)
    matrix = gaussian_weight_matrix(
        demand_xy,
        facility_xy,
        speed_m_per_min=speed_m_per_min,
        tau_minutes=float(scenario["tau_minutes"]),
        cutoff_minutes=float(scenario["cutoff_minutes"]),
    )
    baseline = e2sfca(demand_population, capacity, matrix)
    baseline_gini = weighted_gini(baseline.accessibility, demand_population)
    matrix_all = gaussian_weight_matrix(
        all_xy,
        facility_xy,
        speed_m_per_min=speed_m_per_min,
        tau_minutes=float(scenario["tau_minutes"]),
        cutoff_minutes=float(scenario["cutoff_minutes"]),
    )
    all_accessibility = np.asarray(matrix_all @ baseline.facility_ratio).ravel()

    # Candidate-kernel summaries use only the baseline state.  They do not use
    # post-install accessibility, Gini, delta Gini, or an exact rank.  These
    # summaries give the surrogate the same pre-install catchment geometry that
    # will be available at inference time.
    candidate_to_demand = gaussian_weight_matrix(
        all_xy,
        demand_xy,
        speed_m_per_min=speed_m_per_min,
        tau_minutes=float(scenario["tau_minutes"]),
        cutoff_minutes=float(scenario["cutoff_minutes"]),
    )
    candidate_denominator = np.asarray(
        candidate_to_demand @ demand_population
    ).ravel()
    baseline_weighted = demand_population * baseline.accessibility
    access_mean = np.divide(
        np.asarray(candidate_to_demand @ baseline_weighted).ravel(),
        candidate_denominator,
        out=np.zeros(len(all_xy), dtype=float),
        where=candidate_denominator > 0,
    )
    access_second_moment = np.divide(
        np.asarray(
            candidate_to_demand
            @ (demand_population * np.square(baseline.accessibility))
        ).ravel(),
        candidate_denominator,
        out=np.zeros(len(all_xy), dtype=float),
        where=candidate_denominator > 0,
    )
    access_std = np.sqrt(
        np.maximum(access_second_moment - np.square(access_mean), 0.0)
    )
    low_threshold = _weighted_quantile(
        baseline.accessibility, demand_population, 0.25
    )
    low_mask = baseline.accessibility <= low_threshold
    low_population = np.asarray(
        candidate_to_demand @ (demand_population * low_mask)
    ).ravel()
    low_share = np.divide(
        low_population,
        candidate_denominator,
        out=np.zeros(len(all_xy), dtype=float),
        where=candidate_denominator > 0,
    )
    low_gap = np.maximum(low_threshold - baseline.accessibility, 0.0)
    low_gap_mean = np.divide(
        np.asarray(candidate_to_demand @ (demand_population * low_gap)).ravel(),
        candidate_denominator,
        out=np.zeros(len(all_xy), dtype=float),
        where=candidate_denominator > 0,
    )

    local_population = {
        radius: _radius_sum(all_xy, demand_xy, demand_population, radius)
        for radius in (500.0, 1000.0, 2000.0)
    }
    local_supply = {
        radius: _radius_sum(all_xy, facility_xy, capacity, radius)
        for radius in (500.0, 1000.0, 2000.0)
    }
    nearest_distance = cKDTree(facility_xy).query(all_xy, k=1)[0]
    ratio = local_population[1000.0] / (local_supply[1000.0] + 1.0)

    feature_values = {
        "log_population": np.log1p(all_population),
        "log_local_population_500m": np.log1p(local_population[500.0]),
        "log_local_population_1000m": np.log1p(local_population[1000.0]),
        "log_local_population_2000m": np.log1p(local_population[2000.0]),
        "log_baseline_accessibility": np.log1p(all_accessibility * 1_000_000.0),
        "log_local_supply_500m": np.log1p(local_supply[500.0]),
        "log_local_supply_1000m": np.log1p(local_supply[1000.0]),
        "log_local_supply_2000m": np.log1p(local_supply[2000.0]),
        "log_nearest_facility_distance": np.log1p(nearest_distance),
        "log_demand_supply_ratio_1000m": np.log1p(ratio),
        "log_candidate_weighted_demand": np.log1p(candidate_denominator),
        "log_candidate_baseline_access_mean": np.log1p(
            access_mean * 1_000_000.0
        ),
        "log_candidate_baseline_access_std": np.log1p(
            access_std * 1_000_000.0
        ),
        "candidate_low_access_share": low_share,
        "log_candidate_low_access_gap": np.log1p(
            low_gap_mean * 1_000_000.0
        ),
        "new_capacity": np.full(len(population), float(scenario["new_capacity"])),
        "tau_minutes": np.full(len(population), float(scenario["tau_minutes"])),
        "cutoff_minutes": np.full(
            len(population), float(scenario["cutoff_minutes"])
        ),
        "historical_reference_mode": np.full(
            len(population),
            float(
                scenario["existing_capacity_mode"]
                == "HISTORICAL_REFERENCE_WINSORIZED"
            ),
        ),
    }
    features = pd.DataFrame(feature_values, index=population.index)
    assert_no_target_leakage(features.columns)
    return FeatureResult(
        features=features,
        demand_index=demand_index,
        demand_population=demand_population,
        baseline_demand_accessibility=baseline.accessibility,
        baseline_all_accessibility=all_accessibility,
        baseline_gini=baseline_gini,
        conservation_error=baseline.conservation_error,
    )


def assert_no_target_leakage(columns) -> None:
    normalized = {str(column).strip().lower() for column in columns}
    found = normalized.intersection(FORBIDDEN_POST_INSTALL_FEATURES)
    if found:
        raise ValueError(f"post-install target leakage columns found: {sorted(found)}")
