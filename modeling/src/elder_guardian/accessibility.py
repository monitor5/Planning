"""Sparse E2SFCA and weighted-inequality primitives.

All spatial coordinates are expected in a metric projected CRS.  The model uses
EPSG:5179 and a transparent walking-speed assumption until a validated
multimodal origin/destination matrix is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AccessibilityResult:
    accessibility: np.ndarray
    facility_ratio: np.ndarray
    demand_denominator: np.ndarray
    reachable_supply: float
    supplied_access: float
    conservation_error: float


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the population-weighted Gini coefficient in O(n log n).

    The function returns NaN when the total weight or weighted mean is zero.
    Negative accessibility is invalid and therefore rejected.
    """

    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.shape != w.shape:
        raise ValueError("values and weights must have identical shapes")
    finite = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[finite]
    w = w[finite]
    if x.size == 0 or w.sum() <= 0:
        return float("nan")
    if np.any(x < -1e-12):
        raise ValueError("weighted_gini requires non-negative values")
    x = np.maximum(x, 0.0)
    weighted_sum = float(np.dot(w, x))
    total_weight = float(w.sum())
    if weighted_sum <= 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    cumulative_weight_before = np.cumsum(ws) - ws
    cumulative_weighted_x_before = np.cumsum(ws * xs) - ws * xs
    pair_difference_sum = np.sum(
        ws * (xs * cumulative_weight_before - cumulative_weighted_x_before)
    )
    gini = float(pair_difference_sum / (total_weight * weighted_sum))
    return min(1.0, max(0.0, gini))


def gaussian_weight_matrix(
    origins_xy: np.ndarray,
    destinations_xy: np.ndarray,
    *,
    speed_m_per_min: float,
    tau_minutes: float,
    cutoff_minutes: float,
) -> sparse.csr_matrix:
    """Build sparse Gaussian travel weights from origins to destinations."""

    origins = np.asarray(origins_xy, dtype=np.float64)
    destinations = np.asarray(destinations_xy, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("origins_xy must have shape (n, 2)")
    if destinations.ndim != 2 or destinations.shape[1] != 2:
        raise ValueError("destinations_xy must have shape (m, 2)")
    if speed_m_per_min <= 0 or tau_minutes <= 0 or cutoff_minutes <= 0:
        raise ValueError("speed, tau, and cutoff must be positive")
    if origins.shape[0] == 0 or destinations.shape[0] == 0:
        return sparse.csr_matrix((origins.shape[0], destinations.shape[0]))

    max_distance = speed_m_per_min * cutoff_minutes
    distances = cKDTree(origins).sparse_distance_matrix(
        cKDTree(destinations), max_distance=max_distance, output_type="coo_matrix"
    )
    cost_minutes = distances.data / speed_m_per_min
    weights = np.exp(-np.square(cost_minutes / tau_minutes))
    return sparse.coo_matrix(
        (weights, (distances.row, distances.col)), shape=distances.shape
    ).tocsr()


def e2sfca(
    demand_population: np.ndarray,
    facility_capacity: np.ndarray,
    demand_to_facility_weights: sparse.spmatrix,
) -> AccessibilityResult:
    """Calculate enhanced two-step floating catchment accessibility.

    The conservation invariant is:
        sum_i N_i A_i == sum_j S_j
    for facilities with at least one reachable positive-demand cell.
    """

    population = np.asarray(demand_population, dtype=np.float64)
    capacity = np.asarray(facility_capacity, dtype=np.float64)
    matrix = sparse.csr_matrix(demand_to_facility_weights, dtype=np.float64)
    if matrix.shape != (population.size, capacity.size):
        raise ValueError("weight matrix dimensions do not match population/capacity")
    if np.any(population < 0) or np.any(capacity < 0):
        raise ValueError("population and capacity must be non-negative")

    denominator = np.asarray(matrix.T @ population).ravel()
    ratio = np.divide(
        capacity,
        denominator,
        out=np.zeros_like(capacity),
        where=denominator > 0,
    )
    accessibility = np.asarray(matrix @ ratio).ravel()
    reachable_supply = float(capacity[denominator > 0].sum())
    supplied_access = float(np.dot(population, accessibility))
    error = abs(reachable_supply - supplied_access)
    return AccessibilityResult(
        accessibility=accessibility,
        facility_ratio=ratio,
        demand_denominator=denominator,
        reachable_supply=reachable_supply,
        supplied_access=supplied_access,
        conservation_error=error,
    )

