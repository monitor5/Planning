"""Exact, sparse counterfactual Gini evaluation for one added facility.

The PDF's E2SFCA definition makes a one-facility counterfactual especially
simple.  For demand weights ``N_i``, candidate kernel weights ``w_ip`` and
candidate capacity ``s_p``,

``D_p = sum_i N_i w_ip`` and ``delta_i = s_p w_ip / D_p``.

Adding the candidate does not change any existing facility ratio, so the new
accessibility is ``A'_i = A_i + delta_i``.  Consequently,
``sum_i N_i delta_i == s_p`` whenever ``D_p > 0``.  This is the supply
conservation invariant used throughout this module.

For positive-demand cells, let ``W = sum_i N_i``, ``Y = sum_i N_i A_i`` and

``H(A) = sum_{i<h} N_i N_h |A_i - A_h|``.

The exact population-weighted Gini is ``G(A) = H(A) / (W Y)``.  A candidate
changes only its sparse catchment.  The evaluator sorts the baseline once and
updates ``H`` from that local support in ``O(s log n + s log s)`` time, where
``s`` is the candidate support size.  It never copies or re-sorts all ``n``
accessibility values for each candidate.

The Gini itself is not monotone in added supply: concentrating new access in an
already well-served area can increase inequality.  Do not clip a negative
``delta_gini`` (baseline Gini minus post-installation Gini) to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = [
    "CandidateBatchResult",
    "CandidateGiniResult",
    "CandidateIncrement",
    "ExactCandidateEvaluation",
    "ExactGiniEvaluator",
    "UndefinedBaselineGiniError",
    "ZeroCandidateDemandError",
    "evaluate_candidates_exact",
]

ZeroDenominatorPolicy = Literal["invalid", "raise"]


class UndefinedBaselineGiniError(ValueError):
    """Raised when the baseline weighted mean is zero and Gini is undefined."""


class ZeroCandidateDemandError(ValueError):
    """Raised when a candidate has no positive weighted demand in its catchment."""


@dataclass(frozen=True)
class CandidateIncrement:
    """Sparse accessibility increment produced by one candidate.

    ``demand_indices`` refer to the original arrays passed to
    :class:`ExactGiniEvaluator`.  Zero-population cells are omitted because they
    affect neither the E2SFCA denominator nor the population-weighted Gini.
    """

    demand_indices: np.ndarray
    accessibility_delta: np.ndarray
    capacity: float
    demand_denominator: float
    supplied_access: float
    conservation_error: float
    valid: bool
    status: str


@dataclass(frozen=True)
class CandidateGiniResult:
    """Exact before/after Gini result for one candidate."""

    capacity: float
    support_size: int
    demand_denominator: float
    gini_before: float
    gini_after: float
    delta_gini: float
    weighted_access_before: float
    weighted_access_after: float
    conservation_error: float
    valid: bool
    status: str


@dataclass(frozen=True)
class CandidateBatchResult:
    """Columnar exact results for a candidate-by-demand sparse matrix."""

    gini_after: np.ndarray
    delta_gini: np.ndarray
    demand_denominator: np.ndarray
    weighted_access_after: np.ndarray
    conservation_error: np.ndarray
    support_size: np.ndarray
    valid: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class ExactCandidateEvaluation:
    """Pipeline-facing result returned by :func:`evaluate_candidates_exact`.

    A candidate whose denominator is zero has ``NaN`` in both Gini arrays and
    zero in ``denominators``.  This keeps its row auditable without silently
    treating a disconnected candidate as a valid zero-benefit location.
    """

    baseline_gini: float
    candidate_gini: np.ndarray
    delta_gini: np.ndarray
    denominators: np.ndarray
    runtime_seconds: float


def _unordered_weighted_absolute_sum(
    values: np.ndarray, weights: np.ndarray
) -> float:
    """Return ``sum_{i<h} w_i w_h |x_i-x_h|`` in O(n log n)."""

    if values.size < 2:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    x = values[order]
    w = weights[order]
    cumulative_w = np.cumsum(w)
    cumulative_wx = np.cumsum(w * x)
    return float(
        np.sum(w[1:] * (x[1:] * cumulative_w[:-1] - cumulative_wx[:-1]))
    )


def _absolute_deviation_from_sorted(
    query: np.ndarray,
    sorted_values: np.ndarray,
    cumulative_weights: np.ndarray,
    cumulative_weighted_values: np.ndarray,
) -> np.ndarray:
    """Return ``sum_h w_h |query-x_h|`` using sorted weighted observations."""

    positions = np.searchsorted(sorted_values, query, side="right")
    left_weight = cumulative_weights[positions]
    left_value = cumulative_weighted_values[positions]
    total_weight = cumulative_weights[-1]
    total_value = cumulative_weighted_values[-1]
    return (
        query * left_weight
        - left_value
        + (total_value - left_value)
        - query * (total_weight - left_weight)
    )


def _sorted_absolute_index(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build arrays used by :func:`_absolute_deviation_from_sorted`."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_weights = np.concatenate(
        ([0.0], np.cumsum(sorted_weights, dtype=np.float64))
    )
    cumulative_values = np.concatenate(
        ([0.0], np.cumsum(sorted_weights * sorted_values, dtype=np.float64))
    )
    return sorted_values, cumulative_weights, cumulative_values


class ExactGiniEvaluator:
    """Evaluate sparse one-facility E2SFCA counterfactuals exactly.

    Parameters
    ----------
    baseline_accessibility:
        Baseline E2SFCA accessibility for every demand row.
    demand_population:
        Non-negative population weights for the same rows.  Zero-population
        rows are retained in the external indexing but excluded internally.
    baseline_reachable_supply:
        Optional reachable baseline supply.  When supplied, construction fails
        unless it equals ``sum_i N_i A_i`` within ``conservation_rtol``.  Passing
        this value turns supply conservation from a diagnostic into a hard
        input invariant.
    conservation_rtol:
        Relative tolerance used for the optional baseline invariant.

    Notes
    -----
    The baseline weighted access ``Y`` must be positive.  In the current data,
    the ``STRICT_UNKNOWN`` capacity scenario has zero supply, so its baseline
    Gini is undefined by the PDF and this class deliberately raises
    :class:`UndefinedBaselineGiniError`.
    """

    def __init__(
        self,
        baseline_accessibility: np.ndarray,
        demand_population: np.ndarray,
        *,
        baseline_reachable_supply: float | None = None,
        conservation_rtol: float = 1e-10,
    ) -> None:
        accessibility = np.asarray(baseline_accessibility, dtype=np.float64)
        population = np.asarray(demand_population, dtype=np.float64)
        if accessibility.ndim != 1 or population.ndim != 1:
            raise ValueError("baseline_accessibility and demand_population must be 1-D")
        if accessibility.shape != population.shape:
            raise ValueError(
                "baseline_accessibility and demand_population must have equal length"
            )
        if accessibility.size == 0:
            raise ValueError("at least one demand row is required")
        if not np.all(np.isfinite(accessibility)):
            raise ValueError("baseline_accessibility must contain only finite values")
        if not np.all(np.isfinite(population)):
            raise ValueError("demand_population must contain only finite values")
        if np.any(accessibility < -1e-12):
            raise ValueError("baseline_accessibility must be non-negative")
        if np.any(population < 0):
            raise ValueError("demand_population must be non-negative")
        if not np.isfinite(conservation_rtol) or conservation_rtol < 0:
            raise ValueError("conservation_rtol must be finite and non-negative")

        accessibility = np.maximum(accessibility, 0.0)
        active_mask = population > 0
        if not np.any(active_mask):
            raise UndefinedBaselineGiniError("positive total population is required")

        self._row_count = population.size
        self._active_original_indices = np.flatnonzero(active_mask)
        self._full_to_active = np.full(population.size, -1, dtype=np.int64)
        self._full_to_active[self._active_original_indices] = np.arange(
            self._active_original_indices.size, dtype=np.int64
        )
        self._population = population[active_mask]
        self._accessibility = accessibility[active_mask]
        self.total_population = float(self._population.sum())
        self.weighted_access_before = float(
            np.dot(self._population, self._accessibility)
        )
        if self.weighted_access_before <= 0:
            raise UndefinedBaselineGiniError(
                "baseline weighted mean is zero; weighted Gini is undefined"
            )

        self._baseline_pair_sum = _unordered_weighted_absolute_sum(
            self._accessibility, self._population
        )
        self.gini_before = self._gini_from_pair_sum(
            self._baseline_pair_sum, self.weighted_access_before
        )
        (
            self._sorted_accessibility,
            self._cumulative_population,
            self._cumulative_weighted_access,
        ) = _sorted_absolute_index(self._accessibility, self._population)

        self.baseline_conservation_error = float("nan")
        if baseline_reachable_supply is not None:
            supplied = float(baseline_reachable_supply)
            if not np.isfinite(supplied) or supplied < 0:
                raise ValueError(
                    "baseline_reachable_supply must be finite and non-negative"
                )
            self.baseline_conservation_error = abs(
                self.weighted_access_before - supplied
            )
            if not np.isclose(
                self.weighted_access_before,
                supplied,
                rtol=conservation_rtol,
                atol=conservation_rtol,
            ):
                raise ValueError(
                    "baseline violates E2SFCA supply conservation: "
                    f"weighted access={self.weighted_access_before}, "
                    f"reachable supply={supplied}"
                )

    @property
    def row_count(self) -> int:
        """Number of rows in the original demand arrays."""

        return self._row_count

    @property
    def active_row_count(self) -> int:
        """Number of positive-population rows used by the weighted Gini."""

        return self._population.size

    def _gini_from_pair_sum(self, pair_sum: float, weighted_access: float) -> float:
        value = pair_sum / (self.total_population * weighted_access)
        if -1e-12 <= value < 0:
            return 0.0
        if 1 < value <= 1 + 1e-12:
            return 1.0
        return float(value)

    def _invalid_increment(
        self, capacity: float, *, on_zero_denominator: ZeroDenominatorPolicy
    ) -> CandidateIncrement:
        if on_zero_denominator == "raise":
            raise ZeroCandidateDemandError(
                "candidate E2SFCA denominator is zero; no positive weighted "
                "demand is reachable"
            )
        if on_zero_denominator != "invalid":
            raise ValueError("on_zero_denominator must be 'invalid' or 'raise'")
        return CandidateIncrement(
            demand_indices=np.empty(0, dtype=np.int64),
            accessibility_delta=np.empty(0, dtype=np.float64),
            capacity=capacity,
            demand_denominator=0.0,
            supplied_access=float("nan"),
            conservation_error=float("nan"),
            valid=False,
            status="ZERO_DEMAND_DENOMINATOR",
        )

    def candidate_increment(
        self,
        demand_indices: np.ndarray,
        kernel_weights: np.ndarray,
        capacity: float,
        *,
        on_zero_denominator: ZeroDenominatorPolicy = "invalid",
    ) -> CandidateIncrement:
        """Return the sparse normalized E2SFCA increment for one candidate.

        The returned increment satisfies ``sum_i N_i delta_i == capacity`` to
        floating-point tolerance.  Thus increasing non-negative capacity while
        holding the kernel fixed increases every affected ``delta_i`` linearly.

        A denominator of zero is a connectivity/data error, not a zero-impact
        valid candidate.  The default returns an invalid result that batch
        pipelines can retain and audit; ``on_zero_denominator="raise"`` raises
        :class:`ZeroCandidateDemandError`.
        """

        indices = np.asarray(demand_indices)
        weights = np.asarray(kernel_weights, dtype=np.float64)
        supplied_capacity = float(capacity)
        if indices.ndim != 1 or weights.ndim != 1 or indices.shape != weights.shape:
            raise ValueError("demand_indices and kernel_weights must be equal-length 1-D arrays")
        if not np.issubdtype(indices.dtype, np.integer):
            if np.any(indices != np.floor(indices)):
                raise ValueError("demand_indices must contain integers")
            indices = indices.astype(np.int64)
        else:
            indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self._row_count):
            raise IndexError("candidate demand index is out of bounds")
        if np.unique(indices).size != indices.size:
            raise ValueError("candidate demand_indices must not contain duplicates")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("kernel_weights must be finite and non-negative")
        if not np.isfinite(supplied_capacity) or supplied_capacity < 0:
            raise ValueError("capacity must be finite and non-negative")
        if on_zero_denominator not in ("invalid", "raise"):
            raise ValueError("on_zero_denominator must be 'invalid' or 'raise'")

        active_positions = self._full_to_active[indices]
        retained = (active_positions >= 0) & (weights > 0)
        if not np.any(retained):
            return self._invalid_increment(
                supplied_capacity, on_zero_denominator=on_zero_denominator
            )

        original_indices = indices[retained]
        active_positions = active_positions[retained]
        active_weights = weights[retained]
        population = self._population[active_positions]
        denominator = float(np.dot(population, active_weights))
        if denominator <= 0:
            return self._invalid_increment(
                supplied_capacity, on_zero_denominator=on_zero_denominator
            )

        delta = active_weights * (supplied_capacity / denominator)
        supplied_access = float(np.dot(population, delta))
        return CandidateIncrement(
            demand_indices=original_indices.copy(),
            accessibility_delta=delta,
            capacity=supplied_capacity,
            demand_denominator=denominator,
            supplied_access=supplied_access,
            conservation_error=abs(supplied_access - supplied_capacity),
            valid=True,
            status="VALID",
        )

    def _global_absolute_deviation(self, query: np.ndarray) -> np.ndarray:
        return _absolute_deviation_from_sorted(
            query,
            self._sorted_accessibility,
            self._cumulative_population,
            self._cumulative_weighted_access,
        )

    def evaluate_candidate(
        self,
        demand_indices: np.ndarray,
        kernel_weights: np.ndarray,
        capacity: float,
        *,
        on_zero_denominator: ZeroDenominatorPolicy = "invalid",
    ) -> CandidateGiniResult:
        """Return the exact post-installation Gini and ``delta_gini``.

        ``delta_gini`` is ``gini_before - gini_after``; larger is better under
        the PDF's inequality objective.  The implementation uses the identity

        ``H' = H + T - M + H_S(A) + H_S(A')``,

        where ``S`` is the sparse changed support,
        ``T = sum_{i in S} N_i[L(A'_i)-L(A_i)]``,
        ``L(z)=sum_h N_h|z-A_h|``, and
        ``M=sum_{i in S}N_i sum_{h in S}N_h|A'_i-A_h|``.
        Prefix sums evaluate each ``L`` exactly without a full candidate sort.
        """

        increment = self.candidate_increment(
            demand_indices,
            kernel_weights,
            capacity,
            on_zero_denominator=on_zero_denominator,
        )
        if not increment.valid:
            return CandidateGiniResult(
                capacity=increment.capacity,
                support_size=0,
                demand_denominator=increment.demand_denominator,
                gini_before=self.gini_before,
                gini_after=float("nan"),
                delta_gini=float("nan"),
                weighted_access_before=self.weighted_access_before,
                weighted_access_after=float("nan"),
                conservation_error=increment.conservation_error,
                valid=False,
                status=increment.status,
            )

        active_positions = self._full_to_active[increment.demand_indices]
        old_values = self._accessibility[active_positions]
        new_values = old_values + increment.accessibility_delta
        local_population = self._population[active_positions]

        if increment.capacity == 0:
            pair_sum_after = self._baseline_pair_sum
        else:
            global_change = float(
                np.dot(
                    local_population,
                    self._global_absolute_deviation(new_values)
                    - self._global_absolute_deviation(old_values),
                )
            )
            (
                old_sorted,
                local_cumulative_population,
                local_cumulative_access,
            ) = _sorted_absolute_index(old_values, local_population)
            mixed_sum = float(
                np.dot(
                    local_population,
                    _absolute_deviation_from_sorted(
                        new_values,
                        old_sorted,
                        local_cumulative_population,
                        local_cumulative_access,
                    ),
                )
            )
            local_old_sum = _unordered_weighted_absolute_sum(
                old_values, local_population
            )
            local_new_sum = _unordered_weighted_absolute_sum(
                new_values, local_population
            )
            pair_sum_after = (
                self._baseline_pair_sum
                + global_change
                - mixed_sum
                + local_old_sum
                + local_new_sum
            )

        weighted_access_after = (
            self.weighted_access_before + increment.supplied_access
        )
        gini_after = self._gini_from_pair_sum(
            pair_sum_after, weighted_access_after
        )
        return CandidateGiniResult(
            capacity=increment.capacity,
            support_size=increment.demand_indices.size,
            demand_denominator=increment.demand_denominator,
            gini_before=self.gini_before,
            gini_after=gini_after,
            delta_gini=self.gini_before - gini_after,
            weighted_access_before=self.weighted_access_before,
            weighted_access_after=weighted_access_after,
            conservation_error=increment.conservation_error,
            valid=True,
            status="VALID",
        )

    def evaluate_sparse_candidates(
        self,
        candidate_to_demand_weights: sparse.spmatrix,
        capacities: np.ndarray | float,
        *,
        on_zero_denominator: ZeroDenominatorPolicy = "invalid",
    ) -> CandidateBatchResult:
        """Evaluate every row of a candidate-by-demand sparse kernel matrix.

        This method keeps only one candidate support live at a time.  A caller
        can therefore stream candidate batches instead of materializing a dense
        ``candidate_count x demand_count`` matrix.
        """

        matrix = sparse.csr_matrix(
            candidate_to_demand_weights, dtype=np.float64, copy=True
        )
        if matrix.shape[1] != self._row_count:
            raise ValueError(
                "candidate matrix columns must match the original demand rows"
            )
        if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
            raise ValueError("candidate matrix weights must be finite and non-negative")
        matrix.sum_duplicates()
        candidate_count = matrix.shape[0]
        capacity_array = np.asarray(capacities, dtype=np.float64)
        if capacity_array.ndim == 0:
            capacity_array = np.full(candidate_count, float(capacity_array))
        if capacity_array.shape != (candidate_count,):
            raise ValueError("capacities must be scalar or one value per candidate")
        if not np.all(np.isfinite(capacity_array)) or np.any(capacity_array < 0):
            raise ValueError("capacities must be finite and non-negative")

        gini_after = np.full(candidate_count, np.nan, dtype=np.float64)
        delta_gini = np.full(candidate_count, np.nan, dtype=np.float64)
        denominator = np.zeros(candidate_count, dtype=np.float64)
        weighted_after = np.full(candidate_count, np.nan, dtype=np.float64)
        conservation_error = np.full(candidate_count, np.nan, dtype=np.float64)
        support_size = np.zeros(candidate_count, dtype=np.int32)
        valid = np.zeros(candidate_count, dtype=bool)
        status = np.full(
            candidate_count, "ZERO_DEMAND_DENOMINATOR", dtype="<U23"
        )

        for candidate in range(candidate_count):
            start, end = matrix.indptr[candidate : candidate + 2]
            result = self.evaluate_candidate(
                matrix.indices[start:end],
                matrix.data[start:end],
                capacity_array[candidate],
                on_zero_denominator=on_zero_denominator,
            )
            denominator[candidate] = result.demand_denominator
            support_size[candidate] = result.support_size
            conservation_error[candidate] = result.conservation_error
            valid[candidate] = result.valid
            status[candidate] = result.status
            if result.valid:
                gini_after[candidate] = result.gini_after
                delta_gini[candidate] = result.delta_gini
                weighted_after[candidate] = result.weighted_access_after

        return CandidateBatchResult(
            gini_after=gini_after,
            delta_gini=delta_gini,
            demand_denominator=denominator,
            weighted_access_after=weighted_after,
            conservation_error=conservation_error,
            support_size=support_size,
            valid=valid,
            status=status,
        )


def evaluate_candidates_exact(
    baseline_accessibility: np.ndarray,
    demand_population: np.ndarray,
    demand_xy: np.ndarray,
    candidate_xy: np.ndarray,
    *,
    new_capacity: np.ndarray | float,
    speed_m_per_min: float,
    tau_minutes: float,
    cutoff_minutes: float,
) -> ExactCandidateEvaluation:
    """Build sparse Gaussian catchments and score all candidates exactly.

    Parameters
    ----------
    baseline_accessibility, demand_population:
        Baseline E2SFCA access and population for the same demand rows.
    demand_xy, candidate_xy:
        ``(n, 2)`` and ``(k, 2)`` coordinates in a metric projected CRS such as
        EPSG:5179.  With no validated pedestrian graph this is explicitly a
        straight-line walk-only baseline, not a network walking-time result.
    new_capacity:
        A non-negative scalar shared by all candidates, or one value per
        candidate.
    speed_m_per_min, tau_minutes, cutoff_minutes:
        Pre-registered Gaussian walking parameters.  For distance ``d``, the
        retained kernel is ``exp(-((d/speed)/tau)^2)`` when
        ``d/speed <= cutoff``.

    Returns
    -------
    ExactCandidateEvaluation
        The baseline Gini plus exact post-installation and delta Gini arrays.
        Runtime covers validation, sparse-neighbor construction and scoring.

    Notes
    -----
    Only positive-population demand rows enter the KD-tree.  At the current
    Seoul scale this avoids both a dense ``61k x 61k`` candidate matrix and
    work on rows that have zero Gini weight.  The sparse matrix is
    candidate-by-demand; each row is released after local-delta scoring.
    """

    started = perf_counter()
    accessibility = np.asarray(baseline_accessibility, dtype=np.float64)
    population = np.asarray(demand_population, dtype=np.float64)
    demand_coordinates = np.asarray(demand_xy, dtype=np.float64)
    candidate_coordinates = np.asarray(candidate_xy, dtype=np.float64)
    if accessibility.ndim != 1 or population.ndim != 1:
        raise ValueError("baseline_accessibility and demand_population must be 1-D")
    if accessibility.shape != population.shape:
        raise ValueError(
            "baseline_accessibility and demand_population must have equal length"
        )
    if demand_coordinates.shape != (population.size, 2):
        raise ValueError("demand_xy must have shape (demand_count, 2)")
    if candidate_coordinates.ndim != 2 or candidate_coordinates.shape[1] != 2:
        raise ValueError("candidate_xy must have shape (candidate_count, 2)")
    if not np.all(np.isfinite(demand_coordinates)):
        raise ValueError("demand_xy must contain only finite coordinates")
    if not np.all(np.isfinite(candidate_coordinates)):
        raise ValueError("candidate_xy must contain only finite coordinates")
    parameters = np.array(
        [speed_m_per_min, tau_minutes, cutoff_minutes], dtype=np.float64
    )
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0):
        raise ValueError("speed, tau, and cutoff must be finite and positive")

    evaluator = ExactGiniEvaluator(accessibility, population)
    candidate_count = candidate_coordinates.shape[0]
    if candidate_count == 0:
        empty = np.empty(0, dtype=np.float64)
        return ExactCandidateEvaluation(
            baseline_gini=evaluator.gini_before,
            candidate_gini=empty.copy(),
            delta_gini=empty.copy(),
            denominators=empty.copy(),
            runtime_seconds=perf_counter() - started,
        )

    active_original_indices = np.flatnonzero(population > 0)
    active_coordinates = demand_coordinates[active_original_indices]
    maximum_distance = speed_m_per_min * cutoff_minutes
    distances = cKDTree(candidate_coordinates).sparse_distance_matrix(
        cKDTree(active_coordinates),
        max_distance=maximum_distance,
        output_type="coo_matrix",
    )
    travel_minutes = distances.data / speed_m_per_min
    gaussian_weights = np.exp(-np.square(travel_minutes / tau_minutes))
    candidate_matrix = sparse.coo_matrix(
        (
            gaussian_weights,
            (distances.row, active_original_indices[distances.col]),
        ),
        shape=(candidate_count, population.size),
    ).tocsr()
    batch = evaluator.evaluate_sparse_candidates(
        candidate_matrix,
        new_capacity,
        on_zero_denominator="invalid",
    )
    return ExactCandidateEvaluation(
        baseline_gini=evaluator.gini_before,
        candidate_gini=batch.gini_after,
        delta_gini=batch.delta_gini,
        denominators=batch.demand_denominator,
        runtime_seconds=perf_counter() - started,
    )
