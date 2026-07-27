import numpy as np
import pytest
from scipy import sparse

from elder_guardian.accessibility import e2sfca, weighted_gini
from elder_guardian.exact import (
    ExactGiniEvaluator,
    UndefinedBaselineGiniError,
    ZeroCandidateDemandError,
    evaluate_candidates_exact,
)


def _full_counterfactual_gini(
    baseline_accessibility, population, indices, kernel, capacity
):
    denominator = np.dot(population[indices], kernel)
    after = baseline_accessibility.copy()
    after[indices] += capacity * kernel / denominator
    return weighted_gini(after, population), after


def test_local_delta_matches_full_e2sfca_recompute_and_full_sort():
    population = np.array([10.0, 0.0, 25.0, 15.0, 30.0])
    baseline_weights = sparse.csr_matrix(
        [
            [1.0, 0.0],
            [0.4, 0.0],
            [0.6, 0.2],
            [0.1, 0.9],
            [0.0, 0.7],
        ]
    )
    baseline_capacity = np.array([35.0, 42.0])
    baseline = e2sfca(population, baseline_capacity, baseline_weights)
    evaluator = ExactGiniEvaluator(
        baseline.accessibility,
        population,
        baseline_reachable_supply=baseline.reachable_supply,
    )

    candidate_indices = np.array([0, 1, 2, 4])
    candidate_kernel = np.array([0.3, 0.8, 0.9, 0.2])
    candidate_capacity = 17.0
    result = evaluator.evaluate_candidate(
        candidate_indices, candidate_kernel, candidate_capacity
    )

    candidate_column = sparse.csr_matrix(
        (
            candidate_kernel,
            (candidate_indices, np.zeros(candidate_indices.size, dtype=int)),
        ),
        shape=(population.size, 1),
    )
    full_matrix = sparse.hstack([baseline_weights, candidate_column], format="csr")
    full = e2sfca(
        population,
        np.append(baseline_capacity, candidate_capacity),
        full_matrix,
    )
    expected_gini = weighted_gini(full.accessibility, population)

    assert result.valid
    assert np.isclose(result.gini_before, weighted_gini(baseline.accessibility, population))
    assert np.isclose(result.gini_after, expected_gini, atol=1e-14)
    assert np.isclose(result.weighted_access_after, full.supplied_access)
    assert result.conservation_error < 1e-12


def test_random_sparse_local_delta_matches_full_sort_to_machine_precision():
    rng = np.random.default_rng(20260727)
    population = rng.integers(1, 80, size=80).astype(float)
    baseline_accessibility = rng.uniform(0.01, 2.0, size=80)
    evaluator = ExactGiniEvaluator(baseline_accessibility, population)

    for _ in range(50):
        support_size = int(rng.integers(1, 25))
        indices = np.sort(
            rng.choice(population.size, size=support_size, replace=False)
        )
        kernel = rng.uniform(0.01, 1.0, size=support_size)
        capacity = float(rng.uniform(0.1, 100.0))
        exact = evaluator.evaluate_candidate(indices, kernel, capacity)
        expected_gini, _ = _full_counterfactual_gini(
            baseline_accessibility, population, indices, kernel, capacity
        )
        assert np.isclose(exact.gini_after, expected_gini, atol=5e-14)


def test_increment_is_capacity_monotone_and_conserves_candidate_supply():
    population = np.array([12.0, 20.0, 8.0, 0.0])
    baseline_accessibility = np.array([0.2, 0.7, 0.4, 9.0])
    evaluator = ExactGiniEvaluator(baseline_accessibility, population)
    indices = np.array([0, 1, 2, 3])
    kernel = np.array([1.0, 0.5, 0.2, 0.9])

    low = evaluator.candidate_increment(indices, kernel, 10.0)
    high = evaluator.candidate_increment(indices, kernel, 25.0)

    assert low.valid and high.valid
    assert np.array_equal(low.demand_indices, np.array([0, 1, 2]))
    assert np.all(high.accessibility_delta >= low.accessibility_delta)
    assert np.allclose(high.accessibility_delta, 2.5 * low.accessibility_delta)
    assert np.isclose(low.supplied_access, 10.0)
    assert np.isclose(high.supplied_access, 25.0)
    assert low.conservation_error < 1e-12
    assert high.conservation_error < 1e-12


def test_gini_is_not_incorrectly_clipped_when_added_supply_worsens_inequality():
    population = np.ones(2)
    baseline_accessibility = np.array([1.0, 0.5])
    evaluator = ExactGiniEvaluator(baseline_accessibility, population)

    result = evaluator.evaluate_candidate(np.array([0]), np.array([1.0]), 1.0)

    assert result.gini_after > result.gini_before
    assert result.delta_gini < 0


def test_zero_denominator_is_auditable_or_raises_by_policy():
    population = np.array([10.0, 0.0])
    baseline_accessibility = np.array([0.5, 0.0])
    evaluator = ExactGiniEvaluator(baseline_accessibility, population)

    invalid = evaluator.evaluate_candidate(
        np.array([1]), np.array([0.7]), 20.0
    )
    assert not invalid.valid
    assert invalid.status == "ZERO_DEMAND_DENOMINATOR"
    assert invalid.demand_denominator == 0
    assert np.isnan(invalid.gini_after)
    assert np.isnan(invalid.delta_gini)

    with pytest.raises(ZeroCandidateDemandError):
        evaluator.evaluate_candidate(
            np.array([1]),
            np.array([0.7]),
            20.0,
            on_zero_denominator="raise",
        )


def test_sparse_batch_matches_individual_results_and_marks_empty_row_invalid():
    population = np.array([5.0, 8.0, 13.0])
    baseline_accessibility = np.array([0.4, 0.2, 0.7])
    evaluator = ExactGiniEvaluator(baseline_accessibility, population)
    candidates = sparse.csr_matrix(
        [
            [1.0, 0.3, 0.0],
            [0.0, 0.2, 0.8],
            [0.0, 0.0, 0.0],
        ]
    )
    capacities = np.array([10.0, 15.0, 20.0])

    batch = evaluator.evaluate_sparse_candidates(candidates, capacities)
    first = evaluator.evaluate_candidate(
        np.array([0, 1]), np.array([1.0, 0.3]), 10.0
    )
    second = evaluator.evaluate_candidate(
        np.array([1, 2]), np.array([0.2, 0.8]), 15.0
    )

    assert np.all(batch.valid == np.array([True, True, False]))
    assert np.isclose(batch.gini_after[0], first.gini_after)
    assert np.isclose(batch.gini_after[1], second.gini_after)
    assert np.isnan(batch.gini_after[2])
    assert batch.status[2] == "ZERO_DEMAND_DENOMINATOR"


def test_zero_supply_baseline_and_broken_baseline_conservation_fail_loudly():
    with pytest.raises(UndefinedBaselineGiniError):
        ExactGiniEvaluator(np.zeros(3), np.ones(3))

    with pytest.raises(ValueError, match="supply conservation"):
        ExactGiniEvaluator(
            np.array([0.2, 0.3]),
            np.array([10.0, 20.0]),
            baseline_reachable_supply=999.0,
        )


def test_duplicate_candidate_indices_are_rejected():
    evaluator = ExactGiniEvaluator(
        np.array([0.2, 0.3]), np.array([10.0, 20.0])
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        evaluator.evaluate_candidate(
            np.array([0, 0]), np.array([0.8, 0.8]), 10.0
        )


def test_pipeline_public_api_builds_sparse_gaussian_and_keeps_zero_denominator():
    population = np.array([10.0, 20.0, 0.0])
    baseline_accessibility = np.array([0.2, 0.6, 0.0])
    demand_xy = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 50.0]])
    candidate_xy = np.array([[0.0, 0.0], [5000.0, 0.0]])

    result = evaluate_candidates_exact(
        baseline_accessibility,
        population,
        demand_xy,
        candidate_xy,
        new_capacity=15.0,
        speed_m_per_min=50.0,
        tau_minutes=5.0,
        cutoff_minutes=10.0,
    )

    kernel = np.exp(-np.square(np.array([0.0, 2.0]) / 5.0))
    expected_gini, _ = _full_counterfactual_gini(
        baseline_accessibility,
        population,
        np.array([0, 1]),
        kernel,
        15.0,
    )
    assert np.isclose(result.baseline_gini, weighted_gini(baseline_accessibility, population))
    assert np.isclose(result.candidate_gini[0], expected_gini)
    assert np.isclose(
        result.delta_gini[0], result.baseline_gini - expected_gini
    )
    assert result.denominators[0] > 0
    assert result.denominators[1] == 0
    assert np.isnan(result.candidate_gini[1])
    assert np.isnan(result.delta_gini[1])
    assert result.runtime_seconds >= 0
