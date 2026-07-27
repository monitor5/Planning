import numpy as np

from elder_guardian.accessibility import (
    e2sfca,
    gaussian_weight_matrix,
    weighted_gini,
)


def _brute_weighted_gini(x, w):
    numerator = sum(
        wi * wj * abs(xi - xj)
        for xi, wi in zip(x, w)
        for xj, wj in zip(x, w)
    )
    return numerator / (2 * sum(w) * sum(wi * xi for wi, xi in zip(w, x)))


def test_weighted_gini_matches_bruteforce_and_is_scale_invariant():
    x = np.array([0.2, 1.0, 0.5, 2.0])
    w = np.array([10.0, 2.0, 7.0, 1.0])
    expected = _brute_weighted_gini(x, w)
    assert np.isclose(weighted_gini(x, w), expected)
    assert np.isclose(weighted_gini(9 * x, w), expected)


def test_weighted_gini_is_nan_for_zero_accessibility():
    assert np.isnan(weighted_gini(np.zeros(3), np.ones(3)))


def test_gaussian_decay_and_cutoff():
    origins = np.array([[0.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
    destinations = np.array([[0.0, 0.0]])
    weights = gaussian_weight_matrix(
        origins,
        destinations,
        speed_m_per_min=50.0,
        tau_minutes=5.0,
        cutoff_minutes=10.0,
    ).toarray()[:, 0]
    assert weights[0] > weights[1] > 0
    assert weights[2] == 0


def test_e2sfca_supply_conservation_and_capacity_monotonicity():
    population = np.array([10.0, 20.0, 30.0])
    facilities = np.array([[0.0, 0.0], [200.0, 0.0]])
    origins = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    matrix = gaussian_weight_matrix(
        origins,
        facilities,
        speed_m_per_min=50.0,
        tau_minutes=5.0,
        cutoff_minutes=10.0,
    )
    low = e2sfca(population, np.array([20.0, 30.0]), matrix)
    high = e2sfca(population, np.array([40.0, 60.0]), matrix)
    assert low.conservation_error < 1e-10
    assert np.all(high.accessibility >= low.accessibility)
    assert np.isclose(low.supplied_access, 50.0)

