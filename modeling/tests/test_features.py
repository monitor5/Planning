import pytest

from elder_guardian.features import assert_no_target_leakage


def test_target_leakage_guard_rejects_delta_gini():
    with pytest.raises(ValueError, match="target leakage"):
        assert_no_target_leakage(["population", "delta_gini"])


def test_target_leakage_guard_allows_baseline_accessibility():
    assert_no_target_leakage(["population", "baseline_accessibility"])

