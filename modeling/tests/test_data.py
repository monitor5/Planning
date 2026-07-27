from pathlib import Path

import numpy as np

from elder_guardian.data import load_population_100m


def test_project_population_loader_expected_invariants():
    root = Path(__file__).resolve().parents[2] / "Data" / "서울_격자별노인인구(100M)"
    if not root.exists():
        return
    population, audit = load_population_100m(root)
    assert audit.raw_rows == 64681
    assert audit.unique_grids == 61652
    assert np.isclose(audit.published_population, 1962046.0)
    assert population["gid"].is_unique
    assert np.isclose(population["population_lower_bound"].sum(), 1962046.0)

