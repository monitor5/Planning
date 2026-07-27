import numpy as np

from elder_guardian.surrogate import (
    SpatialGCN,
    normalized_spatial_adjacency,
    ranking_curve,
)


def test_same_district_adjacency_does_not_cross_holdout_boundary():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    groups = np.array(["A", "B", "B"])
    matrix = normalized_spatial_adjacency(xy, groups).tocsr()
    assert matrix[0, 1] == 0
    assert matrix[1, 2] > 0
    assert np.all(matrix.diagonal() > 0)


def test_gcn_accepts_scenario_batch():
    import torch

    xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    groups = np.array(["A", "A"])
    coo = normalized_spatial_adjacency(xy, groups)
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(
        indices, torch.as_tensor(coo.data, dtype=torch.float32), coo.shape
    ).coalesce()
    model = SpatialGCN(input_dim=3, hidden_dim=4, dropout=0.0)
    output = model(torch.ones((2, 2, 3)), adjacency)
    assert output.shape == (2, 2)


def test_regret_is_after_exact_top_k_reranking():
    truth = np.array([[10.0, 9.0, 1.0]])
    prediction = np.array([[9.0, 10.0, 0.0]])
    curve = ranking_curve(
        truth, prediction, np.array(["A", "A", "A"]), top_k_values=(1, 2)
    )
    assert curve["1"]["policy_regret"] == 1.0
    assert curve["2"]["policy_regret"] == 0.0
    assert curve["2"]["best_in_predicted_top_k"] == 1.0
