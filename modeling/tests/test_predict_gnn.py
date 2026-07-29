from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_gnn.py"
SPEC = importlib.util.spec_from_file_location("predict_gnn", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
predict_gnn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predict_gnn)


def test_shortlist_frame_is_descending_and_stable() -> None:
    prediction = np.array([[0.1, 0.3, 0.3, -0.2]], dtype=float)
    training = {
        "gids": np.array(["a", "b", "c", "d"]),
        "groups": np.array(["g1", "g1", "g2", "g2"]),
        "xy": np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        ),
    }
    result = predict_gnn.shortlist_frame(prediction, training, 0, 3)
    assert result["gid"].tolist() == ["b", "c", "a"]
    assert result["screening_rank"].tolist() == [1, 2, 3]
    assert result["gnn_predicted_delta_gini"].is_monotonic_decreasing


def test_shortlist_frame_rejects_invalid_arguments() -> None:
    prediction = np.zeros((1, 1), dtype=float)
    training = {
        "gids": np.array(["a"]),
        "groups": np.array(["g"]),
        "xy": np.zeros((1, 2)),
    }
    with pytest.raises(ValueError, match="positive"):
        predict_gnn.shortlist_frame(prediction, training, 0, 0)
    with pytest.raises(ValueError, match="scenario_index"):
        predict_gnn.shortlist_frame(prediction, training, 1, 1)
