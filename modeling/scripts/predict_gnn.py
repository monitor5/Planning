#!/usr/bin/env python3
"""Run the saved SpatialGCN as an audited, non-final shortlist generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "modeling" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elder_guardian.surrogate import (  # noqa: E402
    SpatialGCN,
    _to_torch_sparse,
    normalized_spatial_adjacency,
)


WARNING = (
    "SCREENING_ONLY_NOT_APPROVED_AT_K20: GNN scores are not final policy "
    "effects; exact E2SFCA/Gini re-evaluation is mandatory."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_artifact(run_dir: Path, manifest: dict, filename: str) -> Path:
    path = run_dir / filename
    expected = manifest["artifact_hashes"].get(filename)
    if expected is None:
        raise RuntimeError(f"{filename} is not registered in run_manifest.json")
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{filename} checksum mismatch")
    return path


def load_and_predict(
    run_dir: Path, *, radius: float = 145.0
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Load verified artifacts and return predictions plus their data contract."""

    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_path = verified_artifact(run_dir, manifest, "gnn_surrogate.pt")
    tensor_path = verified_artifact(
        run_dir, manifest, "surrogate_training_tensor.npz"
    )
    with np.load(tensor_path, allow_pickle=False) as archive:
        training = {name: archive[name] for name in archive.files}
    required = {
        "features",
        "targets",
        "xy",
        "groups",
        "gids",
        "feature_names",
        "scenario_names",
    }
    missing = sorted(required - training.keys())
    if missing:
        raise RuntimeError(f"training tensor is missing fields: {missing}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    features = np.asarray(training["features"], dtype=np.float32)
    xy = np.asarray(training["xy"], dtype=np.float64)
    groups = np.asarray(training["groups"])
    gids = np.asarray(training["gids"])
    tensor_feature_names = training["feature_names"].astype(str).tolist()
    tensor_scenarios = training["scenario_names"].astype(str).tolist()
    if features.ndim != 3 or features.shape[1] != len(xy):
        raise RuntimeError(f"invalid GNN feature shape: {features.shape}")
    if features.shape[1] != len(groups) or len(groups) != len(gids):
        raise RuntimeError("node arrays have inconsistent lengths")
    if tensor_feature_names != list(checkpoint["feature_names"]):
        raise RuntimeError("checkpoint and tensor feature order differ")
    if tensor_scenarios != list(checkpoint["scenario_names"]):
        raise RuntimeError("checkpoint and tensor scenario order differ")
    if features.shape[-1] != int(checkpoint["input_dim"]):
        raise RuntimeError("checkpoint input_dim does not match feature tensor")
    if not np.isfinite(features).all() or not np.isfinite(xy).all():
        raise RuntimeError("GNN inputs contain non-finite values")

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    standardized = (features - feature_mean) / feature_scale
    adjacency = _to_torch_sparse(
        normalized_spatial_adjacency(xy, groups, radius=radius)
    )
    model = SpatialGCN(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        standardized_prediction = model(
            torch.as_tensor(standardized, dtype=torch.float32), adjacency
        ).numpy()
    prediction = (
        standardized_prediction * float(checkpoint["target_scale"])
        + float(checkpoint["target_mean"])
    )
    if prediction.shape != features.shape[:2] or not np.isfinite(prediction).all():
        raise RuntimeError(f"invalid GNN output: shape={prediction.shape}")
    return prediction, training, checkpoint


def shortlist_frame(
    prediction: np.ndarray,
    training: dict[str, np.ndarray],
    scenario_index: int,
    top_k: int,
) -> pd.DataFrame:
    """Return a deterministic top-K table for one saved scenario."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 <= scenario_index < prediction.shape[0]:
        raise ValueError("scenario_index is outside the saved scenario range")
    scores = prediction[scenario_index]
    order = np.argsort(-scores, kind="mergesort")[: min(top_k, len(scores))]
    return pd.DataFrame(
        {
            "screening_rank": np.arange(1, len(order) + 1),
            "gid": training["gids"][order].astype(str),
            "borough": training["groups"][order].astype(str),
            "x_epsg5179": training["xy"][order, 0],
            "y_epsg5179": training["xy"][order, 1],
            "gnn_predicted_delta_gini": scores[order],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the saved SpatialGCN on its verified feature tensor. "
            "This produces a screening shortlist, never a final recommendation."
        )
    )
    parser.add_argument("--artifacts", required=True)
    parser.add_argument(
        "--scenario",
        help="saved scenario name; defaults to the manifest base scenario",
    )
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--radius", type=float, default=145.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    run_dir = Path(arguments.artifacts).resolve()
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    prediction, training, _ = load_and_predict(run_dir, radius=arguments.radius)
    scenario_names = training["scenario_names"].astype(str).tolist()
    scenario = arguments.scenario or manifest["config"]["base_scenario"]
    if scenario not in scenario_names:
        raise ValueError(
            f"unknown scenario {scenario!r}; choose one of {scenario_names}"
        )
    scenario_index = scenario_names.index(scenario)
    shortlist = shortlist_frame(
        prediction, training, scenario_index, arguments.top
    )
    if arguments.output:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shortlist.to_csv(output, index=False)
        print(f"wrote {output}")
    else:
        print(shortlist.to_string(index=False))
    print(
        json.dumps(
            {
                "scenario": scenario,
                "nodes_scored": int(prediction.shape[1]),
                "top_k": int(len(shortlist)),
                "score_min": float(prediction[scenario_index].min()),
                "score_max": float(prediction[scenario_index].max()),
                "warning": WARNING,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
