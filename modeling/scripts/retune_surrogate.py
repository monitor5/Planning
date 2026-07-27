#!/usr/bin/env python3
"""Second-stage self-feature and fixed-ensemble validation on saved exact labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "modeling" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elder_guardian.surrogate import (  # noqa: E402
    fit_final_mlp,
    ranking_curve,
    spatial_cross_validate_mlp,
)


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    arguments = parser.parse_args()
    run_dir = Path(arguments.artifacts).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    training = np.load(run_dir / "surrogate_training_tensor.npz")
    previous = np.load(run_dir / "surrogate_oof_predictions.npz")
    features = training["features"]
    targets = training["targets"]
    groups = training["groups"]
    gcn_predictions = previous["gcn"]

    mlp_metrics, mlp_predictions = spatial_cross_validate_mlp(
        features,
        targets,
        groups,
        hidden_dim=64,
        dropout=0.05,
        epochs=220,
        patience=25,
        learning_rate=0.005,
        weight_decay=1e-4,
        top_loss_weight=40.0,
        folds=5,
        top_k=20,
        random_seed=int(manifest["config"]["random_seed"]),
    )
    # This blend was fixed before inspecting its held-out result.
    ensemble_gcn_weight = 0.5
    ensemble_predictions = (
        ensemble_gcn_weight * gcn_predictions
        + (1.0 - ensemble_gcn_weight) * mlp_predictions
    )
    mlp_curve = ranking_curve(targets, mlp_predictions, groups)
    ensemble_curve = ranking_curve(targets, ensemble_predictions, groups)
    ensemble_metrics = {
        "model": "FixedGCNMLPEnsemble",
        **ensemble_curve["20"],
        "fit_seconds": mlp_metrics.fit_seconds,
    }
    thresholds = manifest["config"]["gates"]

    def approved(metrics: dict) -> bool:
        return bool(
            metrics["spearman"] >= thresholds["min_spearman"]
            and metrics["recall_at_k"] >= thresholds["min_recall_at_20"]
            and metrics["best_in_predicted_top_k"]
            >= thresholds["min_recall_best_at_20"]
            and metrics["relative_policy_regret"]
            <= thresholds["max_relative_policy_regret_at_20"]
        )

    result = {
        "method_change": {
            "reason": "preserve node-local peaks lost by graph smoothing",
            "node_model": "64-64 GELU MLP with top-emphasized Huber loss",
            "ensemble": "fixed 0.5 GCN + 0.5 MLP raw-delta prediction",
            "ensemble_weight_selected_without_test_metric_search": True,
        },
        "metrics": [asdict(mlp_metrics), ensemble_metrics],
        "ranking_curves": {
            "NodeMLP": mlp_curve,
            "FixedGCNMLPEnsemble": ensemble_curve,
        },
        "gates": {
            "NodeMLP": approved(asdict(mlp_metrics)),
            "FixedGCNMLPEnsemble": approved(ensemble_metrics),
            "thresholds": thresholds,
            "final_authority": "exact_e2sfca_weighted_gini",
        },
    }
    np.savez_compressed(
        run_dir / "surrogate_improvement_oof.npz",
        truth=targets,
        mlp=mlp_predictions,
        fixed_ensemble=ensemble_predictions,
    )
    fit_final_mlp(
        features,
        targets,
        feature_names=training["feature_names"].astype(str).tolist(),
        scenario_names=training["scenario_names"].astype(str).tolist(),
        output_path=run_dir / "mlp_surrogate.pt",
        hidden_dim=64,
        dropout=0.05,
        epochs=220,
        learning_rate=0.005,
        weight_decay=1e-4,
        top_loss_weight=40.0,
        random_seed=int(manifest["config"]["random_seed"]),
    )
    write_json(run_dir / "surrogate_improvement.json", result)
    write_json(
        run_dir / "surrogate_ensemble.json",
        {
            "artifact_type": "elder_guardian.FixedGCNMLPEnsemble",
            "gcn_artifact": "gnn_surrogate.pt",
            "mlp_artifact": "mlp_surrogate.pt",
            "gcn_weight": ensemble_gcn_weight,
            "mlp_weight": 1.0 - ensemble_gcn_weight,
            "approved": result["gates"]["FixedGCNMLPEnsemble"],
            "final_predictions_require_exact_top_k_reranking": True,
        },
    )
    print(json.dumps(result["gates"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

