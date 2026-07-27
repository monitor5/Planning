"""Scenario-conditioned GCN surrogate with district-level spatial validation."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


@dataclass
class SurrogateMetrics:
    model: str
    mae: float
    spearman: float
    recall_at_k: float
    best_in_predicted_top_k: float
    policy_regret: float
    relative_policy_regret: float
    relative_policy_regret_p95: float
    recall_best_ci_low: float
    recall_best_ci_high: float
    evaluation_units: int
    fit_seconds: float


class SpatialGCN(torch.nn.Module):
    """Two-layer full-batch graph convolution for multiple scenarios."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
        self.dropout = float(dropout)

    @staticmethod
    def _propagate(adjacency: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: scenarios x nodes x channels
        scenarios, nodes, channels = tensor.shape
        flat = tensor.permute(1, 0, 2).reshape(nodes, scenarios * channels)
        propagated = torch.sparse.mm(adjacency, flat)
        return propagated.reshape(nodes, scenarios, channels).permute(1, 0, 2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self._propagate(adjacency, x)
        hidden = torch.relu(self.input_layer(hidden))
        hidden = torch.nn.functional.dropout(
            hidden, p=self.dropout, training=self.training
        )
        hidden = self._propagate(adjacency, hidden)
        hidden = torch.relu(self.hidden_layer(hidden))
        return self.output_layer(hidden).squeeze(-1)


class NodeMLP(torch.nn.Module):
    """Self-feature benchmark that avoids graph over-smoothing."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def normalized_spatial_adjacency(
    xy: np.ndarray, groups: np.ndarray, *, radius: float = 145.0
) -> sparse.coo_matrix:
    """Create same-district 8-neighbor adjacency with symmetric normalization."""

    pairs = cKDTree(np.asarray(xy, dtype=float)).query_pairs(
        radius, output_type="ndarray"
    )
    group_array = np.asarray(groups)
    if len(pairs):
        pairs = pairs[group_array[pairs[:, 0]] == group_array[pairs[:, 1]]]
    nodes = len(xy)
    row = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(nodes)])
    col = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(nodes)])
    values = np.ones(len(row), dtype=np.float32)
    adjacency = sparse.coo_matrix((values, (row, col)), shape=(nodes, nodes))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree, dtype=float),
        where=degree > 0,
    )
    normalized = sparse.diags(inverse_sqrt) @ adjacency @ sparse.diags(inverse_sqrt)
    return normalized.tocoo()


def _to_torch_sparse(matrix: sparse.coo_matrix) -> torch.Tensor:
    indices = torch.as_tensor(
        np.vstack([matrix.row, matrix.col]), dtype=torch.long
    )
    values = torch.as_tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def _ranking_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    top_k: int,
    groups: np.ndarray | None = None,
    random_seed: int = 20260727,
) -> dict:
    scenario_recalls: list[float] = []
    best_hits: list[float] = []
    regrets: list[float] = []
    relative_regrets: list[float] = []
    group_array = (
        np.asarray(groups)
        if groups is not None
        else np.zeros(truth.shape[1], dtype=np.int8)
    )
    for scenario in range(truth.shape[0]):
        for group in np.unique(group_array):
            valid = (
                np.isfinite(truth[scenario])
                & np.isfinite(prediction[scenario])
                & (group_array == group)
            )
            y = truth[scenario, valid]
            p = prediction[scenario, valid]
            k = min(top_k, len(y))
            if k == 0:
                continue
            true_top = np.argpartition(y, -k)[-k:]
            predicted_top = np.argpartition(p, -k)[-k:]
            intersection = len(np.intersect1d(true_top, predicted_top))
            scenario_recalls.append(intersection / k)
            true_best = int(np.argmax(y))
            best_hits.append(float(true_best in predicted_top))
            # Production re-ranks the surrogate top-K with the exact evaluator.
            # Regret is therefore measured after that exact top-K re-evaluation,
            # not from the single surrogate argmax.
            best_exact_inside_predicted_top = float(np.max(y[predicted_top]))
            regret = float(y[true_best] - best_exact_inside_predicted_top)
            regrets.append(regret)
            relative_regrets.append(
                regret / max(abs(float(y[true_best])), 1e-12)
            )
    flattened_valid = np.isfinite(truth) & np.isfinite(prediction)
    correlation = spearmanr(truth[flattened_valid], prediction[flattened_valid]).statistic
    random = np.random.default_rng(random_seed)
    hit_array = np.asarray(best_hits, dtype=float)
    if hit_array.size:
        bootstrap_means = np.mean(
            random.choice(hit_array, size=(1000, hit_array.size), replace=True),
            axis=1,
        )
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = ci_high = float("nan")
    return {
        "mae": float(np.mean(np.abs(truth[flattened_valid] - prediction[flattened_valid]))),
        "spearman": float(correlation),
        "recall_at_k": float(np.mean(scenario_recalls)),
        "best_in_predicted_top_k": float(np.mean(best_hits)),
        "policy_regret": float(np.mean(regrets)),
        "relative_policy_regret": float(np.mean(relative_regrets)),
        "relative_policy_regret_p95": float(np.quantile(relative_regrets, 0.95)),
        "recall_best_ci_low": float(ci_low),
        "recall_best_ci_high": float(ci_high),
        "evaluation_units": int(len(best_hits)),
    }


def ranking_curve(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10, 20, 50),
    random_seed: int = 20260727,
) -> dict[str, dict]:
    """Return district-by-scenario shortlist metrics for several K values."""

    return {
        str(top_k): _ranking_metrics(
            truth,
            prediction,
            top_k=top_k,
            groups=groups,
            random_seed=random_seed,
        )
        for top_k in top_k_values
    }


def _standardize_features(
    features: np.ndarray, train_node_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = features[:, train_node_mask, :].reshape(-1, features.shape[-1])
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (features - mean) / scale, mean, scale


def _top_emphasis_weights(
    targets: np.ndarray, node_mask: np.ndarray, top_loss_weight: float
) -> np.ndarray:
    """Compute scenario-local training weights without reading held-out labels."""

    weights = np.ones_like(targets, dtype=np.float32)
    if top_loss_weight <= 1:
        return weights
    indices = np.flatnonzero(node_mask)
    for scenario in range(targets.shape[0]):
        values = targets[scenario, indices]
        order = np.argsort(values, kind="mergesort")
        percentile = np.empty(len(values), dtype=np.float32)
        percentile[order] = (
            np.arange(1, len(values) + 1, dtype=np.float32) / len(values)
        )
        weights[scenario, indices] = 1.0 + (top_loss_weight - 1.0) * np.power(
            percentile, 12
        )
    return weights


def spatial_cross_validate_gcn(
    features: np.ndarray,
    targets: np.ndarray,
    xy: np.ndarray,
    groups: np.ndarray,
    *,
    hidden_dim: int = 32,
    dropout: float = 0.15,
    epochs: int = 160,
    patience: int = 20,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    top_loss_weight: float = 20.0,
    folds: int = 5,
    top_k: int = 20,
    random_seed: int = 20260727,
) -> tuple[SurrogateMetrics, np.ndarray]:
    """Return out-of-district predictions; no random row split is used."""

    start = time.perf_counter()
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(groups)
    predictions = np.full(y.shape, np.nan, dtype=np.float64)
    adjacency = _to_torch_sparse(normalized_spatial_adjacency(xy, groups))
    splitter = GroupKFold(n_splits=folds)

    for fold, (train_nodes_initial, test_nodes) in enumerate(
        splitter.split(np.arange(len(groups)), groups=groups)
    ):
        train_groups = np.unique(groups[train_nodes_initial])
        validation_group = train_groups[fold % len(train_groups)]
        validation_nodes = np.flatnonzero(groups == validation_group)
        train_nodes = np.setdiff1d(train_nodes_initial, validation_nodes)
        train_mask = np.zeros(len(groups), dtype=bool)
        validation_mask = np.zeros(len(groups), dtype=bool)
        train_mask[train_nodes] = True
        validation_mask[validation_nodes] = True
        standardized, _, _ = _standardize_features(x, train_mask)
        train_y = y[:, train_mask]
        target_mean = float(train_y.mean())
        target_scale = float(train_y.std())
        if target_scale < 1e-12:
            target_scale = 1.0
        tensor_x = torch.as_tensor(standardized, dtype=torch.float32)
        tensor_y = torch.as_tensor((y - target_mean) / target_scale)
        tensor_loss_weight = torch.as_tensor(
            _top_emphasis_weights(y, train_mask, top_loss_weight)
        )
        model = SpatialGCN(x.shape[-1], hidden_dim, dropout)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(tensor_x, adjacency)
            loss_values = torch.nn.functional.huber_loss(
                output[:, train_mask],
                tensor_y[:, train_mask],
                delta=1.0,
                reduction="none",
            )
            loss = torch.sum(
                loss_values * tensor_loss_weight[:, train_mask]
            ) / torch.sum(tensor_loss_weight[:, train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = torch.nn.functional.mse_loss(
                    model(tensor_x, adjacency)[:, validation_mask],
                    tensor_y[:, validation_mask],
                ).item()
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            fold_prediction = model(tensor_x, adjacency).numpy()
        predictions[:, test_nodes] = (
            fold_prediction[:, test_nodes] * target_scale + target_mean
        )

    values = _ranking_metrics(
        y,
        predictions,
        top_k=top_k,
        groups=groups,
        random_seed=random_seed,
    )
    metrics = SurrogateMetrics(
        model="SpatialGCN",
        fit_seconds=time.perf_counter() - start,
        **values,
    )
    return metrics, predictions


def spatial_cross_validate_tabular(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int = 5,
    top_k: int = 20,
    random_seed: int = 20260727,
) -> tuple[SurrogateMetrics, np.ndarray]:
    """Leakage-safe nonlinear tabular benchmark for the GCN."""

    start = time.perf_counter()
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    predictions = np.full(y.shape, np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for train_nodes, test_nodes in splitter.split(np.arange(len(groups)), groups=groups):
        train_x = x[:, train_nodes, :].reshape(-1, x.shape[-1])
        train_y = y[:, train_nodes].reshape(-1)
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=random_seed,
        )
        model.fit(train_x, train_y)
        test_x = x[:, test_nodes, :].reshape(-1, x.shape[-1])
        fold_prediction = model.predict(test_x).reshape(y.shape[0], len(test_nodes))
        predictions[:, test_nodes] = fold_prediction
    values = _ranking_metrics(
        y,
        predictions,
        top_k=top_k,
        groups=groups,
        random_seed=random_seed,
    )
    metrics = SurrogateMetrics(
        model="HistGradientBoosting",
        fit_seconds=time.perf_counter() - start,
        **values,
    )
    return metrics, predictions


def spatial_cross_validate_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.05,
    epochs: int = 220,
    patience: int = 25,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    top_loss_weight: float = 40.0,
    folds: int = 5,
    top_k: int = 20,
    random_seed: int = 20260727,
) -> tuple[SurrogateMetrics, np.ndarray]:
    """District-held-out MLP baseline with top-emphasized training loss."""

    start = time.perf_counter()
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(groups)
    predictions = np.full(y.shape, np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_nodes_initial, test_nodes) in enumerate(
        splitter.split(np.arange(len(groups)), groups=groups)
    ):
        train_groups = np.unique(groups[train_nodes_initial])
        validation_group = train_groups[fold % len(train_groups)]
        validation_nodes = np.flatnonzero(groups == validation_group)
        train_nodes = np.setdiff1d(train_nodes_initial, validation_nodes)
        train_mask = np.zeros(len(groups), dtype=bool)
        validation_mask = np.zeros(len(groups), dtype=bool)
        train_mask[train_nodes] = True
        validation_mask[validation_nodes] = True
        standardized, _, _ = _standardize_features(x, train_mask)
        train_y = y[:, train_mask]
        target_mean = float(train_y.mean())
        target_scale = max(float(train_y.std()), 1e-12)
        tensor_x = torch.as_tensor(standardized, dtype=torch.float32)
        tensor_y = torch.as_tensor((y - target_mean) / target_scale)
        tensor_loss_weight = torch.as_tensor(
            _top_emphasis_weights(y, train_mask, top_loss_weight)
        )
        model = NodeMLP(x.shape[-1], hidden_dim, dropout)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(tensor_x)
            loss_values = torch.nn.functional.huber_loss(
                output[:, train_mask],
                tensor_y[:, train_mask],
                delta=1.0,
                reduction="none",
            )
            loss = torch.sum(
                loss_values * tensor_loss_weight[:, train_mask]
            ) / torch.sum(tensor_loss_weight[:, train_mask])
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = torch.nn.functional.mse_loss(
                    model(tensor_x)[:, validation_mask],
                    tensor_y[:, validation_mask],
                ).item()
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            fold_prediction = model(tensor_x).numpy()
        predictions[:, test_nodes] = (
            fold_prediction[:, test_nodes] * target_scale + target_mean
        )
    values = _ranking_metrics(
        y,
        predictions,
        top_k=top_k,
        groups=groups,
        random_seed=random_seed,
    )
    return (
        SurrogateMetrics(
            model="NodeMLP",
            fit_seconds=time.perf_counter() - start,
            **values,
        ),
        predictions,
    )


def fit_final_gcn(
    features: np.ndarray,
    targets: np.ndarray,
    xy: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: list[str],
    scenario_names: list[str],
    output_path: str | Path,
    hidden_dim: int = 32,
    dropout: float = 0.15,
    epochs: int = 160,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    top_loss_weight: float = 20.0,
    random_seed: int = 20260727,
) -> dict:
    """Fit all exact labels and save a portable state/scaler bundle."""

    torch.manual_seed(random_seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    node_mask = np.ones(x.shape[1], dtype=bool)
    standardized, feature_mean, feature_scale = _standardize_features(x, node_mask)
    target_mean = float(y.mean())
    target_scale = max(float(y.std()), 1e-12)
    tensor_x = torch.as_tensor(standardized, dtype=torch.float32)
    tensor_y = torch.as_tensor((y - target_mean) / target_scale)
    tensor_loss_weight = torch.as_tensor(
        _top_emphasis_weights(y, node_mask, top_loss_weight)
    )
    adjacency = _to_torch_sparse(normalized_spatial_adjacency(xy, groups))
    model = SpatialGCN(x.shape[-1], hidden_dim, dropout)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(tensor_x, adjacency)
        loss_values = torch.nn.functional.huber_loss(
            prediction, tensor_y, delta=1.0, reduction="none"
        )
        loss = torch.sum(loss_values * tensor_loss_weight) / torch.sum(
            tensor_loss_weight
        )
        loss.backward()
        optimizer.step()
    bundle = {
        "state_dict": model.state_dict(),
        "input_dim": x.shape[-1],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "feature_names": feature_names,
        "scenario_names": scenario_names,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "random_seed": random_seed,
        "capacity_and_travel_semantics": "scenario inputs; not observed operational truth",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"state_dict", "feature_mean", "feature_scale"}
    }
    metadata["output_path"] = str(output)
    metadata["epochs"] = epochs
    return metadata


def fit_final_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    feature_names: list[str],
    scenario_names: list[str],
    output_path: str | Path,
    hidden_dim: int = 64,
    dropout: float = 0.05,
    epochs: int = 220,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    top_loss_weight: float = 40.0,
    random_seed: int = 20260727,
) -> dict:
    """Fit and save the self-feature model used in the fixed 50/50 ensemble."""

    torch.manual_seed(random_seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    node_mask = np.ones(x.shape[1], dtype=bool)
    standardized, feature_mean, feature_scale = _standardize_features(x, node_mask)
    target_mean = float(y.mean())
    target_scale = max(float(y.std()), 1e-12)
    tensor_x = torch.as_tensor(standardized, dtype=torch.float32)
    tensor_y = torch.as_tensor((y - target_mean) / target_scale)
    tensor_loss_weight = torch.as_tensor(
        _top_emphasis_weights(y, node_mask, top_loss_weight)
    )
    model = NodeMLP(x.shape[-1], hidden_dim, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(tensor_x)
        loss_values = torch.nn.functional.huber_loss(
            prediction, tensor_y, delta=1.0, reduction="none"
        )
        loss = torch.sum(loss_values * tensor_loss_weight) / torch.sum(
            tensor_loss_weight
        )
        loss.backward()
        optimizer.step()
    bundle = {
        "state_dict": model.state_dict(),
        "artifact_type": "elder_guardian.NodeMLP",
        "input_dim": x.shape[-1],
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "feature_names": feature_names,
        "scenario_names": scenario_names,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "top_loss_weight": top_loss_weight,
        "random_seed": random_seed,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    return {
        "output_path": str(output),
        "input_dim": x.shape[-1],
        "hidden_dim": hidden_dim,
        "epochs": epochs,
    }


def write_metrics(metrics: list[SurrogateMetrics], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(metric) for metric in metrics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
