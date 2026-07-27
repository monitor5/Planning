"""Human-readable report and diagnostic plots for a completed model run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_wgs84(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        result["x"].to_numpy(dtype=float), result["y"].to_numpy(dtype=float)
    )
    result["latitude"] = latitude
    result["longitude"] = longitude
    return result


def plot_base_candidates(
    candidates: pd.DataFrame,
    population: pd.DataFrame,
    *,
    scenario_name: str,
    output_path: str | Path,
) -> None:
    subset = candidates.loc[candidates["scenario"] == scenario_name].copy()
    best = subset.nlargest(1, "delta_gini")
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    pop = population.loc[population["population"] > 0]
    axis.scatter(
        pop["x"],
        pop["y"],
        s=np.clip(np.sqrt(pop["population"]), 1, 8),
        c="#d9dde3",
        alpha=0.45,
        linewidths=0,
        label="65+ population grid",
    )
    plot = axis.scatter(
        subset["x"],
        subset["y"],
        c=subset["delta_gini"],
        s=5,
        cmap="viridis",
        alpha=0.8,
        linewidths=0,
        label="candidate ΔG",
    )
    axis.scatter(
        best["x"],
        best["y"],
        marker="*",
        s=260,
        c="#e63946",
        edgecolors="white",
        linewidths=1,
        label="exact best",
    )
    figure.colorbar(plot, ax=axis, shrink=0.75, label="Gini reduction (ΔG)")
    axis.set_title(f"Exact E2SFCA candidate screening — {scenario_name}")
    axis.set_xlabel("EPSG:5179 x (m)")
    axis.set_ylabel("EPSG:5179 y (m)")
    axis.set_aspect("equal")
    axis.legend(loc="lower right")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_surrogate_validation(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    output_path: str | Path,
    max_points: int = 30000,
    random_seed: int = 20260727,
) -> None:
    rng = np.random.default_rng(random_seed)
    models = list(predictions)
    figure, axes = plt.subplots(
        1, len(models), figsize=(6 * len(models), 5), constrained_layout=True
    )
    if len(models) == 1:
        axes = [axes]
    for axis, model_name in zip(axes, models):
        prediction = predictions[model_name]
        valid = np.flatnonzero(np.isfinite(truth.ravel()) & np.isfinite(prediction.ravel()))
        if len(valid) > max_points:
            valid = rng.choice(valid, max_points, replace=False)
        y = truth.ravel()[valid]
        p = prediction.ravel()[valid]
        axis.scatter(y, p, s=4, alpha=0.18, linewidths=0)
        minimum = min(float(y.min()), float(p.min()))
        maximum = max(float(y.max()), float(p.max()))
        axis.plot([minimum, maximum], [minimum, maximum], "--", color="#e63946")
        axis.set_title(model_name)
        axis.set_xlabel("Exact ΔG")
        axis.set_ylabel("Out-of-district predicted ΔG")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_run_report(
    *,
    output_path: str | Path,
    run_id: str,
    manifest: dict,
    scenario_summary: list[dict],
    base_top: pd.DataFrame,
    robust_top: pd.DataFrame,
    metrics: list[dict],
    gates: dict,
    capacity_validation: dict | None,
) -> None:
    metric_lines = "\n".join(
        (
            f"- {item['model']}: Spearman={item['spearman']:.4f}, "
            f"Recall@20={item['recall_at_k']:.4f}, "
            f"RecallBest@20={item['best_in_predicted_top_k']:.4f} "
            f"(95% bootstrap CI {item['recall_best_ci_low']:.4f}–"
            f"{item['recall_best_ci_high']:.4f}), "
            f"relative regret@20={item['relative_policy_regret']:.4f}, "
            f"p95 regret@20={item['relative_policy_regret_p95']:.4f}, "
            f"MAE={item['mae']:.8g}"
        )
        for item in metrics
    )
    scenario_lines = "\n".join(
        (
            f"- `{item['name']}`: baseline Gini={item['baseline_gini']:.8f}, "
            f"best after={item['best_candidate_gini']:.8f}, "
            f"ΔG={item['best_delta_gini']:.8f}, "
            f"conservation error={item['conservation_error']:.3g}"
        )
        for item in scenario_summary
    )
    base_lines = "\n".join(
        (
            f"- {row.gid} / {row.borough}: ({row.latitude:.6f}, "
            f"{row.longitude:.6f}), ΔG={row.delta_gini:.8f}"
        )
        for row in base_top.itertuples()
    )
    robust_lines = "\n".join(
        (
            f"- {row.gid} / {row.borough}: ({row.latitude:.6f}, "
            f"{row.longitude:.6f}), 평균 백분위={row.mean_percentile:.6f}"
        )
        for row in robust_top.itertuples()
    )
    capacity_text = (
        json.dumps(capacity_validation, ensure_ascii=False, indent=2)
        if capacity_validation is not None
        else "용량 회귀 검증 모듈 결과 없음"
    )
    text = f"""# Elder Guardian 모델 실행 보고서

실행 ID: `{run_id}`

## 결론

이 실행의 정책 목적함수는 PDF와 동일한 65+ 인구가중 접근성 Gini입니다. 다만
현행 운영 동시수용량, 다중교통 OD, 후보 필지 실현가능성이 없으므로 아래 위치는
**보행·용량 시나리오 기반 100m 격자 스크리닝**이며 최종 부지가 아닙니다.

기본 시나리오 exact 상위 후보:

{base_lines}

전체 민감도 시나리오의 평균 순위가 안정적인 후보:

{robust_lines}

## 정확 평가 결과

{scenario_lines}

## 공간 홀드아웃 surrogate 검증

{metric_lines}

- GCN shortlist 승인: `{gates.get('gcn_approved')}`
- 최종 권위 모델: `exact_e2sfca_weighted_gini`

## 수용량 회귀 검증

```json
{capacity_text}
```

실제 정원과 시설 전용면적의 동시 관측이 없거나 회귀가 중앙값 기준선을 이기지
못하면 용량 모델은 배포 불가로 표시됩니다. 이 경우 위치 모델은 20/40/60명
명시적 시나리오만 사용합니다.

## 데이터 및 재현성

- 100m 공개 하한 인구 합계: {manifest['population_audit']['published_population']:.0f}
- 1km 잔차 민감도 인구 합계: {manifest['population_sensitivity']['sensitivity_total']:.0f}
- 고유 100m 격자: {manifest['population_audit']['unique_grids']}
- 시설 master 행: {manifest['facility_audit']['LEGAL_NOMINAL']['facility_rows']}
- 확인된 운영 동시수용량 행: {manifest['facility_audit']['LEGAL_NOMINAL']['verified_operational_capacity_rows']}

입력 SHA-256, source commit, 설정 전체는 `run_manifest.json`에 저장했습니다.

## 사용 전 필수 확인

1. 추천 격자의 필지 소유권·지가·용도지역·경사·건축 가능 면적을 검증합니다.
2. 3,644개 시설의 운영/휴지/폐지 상태와 현재 동시수용량을 수집합니다.
3. 보행·버스·지하철 OD와 환승/대기비용으로 직선거리 시나리오를 교체합니다.
4. 위 자료로 exact 평가와 공간 홀드아웃을 다시 실행합니다.
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
