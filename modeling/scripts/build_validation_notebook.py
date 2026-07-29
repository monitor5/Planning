"""Build the executed-model revalidation notebook for 14_rewritten (1).pdf."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "modeling" / "notebooks" / "14_rewritten_model_revalidation.ipynb"


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# `14_rewritten (1).pdf` 기반 모델 재검증

**검증 기준일:** 2026-07-30  
**대상:** `modeling/artifacts/run_20260727_improved`  
**목적:** PDF의 수식·데이터·검증 요구사항을 실제 구현 및 저장 산출물과 다시 대조하고, 코드 테스트·체크섬·모델 추론·성능지표를 재실행한다.

> **요약 판정**
>
> - **정확 계산부(E2SFCA + 인구가중 Gini + 후보별 반사실 평가)는 정상**이며, 수학적 보존식·단위 테스트·반복실행 일치성을 통과했다.
> - **GCN/MLP 대리모델은 파일 로드와 추론은 정상**이지만, PDF 목적에 맞춰 미리 정한 `RecallBest@20 ≥ 0.95` 승인 기준을 통과하지 못했다. 최종 순위의 권한은 exact 계산기에 있다.
> - **PDF가 요구한 전체 정책모델은 아직 완성되지 않았다.** 현재 운용정원, 다중교통 OD 일반화비용, 필지·예산·법적 설치 가능성 데이터가 미구현/미확보이므로 결과는 **도보 직선거리 기반 후보 셀 스크리닝**이다.

이 노트북은 성능을 부풀리지 않는다. 결정론적 exact 계산기의 수치 정확성, 통계 대리모델의 예측·순위 성능, 정책 배치 가능성을 서로 다른 층으로 구분한다.
"""
    ),
    code(
        r"""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "modeling").is_dir() and (candidate / "14_rewritten (1).pdf").is_file():
            return candidate
    raise FileNotFoundError("프로젝트 루트 또는 14_rewritten (1).pdf를 찾지 못했습니다.")


ROOT = find_project_root(Path.cwd().resolve())
MODELING = ROOT / "modeling"
ARTIFACTS = MODELING / "artifacts" / "run_20260727_improved"
REFERENCE_RUN = MODELING / "artifacts" / "run_20260727_full"
PDF_PATH = ROOT / "14_rewritten (1).pdf"

sys.path.insert(0, str(MODELING / "src"))

with (ARTIFACTS / "run_manifest.json").open(encoding="utf-8") as handle:
    manifest = json.load(handle)
with (ARTIFACTS / "validation_gates.json").open(encoding="utf-8") as handle:
    gates = json.load(handle)
with (ARTIFACTS / "scenario_summary.json").open(encoding="utf-8") as handle:
    scenario_summary = json.load(handle)
with (ARTIFACTS / "capacity_validation.json").open(encoding="utf-8") as handle:
    capacity_validation = json.load(handle)

assert manifest["status"] == "SCREENING_ONLY_NOT_POLICY_DEPLOYABLE"
assert gates["final_authority"] == "exact_e2sfca_weighted_gini"

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.float_format", lambda value: f"{value:,.8g}")

environment = pd.DataFrame(
    [
        ("project_root", str(ROOT)),
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("artifact_status", manifest["status"]),
        ("deployment_scope", gates["deployment_scope"]),
        ("final_authority", gates["final_authority"]),
    ],
    columns=["항목", "값"],
)
display(environment)
"""
    ),
    markdown(
        r"""
## 1. PDF 식별 및 요구사항 대조

PDF 14쪽 전체를 Poppler로 렌더링하여 표·수식·본문의 잘림 여부까지 시각 검토했다. 아래 셀은 같은 파일의 페이지 수와 SHA-256을 기록한다. 요구사항 표의 판정은 문서의 데이터(5쪽), 정원 회귀(6쪽), 비용·교통(7~8쪽), E2SFCA(8~9쪽), Gini·목적함수(10쪽), GNN·검증(11~12쪽), 산출물·재현성(13쪽)을 구현물과 대조한 결과다.
"""
    ),
    code(
        r"""
from pypdf import PdfReader


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


pdf_reader = PdfReader(str(PDF_PATH))
pdf_identity = pd.DataFrame(
    [
        ("파일", PDF_PATH.name),
        ("페이지", len(pdf_reader.pages)),
        ("SHA-256", sha256_file(PDF_PATH)),
        ("암호화", bool(pdf_reader.is_encrypted)),
    ],
    columns=["항목", "값"],
)
assert len(pdf_reader.pages) == 14
assert not pdf_reader.is_encrypted
display(pdf_identity)

requirements = pd.DataFrame(
    [
        ("5쪽", "100m 고령인구·시설 데이터의 기준일과 계보", "원천 해시·중복·결측·억제값 감사를 저장", "통과"),
        ("6쪽", "검증된 전용면적→현행 동시수용 정원 회귀", "과거 후보 292건 진단만 가능, 상관·CV 기준 미달", "기각"),
        ("6~7쪽", "예산·지가·건축비·용도·필지 설치 가능성 제약", "최종 파이프라인에 없음", "미구현"),
        ("7~8쪽", "도보·버스·지하철 경로의 최소 일반화비용", "51m/분 도보 직선거리만 사용", "미구현"),
        ("8~9쪽", "Gaussian 감쇠 E2SFCA 2단계 계산·분모 0 오류 분리", "희소행렬 구현, walk10에서 기존시설 1곳 분모 0이 조용히 제외됨", "부분 통과"),
        ("10쪽", "인구가중 Gini와 후보별 ΔG exact 평가", "희소 국소갱신 exact 구현·brute-force와 대조", "통과"),
        ("11~12쪽", "GNN 연속 공간/LODO 홀드아웃·회귀+순위손실·교통 edge", "5-fold 자치구 GroupKFold, Huber 회귀만 사용, 교통 edge 없음", "부분·게이트 실패"),
        ("11~12쪽", "GNN shortlist를 exact로 재평가", "최종 권한을 exact로 강제", "통과"),
        ("13쪽", "후보 위치·접근성 변화·교통영향·재현 메타데이터", "지도·후보·해시 존재, 교통영향/필지 확정은 없음", "부분"),
    ],
    columns=["PDF 위치", "PDF 요구", "현재 구현", "판정"],
)
display(requirements)
"""
    ),
    markdown(
        r"""
## 2. 실제 처리 과정

현재 실행된 흐름은 아래와 같다. 5·10단계의 제한 때문에 결과를 “설치 부지 추천”이 아니라 “추가 필지조사를 시작할 100m 후보 셀”로 해석해야 한다.
"""
    ),
    code(
        r"""
process = pd.DataFrame(
    [
        (1, "수집", "서울 100m 고령인구 ZIP, 2025-06 경로당 3,644개", "원천 파일·해시", "완료"),
        (2, "인구 정제", "격자 중복 집계, 공표값/억제·공란 분리", "61,652개 물리 격자", "완료"),
        (3, "시설 정제", "좌표·식별자 정합, 정원 계보 분리", "현행정원 미확인 시나리오", "제한"),
        (4, "후보 생성", "100m 격자, 1km 국지수요 필터", "57,722개 예비 후보", "완료"),
        (5, "이동비용", "직선거리 ÷ 51m/분, Gaussian 감쇠·cutoff", "희소 수요-시설 가중치", "대체값"),
        (6, "기준 접근성", "E2SFCA 2단계", "A₀ 및 기준 Gini", "완료"),
        (7, "정확 반사실", "후보별 신규공급을 국소 갱신", "후보별 Gini, ΔG", "완료"),
        (8, "대리모델", "5-fold 자치구 GroupKFold(버퍼 없음)", "GCN·HGB·MLP OOF 성능", "부분·게이트 실패"),
        (9, "정확 재순위", "대리모델이 아니라 exact 결과로 최종 정렬", "기본·강건 후보 상위 20", "완료"),
        (10, "정책 검토", "필지·예산·법규·다중교통 OD 확인", "실제 설치 가능 부지", "미완료"),
    ],
    columns=["순서", "단계", "처리", "산출물", "상태"],
)
display(process)
"""
    ),
    markdown(
        r"""
## 3. 데이터 감사

“공표 양수 행 수”는 원천 행 기준이고, “양수 고유 격자”는 중복을 합친 모델 입력 기준이므로 숫자가 다르다. 억제/공란은 임의로 0명이라고 단정하지 않고 공표 하한 시나리오와 1km 잔차 민감도 시나리오를 분리했다.
"""
    ),
    code(
        r"""
population = pd.read_csv(ARTIFACTS / "population_model_input.csv")
facilities = pd.read_csv(ARTIFACTS / "facilities_legal_nominal.csv")
exact_columns = ["scenario", "gid", "candidate_gini", "delta_gini", "valid"]
exact_results = pd.read_csv(ARTIFACTS / "exact_candidate_results.csv", usecols=exact_columns)

population_audit = manifest["population_audit"]
facility_audit = manifest["facility_audit"]["LEGAL_NOMINAL"]

actual_audit = {
    "원천 행": population_audit["raw_rows"],
    "물리 고유 격자": len(population),
    "중복으로 추가된 원천 행": int((population["source_row_count"] - 1).sum()),
    "공표 양수 원천 행": population_audit["published_positive_rows"],
    "양수 고유 격자": int((population["population"] > 0).sum()),
    "고령인구 공표 하한 합계": float(population["population"].sum()),
    "시설 행": len(facilities),
    "시설 고유 ID": int(facilities["source_record_id"].nunique()),
    "검증된 현행 운용정원 행": facility_audit["verified_operational_capacity_rows"],
    "exact 결과 행": len(exact_results),
    "exact 시나리오": int(exact_results["scenario"].nunique()),
}

assert len(population) == population_audit["unique_grids"] == 61_652
assert int((population["source_row_count"] - 1).sum()) == population_audit["duplicate_rows"] == 3_029
assert np.isclose(population["population"].sum(), population_audit["published_population"])
assert len(facilities) == facility_audit["facility_rows"] == 3_644
assert facilities["source_record_id"].nunique() == len(facilities)
assert len(exact_results) == 404_054
assert exact_results["scenario"].nunique() == 7

display(pd.DataFrame(actual_audit.items(), columns=["감사항목", "값"]))

sensitivity = manifest["population_sensitivity"]
display(
    pd.DataFrame(
        [
            ("공표 하한", sensitivity["lower_bound_total"], "억제·공란을 임의 배분하지 않음"),
            ("1km 잔차 민감도", sensitivity["sensitivity_total"], f"{sensitivity['allocated_total']:,.0f}명만 별도 배분"),
        ],
        columns=["인구 시나리오", "총인구", "해석"],
    )
)
"""
    ),
    markdown(
        r"""
## 4. 정원 회귀 검증

PDF는 시설별 검증된 전용면적과 동시수용 정원을 요구한다. 현재 자료는 과거 후보 계보이며 현행 운용값으로 확인된 행이 0개다. 따라서 회귀를 강행하지 않고, 자치구 GroupKFold에서 단순 중앙값 기준보다 실제로 나아지는지 진단한 뒤 배치 게이트를 닫았다.
"""
    ),
    code(
        r"""
capacity_metrics = capacity_validation["metrics"]
capacity_rows = []
for model_name, values in capacity_metrics["models"].items():
    capacity_rows.append(
        {
            "모델": model_name,
            "MAE(명)": values["mae"],
            "RMSE(명)": values["rmse"],
            "중앙절대오차(명)": values["median_absolute_error"],
            "R²": values["r2"],
        }
    )
capacity_table = pd.DataFrame(capacity_rows).sort_values("MAE(명)")

assert capacity_validation["selected_model"] == "median_scenario"
assert capacity_validation["deployable"] is False
assert capacity_metrics["correlation"]["sufficient_positive_correlation"] is False
assert facility_audit["verified_operational_capacity_rows"] == 0

display(
    pd.DataFrame(
        [
            ("진단 표본", capacity_metrics["sample"]["valid_positive_rows"]),
            ("자치구 그룹", capacity_metrics["sample"]["borough_groups"]),
            ("면적-정원 Pearson", capacity_metrics["correlation"]["pearson"]),
            ("면적-정원 Spearman", capacity_metrics["correlation"]["spearman"]),
            ("필요 최소 Spearman", capacity_metrics["correlation"]["minimum_required_spearman"]),
            ("배치 승인", capacity_validation["deployable"]),
            ("기각 사유", capacity_validation["reason"]),
        ],
        columns=["항목", "값"],
    )
)
display(capacity_table)

ax = capacity_table.set_index("모델")["MAE(명)"].plot.bar(
    figsize=(8, 3.5), color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"]
)
ax.set_title("Capacity diagnostic: grouped-CV MAE (lower is better)")
ax.set_ylabel("MAE (persons)")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 5. 핵심 수식과 독립 소형 예제 검증

현 구현의 범위에서 사용하는 식은 PDF와 같다.

\[
R_j=\frac{S_j}{\sum_i N_i w_{ij}},\qquad
A_i=\sum_jR_jw_{ij}
\]

\[
w(c)=\exp[-(c/\tau)^2]\mathbf{1}(c\le c_{\max}),\qquad
\Delta G_k=G(A_0)-G(A_k)
\]

아래 소형 예제는 저장 결과를 읽는 것이 아니라 구현 함수를 직접 호출한다. E2SFCA 공급 보존식 \(\sum_iN_iA_i=\sum_jS_j\)을 확인하고, 희소 국소갱신 exact Gini를 전체 배열을 다시 계산한 brute-force Gini와 비교한다.
"""
    ),
    code(
        r"""
from scipy import sparse

from elder_guardian.accessibility import e2sfca, weighted_gini
from elder_guardian.exact import ExactGiniEvaluator

small_population = np.array([100.0, 50.0, 25.0])
small_capacity = np.array([20.0, 10.0])
small_weights = sparse.csr_matrix(
    np.array(
        [
            [1.0, 0.2],
            [0.4, 1.0],
            [0.0, 0.5],
        ]
    )
)
small_access = e2sfca(small_population, small_capacity, small_weights)

evaluator = ExactGiniEvaluator(
    small_access.accessibility,
    small_population,
    baseline_reachable_supply=small_access.reachable_supply,
)
candidate_kernel = np.array([0.8, 0.4, 0.1])
candidate_capacity = 5.0
exact_small = evaluator.evaluate_candidate(
    np.arange(len(small_population)), candidate_kernel, candidate_capacity
)
candidate_denominator = float(np.dot(small_population, candidate_kernel))
candidate_increment = candidate_capacity * candidate_kernel / candidate_denominator
brute_gini_after = weighted_gini(
    small_access.accessibility + candidate_increment, small_population
)

assert small_access.conservation_error < 1e-12
assert exact_small.conservation_error < 1e-12
assert np.isclose(exact_small.gini_after, brute_gini_after, atol=1e-14)

display(
    pd.DataFrame(
        [
            ("E2SFCA reachable supply", small_access.reachable_supply),
            ("E2SFCA population-weighted access", small_access.supplied_access),
            ("E2SFCA conservation error", small_access.conservation_error),
            ("baseline weighted Gini", exact_small.gini_before),
            ("exact post-installation Gini", exact_small.gini_after),
            ("brute-force post-installation Gini", brute_gini_after),
            ("exact-vs-brute absolute error", abs(exact_small.gini_after - brute_gini_after)),
            ("candidate supply conservation error", exact_small.conservation_error),
        ],
        columns=["검증값", "결과"],
    )
)
"""
    ),
    markdown(
        r"""
## 6. exact 시나리오 성능

exact 계산기는 확률적 예측모델이 아니라 정해진 데이터와 가정에 대해 모든 후보의 목적함수를 계산하는 결정론적 평가기다. 따라서 “정확도 %” 대신 수식 일치, 공급 보존오차, 반복실행 일치, 처리시간을 본다.
"""
    ),
    code(
        r"""
scenario_df = pd.DataFrame(scenario_summary)
scenario_df["relative_gini_reduction_pct"] = (
    100.0 * scenario_df["best_delta_gini"] / scenario_df["baseline_gini"]
)

assert np.allclose(
    scenario_df["baseline_gini"] - scenario_df["best_candidate_gini"],
    scenario_df["best_delta_gini"],
    rtol=0,
    atol=2e-15,
)
assert scenario_df["conservation_error"].max() < 1e-7
assert scenario_df["exact_runtime_seconds"].max() < 10

scenario_view = scenario_df[
    [
        "name",
        "baseline_gini",
        "best_candidate_gini",
        "best_delta_gini",
        "relative_gini_reduction_pct",
        "best_gid",
        "valid_candidates",
        "exact_runtime_seconds",
        "conservation_error",
    ]
].rename(
    columns={
        "name": "시나리오",
        "baseline_gini": "Gini 전",
        "best_candidate_gini": "Gini 후",
        "best_delta_gini": "ΔG",
        "relative_gini_reduction_pct": "상대 Gini 감소(%)",
        "best_gid": "최상위 격자",
        "valid_candidates": "유효 후보",
        "exact_runtime_seconds": "전체 후보 처리(초)",
        "conservation_error": "공급 보존오차",
    }
)
display(scenario_view)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
scenario_df.plot.bar(
    x="name", y="best_delta_gini", ax=axes[0], legend=False, color="#4c78a8"
)
axes[0].set_title("Best exact ΔG by scenario")
axes[0].set_xlabel("")
axes[0].set_ylabel("ΔG")
axes[0].tick_params(axis="x", rotation=65)
scenario_df.plot.bar(
    x="name", y="exact_runtime_seconds", ax=axes[1], legend=False, color="#f58518"
)
axes[1].set_title("All-candidate exact runtime")
axes[1].set_xlabel("")
axes[1].set_ylabel("seconds")
axes[1].tick_params(axis="x", rotation=65)
plt.tight_layout()
plt.show()

base = scenario_df.loc[scenario_df["name"] == "legal20_walk15_new40_BASE"].iloc[0]
display(
    Markdown(
        f"기본 시나리오는 **{base['best_gid']}**에서 "
        f"`Gini {base['baseline_gini']:.9f} → {base['best_candidate_gini']:.9f}`이며, "
        f"`ΔG={base['best_delta_gini']:.9f}` "
        f"(상대 감소 **{base['relative_gini_reduction_pct']:.3f}%**)다. "
        "효과가 0.24% 수준이므로 과도한 정책효과로 해석해서는 안 된다."
    )
)
"""
    ),
    markdown(
        r"""
### 6.1 기존시설 분모 0 별도 감사

저장된 `conservation_error`는 **도달 가능한 공급량**과 인구가중 접근성 합계만 비교하므로, 처음부터 도달 불가로 제외된 기존시설 공급은 잡지 못한다. PDF는 분모 0을 결측·연결·파라미터 오류로 분리해 남기라고 요구한다. 아래 셀은 각 시나리오의 기존시설 E2SFCA 분모를 원자료에서 다시 계산한다.
"""
    ),
    code(
        r"""
from elder_guardian.accessibility import gaussian_weight_matrix

sensitivity_values = pd.read_csv(
    ARTIFACTS / "population_1km_residual_sensitivity.csv"
)
sensitivity_population = population.copy()
sensitivity_population["population"] = sensitivity_population["gid"].map(
    sensitivity_values.set_index("gid")["population"]
)
assert sensitivity_population["population"].notna().all()
historical_facilities = pd.read_csv(
    ARTIFACTS / "facilities_historical_reference_winsorized.csv"
)

population_by_mode = {
    "PUBLISHED_LOWER_BOUND": population,
    "ONE_KM_RESIDUAL_SENSITIVITY": sensitivity_population,
}
facilities_by_mode = {
    "LEGAL_NOMINAL": facilities,
    "HISTORICAL_REFERENCE_WINSORIZED": historical_facilities,
}

denominator_audit_rows = []
zero_facility_rows = []
for scenario in manifest["config"]["scenarios"]:
    scenario_population = population_by_mode[scenario["population_mode"]]
    scenario_facilities = facilities_by_mode[scenario["existing_capacity_mode"]]
    weight_matrix = gaussian_weight_matrix(
        scenario_population[["x", "y"]].to_numpy(),
        scenario_facilities[["x", "y"]].to_numpy(),
        speed_m_per_min=manifest["config"]["walking_speed_m_per_min"],
        tau_minutes=scenario["tau_minutes"],
        cutoff_minutes=scenario["cutoff_minutes"],
    )
    denominator = np.asarray(
        weight_matrix.T @ scenario_population["population"].to_numpy()
    ).ravel()
    direct_baseline = e2sfca(
        scenario_population["population"].to_numpy(),
        scenario_facilities["model_capacity"].to_numpy(),
        weight_matrix,
    )
    direct_baseline_gini = weighted_gini(
        direct_baseline.accessibility,
        scenario_population["population"].to_numpy(),
    )
    stored_baseline_gini = next(
        item["baseline_gini"]
        for item in scenario_summary
        if item["name"] == scenario["name"]
    )
    zero_mask = denominator <= 0
    total_supply = float(scenario_facilities["model_capacity"].sum())
    omitted_supply = float(
        scenario_facilities.loc[zero_mask, "model_capacity"].sum()
    )
    denominator_audit_rows.append(
        {
            "시나리오": scenario["name"],
            "기존시설": len(scenario_facilities),
            "분모0 시설": int(zero_mask.sum()),
            "총공급": total_supply,
            "도달공급": total_supply - omitted_supply,
            "조용히 제외된 공급": omitted_supply,
            "재계산 Gini": direct_baseline_gini,
            "저장 Gini 절대차": abs(direct_baseline_gini - stored_baseline_gini),
            "판정": "통과" if not zero_mask.any() else "오류 플래그 필요",
        }
    )
    if zero_mask.any():
        details = scenario_facilities.loc[
            zero_mask,
            ["source_record_id", "facility_name", "borough", "model_capacity"],
        ].copy()
        details.insert(0, "scenario", scenario["name"])
        zero_facility_rows.append(details)

denominator_audit = pd.DataFrame(denominator_audit_rows)
display(denominator_audit)
if zero_facility_rows:
    display(Markdown("**분모 0 기존시설 상세**"))
    display(pd.concat(zero_facility_rows, ignore_index=True))

assert denominator_audit.loc[
    denominator_audit["시나리오"] == "legal20_walk10_new20", "분모0 시설"
].iloc[0] == 1
assert denominator_audit.loc[
    denominator_audit["시나리오"] == "legal20_walk15_new40_BASE", "분모0 시설"
].iloc[0] == 0
assert denominator_audit["저장 Gini 절대차"].max() < 5e-15

display(
    Markdown(
        "**판정:** 기본 시나리오는 기존시설 분모 0이 없어 정상이다. "
        "`legal20_walk10_new20`은 흐능날 경로당 정원 20명이 도달공급에서 빠졌는데 "
        "기존 gate가 이를 검출하지 못했다. 이 민감도 시나리오는 오류상태로 표시하고 "
        "연결성/인구 억제/10분 cutoff 원인을 확인한 뒤 재실행해야 한다."
    )
)
"""
    ),
    markdown(
        r"""
## 7. 전체 후보 반복실행 재현성

서로 다른 두 실행 디렉터리에서 7개 시나리오×후보 결과의 키, 유효성, Gini를 직접 비교한다. 런타임 열은 장비 부하에 따라 달라질 수 있으므로 수치결과 비교에서 제외한다.
"""
    ),
    code(
        r"""
reference_exact = pd.read_csv(
    REFERENCE_RUN / "exact_candidate_results.csv", usecols=exact_columns
)
left = exact_results.sort_values(["scenario", "gid"]).reset_index(drop=True)
right = reference_exact.sort_values(["scenario", "gid"]).reset_index(drop=True)

keys_equal = left[["scenario", "gid"]].equals(right[["scenario", "gid"]])
valid_equal = left["valid"].equals(right["valid"])
max_candidate_gini_diff = float(
    np.nanmax(np.abs(left["candidate_gini"] - right["candidate_gini"]))
)
max_delta_gini_diff = float(
    np.nanmax(np.abs(left["delta_gini"] - right["delta_gini"]))
)

assert len(left) == len(right) == 404_054
assert keys_equal and valid_equal
assert max_candidate_gini_diff == 0.0
assert max_delta_gini_diff == 0.0

display(
    pd.DataFrame(
        [
            ("비교 행", len(left)),
            ("시나리오·격자 키 일치", keys_equal),
            ("유효성 플래그 일치", valid_equal),
            ("candidate_gini 최대 차이", max_candidate_gini_diff),
            ("delta_gini 최대 차이", max_delta_gini_diff),
        ],
        columns=["검증", "결과"],
    )
)
"""
    ),
    code(
        r"""
valid_mask = exact_results["valid"].astype(bool)
invalid_mask = ~valid_mask
identity_error = float(
    np.nanmax(
        np.abs(
            (
                exact_results.loc[valid_mask, "candidate_gini"]
                + exact_results.loc[valid_mask, "delta_gini"]
            ).to_numpy()
            - exact_results.loc[valid_mask, "scenario"].map(
                scenario_df.set_index("name")["baseline_gini"]
            ).to_numpy()
        )
    )
)
exact_diagnostics = pd.DataFrame(
    [
        ("전체 행", len(exact_results)),
        ("중복 (scenario, gid)", int(exact_results.duplicated(["scenario", "gid"]).sum())),
        ("유효 후보행", int(valid_mask.sum())),
        ("분모0 무효 후보행", int(invalid_mask.sum())),
        ("무효행 Gini/ΔG 모두 NaN", bool(exact_results.loc[invalid_mask, ["candidate_gini", "delta_gini"]].isna().all().all())),
        ("음수 ΔG 행(불평등 악화, 클리핑 안 함)", int((exact_results.loc[valid_mask, "delta_gini"] < 0).sum())),
        ("G_before-G_after=ΔG 최대오차", identity_error),
    ],
    columns=["exact 감사", "결과"],
)
assert exact_results.duplicated(["scenario", "gid"]).sum() == 0
assert invalid_mask.sum() == 12_785
assert exact_results.loc[invalid_mask, ["candidate_gini", "delta_gini"]].isna().all().all()
assert identity_error < 2e-15
display(exact_diagnostics)
"""
    ),
    markdown(
        r"""
## 8. 후보 결과와 설치 가능성 경고

기본 시나리오의 exact 1위와 7개 시나리오를 종합한 강건 1위는 서로 다르다. 더 중요하게, 후보는 **필지**가 아니라 **100m 격자 중심점**이다. 후속 point-level 프로토타입에서 기본 1위 중심점은 공식 도시계획 도로 폴리곤 내부로 확인되어 그 점 자체는 제외 대상이다. 셀 전체를 폐기하거나 실제 부지로 확정할 근거는 아니며, 연속지적도 기반 필지 열거가 필요하다.
"""
    ),
    code(
        r"""
base_top = pd.read_csv(ARTIFACTS / "recommendations_base_top20.csv")
robust_top = pd.read_csv(ARTIFACTS / "recommendations_robust_top20.csv")

display(Markdown("### 기본 시나리오 exact 상위 5개"))
display(
    base_top[
        ["gid", "borough", "latitude", "longitude", "baseline_gini", "candidate_gini", "delta_gini"]
    ].head(5)
)
display(Markdown("### 7개 시나리오 강건 상위 5개"))
display(
    robust_top[
        ["gid", "borough", "latitude", "longitude", "mean_percentile", "worst_percentile", "mean_delta_gini"]
    ].head(5)
)

display(Image(filename=str(ARTIFACTS / "map_base_exact.png"), width=900))

feasibility_path = ROOT / "tmp" / "feasibility_research" / "candidate_feasibility_snapshot.json"
if feasibility_path.exists():
    with feasibility_path.open(encoding="utf-8") as handle:
        feasibility = json.load(handle)
    feasibility_rows = []
    for item in feasibility["candidate_results"]:
        assessment = item["assessment"]
        feasibility_rows.append(
            {
                "격자": item["candidate_id"],
                "자치구/동": f"{item['borough']} {item['neighbourhood']}",
                "point-level 판정": assessment["disposition"],
                "강한 제외 증거": " / ".join(assessment["hard_exclusion_evidence"]) or "없음",
                "검토 증거": " / ".join(assessment["review_evidence"]) or "없음",
            }
        )
    display(pd.DataFrame(feasibility_rows))
else:
    display(Markdown("후속 point-level 설치가능성 스냅샷이 없어 이 단계는 재실행하지 않았다."))
"""
    ),
    markdown(
        r"""
## 9. 대리모델의 실제 성능과 승인 게이트

저장 성능표는 일반 무작위 분할이 아니라 5-fold 자치구 GroupKFold의 OOF 예측을 사용한다. 다만 metric 함수가 이를 다시 **25개 자치구×7개 시나리오=175개 자치구 내부 단위**로 쪼개 평가한다. PDF의 정책 목적은 서울 전체 후보집합에서 한 곳을 고르는 것이므로, 이 175단위 수치는 “자치구 내부 진단”이지 서울 전체 정책 성능이 아니다. 다음 절에서 서울 전체 지표를 별도로 재계산한다.

- `MAE`: ΔG 수치 오차
- `Spearman`: 후보 순위 상관
- `Recall@20`: 실제 상위 후보가 예측 상위 20에 얼마나 포함되는지
- `RecallBest@20`: 각 평가단위의 실제 1위가 예측 상위 20 안에 들어오는 비율
- `relative regret@20`: 예측 shortlist 안에서 exact 재평가했을 때 실제 최상 대비 놓친 효과의 비율

승인은 네 기준을 모두 만족해야 한다. 특히 실제 1위를 놓치지 않기 위한 `RecallBest@20 ≥ 0.95`가 병목이다. 또한 PDF의 pairwise 순위손실과 대중교통 그래프 edge는 구현되지 않았고, 현재 학습손실은 top-emphasized Huber 회귀다.
"""
    ),
    code(
        r"""
with (ARTIFACTS / "surrogate_metrics.json").open(encoding="utf-8") as handle:
    original_surrogate_metrics = json.load(handle)
with (ARTIFACTS / "surrogate_improvement.json").open(encoding="utf-8") as handle:
    surrogate_improvement = json.load(handle)
with (ARTIFACTS / "surrogate_ranking_curves.json").open(encoding="utf-8") as handle:
    original_curves = json.load(handle)

thresholds = gates["thresholds"]
all_surrogate_metrics = original_surrogate_metrics + surrogate_improvement["metrics"]
surrogate_table = pd.DataFrame(all_surrogate_metrics)
surrogate_table["gate_pass"] = (
    (surrogate_table["spearman"] >= thresholds["min_spearman"])
    & (surrogate_table["recall_at_k"] >= thresholds["min_recall_at_20"])
    & (surrogate_table["best_in_predicted_top_k"] >= thresholds["min_recall_best_at_20"])
    & (surrogate_table["relative_policy_regret"] <= thresholds["max_relative_policy_regret_at_20"])
)

assert not surrogate_table["gate_pass"].any()
assert gates["gcn_approved"] is False

display(
    surrogate_table[
        [
            "model",
            "mae",
            "spearman",
            "recall_at_k",
            "best_in_predicted_top_k",
            "relative_policy_regret",
            "relative_policy_regret_p95",
            "evaluation_units",
            "fit_seconds",
            "gate_pass",
        ]
    ].rename(
        columns={
            "model": "모델",
            "mae": "MAE",
            "spearman": "Spearman",
            "recall_at_k": "Recall@20",
            "best_in_predicted_top_k": "RecallBest@20",
            "relative_policy_regret": "평균 상대후회@20",
            "relative_policy_regret_p95": "상대후회 p95@20",
            "evaluation_units": "평가단위",
            "fit_seconds": "학습시간(초)",
            "gate_pass": "승인",
        }
    )
)

all_curves = {
    **original_curves,
    **surrogate_improvement["ranking_curves"],
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for model_name, curve in all_curves.items():
    ks = sorted(int(k) for k in curve)
    axes[0].plot(
        ks,
        [curve[str(k)]["best_in_predicted_top_k"] for k in ks],
        marker="o",
        label=model_name,
    )
    axes[1].plot(
        ks,
        [curve[str(k)]["relative_policy_regret_p95"] for k in ks],
        marker="o",
        label=model_name,
    )
axes[0].axhline(thresholds["min_recall_best_at_20"], color="black", linestyle="--", label="approval target")
axes[0].set_title("RecallBest@K")
axes[0].set_xlabel("shortlist K")
axes[0].set_ylabel("hit rate")
axes[0].set_ylim(0, 1.03)
axes[1].set_title("Relative policy regret p95@K")
axes[1].set_xlabel("shortlist K")
axes[1].set_ylabel("relative regret")
axes[1].set_yscale("symlog", linthresh=1e-4)
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 10. 저장 OOF 예측으로 순위지표 재계산

JSON에 적힌 성능표를 그대로 믿지 않고, 저장된 out-of-fold 예측 배열과 정답을 다시 읽어 동일한 평가 함수를 호출한다. 아래 최대 차이는 재계산값과 저장된 ranking curve 사이의 차이다.
"""
    ),
    code(
        r"""
from elder_guardian.surrogate import ranking_curve

training_tensor = np.load(ARTIFACTS / "surrogate_training_tensor.npz", allow_pickle=False)
original_oof = np.load(ARTIFACTS / "surrogate_oof_predictions.npz", allow_pickle=False)
improved_oof = np.load(ARTIFACTS / "surrogate_improvement_oof.npz", allow_pickle=False)

truth = training_tensor["targets"]
groups = training_tensor["groups"]
xy = training_tensor["xy"]
assert np.array_equal(truth, original_oof["truth"])
assert np.array_equal(truth, improved_oof["truth"])

oof_specs = [
    ("SpatialGCN", original_oof["gcn"], original_curves["SpatialGCN"]["20"]),
    (
        "HistGradientBoosting",
        original_oof["hist_gradient_boosting"],
        original_curves["HistGradientBoosting"]["20"],
    ),
    ("NodeMLP", improved_oof["mlp"], surrogate_improvement["ranking_curves"]["NodeMLP"]["20"]),
    (
        "FixedGCNMLPEnsemble",
        improved_oof["fixed_ensemble"],
        surrogate_improvement["ranking_curves"]["FixedGCNMLPEnsemble"]["20"],
    ),
]
metric_fields = [
    "mae",
    "spearman",
    "recall_at_k",
    "best_in_predicted_top_k",
    "policy_regret",
    "relative_policy_regret",
    "relative_policy_regret_p95",
]
recheck_rows = []
for model_name, prediction, saved in oof_specs:
    assert np.isfinite(prediction).all()
    recalculated = ranking_curve(
        truth,
        prediction,
        groups,
        top_k_values=(20,),
        random_seed=manifest["config"]["random_seed"],
    )["20"]
    max_difference = max(abs(recalculated[field] - saved[field]) for field in metric_fields)
    recheck_rows.append(
        {
            "모델": model_name,
            "예측 배열 shape": str(prediction.shape),
            "유한값": bool(np.isfinite(prediction).all()),
            "저장지표 대비 최대차이": max_difference,
        }
    )
    assert max_difference < 1e-12

display(pd.DataFrame(recheck_rows))
"""
    ),
    markdown(
        r"""
### 10.1 정책 모집단과 공간 버퍼 재감사

같은 OOF 예측을 서울 전체 후보집합으로 묶어 `Recall@20`과 `RecallBest@20`을 다시 계산한다. 또한 5-fold GroupKFold의 각 test 격자에서 가장 가까운 train 격자까지의 거리를 측정한다. 자치구가 fold 사이에서 겹치지는 않지만 경계 양쪽 격자는 100m까지 붙어 있어 연속 공간 블록/버퍼 검증과 같지 않다.
"""
    ),
    code(
        r"""
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

seoul_group = np.repeat("서울전체", len(groups))
fold_labels = np.empty(len(groups), dtype=object)
buffer_rows = []
splitter = GroupKFold(n_splits=manifest["config"]["surrogate"]["folds"])
for fold_index, (train_index, test_index) in enumerate(
    splitter.split(xy, groups=groups), start=1
):
    fold_labels[test_index] = f"fold-{fold_index}"
    nearest_distance = cKDTree(xy[train_index]).query(xy[test_index], k=1)[0]
    buffer_rows.append(
        {
            "fold": fold_index,
            "test 격자": len(test_index),
            "train까지 최소거리(m)": float(nearest_distance.min()),
            "150m 이내(%)": 100.0 * float(np.mean(nearest_distance <= 150.0)),
            "765m 이내(%)": 100.0 * float(np.mean(nearest_distance <= 765.0)),
        }
    )

policy_metric_rows = []
for model_name, prediction, _ in oof_specs:
    district_metric = ranking_curve(
        truth, prediction, groups, top_k_values=(20,)
    )["20"]
    fold_metric = ranking_curve(
        truth, prediction, fold_labels, top_k_values=(20,)
    )["20"]
    seoul_metric = ranking_curve(
        truth, prediction, seoul_group, top_k_values=(20,)
    )["20"]
    policy_metric_rows.append(
        {
            "모델": model_name,
            "자치구 Recall@20": district_metric["recall_at_k"],
            "자치구 RecallBest@20": district_metric["best_in_predicted_top_k"],
            "외부fold RecallBest@20": fold_metric["best_in_predicted_top_k"],
            "서울전체 Recall@20": seoul_metric["recall_at_k"],
            "서울전체 RecallBest@20": seoul_metric["best_in_predicted_top_k"],
            "서울전체 평가단위": seoul_metric["evaluation_units"],
        }
    )

policy_metrics = pd.DataFrame(policy_metric_rows)
buffer_audit = pd.DataFrame(buffer_rows)
display(Markdown("**같은 OOF 예측, 서로 다른 정책 모집단**"))
display(policy_metrics)
display(Markdown("**train–test 공간거리 감사**"))
display(buffer_audit)

assert policy_metrics["서울전체 평가단위"].eq(7).all()
assert buffer_audit["train까지 최소거리(m)"].min() <= 100.000001
assert policy_metrics["서울전체 RecallBest@20"].max() < 0.95
"""
    ),
    markdown(
        r"""
## 11. 실제 모델 가중치 로드 및 추론

저장된 GCN과 개선 MLP 체크포인트를 CPU에서 로드하고, 7개 시나리오×52,736개 후보 전체에 forward pass를 수행한다. 이 검증은 “모델 파일이 깨지지 않았고 실제로 정상 추론한다”는 뜻이지, 앞 절의 승인 게이트 실패를 뒤집는 것은 아니다.
"""
    ),
    code(
        r"""
import torch

from elder_guardian.surrogate import NodeMLP, SpatialGCN, normalized_spatial_adjacency

features = training_tensor["features"]
xy = training_tensor["xy"]
model_groups = training_tensor["groups"]
adjacency_coo = normalized_spatial_adjacency(xy, model_groups)
adjacency_indices = torch.tensor(
    np.vstack([adjacency_coo.row, adjacency_coo.col]), dtype=torch.long
)
adjacency_values = torch.tensor(adjacency_coo.data, dtype=torch.float32)
adjacency = torch.sparse_coo_tensor(
    adjacency_indices,
    adjacency_values,
    adjacency_coo.shape,
    check_invariants=True,
).coalesce()


def standardized_features(checkpoint: dict) -> torch.Tensor:
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    return torch.tensor((features - mean) / scale, dtype=torch.float32)


inference_rows = []

gcn_checkpoint = torch.load(
    ARTIFACTS / "gnn_surrogate.pt", map_location="cpu", weights_only=True
)
gcn = SpatialGCN(
    gcn_checkpoint["input_dim"],
    gcn_checkpoint["hidden_dim"],
    gcn_checkpoint["dropout"],
)
gcn.load_state_dict(gcn_checkpoint["state_dict"], strict=True)
gcn.eval()
with torch.no_grad():
    gcn_prediction = gcn(standardized_features(gcn_checkpoint), adjacency).numpy()
    gcn_prediction = (
        gcn_prediction * gcn_checkpoint["target_scale"] + gcn_checkpoint["target_mean"]
    )
inference_rows.append(
    {
        "모델": "SpatialGCN",
        "출력 shape": str(gcn_prediction.shape),
        "유한값": bool(np.isfinite(gcn_prediction).all()),
        "최솟값": float(gcn_prediction.min()),
        "최댓값": float(gcn_prediction.max()),
        "파라미터 텐서": len(gcn_checkpoint["state_dict"]),
    }
)

mlp_checkpoint = torch.load(
    ARTIFACTS / "mlp_surrogate.pt", map_location="cpu", weights_only=True
)
mlp = NodeMLP(
    mlp_checkpoint["input_dim"],
    mlp_checkpoint["hidden_dim"],
    mlp_checkpoint["dropout"],
)
mlp.load_state_dict(mlp_checkpoint["state_dict"], strict=True)
mlp.eval()
with torch.no_grad():
    mlp_prediction = mlp(standardized_features(mlp_checkpoint)).numpy()
    mlp_prediction = (
        mlp_prediction * mlp_checkpoint["target_scale"] + mlp_checkpoint["target_mean"]
    )
inference_rows.append(
    {
        "모델": "NodeMLP",
        "출력 shape": str(mlp_prediction.shape),
        "유한값": bool(np.isfinite(mlp_prediction).all()),
        "최솟값": float(mlp_prediction.min()),
        "최댓값": float(mlp_prediction.max()),
        "파라미터 텐서": len(mlp_checkpoint["state_dict"]),
    }
)

assert gcn_prediction.shape == truth.shape == (7, 52_736)
assert mlp_prediction.shape == truth.shape
assert np.isfinite(gcn_prediction).all()
assert np.isfinite(mlp_prediction).all()
display(pd.DataFrame(inference_rows))
"""
    ),
    markdown(
        r"""
## 12. 코드 테스트와 산출물 체크섬 재실행

테스트는 notebook 실행 위치와 무관하도록 `modeling/src`를 `PYTHONPATH`에 명시한다. 체크섬 검증은 run manifest에 등록된 파일이 생성 이후 바뀌거나 손상됐는지 확인한다.
"""
    ),
    code(
        r"""
test_environment = os.environ.copy()
test_environment["PYTHONPATH"] = str(MODELING / "src")
test_run = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", str(MODELING / "tests")],
    cwd=ROOT,
    env=test_environment,
    capture_output=True,
    text=True,
)
print(test_run.stdout)
if test_run.stderr:
    print(test_run.stderr)
assert test_run.returncode == 0, "pytest 실패"
"""
    ),
    code(
        r"""
verify_run = subprocess.run(
    [
        sys.executable,
        str(MODELING / "scripts" / "verify.py"),
        "--artifacts",
        str(ARTIFACTS),
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
print(verify_run.stdout)
if verify_run.stderr:
    print(verify_run.stderr)
assert verify_run.returncode == 0, "artifact checksum 검증 실패"
assert "0 checksum mismatches" in verify_run.stdout
"""
    ),
    markdown(
        r"""
### 12.1 배포 묶음과 문서 버전 대조

live canonical 디렉터리의 체크섬 통과와 별개로, 전달용 tar.gz가 같은 개선 버전인지 확인한다. 개선 MLP와 재조정 스크립트가 빠진 과거 스냅샷을 “improved”로 전달하면 성능표를 재현할 수 없다.
"""
    ),
    code(
        r"""
import re
import tarfile

archive_path = MODELING / "artifacts" / "elder_guardian_model_run_20260727_improved.tar.gz"
required_archive_members = {
    "modeling/scripts/retune_surrogate.py",
    "modeling/artifacts/run_20260727_improved/mlp_surrogate.pt",
    "modeling/artifacts/run_20260727_improved/surrogate_ensemble.json",
    "modeling/artifacts/run_20260727_improved/surrogate_improvement.json",
    "modeling/artifacts/run_20260727_improved/surrogate_improvement_oof.npz",
}

with tarfile.open(archive_path, "r:gz") as archive:
    member_names = set(archive.getnames())
    missing_archive_members = sorted(required_archive_members - member_names)
    archived_manifest_member = archive.extractfile(
        "modeling/artifacts/run_20260727_improved/run_manifest.json"
    )
    archived_manifest = json.load(archived_manifest_member)
    archived_surrogate_member = archive.extractfile(
        "modeling/src/elder_guardian/surrogate.py"
    )
    archived_surrogate_hash = hashlib.sha256(archived_surrogate_member.read()).hexdigest()

live_surrogate_hash = sha256_file(MODELING / "src" / "elder_guardian" / "surrogate.py")
archive_current = (
    not missing_archive_members
    and archived_surrogate_hash == live_surrogate_hash
    and len(archived_manifest["artifact_hashes"]) == len(manifest["artifact_hashes"])
    and len(archived_manifest["sources"]["modeling_code_hashes"])
    == len(manifest["sources"]["modeling_code_hashes"])
)

readme_text = (MODELING / "README.md").read_text(encoding="utf-8")
readme_spearman_match = re.search(
    r"Spearman[^0-9]*([0-9]+\.[0-9]+)", readme_text
)
readme_spearman = (
    float(readme_spearman_match.group(1)) if readme_spearman_match else float("nan")
)
configured_spearman = float(thresholds["min_spearman"])

display(
    pd.DataFrame(
        [
            ("배포 tar SHA-256", sha256_file(archive_path)),
            ("tar artifact hash 수", len(archived_manifest["artifact_hashes"])),
            ("live artifact hash 수", len(manifest["artifact_hashes"])),
            ("tar source hash 수", len(archived_manifest["sources"]["modeling_code_hashes"])),
            ("live source hash 수", len(manifest["sources"]["modeling_code_hashes"])),
            ("필수 누락 member", ", ".join(missing_archive_members) or "없음"),
            ("surrogate.py live와 일치", archived_surrogate_hash == live_surrogate_hash),
            ("배포 묶음 최신", archive_current),
            ("README Spearman 기준", readme_spearman),
            ("config/manifest Spearman 기준", configured_spearman),
            ("문서-설정 기준 일치", np.isclose(readme_spearman, configured_spearman)),
        ],
        columns=["배포 감사", "결과"],
    )
)
"""
    ),
    markdown(
        r"""
## 13. 최종 검증표와 해석
"""
    ),
    code(
        r"""
scorecard = pd.DataFrame(
    [
        ("PDF 14쪽 전체 렌더·내용 대조", "통과", "페이지/해시 기록, 요구사항 매트릭스 작성"),
        ("데이터 행수·중복·총합·식별자", "통과", "61,652 격자, 3,644 시설, exact 404,054행"),
        ("E2SFCA 도달공급 보존식", "통과", f"실데이터 최대오차 {scenario_df['conservation_error'].max():.3e}"),
        ("기존시설 분모0 오류 분리", "부분 실패", "walk10에서 1시설·정원20 누락, 기존 gate가 미검출"),
        ("exact 국소갱신 vs brute-force", "통과", "소형 예제 절대차 < 1e-14"),
        ("전체 후보 반복실행", "통과", "404,054행 Gini/ΔG 최대차이 0"),
        ("저장 모델 로드·전체 forward", "통과", "GCN/MLP 모두 (7, 52,736), 유한 출력"),
        ("OOF 성능표 독립 재계산", "통과", "4개 모델 저장지표와 최대차이 < 1e-12"),
        ("서울전체 GCN/MLP shortlist", "실패", f"최고 RecallBest@20={policy_metrics['서울전체 RecallBest@20'].max():.3f} < 0.95"),
        ("공간 CV 버퍼", "실패", f"train-test 최소 {buffer_audit['train까지 최소거리(m)'].min():.0f}m; 연속 블록/LODO 아님"),
        ("순위손실·교통 그래프 edge", "미구현", "top-emphasized Huber 회귀와 물리 인접 edge만 사용"),
        ("현행 정원 회귀 배치", "기각", "검증 현행정원 0건, Spearman 0.086, 기준모델 우위 없음"),
        ("다중교통 일반화비용", "미구현", "현재 도보 직선거리 대체값"),
        ("필지·예산·법규 최종 타당성", "미구현", "후보는 100m 셀; exact 1위 중심점은 도로 증거"),
        ("전달용 tar와 live 버전", "통과" if archive_current else "실패", "누락 없음·해시 일치" if archive_current else "개선 MLP/소스가 빠진 이전 스냅샷"),
        ("정책 배치", "불가", "SCREENING_ONLY_NOT_POLICY_DEPLOYABLE"),
    ],
    columns=["검증 항목", "판정", "근거"],
)
display(scorecard)

best_surrogate = surrogate_table.sort_values(
    ["best_in_predicted_top_k", "relative_policy_regret"], ascending=[False, True]
).iloc[0]
display(
    Markdown(
        f'''
### 결론

1. **정확 계산 수학과 기본 시나리오는 정상이다.** 수식, 도달공급 보존식, 테스트 26개, 전체 후보 반복실행, 체크섬을 통과했고 기본 시나리오의 기존시설 분모 0도 없다. 전체 후보 평가는 **{base['exact_runtime_seconds']:.2f}초**였다.
2. **walk10 민감도 시나리오는 재실행이 필요하다.** 기존시설 1곳의 분모가 0이라 정원 20명이 조용히 도달공급에서 제외됐지만 기존 gate가 이를 잡지 못했다.
3. **대리모델은 실행되지만 승인되지 않았다.** 자치구 내부 진단에서 가장 좋은 `{best_surrogate['model']}`의 `RecallBest@20`은 **{best_surrogate['best_in_predicted_top_k']:.3f}**, 서울 전체 최고값은 **{policy_metrics['서울전체 RecallBest@20'].max():.3f}**로 기준 0.95보다 낮다. K=50 자치구 지표를 보고 사후에 기준을 바꾸지 않았다.
4. **기본 시나리오의 형평성 개선은 작다.** 최상 후보의 상대 Gini 감소는 **{base['relative_gini_reduction_pct']:.3f}%**다.
5. **현재 결과는 정책 확정안이 아니다.** 현행 정원, 도로망·대중교통 OD, 필지·지가·건축비·법적 가능성을 결합하기 전에는 설치 위치나 기대효과를 확정할 수 없다.

### 개선 우선순위

1. 시설별 현행 동시수용 정원과 검증된 전용면적을 동일 기준일로 수집한다.
2. 도보 네트워크와 버스·지하철 GTFS/환승을 결합한 시간대별 OD 일반화비용을 만든다.
3. 연속지적도·용도지역·도시계획시설·지가·건축비로 후보 필지와 예산 제약을 exact 계산 전에 적용한다.
4. walk10 기존시설 분모 0을 오류상태로 강제하고, 연속 공간 블록/LODO와 이동권 버퍼를 둔 OOF를 다시 수행한다.
5. pairwise 순위손실과 서울 전체 정책지표를 추가한 뒤 새 입력으로 exact 정답을 다시 만들고 재학습한다.
6. 대리모델은 `RecallBest@20 ≥ 0.95`를 통과할 때까지 속도 최적화용으로만 사용하고, 최종 후보는 항상 exact로 재평가한다.
'''
    )
)
"""
    ),
    markdown(
        r"""
## 재실행 방법

프로젝트 루트에서:

```bash
python3 modeling/scripts/build_validation_notebook.py
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  modeling/notebooks/14_rewritten_model_revalidation.ipynb
```

전체 모델을 원천 데이터부터 다시 학습하려면 `modeling/README.md`의 학습 명령을 사용한다. 이 노트북은 이미 생성된 106MB exact 결과를 검증하는 용도이며, 매번 전체 원천 파이프라인을 재학습하지 않는다.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3.11 (Elder Guardian)",
            "language": "python",
            "name": "elder-guardian-py311",
        },
        "language_info": {
            "name": "python",
            "version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        },
        "title": "14_rewritten (1).pdf 모델 재검증",
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
