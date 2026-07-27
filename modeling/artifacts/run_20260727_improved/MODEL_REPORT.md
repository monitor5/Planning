# Elder Guardian 모델 실행 보고서

실행 ID: `run_20260727_improved`

## 결론

이 실행의 정책 목적함수는 PDF와 동일한 65+ 인구가중 접근성 Gini입니다. 다만
현행 운영 동시수용량, 다중교통 OD, 후보 필지 실현가능성이 없으므로 아래 위치는
**보행·용량 시나리오 기반 100m 격자 스크리닝**이며 최종 부지가 아닙니다.

기본 시나리오 exact 상위 후보:

- 다사638441 / 강남구: (37.495900, 127.091041), ΔG=0.00067231
- 다사639441 / 강남구: (37.495904, 127.092172), ΔG=0.00067053
- 다사637441 / 강남구: (37.495896, 127.089909), ΔG=0.00066669
- 다사637442 / 강남구: (37.496798, 127.089905), ΔG=0.00066641
- 다사638442 / 강남구: (37.496801, 127.091036), ΔG=0.00066531

전체 민감도 시나리오의 평균 순위가 안정적인 후보:

- 다사635507 / 광진구: (37.555377, 127.087319), 평균 백분위=0.999647
- 다사641506 / 광진구: (37.554499, 127.094117), 평균 백분위=0.999609
- 다사642506 / 광진구: (37.554503, 127.095249), 평균 백분위=0.999602
- 다사641505 / 광진구: (37.553598, 127.094122), 평균 백분위=0.999579
- 다사642505 / 광진구: (37.553601, 127.095254), 평균 백분위=0.999576

## 정확 평가 결과

- `legal20_walk10_new20`: baseline Gini=0.33677020, best after=0.33641913, ΔG=0.00035107, conservation error=5.82e-11
- `legal20_walk15_new20`: baseline Gini=0.27622446, best after=0.27587979, ΔG=0.00034467, conservation error=1.46e-11
- `legal20_walk15_new40_BASE`: baseline Gini=0.27622446, best after=0.27555215, ΔG=0.00067231, conservation error=1.46e-11
- `legal20_walk15_new60`: baseline Gini=0.27622446, best after=0.27526429, ΔG=0.00096017, conservation error=1.46e-11
- `legal20_walk20_new40`: baseline Gini=0.24395744, best after=0.24329519, ΔG=0.00066225, conservation error=8.73e-11
- `historical_reference_walk15_new40`: baseline Gini=0.29278496, best after=0.29228887, ΔG=0.00049610, conservation error=1.46e-11
- `legal20_walk15_new40_1km_residual`: baseline Gini=0.27598118, best after=0.27530955, ΔG=0.00067163, conservation error=0

## 공간 홀드아웃 surrogate 검증

- SpatialGCN: Spearman=0.9828, Recall@20=0.6674, RecallBest@20=0.7886 (95% bootstrap CI 0.7257–0.8514), relative regret@20=0.0039, p95 regret@20=0.0325, MAE=2.8424255e-05
- HistGradientBoosting: Spearman=0.9757, Recall@20=0.4340, RecallBest@20=0.5257 (95% bootstrap CI 0.4571–0.6000), relative regret@20=0.0129, p95 regret@20=0.0701, MAE=9.3703253e-05

- GCN shortlist 승인: `False`
- 최종 권위 모델: `exact_e2sfca_weighted_gini`

## 수용량 회귀 검증

```json
{
  "selected_model": "median_scenario",
  "deployable": false,
  "reason": "insufficient_positive_correlation:spearman=0.086479<0.200000",
  "metrics": {
    "sample": {
      "input_rows": 292,
      "valid_positive_rows": 292,
      "dropped_rows": 0,
      "borough_groups": 23,
      "unique_areas": 272,
      "unique_capacities": 81
    },
    "correlation": {
      "pearson": -0.006741528166304999,
      "spearman": 0.08647879918590513,
      "minimum_required_spearman": 0.2,
      "sufficient_positive_correlation": false
    },
    "cv": {
      "scheme": "GroupKFold",
      "group_column": "borough",
      "n_splits": 5,
      "status": "completed",
      "all_preprocessing_fit_within_fold": true
    },
    "models": {
      "median_scenario": {
        "status": "ok",
        "mae": 21.174657534246574,
        "rmse": 28.964964665647216,
        "median_absolute_error": 17.0,
        "r2": -0.07366398574873334,
        "folds": [
          {
            "fold": 1,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 21.305084745762713,
            "rmse": 29.91428432671691,
            "median_absolute_error": 14.0,
            "r2": -0.3258813534360716
          },
          {
            "fold": 2,
            "train_rows": 235,
            "test_rows": 57,
            "mae": 17.228070175438596,
            "rmse": 22.765779181797974,
            "median_absolute_error": 15.0,
            "r2": -0.1759939660169425
          },
          {
            "fold": 3,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 23.338983050847457,
            "rmse": 33.45399066666672,
            "median_absolute_error": 16.0,
            "r2": -0.1640640406747087
          },
          {
            "fold": 4,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 19.338983050847457,
            "rmse": 24.686852330058233,
            "median_absolute_error": 17.0,
            "r2": -0.008155198255763274
          },
          {
            "fold": 5,
            "train_rows": 234,
            "test_rows": 58,
            "mae": 24.586206896551722,
            "rmse": 32.3376582010979,
            "median_absolute_error": 18.0,
            "r2": -0.2196851813327787
          }
        ]
      },
      "nonnegative_linear": {
        "status": "ok",
        "mae": 22.34317224585621,
        "rmse": 28.736844153361297,
        "median_absolute_error": 18.871373023733536,
        "r2": -0.056818786770651775,
        "folds": [
          {
            "fold": 1,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 20.559732789760996,
            "rmse": 27.857182085445725,
            "median_absolute_error": 15.811322369696846,
            "r2": -0.1497986500434505
          },
          {
            "fold": 2,
            "train_rows": 235,
            "test_rows": 57,
            "mae": 23.02053004852557,
            "rmse": 26.865673428343992,
            "median_absolute_error": 21.95744680851064,
            "r2": -0.6377044075006695
          },
          {
            "fold": 3,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 22.64072161198807,
            "rmse": 31.868957483604436,
            "median_absolute_error": 16.197424892703864,
            "r2": -0.056371618135130364
          },
          {
            "fold": 4,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 21.813050120026183,
            "rmse": 26.06528905194101,
            "median_absolute_error": 20.566523605150216,
            "r2": -0.1238828553080602
          },
          {
            "fold": 5,
            "train_rows": 234,
            "test_rows": 58,
            "mae": 23.72826407309166,
            "rmse": 30.570087537943945,
            "median_absolute_error": 22.5,
            "r2": -0.08999367865921126
          }
        ],
        "relative_mae_improvement_vs_median": -0.05518458608927924,
        "fold_win_rate_vs_median": 0.6,
        "rmse_improved_vs_median": true
      },
      "huber": {
        "status": "ok",
        "mae": 21.847774254649284,
        "rmse": 29.209513825373655,
        "median_absolute_error": 17.943676132664624,
        "r2": -0.09187025885833,
        "folds": [
          {
            "fold": 1,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 21.1228341224945,
            "rmse": 29.555926261232287,
            "median_absolute_error": 14.76395726949383,
            "r2": -0.2943048463018496
          },
          {
            "fold": 2,
            "train_rows": 235,
            "test_rows": 57,
            "mae": 20.148258864151465,
            "rmse": 24.6357588490767,
            "median_absolute_error": 18.55016761986493,
            "r2": -0.3771205016537431
          },
          {
            "fold": 3,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 23.200837208671793,
            "rmse": 33.25487746175934,
            "median_absolute_error": 15.684783005238167,
            "r2": -0.15024860308074217
          },
          {
            "fold": 4,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 20.24645796821132,
            "rmse": 25.146328903268387,
            "median_absolute_error": 19.328827689225342,
            "r2": -0.04603240471225711
          },
          {
            "fold": 5,
            "train_rows": 234,
            "test_rows": 58,
            "mae": 24.50796031789101,
            "rmse": 32.28210664047694,
            "median_absolute_error": 18.398156135334226,
            "r2": -0.21549828457490094
          }
        ],
        "relative_mae_improvement_vs_median": -0.031788788995243636,
        "fold_win_rate_vs_median": 0.6,
        "rmse_improved_vs_median": false
      },
      "random_forest": {
        "status": "ok",
        "mae": 23.323524721801366,
        "rmse": 30.289659439904874,
        "median_absolute_error": 19.003368178262576,
        "r2": -0.1741164164593909,
        "folds": [
          {
            "fold": 1,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 21.223711043178827,
            "rmse": 27.99307494916868,
            "median_absolute_error": 15.829437715742841,
            "r2": -0.16104390368389132
          },
          {
            "fold": 2,
            "train_rows": 235,
            "test_rows": 57,
            "mae": 24.951034757136185,
            "rmse": 29.8910860347388,
            "median_absolute_error": 20.846830702487615,
            "r2": -1.027325319729989
          },
          {
            "fold": 3,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 24.873082779637258,
            "rmse": 35.68548771657836,
            "median_absolute_error": 18.07046728611239,
            "r2": -0.32453757822740026
          },
          {
            "fold": 4,
            "train_rows": 233,
            "test_rows": 59,
            "mae": 21.364671420657174,
            "rmse": 25.926024507042076,
            "median_absolute_error": 20.30689001307917,
            "r2": -0.11190532414123844
          },
          {
            "fold": 5,
            "train_rows": 234,
            "test_rows": 58,
            "mae": 24.276444624901956,
            "rmse": 31.058028574118914,
            "median_absolute_error": 19.349772956350588,
            "r2": -0.12506699649673658
          }
        ],
        "relative_mae_improvement_vs_median": -0.10148297246741055,
        "fold_win_rate_vs_median": 0.4,
        "rmse_improved_vs_median": false
      }
    },
    "selection": {
      "minimum_relative_mae_improvement": 0.05,
      "minimum_fold_win_rate": 0.6,
      "selected_model": "median_scenario",
      "deployable": false,
      "reason": "insufficient_positive_correlation:spearman=0.086479<0.200000"
    }
  },
  "metadata": {
    "feature_column": "area_m2",
    "target_column": "capacity",
    "group_column": "borough",
    "target_policy": "positive_observed_values_only",
    "generated_capacity_labels_used": false,
    "prohibited_target_columns": [
      "capacity_area_fire_proxy",
      "capacity_legal_nominal",
      "capacity_strict_unknown",
      "estimated_capacity",
      "legacy_capacity_proxy"
    ],
    "deployment_trust_requested": false,
    "trusted_for_deployment": false,
    "historical_legacy_lineage": true,
    "training_area_min_m2": 23.98,
    "training_area_max_m2": 1649.17,
    "random_state": 20260727,
    "preparation": {
      "master_rows": 4281,
      "historical_rows": 1854,
      "joined_rows": 1854,
      "positive_finite_rows": 292,
      "dropped_missing_or_nonpositive_area": 1372,
      "dropped_missing_or_nonpositive_capacity": 692,
      "dropped_ineligible_p0": 0,
      "all_rows_require_current_register_verification": true,
      "any_row_marked_operationally_usable": false,
      "label_source": "capacity_registered_historical_candidate",
      "feature_source": "legacy_usable_area_m2",
      "generated_capacity_labels_used": false
    },
    "selection_reason": "insufficient_positive_correlation:spearman=0.086479<0.200000",
    "facility_master_csv": "${PROJECT_ROOT}/tmp/planning_data/data/facilities/processed/facility-master.csv",
    "historical_candidates_csv": "${PROJECT_ROOT}/tmp/planning_data/data/facilities/processed/historical-capacity-candidates.csv",
    "lineage_policy": "historical capacity and legacy area are diagnostic-only until both are verified against current authoritative records",
    "source_workflow_deployable": false,
    "artifact_path": "${PROJECT_ROOT}/modeling/artifacts/run_20260727_improved/capacity_diagnostic.joblib"
  }
}
```

실제 정원과 시설 전용면적의 동시 관측이 없거나 회귀가 중앙값 기준선을 이기지
못하면 용량 모델은 배포 불가로 표시됩니다. 이 경우 위치 모델은 20/40/60명
명시적 시나리오만 사용합니다.

## 데이터 및 재현성

- 100m 공개 하한 인구 합계: 1962046
- 1km 잔차 민감도 인구 합계: 1969965
- 고유 100m 격자: 61652
- 시설 master 행: 3644
- 확인된 운영 동시수용량 행: 0

입력 SHA-256, source commit, 설정 전체는 `run_manifest.json`에 저장했습니다.

## 사용 전 필수 확인

1. 추천 격자의 필지 소유권·지가·용도지역·경사·건축 가능 면적을 검증합니다.
2. 3,644개 시설의 운영/휴지/폐지 상태와 현재 동시수용량을 수집합니다.
3. 보행·버스·지하철 OD와 환승/대기비용으로 직선거리 시나리오를 교체합니다.
4. 위 자료로 exact 평가와 공간 홀드아웃을 다시 실행합니다.
