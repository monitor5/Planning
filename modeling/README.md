# Elder Guardian 입지 스크리닝 모델

이 디렉터리는 `14_rewritten (1).pdf`의 목적함수에 맞춘 실행형 모델입니다.
100m 고령인구 가중 접근성 Gini를 최소화하는 신규 경로당 후보를 다음 순서로
평가합니다.

1. 25개 자치구의 100m 격자를 `gid`별로 합산합니다.
2. 현재 확인할 수 없는 운영 동시수용량은 실제값으로 위장하지 않고, 명명된
   민감도 시나리오로만 사용합니다.
3. 보행 직선거리 기반 Gaussian E2SFCA를 계산합니다.
4. 모든 허용 후보의 설치 후 가중 Gini를 정확하게 재계산합니다.
5. 정확 결과만 라벨로 쓰는 GCN 후보 압축기를 자치구 공간 홀드아웃으로
   검증합니다.
6. 최종 추천은 GCN 점수를 그대로 쓰지 않고 exact evaluator로 다시 검산합니다.

## 빠른 실행

프로젝트 루트에서:

```bash
python3 -m pip install -e 'modeling[test]'
python3 -m pytest modeling/tests
python3 modeling/scripts/train.py \
  --config modeling/config/base.json \
  --run-id run_20260727_full
python3 modeling/scripts/predict.py \
  --artifacts modeling/artifacts/run_20260727_full \
  --top 10
```

기본 입력은 다음 위치를 자동으로 찾습니다.

- 인구: `Data/서울_격자별노인인구(100M)`
- 시설 master:
  `tmp/planning_data/data/facilities/processed/facility-master.csv`
- 과거 정원 참고:
  `tmp/planning_data/data/facilities/processed/historical-capacity-candidates.csv`

입력 위치는 `train.py` 옵션으로 바꿀 수 있습니다.

## 결과의 정확한 해석

현재 데이터에는 3,644개 경로당의 확인된 운영 동시수용량이 한 건도 없습니다.
따라서 `STRICT_UNKNOWN`은 공급합이 0이고 Gini가 정의되지 않아 자동으로
실패합니다. 기본 `LEGAL_NOMINAL=20`은 법정 최소 프록시일 뿐 실제 정원이
아닙니다. 과거 정원도 현행 운영값이 아닌 낡은 참고치로 별도 표시됩니다.

교통 OD가 없으므로 결과는 노인 보행속도를 실측한 모델이 아니라, 설정 파일에
고정한 직선거리·분당 51m 시나리오입니다. 후보지의 소유권, 지가, 용도지역,
경사, 건축 가능성도 입력에 없으므로 결과는 **토지 실현가능성 검토 전 100m
격자 스크리닝**입니다. 행정 의사결정용 최종 부지 추천으로 사용하면 안 됩니다.

`연면적_결과_간단.csv`는 건물 전체 연면적을 시설 전용면적으로 잘못 재사용해
최대 수만 명의 비현실적 정원을 만들므로 완전히 배제합니다. `results_full.csv`의
`estimated_capacity`도 면적 공식으로 생성된 값이어서 회귀 정답으로 사용하지
않습니다.

## 모델 승인 게이트

GCN은 자치구 공간 홀드아웃에서 다음 조건을 모두 만족할 때만 shortlist
surrogate로 승인됩니다.

- Spearman 순위상관 ≥ 0.75
- Recall@20 ≥ 0.50
- RecallBest@20 ≥ 0.95
- top-20 exact 재평가 평균 상대 정책 regret ≤ 0.02

통과 여부와 관계없이 exact evaluator가 최종 권위 모델입니다. 실행 결과에는
입력 SHA-256, 설정, 모델 가중치, exact 후보 결과, 공간 검증 지표, 공급보전
오차, 추천 후보와 민감도 순위를 함께 저장합니다.

## 추가 데이터가 들어오면 바꿀 부분

- 시설별 `운영/휴지/폐지` 상태와 현행 동시수용량
- 시설 전용 순사용면적과 동일 시점의 정원 쌍
- 보행·버스·지하철 OD 및 환승/대기시간
- 후보 필지, 지가·예산, 용도지역, 경사, 소유권, 건축 가능 면적
- 억제 격자의 승인된 통계적 보정 규칙

이 자료가 확보되면 용량 회귀와 다중교통 비용행렬을 재학습한 뒤 같은 exact
평가기와 공간 검증 게이트를 재실행할 수 있습니다.
