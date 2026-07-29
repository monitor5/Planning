# SpatialGCN 모델 카드

## 1. 결론

`SpatialGCN`은 신규 경로당 후보의 exact 형평성 개선량
\(\Delta G=G(A_0)-G(A_k)\)을 근사해 후보 수를 줄이는 대리모델이다. 모델
파일은 정상 로드되고 전체 후보에 대해 유한한 값을 출력하지만, 사전 정의된
`RecallBest@20 >= 0.95` 기준을 통과하지 못했다.

- **허용 용도:** 동일 데이터 계약 안에서 넓은 후보 shortlist를 만드는 실험적
  가속기
- **금지 용도:** GNN 점수를 최종 입지 순위로 사용하거나 실제 정책 효과로 해석
- **최종 권한:** `exact_e2sfca_weighted_gini`
- **상태:** `NOT_APPROVED_FOR_SHORTLIST_AT_K20`

## 2. 예측 대상과 데이터 계약

| 항목 | 값 |
|---|---|
| target | 후보별 exact `delta_gini` |
| 후보 노드 | 52,736개 |
| 시나리오 | 7개 |
| 입력 shape | `(7, 52_736, 19)` |
| 출력 shape | `(7, 52_736)` |
| 정답 의미 | 현재 가정 아래의 모의 exact 결과 |
| 실제 관측 정책성과 | 없음 |
| 좌표계 | EPSG:5179 미터 좌표 |
| 이동비용 | 51m/분 도보 직선거리 대체값 |
| 정원 의미 | 법정 최소 또는 과거 참고 시나리오, 현행 운용정원 아님 |

19개 입력 특성은 모두 설치 이전에 계산된다.

1. 수요: 현재격자 인구, 500m·1km·2km 국지 인구
2. 기존 접근성·공급: 기준 접근성, 반경별 공급, 최근접 시설거리,
   수요-공급비
3. 후보 catchment: 가중수요, 기준 접근성 평균·표준편차, 저접근 비중·격차
4. 시나리오: 신규정원, Gaussian `tau`, cutoff, 과거정원 모드

`post_install_accessibility`, `post_install_gini`, `delta_gini`, exact rank는
입력에서 금지된다.

## 3. 그래프와 신경망 구조

그래프는 같은 자치구에 속한 후보 중 145m 이내 이웃을 연결하고 self-loop를
추가한 뒤 대칭 정규화한다.

\[
\hat A=D^{-1/2}(A+I)D^{-1/2}
\]

\[
H_1=\operatorname{ReLU}(W_1\hat AX+b_1),\qquad
H_2=\operatorname{ReLU}(W_2\hat AH_1+b_2),\qquad
\hat y=W_oH_2+b_o
\]

| 구조 항목 | 값 |
|---|---:|
| 후보 노드 | 52,736 |
| 무방향 non-self edge | 200,767 |
| self-loop 포함 sparse nnz | 454,270 |
| 연결성분 | 36 |
| self-loop만 있는 노드 | 2 |
| 입력 차원 | 19 |
| hidden 차원 | 32 |
| graph propagation | 2회 |
| dropout | 0.15 |
| 학습 파라미터 | 1,729 |
| 체크포인트 크기 | 11,549 bytes |

현재 edge는 물리 인접만 나타낸다. 버스·지하철 연결, 환승, 이동시간 edge는
없다.

## 4. 학습

| 항목 | 설정 |
|---|---|
| optimizer | Adam |
| learning rate | 0.01 |
| weight decay | 0.0001 |
| 최대 epoch | 160 |
| early-stopping patience | 20 |
| CV fold | 자치구 GroupKFold 5개 |
| validation | 각 fold의 train 자치구 중 1개를 별도 사용 |
| loss | top-emphasized Huber |
| top loss weight | 20 |
| random seed | 20260727 |

특성 평균·표준편차와 target scale은 fold별 train 노드에서만 적합한다. OOF
지표를 만든 fold 모델은 저장되지 않는다. 배포 checkpoint는 전체 exact 라벨로
160 epoch 고정 재학습하며 early stopping, validation score, epoch별 loss
기록이 없다. checkpoint bundle에도 learning rate, weight decay, top loss
weight, epoch 수, adjacency radius가 들어 있지 않아 파일 단독으로 학습을
완전히 재현할 수 없다. PDF가 제안한 pairwise ranking loss도 없으며, 저장
성능은 단일 seed 결과다.

## 5. OOF 성능

### 자치구 내부 진단 175단위

| 지표 | SpatialGCN | 승인 기준 | 판정 |
|---|---:|---:|---|
| MAE | `2.8424e-05` | - | 참고 |
| Spearman | `0.98284` | `>= 0.85` | 통과 |
| Recall@20 | `0.66743` | `>= 0.50` | 통과 |
| RecallBest@20 | `0.78857` | `>= 0.95` | **실패** |
| 평균 상대 policy regret@20 | `0.003945` | `<= 0.02` | 통과 |
| 상대 regret p95@20 | `0.032455` | - | 꼬리위험 존재 |

이 175단위 지표는 자치구 안에서 후보를 고르는 진단이다. 서울 전체에서 한
곳을 고르는 PDF 정책목표와 동일하지 않다.

### 서울 전체 7개 시나리오

| K | Recall@K | RecallBest@K | 평균 상대 regret |
|---:|---:|---:|---:|
| 10 | 0.3571 | 0.5714 | 0.003407 |
| 20 | 0.4214 | 0.5714 | 0.000567 |
| 50 | 0.6771 | 1.0000 | 0.000000 |

K=50에서는 저장된 7개 OOF 시나리오의 실제 1위를 모두 포함했지만, 이는 같은
결과를 본 뒤 발견한 operating point다. 독립 검증셋 없이 K=50을 새로운 승인
기준으로 채택할 수 없다.

### 시나리오별 실제 1위의 GCN 예측순위

| 시나리오 | 실제 1위 격자 | GCN 예측순위 | top-20 포함 |
|---|---|---:|---|
| walk10/new20 | 다사637437 | 10 | 예 |
| walk15/new20 | 다사639440 | 4 | 예 |
| base walk15/new40 | 다사638441 | 6 | 예 |
| walk15/new60 | 다사640506 | 35 | 아니오 |
| walk20/new40 | 다사641507 | 33 | 아니오 |
| historical walk15/new40 | 다사642505 | 34 | 아니오 |
| 1km residual walk15/new40 | 다사638441 | 6 | 예 |

전체 OOF 회귀 성능은 `RMSE=4.4393e-05`, `R²=0.96636`이다. 회귀 적합도와
Spearman이 높아도 최상위 국소 후보를 정확히 잡는 능력은 별개이므로
RecallBest를 승인 게이트로 유지한다.

동일한 자치구 holdout에서 graph를 사용하지 않는 `NodeMLP`가
`MAE=2.5724e-05`, `R²=0.97257`로 GCN보다 좋았고, 서울 전체 실제 1위의
예측순위도 `6, 6, 11, 20, 27, 8, 11`로 top-20 포함 6/7이었다. 따라서 현재
실험은 graph aggregation의 추가 이득을 입증하지 못했다. exact 1위와 20위의
시나리오별 ΔG 간격은 `1.93e-06~1.31e-05`인데 GCN 전체 MAE는 이 간격의
`2.24~12.42배`여서 미세 최상위 순위를 직접 확정할 해상도가 부족하다.

전체 라벨로 학습한 최종 checkpoint를 같은 학습표본에 다시 적용하면 실제
1위의 예측순위는 `13, 4, 5, 43, 8, 21, 5`이고 top-20 포함은 5/7이다.
이는 학습내 smoke test이므로 OOF 성능이나 외부성능으로 사용할 수 없다.

## 6. 실사용 판단

현재 GNN은 **독립 추천기로는 쓸 수 없고, 넓은 shortlist 가속기로만 제한적
가능성**이 있다.

1. top-20은 서울 전체 7개 시나리오 중 3개에서 실제 1위를 놓쳤다.
2. K=50 OOF 결과는 유망하지만 시나리오가 7개뿐이고 서로 독립적인 외부
   표본이 아니다.
3. 공간 CV는 자치구 5-fold이며 연속 공간 block, 이동권 buffer, LODO가 아니다.
4. 다른 기준일·도시·교통망·현행 정원에 대한 외부 검증이 없다.
5. GNN target 자체가 직선거리·대체 정원 기반 exact 결과이므로 현실 정책성과를
   검증한 것이 아니다.
6. 현재 exact 전체평가도 시나리오당 약 3.6~7.0초라서, 단발성 서울 분석에서는
   GNN의 위험을 감수할 만큼 계산 절감이 절대적으로 크지 않다.

저장된 tensor를 대상으로 한 CPU forward 자체는 약 0.05초다. 그러나 새
시나리오의 특성 생성까지 포함한 실제 base 측정에서는 GNN Top-50 + exact
재순위가 약 4.36초, 특성 생성 + 전체 exact가 약 9.27초로 약 2.1배
가속이었다. 모델 학습 약 219초까지 회수하려면 반복 시나리오가 필요하다.
이 수치는 현재 장비·데이터의 실측값이며 일반 SLA가 아니다.

따라서 운영 절차는 다음으로 제한한다.

```text
GNN broad shortlist (실험적으로 K>=50)
  -> shortlist의 exact E2SFCA/Gini 전수 재평가
  -> 필지·예산·교통·현행정원 gate
  -> 사람의 정책 검토
```

새 데이터로 재학습한 뒤에는 서울전체 RecallBest, 외부 시점/도시 holdout,
연속 공간 buffer, 여러 seed 안정성, 시나리오 holdout을 다시 통과해야 한다.

## 7. 실행 인터페이스와 배포 한계

체크섬 검증, feature/scenario 순서 검증, 유한값·shape 검증을 포함한 제한형
CLI가 제공된다.

```bash
python modeling/scripts/predict_gnn.py \
  --artifacts modeling/artifacts/run_20260727_improved \
  --scenario legal20_walk15_new40_BASE \
  --top 50 \
  --output /tmp/gnn_shortlist.csv
```

이 CLI는 저장된 7개 시나리오 tensor만 점수화한다. 임의의 원자료에서 새
feature tensor를 만드는 온라인 API가 아니며 exact 재평가도 자동 수행하지
않는다. `gnn_surrogate.pt` 단독으로는 부족하고
`surrogate_training_tensor.npz`, 동일 graph 규칙, manifest가 함께 필요하다.
lockfile, container, torch ABI 선언, latency/memory SLA와 drift monitor도 아직
없으므로 운영 배포 패키지가 아니라 감사 가능한 연구 artifact다.
