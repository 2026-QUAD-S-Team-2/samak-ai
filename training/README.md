# 학습(Training)

## 워크플로우

1. `python training/preprocess.py` → 전처리된 CSV + `preprocess_signature.json` 생성
2. `python training/train_baseline.py train` → TF-IDF + Logistic Regression 학습, `training/runs/tfidf_lr/`에 저장
3. `python training/evaluate_baseline.py --tune` → PR curve 기반 best threshold 탐색, `metadata.json` 업데이트
4. `python training/train_baseline.py export` → `training/runs/tfidf_lr/`에서 `models/fraud-baseline/`로 원자적(atomic) 교체

## 한 번에 실행(추천)

전처리 → 학습 → export 를 한 번에:

```bash
python training/train_baseline.py all --export-dir models/fraud-baseline
```

평가(threshold 튜닝 포함):

```bash
python training/evaluate_baseline.py --model-dir models/fraud-baseline --tune
```

> 모듈 방식으로도 실행 가능:
> - `python -m training.train_baseline all --export-dir models/fraud-baseline`
> - `python -m training.evaluate_baseline --model-dir models/fraud-baseline --tune`

## 데이터셋 정책

입력 파일(존재할 경우 자동 선택):
- `training/data/raw/RecruitmentScam.csv` — **필수**. train/test split 기준
- `training/data/raw/FakeJobPostings.csv` — all-positive면 `FakeJobPostings_PosOnly`로 취급, **train 증강만**
- `training/data/raw/LinkedInPostings.csv` — all-negative(`LinkedIn_NegOnly`), **train 증강만**

역할 분리:
- **test set**은 반드시 `RecruitmentScam` 에서만 생성 (stratified split)
- `FakeJobPostings_PosOnly`, `LinkedIn_NegOnly`는 train 증강 전용(test 금지)
- 증강 비율: `--fakepos-multiplier`(default 2.0), `--linkedin-multiplier`(default 3.0)

누수 방지:
- 전처리 후 `text_hash`(SHA256) 기준으로 중복 제거
- train/test 해시 교집합이 있으면 오류로 중단

## 전처리 출력 스키마

`training/data/processed/` 아래:

| 파일 | 설명 |
|------|------|
| `combined_train.csv` / `train.csv` | 학습용 (RecruitmentScam train + augment) |
| `test.csv` | 평가용 (RecruitmentScam test only) |
| `valid.csv` | `--valid-size` > 0 일 때만 생성 |
| `preprocess_signature.json` | 전처리 파라미터 재현성 기록 |

처리된 CSV 컬럼:

```
text            # 통합 문서 텍스트 ([TITLE] ... [DESCRIPTION] ... 형태)
fraudulent      # 라벨 (0/1)
salary_low      # 파싱된 최저 급여
salary_high     # 파싱된 최고 급여
salary_mid      # 중간값
has_salary      # 급여 정보 존재 여부 (0/1)
has_location    # 위치 정보 존재 여부 (0/1)
location_len    # 위치 문자열 길이
source          # 데이터 출처 (RecruitmentScam / FakeJobPostings_PosOnly / LinkedIn_NegOnly)
text_len        # 문서 길이
text_hash       # SHA256 해시 (중복 제거/누수 검사용)
```

## 텍스트 통합 템플릿

학습·평가·서빙에서 동일하게 사용:

```
[TITLE] ...
[LOCATION] ...
[EMPLOYMENT_TYPE] ...
[INDUSTRY] ...
[SALARY] ...
[COMPANY_PROFILE] ...
[DESCRIPTION] ...
[REQUIREMENTS] ...
[BENEFITS] ...
```

클린업: HTML 제거, URL/이메일/전화번호 마스킹(`<URL>` `<EMAIL>` `<PHONE>`), 라벨 누수 키워드 마스킹(`<LEAK>`), 소문자, 다중 공백 제거, 최대 6000자 제한.

급여/위치 피처는 토큰 형태(`has_salary=1`, `salary_low=50000` 등)로 문서 끝에 주입되어 TF-IDF에서도 활용됩니다.

## 모델 아티팩트

학습 결과물(`training/runs/tfidf_lr/`):

```
training/runs/tfidf_lr/
├── vectorizer.joblib   # TF-IDF 벡터라이저 (max_features=50000, ngram 1-2)
├── model.joblib        # Logistic Regression (class_weight=balanced)
└── metadata.json       # threshold, metrics, preprocess_signature, dataset_signature
```

export 후 inference가 읽는 경로(`models/fraud-baseline/`):

```
models/fraud-baseline/
├── vectorizer.joblib
├── model.joblib
└── metadata.json       # evaluate 단계에서 best threshold가 기록됨
```

## 평가 지표

`evaluate_baseline.py --tune` 실행 시:
- PR-AUC (주 지표)
- `threshold_f1_max`: PR curve에서 F1이 최대가 되는 threshold
- `threshold_precision_90`: precision ≥ 0.9 조건을 만족하면서 recall이 최대인 threshold
- source별 세부 지표 (`metrics_by_source`)
- `metadata.json`에 자동 저장
