# 학습(Training) (스켈레톤)

이 레포지토리는 **추론(inference)** 이 중심이며, 학습 파이프라인은 MVP 단계에서 **뼈대(skeleton)** 만 제공합니다.

## 예정 워크플로우

1. 원천/가공 데이터는 `training/data/` 아래에 둡니다. (기본적으로 gitignored)
2. `python training/preprocess.py` 실행 → **평가 가능한 test 셋 생성(RecruitmentScam only)** + train 증강(FakePos/LinkedIn)
3. `python training/train_baseline.py train` 실행 → Baseline 모델 학습(TF-IDF + Logistic Regression) + 저장
4. `python training/evaluate_baseline.py --tune` 실행 → PR-AUC 중심 평가 + best threshold 저장(metadata.json)
5. `python training/train_baseline.py export` 실행 → 아티팩트를 `models/fraud-baseline/`로 원자적(atomic) 교체 내보내기:
   - `vectorizer.joblib`
   - `model.joblib`
   - `metadata.json`

## 한 번에 실행(추천)

전처리 → 학습 → (옵션) export 를 한 번에 하려면:

```bash
python training/train_baseline.py all --export-dir models/fraud-baseline
```

저장된 모델을 test set으로 평가(+accuracy 포함):

```bash
python training/evaluate_baseline.py --model-dir models/fraud-baseline --test-path training/data/processed/test.csv
```

> 팁: 모듈 방식으로도 실행할 수 있습니다.
> - `python -m training.train_baseline all --export-dir models/fraud-baseline`
> - `python -m training.evaluate_baseline --model-dir models/fraud-baseline`

## make-dataset 통합 스키마

전처리 스크립트(`training/preprocess.py`)는 아래 공통 컬럼만 남기고 통합합니다:

```text
title
description
requirements
company_profile
location
salary_range
employment_type
industry
benefits
fraudulent
```

MVP 1차 기준으로 아래 컬럼들은 제거합니다:
- `department`, `function`
- `telecommuting`, `has_company_logo`, `has_questions`

기본 입력 파일(존재할 경우 자동 선택):
- `training/data/raw/FakeJobPostings.csv`
- `training/data/raw/RecruitmentScam.csv`
- `training/data/raw/LinkedInPostings.csv` (정상(0-only) 증강용)

## 전처리 핵심 규칙(요약)

- **역할 분리(중요)**:
  - `RecruitmentScam`: train/test 생성(평가의 기준)
  - `FakeJobPostings`가 all-positive면: `FakeJobPostings_PosOnly`로 취급, **train 증강만**(test 금지)
  - `LinkedInPostings`: `LinkedIn_NegOnly`로 취급, **train 증강만**(test 금지)
- 텍스트 통합 템플릿(학습/평가/서빙 공통 목표):
  - `[TITLE] ...`, `[LOCATION] ...`, `[EMPLOYMENT_TYPE] ...`, `[INDUSTRY] ...`, `[SALARY] ...`
  - `[COMPANY_PROFILE] ...`, `[DESCRIPTION] ...`, `[REQUIREMENTS] ...`, `[BENEFITS] ...`
- 클린업: HTML 제거, URL/이메일/전화번호 마스킹, 소문자, 공백 정리, 최대 길이 제한(기본 6000자).
- salary_range 파싱: `salary_low/high/mid/has_salary` 생성 + TF-IDF가 쓰도록 토큰 형태로도 주입합니다.
- split: **RecruitmentScam에서 stratified split**로 `test.csv`를 만듭니다. test는 반드시 pos/neg가 둘 다 있어야 합니다.
- 출력:
  - `combined_train.csv` / `train.csv`: 학습용(RecruitmentScam train + augment)
  - `test.csv`: 평가용(RecruitmentScam test only)

## 모델 파일 구조

```
models/fraud-baseline/
├── vectorizer.joblib   # TF-IDF 벡터라이저
├── model.joblib        # Logistic Regression 모델
└── metadata.json       # threshold, version 정보
```

## 참고 사항

- 외부 사실 기반 feature(회사 규모, 지도/리뷰 등)는 이 AI 서비스의 범위 밖입니다.
- 확률 보정(calibration: Platt scaling / isotonic 등)은 추후 추가할 수 있으며, 적용 시 `metadata.json`에 기록합니다.
