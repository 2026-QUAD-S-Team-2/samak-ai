# Samak AI (Inference Service)

“취업 사기 공고 분석 서비스”의 **ML baseline 추론 서비스(FastAPI)** 레포지토리입니다.

## What this repo does

백엔드가 전달한 공고 텍스트(또는 URL에서 추출된 텍스트)를 입력으로 받아
  **사기 확률 / 위험 점수 / 위험 레벨**등을 자연어로 변환하여 반환합니다.


## 레포지토리 파이프라인
- 데이터 넣기: `*.csv`
- 전처리/통합 + split: `preprocess.py` → `combined.csv` + `train.csv` + `test.csv`
  - 텍스트 통합/클린업/급여 파싱/LinkedIn 정상 샘플링/stratified split 포함
- 학습: `training/train_baseline.py train` → `{vectorizer.joblib, model.joblib, metadata.json}`
- 배포용 export: `training/train_baseline.py export` 또는 `training/train_baseline.py all --export-dir ...` → `models/fraud-baseline/`로 원자적 교체
- 추론: FastAPI가 `models/fraud-baseline/`에서 아티팩트 로딩 후 `/v1/infer`로 확률/점수/레벨 반환

## API

### `GET /healthz`

Response:
```json
{ "status": "ok" }
```

### `POST /v1/infer`

Request:
```json
{
  "analysisId": "string",
  "text": "string",
  "meta": {
    "sourceType": "TEXT|URL|CHAT",
    "language": "en|ko|etc",
    "offeredCompensation": {
      "amount": 52000,
      "currency": "GBP",
      "period": "YEAR|MONTH|HOUR"
    }
  }
}
```

Response:
```json
{
  "analysisId": "string",
  "modelVersion": "fraud-baseline-v1.0.0",
  "fraudProbability": 0.42,
  "riskScore": 42,
  "riskLevel": "LOW",
  "notes": "string"
}
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000 # 추론 서비스 켜기
```

## Docker run

```bash
docker build -t samak-ai .
docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models/fraud-baseline samak-ai
```

## cURL example

```bash
curl -sS http://localhost:8000/v1/infer \
  -H 'content-type: application/json' \
  -d '{"analysisId":"a-1","text":"We will hire you immediately. Pay upfront..."}'
```

## Model files

- 기본 경로: `models/fraud-baseline/`
- 환경변수로 변경: `MODEL_DIR=/path/to/fraud-baseline`
- 파일 형식:
  - `vectorizer.joblib`
  - `model.joblib`
  - (optional) `metadata.json`

모델 파일이 없거나 로딩 실패 시 `/v1/infer`는 **503**을 반환합니다.

## Training (preprocess / train / export)

- 데이터 위치: `training/data/` (raw/processed, gitignored 되어있음)
- 전처리(평가용 test는 RecruitmentScam만 사용): `python training/preprocess.py`
- 학습: `python training/train_baseline.py train --train-path training/data/processed/train.csv --out-dir training/runs/tfidf_lr`
- 평가(+threshold 저장): `python training/evaluate_baseline.py --model-dir training/runs/tfidf_lr --train-path training/data/processed/train.csv --test-path training/data/processed/test.csv --tune`
- export(원자적 교체): `python training/train_baseline.py export --model-dir training/runs/tfidf_lr --export-dir models/fraud-baseline`
- 서빙: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Quick local test (no backend)

`.txt` 파일에 공고 텍스트를 넣고 바로 추론해볼 수 있습니다:

```bash
python3 scripts/run_infer_txt.py --file posting.txt
```

### `POST /v1/external/maps/geocode`

Request:
```json
{ "address": "London, UK" }
```

### `POST /v1/external/gemini/generate`

Request:
```json
{ "prompt": "Summarize fraud risks in this job posting: ..." }
```

환경변수 설정은 `.env.example` 참고.


## 실제 공고 테스트
1. 먼저 모델을 models/fraud-baseline/에 준비(학습+export): 
`python3 training/train_baseline.py all --export-dir models/fraud-baseline`
2. 공고 텍스트를 posting.txt에 넣고 실행:
`python3 scripts/run_infer_txt.py --file scripts/posting.txt`
