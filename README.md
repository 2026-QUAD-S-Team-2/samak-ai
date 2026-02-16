# Samak AI

“취업 사기 공고 분석 서비스”의 **이미지 기반 분석 + Gemini 자연어 요약(FastAPI)** MVP 레포지토리입니다.

<br>

## What this repo does

입력은 **무조건 이미지(채팅 캡처/공고 캡처)** 입니다.

파이프라인:
이미지(URL 또는 multipart) → OCR(EasyOCR) → ML baseline(TF‑IDF+LR) → 위험도 점수/레벨 산출 → Gemini로 자연어 문장 생성 → JSON 반환

<br>


## 레포지토리 파이프라인
- 데이터 넣기: `*.csv`
- 전처리/통합 + split: `preprocess.py` → `combined.csv` + `train.csv` + `test.csv`
  - 텍스트 통합/클린업/급여 파싱/LinkedIn 정상 샘플링/stratified split 포함
- 학습: `training/train_baseline.py train` → `{vectorizer.joblib, model.joblib, metadata.json}`
- 배포용 export: `training/train_baseline.py export` 또는 `training/train_baseline.py all --export-dir ...` → `models/fraud-baseline/`로 원자적 교체
- 추론: FastAPI가 `models/fraud-baseline/`에서 아티팩트 로딩 후 `/v1/analyze/image`로 결과 반환

<br>

## 환경변수

환경변수 설정은 `.env.example` 참고.
- `MODEL_DIR` (기본 `models/fraud-baseline`)
- `GEMINI_API_KEY` (있으면 요약 문장을 Gemini가 다듬음, 실패 시 템플릿 문장 그대로 반환)
- `GEMINI_MODEL` (기본 `gemini-2.5-flash`)

<br>

## Quick local test (no backend)

FastAPI 없이 내부 함수로 전체 플로우를 바로 확인할 수 있습니다:

```bash
python scripts/local_test.py --file scripts/recruitment2.png --company-name "OO회사"
```

<br>

## API

### `GET /healthz`

Response:
```json
{ "status": "ok" }
```

### `POST /v1/analyze/image`

Request (A: JSON / imageUrl):
```json
{
  "imageUrl": "https://...",
  "meta": { "companyName": "삼성중공업", "countryCode": "KR", "sourceUrl": "https://..." }
}
```

Response:
```json
{
  "analysisId": "uuid",
  "type": "JOB_POST|MESSAGE",
  "ocr": { "textPreview": "…", "textLength": 1234, "languageGuess": "ko|en", "confidenceAvg": 0.81 },
  "mlPrediction": {
    "modelVersion": "fraud-baseline-v1.0.0",
    "fraudProbability": 0.42,
    "riskScore": 42,
    "riskLevel": "LOW|MEDIUM|HIGH",
    "thresholdUsed": 0.346
  },
  "analysisSummary": { "score": 58, "label": "Warning", "message": "화면 표시용 문장(3문장)" }
}
```

<br>

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
curl -sS http://localhost:8000/v1/analyze/image \
  -H 'content-type: application/json' \
  -d '{"imageUrl":"https://example.com/sample.png","meta":{"companyName":"OO","countryCode":"KR"}}'
```

<br>

## Model files

- 기본 경로: `models/fraud-baseline/`
- 환경변수로 변경: `MODEL_DIR=/path/to/fraud-baseline`
- 파일 형식:
  - `vectorizer.joblib`
  - `model.joblib`
  - (optional) `metadata.json`

모델 파일이 없거나 로딩 실패 시에도 API는 200을 유지하되, 점수는 보수적으로(0.0) 처리되고 템플릿 메시지가 반환됩니다.

<br>

## Training (preprocess / train / export)

- 데이터 위치: `training/data/` (raw/processed, gitignored 되어있음)
- 전처리(평가용 test는 RecruitmentScam만 사용): `python training/preprocess.py`
- 학습: `python training/train_baseline.py train --train-path training/data/processed/train.csv --out-dir training/runs/tfidf_lr`
- 평가(+threshold 저장): `python training/evaluate_baseline.py --model-dir training/runs/tfidf_lr --train-path training/data/processed/train.csv --test-path training/data/processed/test.csv --tune`
- export(원자적 교체): `python training/train_baseline.py export --model-dir training/runs/tfidf_lr --export-dir models/fraud-baseline`
- 서빙: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

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
