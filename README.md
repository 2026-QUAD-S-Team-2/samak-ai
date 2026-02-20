# Samak AI

“취업 사기 공고 분석 서비스”의 **이미지 기반 분석 + Gemini 자연어 요약(FastAPI)** MVP 레포지토리입니다.

<br>

## What this repo does

입력은 **무조건 이미지(채팅 캡처/공고 캡처)** 입니다.

파이프라인:
이미지 URL(JSON) → 이미지 다운로드 → OCR(EasyOCR) → ML baseline(TF‑IDF+LR) → 위험도 점수/레벨 산출 → Gemini로 자연어 문장 생성 → JSON 반환

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

## 백엔드 Push 설정(하드코딩)

분석 결과를 백엔드로 자동 전송(push)하려면 `app/backend_push_config.py`를 수정하세요.

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

### Swagger / OpenAPI

- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`
- ReDoc: `/redoc`

서버 없이 스펙 파일만 공유하려면:

```bash
python3 scripts/export_openapi.py --out openapi.json
```

### `POST /v1/analyze/image`

Request (JSON only):
	```json
	{
	  "debug": false,
	  "imageUrls": ["https://example.com/sample.png"],
	  "countryCode": "UA",
	  "salaryText": "3000000 KRW"
	}
	```

Response:
```json
{
  "analysisId": "uuid",
  "fraudProbability": 0.87,
  "riskScore": 87,
  "riskLevel": "HIGH",
  "riskSignals": ["bitcoin", "upfront payment"],
  "travelBanRegionsMatched": ["우크라이나"],
  "message": "..."
}
```

Response (요청이 여러 장인 경우):
```json
{
  "results": [
    {
      "analysisId": "uuid",
      "fraudProbability": 0.87,
      "riskScore": 87,
      "riskLevel": "HIGH",
      "riskSignals": ["bitcoin", "upfront payment"],
      "travelBanRegionsMatched": ["우크라이나"],
      "message": "..."
    }
  ]
}
```

호환성:
- 기존 단일 입력인 `imageUrl`도 계속 동작합니다(서버에서 `imageUrls`로 정규화).

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
	  -d '{"imageUrls":["https://example.com/sample.png"],"countryCode":"UA","salaryText":"3000000 KRW","debug":false}'
	```

	<br>

### `POST /v1/wage-warning`

설명:
- `salaryText`가 없으면, 내부에 저장된 해당 국가 최저 시급이 있으면 안내 문구를 반환하고 없으면 null을 반환합니다.
- `salaryText`는 **시급(hourly)** 만 지원합니다. (예: `KRW 12000/h`, `USD 25/hour`)
- 고임금 경고는 `시급 >= (해당 국가 최저 시급 * 4)` 일 때 트리거됩니다.

Request:
```json
{ "countryCode": "UA", "salaryText": "UAH 150/h" }
```

Response:
```json
{
  "code": "200",
  "message": "API 요청 성공",
  "data": { "warningMessage": "..." }
}
```

```bash
curl -sS http://localhost:8000/v1/wage-warning \
  -H 'content-type: application/json' \
  -d '{"countryCode":"UA","salaryText":"UAH 150/h"}'

curl -sS http://localhost:8000/v1/wage-warning \
  -H 'content-type: application/json' \
  -d '{"countryCode":"UA"}'
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
