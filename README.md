# Samak AI — 채용 사기 탐지 서비스

**해외 채용 공고 이미지**를 받아 사기 여부를 자동으로 분석하는 API 서비스입니다.
이미지에서 텍스트를 추출(OCR)하고, 머신러닝 모델과 Gemini AI를 함께 사용해 사기 확률과 위험 신호를 반환합니다.

메시지 브로커로 **Google Cloud Pub/Sub**을 사용합니다.

---

## 전체 파이프라인

```
요청 (Google Cloud Pub/Sub)
        │  analysis-request-subscription
        ▼
┌─────────────────────────────────────────────────────┐
│  병렬 처리 (asyncio.gather)                           │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │  OCR (EasyOCR)      │  │  Gemini 이미지 분석     │  │
│  │  이미지 → 텍스트 추출   │  │  (멀티모달 입력)         │  │
│  └─────────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
        │                           │
        ▼                           │
┌───────────────────────┐           │
│  ML 판단               │           │
│  TF-IDF +             │           │
│  Logistic Regression  │           │
│  사기 확률 계산          │           │
└───────────────────────┘           │
        │                           │
        └──────────┬────────────────┘
                   ▼
        ┌─────────────────────┐
        │  앙상블               │
        │  ML × 0.4           │
        │  + Gemini × 0.6     │
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  메시지 생성           │
        │  템플릿               │
        │  + Gemini polish    │
        └─────────────────────┘
                   │
                   ▼
        결과 발행 (Google Cloud Pub/Sub)
           analysis-result-topic
```

---

## 핵심 기술 포인트

### 1. Gemini Vision — 이미지 직접 분석

`gemini-2.5-flash` 모델에 이미지를 직접 입력해 사기 확률을 산출합니다. 모든 요청에서 실행됩니다.

탐지 기준:
- 비정상적으로 높은 급여 약속
- 선입금/보증금/장비 구매 요구
- 개인정보 즉시 요구 (계좌번호, 주민번호 등)
- 회사 정보 불명확 (이름/주소/연락처 없음)
- 문법 오류, 번역 투 문체
- 여행금지 국가·지역 파견 근무 제안
- 카카오톡/텔레그램 등 비공식 채널 연락 요구
- 레이아웃이 조잡하거나 로고가 위조처럼 보임

### 2. Hybrid ML + Gemini Ensemble

OCR → ML과 Gemini 이미지 분석이 **병렬로** 각각 사기 확률을 산출하고, **가중 평균(40:60)** 으로 최종 확률을 계산합니다.

```
최종 확률 = ML확률 × 0.4 + Gemini확률 × 0.6
```

### 3. OCR + Gemini 병렬 실행 (asyncio.gather)

EasyOCR과 Gemini API 호출을 `asyncio.to_thread` + `asyncio.gather`로 동시에 실행합니다.

### 4. Structured LLM Output (JSON mode)

Gemini에게 자유형 텍스트가 아닌 **구조화된 JSON 출력**을 요청합니다.
`fraud_probability`, `risk_signals`, `reasoning` 각 필드를 독립적으로 활용합니다.

```json
{
  "fraud_probability": 0.73,
  "risk_signals": ["선불 요구 패턴", "해외 근무 조건"],
  "reasoning": "해당 공고는 선불 비용을 요구하고 있어 전형적인 사기 패턴과 일치합니다..."
}
```

### 5. Gemini 메시지 polish

분석 결과를 바탕으로 생성된 템플릿 메시지를 Gemini가 자연스러운 한국어로 다듬습니다.

### 6. OCR 파이프라인

EasyOCR로 영어 이미지에서 텍스트를 추출합니다.
이미지 URL과 multipart 파일 업로드 모두 지원하며, 글로벌 캐싱으로 초기화 비용을 최소화합니다.

### 7. 위험 지역 자동 탐지

텍스트에서 **대한민국 외교부가 지정한 여행금지 지역**이 언급되면 자동으로 탐지해 결과에 포함합니다.

### 8. 병렬 이미지 처리 (asyncio.gather)

`imageUrls` 필드로 여러 이미지를 한 번에 요청할 수 있습니다.
5장 이상은 `asyncio.gather`로 병렬 처리합니다.

### 9. Graceful Degradation

- Gemini API key 없어도 서비스 정상 동작 (ML 단독 결과로 fallback)
- ML 모델 파일 없어도 200 응답 유지 (보수적 0% 확률 반환)
- Gemini 응답 실패 시 ML 단독 결과 사용, polish 실패 시 템플릿 메시지 사용

---

## API 레퍼런스

### `GET /healthz`

```json
{
  "status": "ok",
  "checks": {
    "model": "ok",
    "minWageData": "ok",
    "pubsub": "configured"
  }
}
```

---

### `POST /v1/analyze/image`

이미지를 분석해 사기 여부와 위험 신호를 반환합니다.

**단일 이미지 (JSON)**
```json
{
  "imageUrl": "https://example.com/job_posting.png",
  "meta": {
    "companyName": "OO회사",
    "countryCode": "SG"
  }
}
```

**복수 이미지 (JSON) — 병렬 처리**
```json
{
  "imageUrls": [
    "https://example.com/page1.png",
    "https://example.com/page2.png"
  ],
  "meta": { "countryCode": "SG" }
}
```

**파일 업로드 (multipart/form-data)**
```
multipartFile : [이미지 파일]
companyName   : OO회사
countryCode   : SG
```

**응답 예시**
```json
{
  "analysisId": "uuid",
  "type": "JOB_POST",
  "ocr": {
    "textPreview": "Earn $5,000/week working from home...",
    "textLength": 847,
    "languageGuess": "en",
    "confidenceAvg": 0.91
  },
  "mlPrediction": {
    "modelVersion": "fraud-baseline-v1.0.0",
    "fraudProbability": 0.63,
    "riskScore": 63,
    "riskLevel": "MEDIUM",
    "thresholdUsed": 0.346
  },
  "explanation": {
    "riskSignals": ["upfront payment", "no interview required"],
    "note": "Signals are matched against predefined scam-pattern rules."
  },
  "ui": {
    "riskLevel": "MEDIUM",
    "trustLabel": "Warning",
    "trustScore": 37
  },
  "analysisSummary": {
    "score": 37,
    "label": "Warning",
    "message": "해당 공고는 선불 비용을 요구하고 있어 사기 패턴과 일치합니다. ML 모델과 AI 분석 결과 63%의 사기 확률이 확인되었습니다. 지원 전 업체 정보를 반드시 검증하시기 바랍니다."
  }
}
```

**복수 이미지 응답**
```json
{
  "results": [ { ... }, { ... } ]
}
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_DIR` | `models/fraud-baseline` | ML 모델 파일 경로 |
| `GEMINI_API_KEY` | (없음) | Google Gemini API 키. 없으면 ML 단독으로 동작 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | 사용할 Gemini 모델 |
| `LOG_LEVEL` | `INFO` | Python 로깅 레벨 |
| `GCP_PROJECT_ID` | (없음) | Google Cloud 프로젝트 ID |
| `PUBSUB_REQUEST_SUBSCRIPTION` | `analysis-request-subscription` | 분석 요청을 수신할 Pub/Sub 구독 이름 |
| `PUBSUB_RESULT_TOPIC` | `analysis-result-topic` | 분석 결과를 발행할 Pub/Sub 토픽 이름 |
| `PUBSUB_DLQ_SUBSCRIPTION` | `analysis-request-dead-letter-subscription` | 처리 실패 메시지를 수신할 DLQ 구독 이름 |
| `PUBSUB_RECONNECT_DELAY` | `5` | 연결 끊김 후 재연결 대기 시간(초) |
| `GOOGLE_APPLICATION_CREDENTIALS` | (없음) | 서비스 계정 키 파일 경로 (GCP 환경에서는 ADC 자동 적용) |

`.env.example`을 복사해 `.env`로 사용하세요.

```bash
cp .env.example .env
```

---

## Pub/Sub 구성

GCP 콘솔에 아래 리소스가 사전 설정되어 있어야 합니다.

| 리소스 | 이름 | 설명 |
|--------|------|------|
| Topic | `analysis-request-topic` | 백엔드가 분석 요청을 발행하는 토픽 |
| Subscription | `analysis-request-subscription` | AI 서버가 요청을 수신하는 구독 |
| Topic | `analysis-result-topic` | AI 서버가 분석 결과를 발행하는 토픽 |
| Subscription | `analysis-result-subscription` | 백엔드가 결과를 수신하는 구독 |
| Topic | `analysis-request-dead-letter-topic` | 처리 실패 메시지 보관 토픽 |
| Subscription | `analysis-request-dead-letter-subscription` | DLQ 모니터링 구독 |

> `analysis-request-subscription`의 **Dead letter policy**가 `analysis-request-dead-letter-topic`으로 설정되어 있어야 합니다.

---

## 실행 방법

### 구성 개요

AI 서버 단일 컨테이너를 띄우고, 메시지 브로커는 **Google Cloud Pub/Sub**에 위임합니다.

```
VM 인스턴스
└── samak-ai 컨테이너  (포트 8000)
    └── 시작 시 Google Cloud Pub/Sub 구독 자동 연결
```

Spring Boot 백엔드는 동일한 GCP 프로젝트의 Pub/Sub 토픽을 통해 통신합니다.

---

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

`.env`에 아래 항목을 채워 넣습니다:

```bash
GCP_PROJECT_ID=your-gcp-project-id

GEMINI_API_KEY=your_gemini_api_key

# 로컬 개발 시 서비스 계정 키 파일 경로 (GCP 환경에서는 불필요)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

### 2. 서버 실행

```bash
docker compose up -d --build
```

---

### 3. 동작 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# AI 서버 헬스 체크
curl http://localhost:8000/healthz

# AI 서버 로그 확인 (Pub/Sub 구독 연결 메시지 확인)
docker compose logs samak-ai
```

정상 동작 시 아래 로그가 출력됩니다:

```
Pub/Sub 구독 시작: projects/.../subscriptions/analysis-request-subscription (DLQ: ...)
```

---

### 4. 서버 중지

```bash
docker compose down
```

---

## 프로젝트 구조

```
app/
├── main.py                  # FastAPI 진입점, lifespan 관리
├── pubsub/                  # Google Cloud Pub/Sub 연동
│   ├── consumer.py          # Streaming Pull 구독, 메시지 처리
│   ├── producer.py          # 분석 결과 발행
│   ├── pubsub_config.py     # 환경 변수 기반 경로 설정
│   └── schemas.py           # 메시지 스키마 (Pydantic)
├── routes/
│   ├── analyze.py           # /v1/analyze/image 엔드포인트
│   └── wage_warning.py      # 최저임금 경고 엔드포인트
├── ml/
│   ├── ml_baseline.py       # TF-IDF + LR 모델 추론
│   ├── message_risk_rules.py# 정규식 기반 위험 신호 탐지
│   ├── risk_patterns.txt    # 위험 패턴 정의 파일
│   └── risk_regions.py      # 여행금지 지역 탐지
└── services/
    ├── gemini_service.py    # Gemini Vision 분석, 메시지 polish
    ├── ocr_service.py       # EasyOCR 래퍼
    ├── scoring_service.py   # 확률 → 점수/레벨 변환
    └── summary_builder.py   # 템플릿 메시지 생성

models/
└── fraud-baseline/
    ├── model.joblib         # 학습된 Logistic Regression
    ├── vectorizer.joblib    # TF-IDF 벡터라이저
    └── metadata.json        # 모델 버전, threshold, 학습 정보

training/
├── train_baseline.py        # 모델 학습 스크립트
├── evaluate_baseline.py     # 성능 평가 스크립트 (PR-AUC, F1, Recall)
└── preprocess.py            # 데이터 전처리 스크립트
```
