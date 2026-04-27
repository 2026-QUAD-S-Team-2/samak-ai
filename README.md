# Samak AI — 채용 사기 탐지 서비스

채용 공고 이미지를 받아 **사기 여부를 자동으로 분석**하는 API 서비스입니다.
이미지에서 텍스트를 추출(OCR)하고, 머신러닝 모델과 Gemini AI를 함께 사용해 사기 확률과 위험 신호를 반환합니다.

---

## 전체 파이프라인

```
요청 (이미지 URL 또는 파일 업로드)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  병렬 처리 (asyncio.gather)                           │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │  OCR (EasyOCR)      │  │  Gemini 이미지 분석     │  │
│  │  이미지 → 텍스트       │  │  (멀티모달 입력)         │  │
│  │  추출                │  │  사기 확률 산출          │  │
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
             JSON 응답 반환
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

EasyOCR로 한국어 + 영어 이미지에서 텍스트를 추출합니다.
이미지 URL과 multipart 파일 업로드 모두 지원하며, 글로벌 캐싱으로 초기화 비용을 최소화합니다.

### 7. 위험 지역 자동 탐지

텍스트에서 **대한민국 외교부가 지정한 여행금지 지역**이 언급되면 자동으로 탐지해 결과에 포함합니다.

### 8. 병렬 이미지 처리 (asyncio.gather)

`imageUrls` 필드로 여러 이미지를 한 번에 요청할 수 있습니다.
장수와 무관하게 `asyncio.gather`로 병렬 처리합니다.

### 9. Graceful Degradation

- Gemini API key 없어도 서비스 정상 동작 (ML 단독 결과로 fallback)
- ML 모델 파일 없어도 200 응답 유지 (보수적 0% 확률 반환)
- Gemini 응답 실패 시 ML 단독 결과 사용, polish 실패 시 템플릿 메시지 사용

---

## API 레퍼런스

### `GET /healthz`

```json
{ "status": "ok" }
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
    "countryCode": "KR"
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
  "meta": { "countryCode": "KR" }
}
```

**파일 업로드 (multipart/form-data)**
```
multipartFile : [이미지 파일]
companyName   : OO회사
countryCode   : KR
```

**응답 예시**
```json
{
  "analysisId": "uuid",
  "type": "JOB_POST",
  "ocr": {
    "textPreview": "당신의 꿈을 실현하세요...",
    "textLength": 847,
    "languageGuess": "ko",
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
    "riskSignals": ["선불 요구", "연락처 없음"],
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
| `RABBITMQ_USERNAME` | `guest` | RabbitMQ 접속 계정 |
| `RABBITMQ_PASSWORD` | `guest` | RabbitMQ 접속 비밀번호 |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ 연결 URL (Docker Compose 사용 시 자동 주입) |
| `RABBITMQ_PREFETCH` | `1` | 동시에 처리할 최대 메시지 수 |
| `RABBITMQ_RECONNECT_DELAY` | `5` | 연결 끊김 후 재연결 대기 시간(초) |

`.env.example`을 복사해 `.env`로 사용하세요.

---

## 실행 방법

### 구성 개요

동일한 VM 인스턴스에서 **RabbitMQ**와 **AI 서버** 두 컨테이너를 함께 띄웁니다.

```
VM 인스턴스
├── rabbitmq 컨테이너  (포트 5672, 15672)
└── samak-ai 컨테이너  (포트 8000)
    └── 시작 시 rabbitmq:5672 로 자동 연결
```

Spring Boot 백엔드는 같은 VM에서 `localhost:5672` 로 RabbitMQ에 접속합니다.

---

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

`.env`에 아래 항목을 채워 넣습니다:

```bash
GEMINI_API_KEY=your_gemini_api_key

RABBITMQ_USERNAME=your_username
RABBITMQ_PASSWORD=your_password
```

---

### 2. 서버 실행

```bash
docker compose up -d --build
```

내부 실행 순서:
1. `rabbitmq` 컨테이너 시작 → `rabbitmq-diagnostics ping` 헬스체크 통과 대기 (최대 50초)
2. `samak-ai` 이미지 빌드 후 컨테이너 시작 → consumer가 `rabbitmq:5672`로 자동 연결

---

### 3. 동작 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# AI 서버 헬스 체크
curl http://localhost:8000/healthz

# AI 서버 로그 확인 (RabbitMQ 연결 메시지 확인)
docker compose logs samak-ai

# RabbitMQ Management UI (Exchange/Queue 선언 확인)
# http://<VM_EXTERNAL_IP>:15672
```

---

### 4. 서버 중지

```bash
docker compose down
```

RabbitMQ 데이터(메시지 큐)까지 초기화하려면:

```bash
docker compose down -v
```