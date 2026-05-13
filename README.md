# Samak AI — 채용 사기 탐지 서비스

**해외 채용 공고 이미지**를 받아 사기 여부를 자동으로 분석하는 API 서비스입니다.
**Gemini Vision**으로 이미지를 직접 분석하고, **규칙 기반 패턴 매칭**으로 사기 확률과 위험 신호를 반환합니다.

메시지 브로커로 **Google Cloud Pub/Sub**을 사용합니다.

---

## 전체 파이프라인

```
요청 (Google Cloud Pub/Sub)
        │  analysis-request-subscription
        ▼
        ┌──────────────────────┐
        │  Gemini Vision 분석    │
        │  (이미지 직접 입력)      │
        │  → fraud_probability  │
        │  → risk_signals       │
        │  → domains_found      │
        │  → regions_mentioned  │
        └──────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────┐
│  규칙 기반 검증                                      │
│  scam_domains 매칭 / travel_ban_regions 매칭         │
└───────────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────────┐
│  Google Maps 위치 조회 (GOOGLE_MAPS_API_KEY 설정 시)  │
│  companyName → Places API                          │
│    미검색 시 → Geocoding API (regions_mentioned)    │
│  → location { lat, lng, viewport, status }         │
└───────────────────────────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  사기 확률 결정        │
        │  Gemini Vision 100%  │
        │  (scam domain → 1.0) │
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

### 2. Structured LLM Output (JSON mode)

Gemini에게 자유형 텍스트가 아닌 **구조화된 JSON 출력**을 요청합니다.
`fraud_probability`, `risk_signals`, `domains_found`, `regions_mentioned` 필드를 독립적으로 활용합니다.

```json
{
  "fraud_probability": 0.73,
  "risk_signals": ["선불 요구 패턴", "해외 근무 조건"],
  "reasoning": "해당 공고는 선불 비용을 요구하고 있어 전형적인 사기 패턴과 일치합니다...",
  "domains_found": ["aloisstaffing.com"],
  "regions_mentioned": ["myanmar", "myawaddy"]
}
```

### 3. 규칙 기반 검증

Gemini가 추출한 `domains_found`를 알려진 사기 도메인 목록과 매칭합니다. 탐지되면 사기 확률을 즉시 1.0으로 확정합니다.
`regions_mentioned`는 **대한민국 외교부 여행금지 지역 목록**과 매칭해 결과에 포함합니다.

### 4. Gemini 메시지 polish

분석 결과를 바탕으로 생성된 템플릿 메시지를 Gemini가 자연스러운 한국어로 다듬습니다.

### 5. Google Maps 위치 조회

`GOOGLE_MAPS_API_KEY` 설정 시 분석 결과에 `location` 객체가 추가됩니다.

- **회사명 있음** → Places API로 검색. 미검색 시 위험 신호 추가("Google Maps에서 회사명이 검색되지 않습니다."), 위치 불일치 시 별도 신호 추가
- **회사명 없거나 검색 실패** → Geocoding API로 `regions_mentioned` 첫 번째 지역 검색 → viewport(bounds) 반환
- Flutter는 `status: "company"`면 마커 핀, `status: "region"`이면 `fitBounds`로 지역 영역 렌더링
- API 키 미설정 시 `location: null`로 graceful degradation

### 7. 병렬 이미지 처리 (asyncio.gather)

`imageUrls` 필드로 여러 이미지를 한 번에 요청할 수 있습니다.
5장 이상은 `asyncio.gather`로 병렬 처리합니다.

### 8. Graceful Degradation

- Gemini API key 없으면 fraud_probability 0.5로 UNKNOWN 처리
- Gemini 응답 실패 시 보수적 0.5 확률 사용, polish 실패 시 템플릿 메시지 사용

---

## API 레퍼런스

### `GET /healthz`

```json
{
  "status": "ok",
  "checks": {
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
  "travelBanRegionsMatched": [],
  "scamDomainsMatched": [],
  "mlPrediction": {
    "modelVersion": "gemini-rule-v1.0.0",
    "fraudProbability": 0.83,
    "riskScore": 83,
    "riskLevel": "HIGH",
    "thresholdUsed": 0.6242
  },
  "explanation": {
    "riskSignals": ["upfront payment", "no interview required"],
    "note": "Signals detected by Gemini Vision analysis."
  },
  "ui": {
    "riskLevel": "HIGH",
    "trustLabel": "Danger",
    "trustScore": 17
  },
  "analysisSummary": {
    "score": 17,
    "label": "Danger",
    "message": "해당 공고는 선불 비용을 요구하고 있어 사기 패턴과 일치합니다. 지원 전 업체 정보를 반드시 검증하시기 바랍니다."
  },
  "location": {
    "rawText": "OO회사",
    "lat": 37.5665,
    "lng": 126.9780,
    "adminLevel": "대한민국",
    "zoom": 14,
    "status": "company",
    "viewportNe": null,
    "viewportSw": null
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
| `GEMINI_API_KEY` | (없음) | Google Gemini API 키. 없으면 fraud_probability 0.5로 처리 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | 사용할 Gemini 모델 |
| `GOOGLE_MAPS_API_KEY` | (없음) | Google Maps API 키 (Places API + Geocoding API). 없으면 location 조회 비활성화 |
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
├── api_models.py            # Swagger(OpenAPI) 문서용 응답 스키마
├── backend_push_config.py   # 백엔드 Push 설정
├── env.py                   # .env 로딩 유틸 (로컬 개발 편의)
├── integrations/            # 외부 서비스 연동 모듈
├── pubsub/                  # Google Cloud Pub/Sub 연동
│   ├── consumer.py          # Streaming Pull 구독, 메시지 처리
│   ├── producer.py          # 분석 결과 발행
│   ├── pubsub_config.py     # 환경 변수 기반 경로 설정
│   └── schemas.py           # 메시지 스키마 (Pydantic)
├── routes/
│   ├── analyze.py           # /v1/analyze/image 엔드포인트
│   └── wage_warning.py      # 최저임금 경고 엔드포인트
├── ml/
│   ├── message_risk_rules.py# 정규식 기반 위험 신호 탐지
│   ├── risk_patterns.txt    # 위험 패턴 정의 파일
│   ├── risk_regions.py      # 여행금지 지역 탐지
│   ├── risk_regions.txt     # 여행금지 지역 목록
│   └── scam_domains.py      # 알려진 사기 도메인 목록
├── schemas/
│   └── wage.py              # 최저임금 관련 Pydantic 스키마
└── services/
    ├── gemini_service.py    # Gemini Vision 분석, 메시지 polish
    ├── maps_service.py      # Google Maps 위치 조회 (Places API + Geocoding API)
    ├── scoring_service.py   # 확률 → 점수/레벨 변환
    ├── summary_builder.py   # 템플릿 메시지 생성
    ├── backend_push.py      # 백엔드로 분석 결과 POST 전송
    ├── min_wage_store.py    # 국가별 최저임금 데이터 로딩/조회
    └── wage_service.py      # 임금 경고 비즈니스 로직
resources/
└── min_wage_hourly.json     # 국가별 최저 시급 데이터
scripts/
├── batch_test.py            # 배치 테스트 스크립트
├── local_test.py            # 로컬 테스트 스크립트
└── recruiting_examples/     # 테스트용 채용 공고 이미지 샘플
tests/
├── conftest.py
├── test_api.py
├── test_min_wage_store.py
├── test_preprocess_policy.py
└── test_wage_warning_api.py
```
