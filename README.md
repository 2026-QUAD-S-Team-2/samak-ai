# Samak AI — 채용 사기 탐지 서비스

해외 채용 공고 이미지를 받아 사기 여부를 분석하는 AI 서비스입니다.<br>
Google Cloud Pub/Sub으로 요청을 수신하고 결과를 발행합니다.

---

## 파이프라인

```
Pub/Sub 요청
    ↓
Google Maps 회사 검증 (companyName → Places API)
    ↓
Gemini Vision 분석 (Maps 결과 컨텍스트 포함)
    ↓
규칙 기반 검증 (scam_domains / travel_ban_regions)
    ↓
메시지 생성
    ↓
Pub/Sub 결과 발행
```

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `GEMINI_MODEL` | 사용할 모델 (기본값: `gemini-2.5-flash`) |
| `GOOGLE_MAPS_API_KEY` | Google Maps API 키 (없으면 위치 조회 비활성화) |
| `GCP_PROJECT_ID` | Google Cloud 프로젝트 ID |
| `PUBSUB_REQUEST_SUBSCRIPTION` | 분석 요청 구독 이름 (기본값: `analysis-request-subscription`) |
| `PUBSUB_RESULT_TOPIC` | 분석 결과 토픽 이름 (기본값: `analysis-result-topic`) |
| `PUBSUB_DLQ_SUBSCRIPTION` | DLQ 구독 이름 (기본값: `analysis-request-dead-letter-subscription`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | 서비스 계정 키 파일 경로 (GCP 환경에서는 불필요) |

```bash
cp .env.example .env
```

---

## 실행

```bash
docker compose up -d --build
```
