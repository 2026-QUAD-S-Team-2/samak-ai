# Samak AI — 채용 사기 탐지 서비스

채용 공고 이미지를 받아 **사기 여부를 자동으로 분석**하는 API 서비스입니다.
이미지에서 텍스트를 추출(OCR)하고, 머신러닝 모델과 Gemini AI를 함께 사용해 사기 확률과 위험 신호를 반환합니다.

---

## 전체 파이프라인

```
요청 (이미지 URL 또는 파일 업로드)
        │
        ▼
┌───────────────────────────────────┐
│  OCR (EasyOCR)                    │
│  이미지 → 텍스트 추출              │
│  30자 미만이면 UNKNOWN 즉시 반환   │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  ML 1차 판단                       │
│  TF-IDF + Logistic Regression     │
│  사기 확률 (0.0 ~ 1.0) 계산        │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Confidence Gating  ◄ LLM as Judge│
│  확률 < 20%  → ML 결과 확정        │
│  확률 > 80%  → ML 결과 확정        │
│  20% ~ 80%  → Gemini 심층 분석    │
└───────────────────────────────────┘
        │ (경계 구간만)
        ▼
┌───────────────────────────────────┐
│  Gemini 심층 분석                  │
│  OCR 텍스트 + ML 판단 결과 전달    │
│  → 구조화 JSON 응답               │
│     fraud_probability             │
│     risk_signals                  │
│     reasoning (판단 근거)         │
│  최종 확률 = ML×0.4 + Gemini×0.6  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  메시지 생성                       │
│  Gemini 분석 사용?                 │
│    YES → reasoning을 메시지로 사용 │
│    NO  → 템플릿 + Gemini polish    │
└───────────────────────────────────┘
        │
        ▼
  JSON 응답 반환
```

---

## 핵심 기술 포인트

### 1. LLM as a Judge (Confidence Gating)

ML 모델이 사기 여부를 판단할 때 **확신이 낮은 경계 구간(20%~80%)** 의 케이스만 Gemini에게 재판단을 요청합니다.
확실히 안전하거나 확실히 사기인 케이스는 ML만으로 처리해 속도와 비용을 절약합니다.

```
확률 < 20%  → "거의 정상"  → ML 결과 확정 (Gemini 호출 없음)
20% ~ 80%   → "경계 구간"  → Gemini가 재분석 후 ML과 블렌딩
확률 > 80%  → "거의 사기"  → ML 결과 확정 (Gemini 호출 없음)
```

### 2. Hybrid ML + LLM Ensemble

경계 구간에서 Gemini가 판단을 내리면, ML 판단과 **가중 평균(40:60)** 으로 최종 확률을 계산합니다.
ML은 학습된 통계적 패턴에 강하고, Gemini는 맥락과 의미 파악에 강합니다.

```
최종 확률 = ML확률 × 0.4 + Gemini확률 × 0.6
```

### 3. Structured LLM Output (JSON mode)

Gemini에게 자유형 텍스트가 아닌 **구조화된 JSON 출력**을 요청합니다.
`fraud_probability`, `risk_signals`, `reasoning` 각 필드를 독립적으로 활용합니다.

```json
{
  "fraud_probability": 0.73,
  "risk_signals": ["선불 요구 패턴", "해외 근무 조건"],
  "reasoning": "해당 공고는 선불 비용을 요구하고 있어 전형적인 사기 패턴과 일치합니다..."
}
```

### 4. OCR 파이프라인

EasyOCR로 한국어 + 영어 이미지에서 텍스트를 추출합니다.
이미지 URL과 multipart 파일 업로드 모두 지원하며, 글로벌 캐싱으로 초기화 비용을 최소화합니다.

### 5. 위험 지역 자동 탐지

텍스트에서 **대한민국 외교부가 지정한 여행금지 지역**이 언급되면 자동으로 탐지해 결과에 포함합니다.

### 6. 병렬 이미지 처리 (asyncio.gather)

`imageUrls` 필드로 여러 이미지를 한 번에 요청할 수 있습니다.
**5장 이상**일 경우 `asyncio.gather`로 병렬 처리해 총 응답 시간을 단축합니다.

```
2~4장 : 순차 처리
5장 이상 : asyncio.gather 병렬 처리 (이미지 다운로드 + Gemini 호출 I/O 구간 중첩)
```

### 7. Graceful Degradation

- Gemini API key 없어도 서비스 정상 동작 (ML + 템플릿 메시지로 fallback)
- ML 모델 파일 없어도 200 응답 유지 (보수적 0% 확률 반환)
- Gemini 응답 검증 실패 시 자동으로 템플릿 메시지 사용

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

**복수 이미지 (JSON) — 5장 이상 시 병렬 처리**
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
| `GEMINI_API_KEY` | (없음) | Google Gemini API 키. 없으면 ML + 템플릿으로만 동작 |
| `GEMINI_MODEL` | `gemini-2.5-flash` | 사용할 Gemini 모델 |
| `LOG_LEVEL` | `INFO` | Python 로깅 레벨 |

`.env.example`을 복사해 `.env`로 사용하세요.

---

## 실행 방법

### Docker 실행

```bash
docker build -t samak-ai .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  -e MODEL_DIR=/app/models/fraud-baseline \
  samak-ai
```

### 동작 확인

```bash
# 헬스 체크
curl http://localhost:8000/healthz

# 분석 요청
curl -sS http://localhost:8000/v1/analyze/image \
  -H 'content-type: application/json' \
  -d '{"imageUrl":"https://example.com/sample.png","meta":{"countryCode":"KR"}}'

# 로컬 파일 테스트 (서버 없이)
python scripts/local_test.py --file ./scripts/sample.png
```