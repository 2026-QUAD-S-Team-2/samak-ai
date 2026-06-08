# Samak AI — 채용 사기 탐지 서비스

해외 채용 공고 이미지를 받아 사기 여부를 분석하는 AI 서비스입니다.<br>
Google Cloud Pub/Sub으로 요청을 수신하고 결과를 발행합니다.

---
## 전체 서비스 아키텍처
<img width="800" height="629" alt="아키텍처" src="https://github.com/user-attachments/assets/6b5e2745-2f38-4628-83fe-477ea83681c8" />



## AI파트 파이프라인

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

## 실행

```bash
docker compose up -d --build
```
