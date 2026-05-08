from __future__ import annotations

"""
로컬에서 FastAPI 없이 전체 플로우를 테스트하는 스크립트.

사용 예:
  python3 scripts/local_test.py --file ./scripts/sample.png --company-name "OO Company"
  python3 scripts/local_test.py --image-url "https://..." --company-name "OO Company"
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv_once
from app.ml.message_risk_rules import extract_risk_signals
from app.ml.risk_regions import find_risk_regions
from app.ml.scam_domains import find_scam_domains
from app.routes.analyze import FRAUD_THRESHOLD, MODEL_VERSION
from app.services.gemini_service import analyze_image_with_gemini_vision, polish_with_gemini
from app.services.ocr_service import ocr_from_bytes, ocr_from_url
from app.services.scoring_service import score_prediction
from app.services.summary_builder import build_template_message


def main() -> int:
    load_dotenv_once()
    parser = argparse.ArgumentParser()
    parser.add_argument("file_pos", nargs="?", default=None, help="(옵션) 이미지 파일 경로")
    parser.add_argument("--image-url", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--company-name", default=None)
    parser.add_argument("--type", default="JOB_POST", help="JOB_POST|MESSAGE")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    file_path = args.file or args.file_pos
    if not args.image_url and not file_path:
        raise SystemExit("--image-url 또는 --file 중 하나는 필수입니다.")

    if file_path:
        img_bytes = Path(file_path).read_bytes()
        ocr = ocr_from_bytes(img_bytes)
    else:
        img_bytes = None
        ocr = ocr_from_url(args.image_url)

    print("=== 1) OCR ===")
    print("- languageGuess:", ocr.language_guess)
    print("- confidenceAvg:", ocr.confidence_avg)
    print("- textLength:", ocr.text_length)
    print("- textPreview:", ocr.text_preview)
    if ocr.error:
        print("- error:", ocr.error)

    if ocr.text_length < 30:
        print("\n=== 2) 규칙 기반 신호 ===")
        print("⚠️ OCR 텍스트가 너무 짧아 신호 추출을 건너뜁니다.")
        print("\n=== 3) Message ===")
        msg = "텍스트 추출에 실패했습니다. 더 선명한 이미지로 다시 시도해 주세요."
        print(msg)
        out = {
            "analysisId": "local",
            "type": args.type,
            "ocr": {
                "textPreview": ocr.text_preview,
                "textLength": ocr.text_length,
                "languageGuess": ocr.language_guess,
                "confidenceAvg": ocr.confidence_avg,
            },
            "mlPrediction": {
                "modelVersion": MODEL_VERSION,
                "fraudProbability": None,
                "riskScore": None,
                "riskLevel": None,
                "thresholdUsed": None,
            },
            "ui": {"riskLevel": "UNKNOWN", "trustLabel": None, "trustScore": None},
            "analysisSummary": {"score": None, "label": None, "message": msg},
        }
        print("\n=== 4) Final JSON ===")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # 규칙 기반 신호 추출
    risk_signals = extract_risk_signals(ocr.text, top_k=3)
    risk_regions = find_risk_regions(ocr.text, top_k=5)
    scam_domains = find_scam_domains(ocr.text)

    print("\n=== 2) 규칙 기반 신호 ===")
    print("- riskSignals:", risk_signals)
    print("- travelBanRegionsMatched:", risk_regions)
    print("- scamDomainsMatched:", scam_domains)

    # Gemini Vision
    fraud_prob: float = 0.5
    vision_signals: list[str] = []
    if img_bytes:
        vision = analyze_image_with_gemini_vision(img_bytes)
        print("\n=== 3) Gemini Vision ===")
        print("- used_gemini:", vision.used_gemini)
        print("- fraud_probability:", vision.fraud_probability)
        print("- risk_signals:", vision.risk_signals)
        print("- reasoning:", vision.reasoning)
        if vision.error:
            print("- error:", vision.error)

        if vision.used_gemini and not vision.error:
            fraud_prob = vision.fraud_probability
            if vision.risk_signals:
                seen = {s.lower() for s in vision.risk_signals}
                extra = [s for s in risk_signals if s.lower() not in seen]
                risk_signals = (vision.risk_signals + extra)[:5]
            vision_signals = vision.risk_signals
    else:
        print("\n=== 3) Gemini Vision ===")
        print("- (URL 입력 시 Vision 미실행)")

    if scam_domains:
        fraud_prob = 1.0

    scores = score_prediction(fraud_prob, FRAUD_THRESHOLD)

    print("\n=== 4) 점수 ===")
    print("- modelVersion:", MODEL_VERSION)
    print("- thresholdUsed:", FRAUD_THRESHOLD)
    print("- fraudProbability:", fraud_prob)
    print("- riskScore:", scores.risk_score)
    print("- riskLevel(model):", scores.model_risk_level)
    print("- uiRiskLevel:", scores.ui_risk_level)
    print("- trustScore:", scores.trust_score)
    print("- uiTrustLabel:", scores.ui_trust_label)
    print("- riskSignals (최종):", risk_signals)

    template = build_template_message(
        company_name=args.company_name,
        trust_score=scores.trust_score,
        risk_score=scores.risk_score,
        ui_trust_label=scores.ui_trust_label,
        has_signals=bool(risk_signals),
        risk_signals=risk_signals,
        travel_ban_regions=risk_regions,
        scam_domains=scam_domains,
    )
    print("\n=== 5) Template ===")
    print(template)

    gem = polish_with_gemini(
        template_message=template,
        trust_score=scores.trust_score,
        trust_label=scores.ui_trust_label,
        fraud_probability=fraud_prob,
        risk_score=scores.risk_score,
        risk_signals=risk_signals,
    )
    final_msg = gem.message

    print("\n=== 6) Gemini Polish ===")
    print("- used_gemini:", gem.used_gemini)
    print("- fallback_to_template:", gem.fallback_to_template)
    print("- no_change:", gem.no_change)
    if gem.error:
        print("- error:", gem.error)
    print(final_msg)

    out = {
        "analysisId": "local",
        "type": args.type,
        "ocr": {
            "textPreview": ocr.text_preview,
            "textLength": ocr.text_length,
            "languageGuess": ocr.language_guess,
            "confidenceAvg": ocr.confidence_avg,
        },
        "mlPrediction": {
            "modelVersion": MODEL_VERSION,
            "fraudProbability": fraud_prob,
            "riskScore": scores.risk_score,
            "riskLevel": scores.model_risk_level,
            "thresholdUsed": FRAUD_THRESHOLD,
        },
        "explanation": {
            "riskSignals": risk_signals,
            "note": "Signals are matched against predefined scam-pattern rules.",
        },
        "ui": {
            "riskLevel": scores.ui_risk_level,
            "trustLabel": scores.ui_trust_label,
            "trustScore": scores.trust_score,
        },
        "analysisSummary": {
            "score": scores.trust_score,
            "label": scores.ui_trust_label,
            "message": final_msg,
        },
    }
    if args.debug:
        out["debug"] = {
            "visionSignals": vision_signals,
            "promptUsed": gem.prompt_used,
            "usedGemini": gem.used_gemini,
            "fallbackToTemplate": gem.fallback_to_template,
            "noChange": gem.no_change,
            "geminiError": gem.error,
        }
    print("\n=== 7) Final JSON ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
