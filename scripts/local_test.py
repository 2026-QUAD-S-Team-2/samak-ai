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

# `python scripts/local_test.py`처럼 파일로 직접 실행할 때도 import가 되도록 경로 보정
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv_once
from app.ml.ml_baseline import BaselineModel
from app.ml.risk_regions import find_risk_regions
from app.services.gemini_service import polish_with_gemini
from app.services.ocr_service import ocr_from_bytes, ocr_from_url
from app.services.scoring_service import score_prediction
from app.services.summary_builder import build_template_message


def main() -> int:
    load_dotenv_once()
    parser = argparse.ArgumentParser()
    # 사용자가 `python3 scripts/local_test.py ./scripts/a.png ...`처럼 실행하는 실수를 줄이기 위해
    # 파일 경로를 positional로도 받을 수 있게 합니다. (--file도 계속 지원)
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
        ocr = ocr_from_url(args.image_url)

    print("=== 1) OCR ===")
    print("- languageGuess:", ocr.language_guess)
    print("- confidenceAvg:", ocr.confidence_avg)
    print("- textLength:", ocr.text_length)
    print("- textPreview:", ocr.text_preview)
    if ocr.error:
        print("- error:", ocr.error)

    if ocr.text_length < 30:
        print("\n=== 2) ML ===")
        print("⚠️ OCR 텍스트가 너무 짧아 ML 추론을 건너뜁니다.")
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
                "modelVersion": "fraud-baseline-v1.0.0",
                "fraudProbability": None,
                "riskScore": None,
                "riskLevel": None,
                "thresholdUsed": None,
            },
            "ui": {"riskLevel": "UNKNOWN", "trustLabel": None, "trustScore": None},
            "analysisSummary": {"score": None, "label": None, "message": msg},
        }
        print("\n=== 5) Final JSON ===")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    model = BaselineModel.load_default()
    prob = model.predict_proba_from_ocr(ocr.text)
    risk_signals = model.risk_signals_from_ocr(ocr.text, top_k=3)
    risk_regions = find_risk_regions(model.get_cleaned_input_from_ocr(ocr.text), top_k=5)
    scores = score_prediction(prob, model.threshold)

    print("\n=== 2) ML ===")
    print("- modelVersion:", model.model_version)
    print("- thresholdUsed:", model.threshold)
    print("- fraudProbability:", prob)
    print("- riskScore:", scores.risk_score)
    print("- riskLevel(modelPolicy):", scores.model_risk_level)
    print("- uiRiskLevel:", scores.ui_risk_level)
    print("- trustScore:", scores.trust_score)
    print("- uiTrustLabel:", scores.ui_trust_label)
    print("- riskSignals:", risk_signals)
    print("- travelBanRegionsMatched:", risk_regions)

    template = build_template_message(
        company_name=args.company_name,
        trust_score=scores.trust_score,
        risk_score=scores.risk_score,
        ui_trust_label=scores.ui_trust_label,
        has_signals=bool(risk_signals),
        travel_ban_regions=risk_regions,
    )
    print("\n=== 3) Template ===")
    print(template)

    gem = polish_with_gemini(
        template_message=template,
        trust_score=scores.trust_score,
        trust_label=scores.ui_trust_label,
        fraud_probability=prob,
        risk_score=scores.risk_score,
        risk_signals=risk_signals,
    )
    final_msg = gem.message

    print("\n=== 4) Gemini ===")
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
            "modelVersion": model.model_version,
            "fraudProbability": prob,
            "riskScore": scores.risk_score,
            "riskLevel": scores.model_risk_level,
            "thresholdUsed": model.threshold,
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
            "inputStructured": model.structure_ocr_text(ocr.text),
            "inputCleaned": model.get_cleaned_input_from_ocr(ocr.text),
            "promptUsed": gem.prompt_used,
            "usedGemini": gem.used_gemini,
            "fallbackToTemplate": gem.fallback_to_template,
            "noChange": gem.no_change,
            "geminiError": gem.error,
            "explanation": {"riskSignals": risk_signals},
        }
    print("\n=== 5) Final JSON ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
