from __future__ import annotations

"""
로컬에서 .txt 공고 텍스트로 모델 추론을 빠르게 돌려보는 스크립트.

백엔드 연동 없이 파일만으로 확인하고 싶을 때 사용합니다.

사용 예:
  python3 scripts/run_infer_txt.py --file examples/posting.txt
  MODEL_DIR=models/fraud-baseline python3 scripts/run_infer_txt.py --file posting.txt
"""

import argparse
import json
from pathlib import Path
import sys

# `python scripts/run_infer_txt.py`처럼 파일로 직접 실행할 때도 import가 되도록 경로 보정
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml_baseline import TfidfLogRegPredictor, clean_posting_text, is_structured_posting_text, make_structured_posting_text
from app.diagnostics import analyze_input
from app.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local inference on a job posting text file.")
    parser.add_argument("--file", required=True, help="Path to .txt file containing job posting text.")
    parser.add_argument("--analysis-id", default="local-1")
    parser.add_argument(
        "--show-structured",
        action="store_true",
        help="추론에 넣기 위해 템플릿으로 감싼 입력(구조화된 텍스트)을 함께 출력합니다.",
    )
    parser.add_argument(
        "--show-cleaned",
        action="store_true",
        help="clean_posting_text()를 적용한 최종 입력(모델에 실제로 들어가는 텍스트)을 함께 출력합니다.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="show 옵션 사용 시 출력할 최대 문자 수(기본 1200).",
    )
    args = parser.parse_args()

    settings = get_settings()
    predictor = TfidfLogRegPredictor.from_dir(settings.model_dir)

    text = Path(args.file).read_text(encoding="utf-8")
    if text.strip() == "":
        raise SystemExit("입력 텍스트가 비어 있습니다.")

    # posting.txt가 이미 [TITLE] ... 형태면 이중 감싸기를 방지합니다.
    already_structured = is_structured_posting_text(text)
    augmented = make_structured_posting_text(text)
    cleaned = clean_posting_text(augmented, max_chars=settings.max_text_chars)

    if args.show_structured:
        preview = augmented if args.preview_chars <= 0 else augmented[: args.preview_chars]
        print("\n=== [구조화된 입력(템플릿)] ===")
        print(f"- already_structured: {already_structured}")
        print(f"- length: {len(augmented)} chars")
        print(preview + ("" if len(preview) == len(augmented) else "\n... (truncated)"))

    if args.show_cleaned:
        preview = cleaned if args.preview_chars <= 0 else cleaned[: args.preview_chars]
        print("\n=== [클린업 후 입력(모델 입력)] ===")
        print(f"- length: {len(cleaned)} chars (MAX_TEXT_CHARS={settings.max_text_chars})")
        print(preview + ("" if len(preview) == len(cleaned) else "\n... (truncated)"))

    proba = predictor.predict_proba(cleaned)
    risk_score = int(round(proba * 100))

    # riskLevel은 FastAPI와 동일하게 riskScore 구간으로만 산출
    risk_level = "HIGH" if risk_score >= 80 else ("MEDIUM" if risk_score >= 50 else "LOW")

    diag = analyze_input(text)

    out = {
        "analysisId": args.analysis_id,
        "modelVersion": settings.model_version,
        "fraudProbability": proba,
        "riskScore": risk_score,
        "riskLevel": risk_level,
        # 배포 폴더(models/fraud-baseline/metadata.json)의 threshold를 그대로 노출
        "modelPolicy": {
            "threshold": float(predictor.threshold),
            "highPrecisionThreshold": predictor.high_precision_threshold,
        },
        "inputDiagnostics": {
            "language": diag.language,
            "in_domain": diag.in_domain,
            "input_confidence": diag.input_confidence,
            "note": diag.note,
        },
    }
    if args.show_structured:
        out["inputStructured"] = augmented
    if args.show_cleaned:
        out["inputCleaned"] = cleaned
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
