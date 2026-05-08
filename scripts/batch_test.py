from __future__ import annotations

"""
디렉토리 내 이미지를 공고 단위로 분석하고 결과를 출력하는 스크립트.

파일명 규칙: postingN_MM.png (같은 N = 같은 공고, MM = 스크린샷 순서)
라벨 파일:  labels.json  {"posting1": "scam", "posting5": "normal", ...}

사용 예:
  # labels.json이 있는 디렉토리 (정확도 자동 계산)
  python3 scripts/batch_test.py --dir ./scripts/recruiting_examples

  # 라벨 없이 결과만 확인
  python3 scripts/batch_test.py --dir ./test_images/

  # CSV로 저장
  python3 scripts/batch_test.py --dir ./scripts/recruiting_examples --out results.csv
"""

import argparse
import csv
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.env import load_dotenv_once
from app.ml.message_risk_rules import extract_risk_signals
from app.ml.risk_regions import find_risk_regions
from app.ml.scam_domains import find_scam_domains
from app.routes.analyze import FRAUD_THRESHOLD
from app.services.gemini_service import analyze_image_with_gemini_vision
from app.services.ocr_service import ocr_from_bytes
from app.services.scoring_service import score_prediction
from app.services.summary_builder import build_template_message

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POSTING_RE = re.compile(r"^(posting\d+)_\d+", re.IGNORECASE)


def _group_by_posting(img_dir: Path) -> dict[str, list[Path]]:
    """postingN_MM.png 파일을 posting 단위로 그룹핑. 매칭 안 되는 파일은 파일명을 key로."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = POSTING_RE.match(p.name)
        key = m.group(1) if m else p.stem
        groups[key].append(p)
    return dict(sorted(groups.items(), key=lambda x: (
        int(re.search(r"\d+", x[0]).group()) if re.search(r"\d+", x[0]) else 0, x[0]
    )))


def _analyze(image_paths: list[Path]) -> dict:
    """한 공고(1장 이상의 이미지)를 분석."""
    all_ocr_texts: list[str] = []
    all_image_bytes: list[bytes] = []

    for path in image_paths:
        img_bytes = path.read_bytes()
        all_image_bytes.append(img_bytes)
        ocr = ocr_from_bytes(img_bytes)
        if ocr.text_length >= 10:
            all_ocr_texts.append(ocr.text)

    combined_text = "\n".join(all_ocr_texts)
    combined_len = sum(len(t) for t in all_ocr_texts)

    vision = analyze_image_with_gemini_vision(all_image_bytes[0])

    if combined_len < 30:
        return {
            "ocr_len": combined_len,
            "ocr_preview": combined_text[:80],
            "fraud_prob": None,
            "risk_level": "UNKNOWN",
            "trust_score": None,
            "scam_domains": [],
            "risk_signals": [],
            "message": "OCR 실패 (텍스트 부족)",
            "num_images": len(image_paths),
        }

    risk_signals = extract_risk_signals(combined_text, top_k=3)
    risk_regions = find_risk_regions(combined_text, top_k=5)
    scam_domains = find_scam_domains(combined_text)

    if vision.used_gemini and not vision.error:
        prob = vision.fraud_probability
        if vision.risk_signals:
            seen = {s.lower() for s in vision.risk_signals}
            extra = [s for s in risk_signals if s.lower() not in seen]
            risk_signals = (vision.risk_signals + extra)[:5]
    else:
        prob = 0.5

    if scam_domains:
        prob = 1.0

    scores = score_prediction(prob, FRAUD_THRESHOLD)
    message = build_template_message(
        company_name=None,
        trust_score=scores.trust_score,
        risk_score=scores.risk_score,
        ui_trust_label=scores.ui_trust_label,
        risk_signals=risk_signals,
        travel_ban_regions=risk_regions,
        scam_domains=scam_domains,
    )

    return {
        "ocr_len": combined_len,
        "ocr_preview": combined_text[:80],
        "fraud_prob": round(prob, 4),
        "risk_level": scores.ui_risk_level,
        "trust_score": scores.trust_score,
        "scam_domains": scam_domains,
        "risk_signals": risk_signals,
        "message": message,
        "num_images": len(image_paths),
    }


def _risk_emoji(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}.get(level, "⚪")


def main() -> int:
    load_dotenv_once()

    parser = argparse.ArgumentParser(description="배치 이미지 분석 테스트")
    parser.add_argument("--dir", required=True, help="이미지 디렉토리 경로")
    parser.add_argument("--out", default=None, help="결과 CSV 저장 경로 (옵션)")
    args = parser.parse_args()

    img_dir = Path(args.dir)
    if not img_dir.is_dir():
        raise SystemExit(f"디렉토리를 찾을 수 없습니다: {img_dir}")

    # labels.json 로드 (있으면)
    labels: dict[str, str] = {}
    label_path = img_dir / "labels.json"
    if label_path.exists():
        with label_path.open(encoding="utf-8") as f:
            labels = json.load(f)
        print(f"[라벨] {label_path.name} 로드 ({len(labels)}개)")

    groups = _group_by_posting(img_dir)
    if not groups:
        raise SystemExit(f"이미지 파일이 없습니다: {img_dir}")

    print(f"\n[배치 테스트] {len(groups)}개 공고 ({sum(len(v) for v in groups.values())}장) 분석 중...")
    print(f"[설정] threshold={FRAUD_THRESHOLD}\n")

    rows: list[dict] = []
    for i, (posting_id, paths) in enumerate(groups.items(), 1):
        img_str = f"{len(paths)}장" if len(paths) > 1 else "1장"
        print(f"[{i}/{len(groups)}] {posting_id} ({img_str}) ... ", end="", flush=True)
        try:
            result = _analyze(paths)
            result["posting_id"] = posting_id
            result["label"] = labels.get(posting_id, "")
            rows.append(result)
            lvl = result["risk_level"]
            prob = result["fraud_prob"]
            domains = result["scam_domains"]
            domain_str = f"  도메인:{domains}" if domains else ""
            print(f"{_risk_emoji(lvl)} {lvl}  prob={prob}{domain_str}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            rows.append({
                "posting_id": posting_id,
                "label": labels.get(posting_id, ""),
                "num_images": len(paths),
                "ocr_len": 0,
                "ocr_preview": "",
                "fraud_prob": None,
                "risk_level": "ERROR",
                "trust_score": None,
                "scam_domains": [],
                "risk_signals": [],
                "message": str(e),
            })

    # 결과 테이블
    print("\n" + "=" * 80)
    print(f"{'공고ID':<12} {'라벨':<8} {'장수':>3}  {'확률':>6}  {'위험도':<8} {'사기도메인'}  {'위험신호'}")
    print("-" * 80)
    for r in rows:
        prob_str = f"{r['fraud_prob']:.3f}" if r["fraud_prob"] is not None else "  N/A"
        domains_str = ", ".join(r["scam_domains"]) if r["scam_domains"] else "-"
        signals_str = " / ".join(r["risk_signals"][:2]) if r["risk_signals"] else "-"
        label_str = r.get("label", "")
        print(f"{r['posting_id']:<12} {label_str:<8} {r['num_images']:>3}장  {prob_str:>6}  {r['risk_level']:<8} {domains_str:<18} {signals_str}")

    # 요약
    print("=" * 80)
    total = len(rows)
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0, "ERROR": 0}
    for r in rows:
        lvl = r["risk_level"] if r["risk_level"] in counts else "ERROR"
        counts[lvl] += 1

    print(f"\n[요약] 총 {total}개 공고")
    for lvl, cnt in counts.items():
        if cnt:
            pct = cnt / total * 100
            print(f"  {_risk_emoji(lvl)} {lvl:<8}: {cnt}개 ({pct:.0f}%)")

    # 정확도 (labels.json 있는 경우)
    if labels:
        labeled = [r for r in rows if r.get("label") in {"scam", "normal"}]
        if labeled:
            scam_rows = [r for r in labeled if r["label"] == "scam"]
            normal_rows = [r for r in labeled if r["label"] == "normal"]
            tp = sum(1 for r in scam_rows if r["risk_level"] == "HIGH")
            fp = sum(1 for r in normal_rows if r["risk_level"] == "HIGH")
            fn = len(scam_rows) - tp
            tn = len(normal_rows) - fp
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

            print(f"\n[정확도]  scam={len(scam_rows)}개  normal={len(normal_rows)}개")
            print(f"  Precision : {precision:.3f}  ({tp}/(TP{tp}+FP{fp}))")
            print(f"  Recall    : {recall:.3f}  ({tp}/(TP{tp}+FN{fn}))")
            print(f"  F1        : {f1:.3f}")

            missed = [r["posting_id"] for r in scam_rows if r["risk_level"] != "HIGH"]
            if missed:
                print(f"  미탐지 사기 공고: {', '.join(missed)}")
            fp_list = [r["posting_id"] for r in normal_rows if r["risk_level"] == "HIGH"]
            if fp_list:
                print(f"  오탐지 정상 공고: {', '.join(fp_list)}")

    # 메시지 샘플
    print("\n[메시지 샘플 - 상위 3개]")
    for r in rows[:3]:
        print(f"\n  {r['posting_id']} ({r.get('label', '')})")
        msg_wrapped = textwrap.fill(r["message"], width=66, initial_indent="  ", subsequent_indent="  ")
        print(msg_wrapped)

    # CSV 저장
    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "posting_id", "label", "num_images", "fraud_prob", "risk_level",
                "trust_score", "scam_domains", "risk_signals", "ocr_len", "ocr_preview", "message",
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    **r,
                    "scam_domains": "; ".join(r["scam_domains"]),
                    "risk_signals": "; ".join(r["risk_signals"]),
                })
        print(f"\n[저장] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
