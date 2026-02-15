# 저장된 모델로 정확도 평가

from __future__ import annotations

"""
Baseline 평가 1파일 스크립트: 저장된 모델로 metrics(+accuracy) 출력.

- 학습된 아티팩트(`vectorizer.joblib`, `model.joblib`)를 로딩
- test.csv를 읽어서 성능을 계산
- (기본) best threshold를 탐색하고 metadata.json에 저장
"""

import argparse
from pathlib import Path
import sys
import json

# `python training/evaluate_baseline.py`처럼 파일로 직접 실행할 때도 import가 되도록 경로 보정
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved baseline model on a test set.")
    parser.add_argument("--model-dir", default="models/fraud-baseline", help="Directory with joblib artifacts.")
    parser.add_argument(
        "--train-path",
        default=None,
        help="Optional train.csv path to record dataset_signature stats (recommended).",
    )
    parser.add_argument("--test-path", default="training/data/processed/test.csv", help="CSV with text/fraudulent.")
    parser.add_argument("--threshold", type=float, default=None, help="If set, evaluate at this threshold.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="PR curve에서 F1이 최대가 되는 threshold를 찾고 metadata.json에 저장합니다.",
    )
    parser.add_argument(
        "--precision-policy",
        type=float,
        default=0.9,
        help="(옵션) 정책 임계치: precision이 이 값 이상이 되는 threshold를 함께 저장합니다. (default: 0.9)",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write metadata.json (debug only).")
    args = parser.parse_args()

    import joblib
    import pandas as pd  # type: ignore
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        precision_recall_curve,
    )

    model_dir = Path(args.model_dir)
    test_path = Path(args.test_path)

    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    clf = joblib.load(model_dir / "model.joblib")

    df = pd.read_csv(test_path)
    x = vectorizer.transform(df["text"].fillna("").astype(str).tolist())
    y = df["fraudulent"].fillna(0).astype(int).to_numpy()

    proba = clf.predict_proba(x)[:, 1]

    pr_auc = float(average_precision_score(y, proba))
    roc_auc = float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else None

    def eval_at(th: float) -> dict[str, object]:
        y_pred = (proba >= float(th)).astype(int)
        return {
            "threshold": float(th),
            "pr_auc": pr_auc,
            "f1": float(f1_score(y, y_pred)),
            "precision_pos": float(precision_score(y, y_pred, zero_division=0)),
            "recall_pos": float(recall_score(y, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "roc_auc_ref": roc_auc,
            "accuracy_ui": float(accuracy_score(y, y_pred)),
        }

    # threshold 선택: (1) 명시된 threshold (2) tune 모드 (3) metadata.json에 있으면 그것 (4) 0.5
    meta_path = model_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = loaded if isinstance(loaded, dict) else {}
        except Exception:  # noqa: BLE001
            meta = {}

    chosen: float
    if args.threshold is not None:
        chosen = float(args.threshold)
    elif args.tune:
        chosen = 0.5
    elif isinstance(meta, dict) and isinstance(meta.get("threshold"), (int, float)):
        chosen = float(meta["threshold"])
    else:
        chosen = 0.5

    best_metrics = eval_at(chosen)

    tuned_info = None
    thresholds_saved: dict[str, float] = {}
    if args.tune:
        precision, recall, thresholds = precision_recall_curve(y, proba)
        # thresholds 길이는 (len(precision)-1). 각 threshold에 대해 precision[1:], recall[1:]가 대응
        if thresholds is not None and len(thresholds) > 0:
            p = precision[1:]
            r = recall[1:]
            denom = (p + r)
            f1s = (2 * p * r) / (denom + 1e-12)
            best_f1 = float(f1s.max())
            # 동일 f1이면 더 작은 threshold(=recall 방향) 선택
            best_idxs = (f1s >= best_f1 - 1e-12).nonzero()[0]
            best_idx = int(best_idxs[0]) if len(best_idxs) > 0 else int(f1s.argmax())
            best_th = float(thresholds[best_idx])
            best_metrics = eval_at(best_th)
            tuned_info = {"method": "pr_curve_f1_max", "best_f1": best_f1}
            thresholds_saved["threshold_f1_max"] = best_th

            # 정책 임계치(precision >= X): 그 조건을 만족하면서 recall이 최대(=threshold 낮은 쪽)인 임계치 선택
            target_p = float(args.precision_policy)
            ok = (p >= target_p).nonzero()[0]
            if len(ok) > 0:
                # recall이 큰 것 우선(동률이면 threshold 더 작은 것)
                ok_sorted = sorted(ok.tolist(), key=lambda i: (-float(r[i]), float(thresholds[i])))
                pol_idx = int(ok_sorted[0])
                thresholds_saved[f"threshold_precision_{int(target_p*100)}"] = float(thresholds[pol_idx])

    # source별 성능(도메인 편향 확인용)
    metrics_by_source: dict[str, object] = {}
    if "source" in df.columns:
        for source_name, subdf in df.groupby("source"):
            try:
                xs = vectorizer.transform(subdf["text"].fillna("").astype(str).tolist())
                ys = subdf["fraudulent"].fillna(0).astype(int).to_numpy()
                ps = clf.predict_proba(xs)[:, 1]
                pos = int((ys == 1).sum())
                neg = int((ys == 0).sum())
                label_diverse = pos > 0 and neg > 0
                if label_diverse:
                    y_pred_s = (ps >= float(best_metrics["threshold"])).astype(int)
                    cm_s = confusion_matrix(ys, y_pred_s).tolist()
                    metrics_by_source[str(source_name)] = {
                        "count": int(len(subdf)),
                        "pos": pos,
                        "neg": neg,
                        "note": None,
                        "pr_auc": float(average_precision_score(ys, ps)),
                        "f1": float(f1_score(ys, y_pred_s)),
                        "precision_pos": float(precision_score(ys, y_pred_s, zero_division=0)),
                        "recall_pos": float(recall_score(ys, y_pred_s, zero_division=0)),
                        "confusion_matrix": cm_s,
                        "roc_auc_ref": float(roc_auc_score(ys, ps)),
                        "accuracy_ui": float(accuracy_score(ys, y_pred_s)),
                    }
                else:
                    metrics_by_source[str(source_name)] = {
                        "count": int(len(subdf)),
                        "pos": pos,
                        "neg": neg,
                        "note": "단일 클래스(source 내 pos=0 또는 neg=0)라 N/A",
                        "pr_auc": None,
                        "f1": None,
                        "precision_pos": None,
                        "recall_pos": None,
                        "confusion_matrix": None,
                        "roc_auc_ref": None,
                        "accuracy_ui": None,
                    }
            except Exception:  # noqa: BLE001
                continue

    print("[평가] 메인 지표(PR-AUC):", best_metrics["pr_auc"])
    print("[평가] best threshold:", best_metrics["threshold"])
    print("[평가] 상세 지표:", best_metrics)

    if not args.no_write:
        # evaluate 실행 시 threshold + metrics는 항상 metadata.json에 채웁니다(덮어쓰기).
        # metadata.json이 없거나 깨졌어도 새로 생성합니다.
        if not isinstance(meta, dict):
            meta = {}
        meta["threshold"] = float(best_metrics["threshold"])
        if thresholds_saved:
            meta["threshold_policies"] = thresholds_saved
        meta["metrics"] = {k: best_metrics[k] for k in ["pr_auc", "f1", "precision_pos", "recall_pos", "confusion_matrix", "roc_auc_ref", "accuracy_ui"]}
        meta["metrics_by_source"] = metrics_by_source
        # (옵션) 제품 MVP 기준으로 가장 해석 가능한 supervised test: RecruitmentScam only
        if "RecruitmentScam" in metrics_by_source:
            meta["metrics_recruitmentscam_test"] = metrics_by_source["RecruitmentScam"]
        if tuned_info is not None:
            meta["threshold_tuning"] = tuned_info

        # dataset_signature 최소 정보 기록(재현성)
        def _split_stats(split_df) -> dict[str, object]:
            pos = int((split_df["fraudulent"].fillna(0).astype(int) == 1).sum())
            n = int(len(split_df))
            out = {"size": n, "pos": pos, "neg": n - pos}
            if "source" in split_df.columns:
                by_source: dict[str, object] = {}
                for s, g in split_df.groupby("source"):
                    p = int((g["fraudulent"].fillna(0).astype(int) == 1).sum())
                    by_source[str(s)] = {"size": int(len(g)), "pos": p, "neg": int(len(g)) - p}
                out["sources"] = by_source
            return out

        try:
            train_df = None
            if args.train_path:
                train_df = pd.read_csv(Path(args.train_path))
            elif isinstance(meta.get("config"), dict) and meta["config"].get("train_path"):
                train_df = pd.read_csv(Path(str(meta["config"]["train_path"])))
        except Exception:  # noqa: BLE001
            train_df = None

        sig: dict[str, object] = {}
        if train_df is not None:
            sig["train_size"] = int(len(train_df))
            sig["train_pos"] = int((train_df["fraudulent"].fillna(0).astype(int) == 1).sum())
            sig["train_neg"] = int(sig["train_size"]) - int(sig["train_pos"])
            sig["sources"] = {"train": _split_stats(train_df).get("sources", {})}
        sig["test_size"] = int(len(df))
        sig["test_pos"] = int((df["fraudulent"].fillna(0).astype(int) == 1).sum())
        sig["test_neg"] = int(sig["test_size"]) - int(sig["test_pos"])
        sig_sources_test = _split_stats(df).get("sources", {})
        if "sources" not in sig:
            sig["sources"] = {}
        if isinstance(sig["sources"], dict):
            sig["sources"]["test"] = sig_sources_test
        meta["dataset_signature"] = sig

        # preprocess_signature도 가능한 경우 채웁니다.
        if "preprocess_signature" not in meta or meta.get("preprocess_signature") is None:
            try:
                processed_dir = Path(args.test_path).resolve().parent
                sig_path = processed_dir / "preprocess_signature.json"
                if sig_path.exists():
                    meta["preprocess_signature"] = json.loads(sig_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass

        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[평가] metadata.json 업데이트 완료: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
