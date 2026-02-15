# 학습해서 저장(전처리→학습→(옵션)export) 

from __future__ import annotations

"""
Baseline 학습/내보내기 1파일 스크립트.

원하는 형태:
- `python training/train_baseline.py train ...`
- `python training/train_baseline.py export ...`
- `python training/train_baseline.py all ...` (전처리 + 학습 + export)

기본 흐름:
1) `training/preprocess.py` 로직으로 통합/클린업/스플릿 생성
2) (이 파일 내부) TF-IDF + LogisticRegression 학습
3) (옵션) `models/fraud-baseline/`로 export 해서 inference가 바로 읽게 함
"""

import argparse
from pathlib import Path
import sys
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

# `python training/train_baseline.py`처럼 파일로 직접 실행할 때도 import가 되도록 경로 보정
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preprocess


@dataclass(frozen=True)
class TrainConfig:
    train_path: Path
    test_path: Path | None
    out_dir: Path
    model_version: str
    max_features: int
    ngram_max: int
    min_df: int
    class_weight: str
    random_state: int
    max_iter: int


def _load_df(path: Path):
    import pandas as pd  # type: ignore

    return pd.read_csv(path)


def _metrics(y_true, y_proba, *, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_proba >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    label_diverse = len(set(list(map(int, y_true.tolist())))) > 1
    pos = int((y_true == 1).sum())
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)) if pos > 0 else None,
        "f1": float(f1_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "roc_auc_ref": float(roc_auc_score(y_true, y_proba)) if label_diverse else None,
        "accuracy_ui": float(accuracy_score(y_true, y_pred)),
    }


def _dataset_signature(paths: list[Path]) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for p in paths:
        try:
            st = p.stat()
            sig.append(
                {
                    "path": str(p),
                    "size_bytes": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            )
        except FileNotFoundError:
            sig.append({"path": str(p), "missing": True})
    return sig


def cmd_train(args: argparse.Namespace) -> int:
    """
    TF-IDF + LogisticRegression 학습 후 아티팩트 저장.

    - 출력: {out_dir}/vectorizer.joblib, {out_dir}/model.joblib, {out_dir}/metadata.json
    """
    cfg = TrainConfig(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path) if args.test_path else None,
        out_dir=Path(args.out_dir),
        model_version=str(args.model_version),
        max_features=int(args.max_features),
        ngram_max=int(args.ngram_max),
        min_df=int(args.min_df),
        class_weight=str(args.class_weight),
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
    )

    train_df = _load_df(cfg.train_path)
    test_df = _load_df(cfg.test_path) if (cfg.test_path and cfg.test_path.exists()) else None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.min_df,
        lowercase=False,  # 전처리에서 이미 lower 처리
    )

    cw = "balanced" if cfg.class_weight == "balanced" else None
    clf = LogisticRegression(
        class_weight=cw,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )

    x_train = vectorizer.fit_transform(train_df["text"].fillna("").astype(str).tolist())
    y_train = train_df["fraudulent"].fillna(0).astype(int).to_numpy()
    clf.fit(x_train, y_train)

    # 학습 직후에는 threshold를 0.5(또는 입력값)로 기록합니다.
    # 최종 threshold/metrics는 evaluate 단계에서 업데이트하는 것을 권장합니다.
    threshold = float(getattr(args, "threshold", 0.5))
    metrics = None
    if test_df is not None:
        # 학습 단계에서는 정확도(참고용)만 간단히 출력합니다.
        # 실제 모델 선택 지표(PR-AUC/F1 등)와 threshold 튜닝은 evaluate 단계에서 수행하세요.
        x_test = vectorizer.transform(test_df["text"].fillna("").astype(str).tolist())
        y_test = test_df["fraudulent"].fillna(0).astype(int).to_numpy()
        proba = clf.predict_proba(x_test)[:, 1]
        m = _metrics(y_test, proba, threshold=threshold)
        print(f"[학습] (참고) test 정확도(accuracy_ui, threshold={threshold:.3f}): {m['accuracy_ui']:.6f}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(vectorizer, cfg.out_dir / "vectorizer.joblib")
    joblib.dump(clf, cfg.out_dir / "model.joblib")

    cfg_dict = asdict(cfg)
    cfg_dict["train_path"] = str(cfg.train_path)
    cfg_dict["test_path"] = str(cfg.test_path) if cfg.test_path is not None else None
    cfg_dict["out_dir"] = str(cfg.out_dir)

    preprocess_cfg = getattr(args, "preprocess_config", None)
    if preprocess_cfg is None:
        # preprocess.py가 남긴 signature를 자동으로 끌어옵니다.
        sig_path = cfg.train_path.parent / "preprocess_signature.json"
        if sig_path.exists():
            try:
                preprocess_cfg = json.loads(sig_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                preprocess_cfg = None
    dataset_sig = getattr(args, "dataset_signature", None)

    # 최소 재현성 시그니처: train/test 규모/클래스 분포/출처 분포
    try:
        sig: dict[str, Any] = {}
        sig["train_size"] = int(len(train_df))
        sig["train_pos"] = int((train_df["fraudulent"].fillna(0).astype(int) == 1).sum())
        sig["train_neg"] = int(sig["train_size"]) - int(sig["train_pos"])
        if test_df is not None:
            sig["test_size"] = int(len(test_df))
            sig["test_pos"] = int((test_df["fraudulent"].fillna(0).astype(int) == 1).sum())
            sig["test_neg"] = int(sig["test_size"]) - int(sig["test_pos"])
        sources: dict[str, Any] = {}
        if "source" in train_df.columns:
            sources_train: dict[str, Any] = {}
            for s, g in train_df.groupby("source"):
                p = int((g["fraudulent"].fillna(0).astype(int) == 1).sum())
                sources_train[str(s)] = {"size": int(len(g)), "pos": p, "neg": int(len(g)) - p}
            sources["train"] = sources_train
        if test_df is not None and "source" in test_df.columns:
            sources_test: dict[str, Any] = {}
            for s, g in test_df.groupby("source"):
                p = int((g["fraudulent"].fillna(0).astype(int) == 1).sum())
                sources_test[str(s)] = {"size": int(len(g)), "pos": p, "neg": int(len(g)) - p}
            sources["test"] = sources_test
        if sources:
            sig["sources"] = sources
        # 기존 raw 파일 signature(있다면)도 함께 묶어서 저장
        if isinstance(dataset_sig, list):
            sig["raw_files"] = dataset_sig
        dataset_sig = sig
    except Exception:  # noqa: BLE001
        pass

    metadata = {
        "model_version": cfg.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "notes": "tfidf + logistic regression (threshold tuned in evaluate)",
        "preprocess_signature": preprocess_cfg,
        "dataset_signature": dataset_sig,
        "config": cfg_dict,
        "metrics": metrics,
    }
    with open(cfg.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[학습] 아티팩트 저장 완료: {cfg.out_dir}")
    return 0


def _atomic_export_dir(src: Path, dst: Path) -> None:
    """
    export 원자성(atomicity) 보장:
    - dst.__tmp__에 먼저 복사
    - dst가 있으면 dst.__bak__으로 이동
    - tmp → dst로 rename
    - bak 정리
    """
    tmp = dst.parent / f"{dst.name}.__tmp__"
    bak = dst.parent / f"{dst.name}.__bak__"

    if tmp.exists():
        shutil.rmtree(tmp)
    if bak.exists():
        shutil.rmtree(bak)

    tmp.mkdir(parents=True, exist_ok=True)
    for name in ["vectorizer.joblib", "model.joblib", "metadata.json"]:
        shutil.copy2(src / name, tmp / name)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.replace(dst, bak)
    os.replace(tmp, dst)
    if bak.exists():
        shutil.rmtree(bak)


def cmd_export(args: argparse.Namespace) -> int:
    src = Path(args.model_dir)
    dst = Path(args.export_dir)
    _atomic_export_dir(src, dst)
    print(f"[내보내기] 아티팩트 원자적 교체 완료: {src} -> {dst}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    # 1) preprocess + split
    processed_dir = Path(args.processed_dir)
    combined_path = processed_dir / "combined_train.csv"

    if args.inputs is None or len(args.inputs) == 0:
        input_paths = preprocess.default_input_paths()
    else:
        input_paths = [Path(p) for p in args.inputs]

    preprocess.make_dataset(
        inputs=input_paths,
        out=combined_path,
        nrows=int(args.nrows) if args.nrows is not None else None,
        max_text_chars=int(args.max_text_chars),
        min_text_chars=int(args.min_text_chars),
        mask_leakage=not bool(args.no_leakage_mask),
        dedupe=not bool(args.no_dedupe),
        random_state=int(args.random_state),
        test_size=float(args.test_size),
        valid_size=float(args.valid_size),
        fakepos_multiplier=float(args.fakepos_multiplier),
        linkedin_multiplier=float(args.linkedin_multiplier),
    )

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    # 2) train
    train_args = argparse.Namespace(
        train_path=str(train_path),
        test_path=str(test_path),
        out_dir=str(args.out_dir),
        model_version=str(args.model_version),
        max_features=int(args.max_features),
        ngram_max=int(args.ngram_max),
        min_df=int(args.min_df),
        class_weight=str(args.class_weight),
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
        threshold=float(args.threshold),
        preprocess_config={
            "max_text_chars": int(args.max_text_chars),
            "min_text_chars": int(args.min_text_chars),
            "test_size": float(args.test_size),
            "valid_size": float(args.valid_size),
            "random_state": int(args.random_state),
            "fakepos_multiplier": float(args.fakepos_multiplier),
            "linkedin_multiplier": float(args.linkedin_multiplier),
            "mask_leakage": not bool(args.no_leakage_mask),
            "dedupe": not bool(args.no_dedupe),
        },
        dataset_signature=_dataset_signature(input_paths),
    )
    cmd_train(train_args)

    # 3) export
    if args.export_dir:
        export_args = argparse.Namespace(model_dir=str(args.out_dir), export_dir=str(args.export_dir))
        cmd_export(export_args)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline TF-IDF+LR: train/export/all.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and save artifacts")
    p_train.add_argument("--train-path", default="training/data/processed/train.csv")
    p_train.add_argument("--test-path", default="training/data/processed/test.csv")
    p_train.add_argument("--out-dir", default="training/runs/tfidf_lr")
    p_train.add_argument("--model-version", default="fraud-baseline-v1.0.0")
    p_train.add_argument("--max-features", type=int, default=50000)
    p_train.add_argument("--ngram-max", type=int, default=2)
    p_train.add_argument("--min-df", type=int, default=2)
    p_train.add_argument("--class-weight", default="balanced", help="balanced | none")
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--max-iter", type=int, default=2000)
    p_train.add_argument("--threshold", type=float, default=0.5, help="Only for initial 기록용 (default: 0.5).")
    p_train.set_defaults(func=cmd_train)

    p_export = sub.add_parser("export", help="Atomic export to models/ for inference")
    p_export.add_argument("--model-dir", default="training/runs/tfidf_lr")
    p_export.add_argument("--export-dir", default="models/fraud-baseline")
    p_export.set_defaults(func=cmd_export)

    p_all = sub.add_parser("all", help="Preprocess + train + (optional) export")
    p_all.add_argument("--inputs", nargs="*", default=None)
    p_all.add_argument("--processed-dir", default="training/data/processed")
    p_all.add_argument("--nrows", type=int, default=None, help="(debug) Read only first N rows per input file.")
    p_all.add_argument("--max-text-chars", type=int, default=6000)
    p_all.add_argument("--min-text-chars", type=int, default=50)
    p_all.add_argument("--test-size", type=float, default=0.2)
    p_all.add_argument("--valid-size", type=float, default=0.0)
    p_all.add_argument("--random-state", type=int, default=42)
    p_all.add_argument("--fakepos-multiplier", type=float, default=2.0)
    p_all.add_argument("--linkedin-multiplier", type=float, default=3.0)
    p_all.add_argument("--no-leakage-mask", action="store_true")
    p_all.add_argument("--no-dedupe", action="store_true")

    p_all.add_argument("--out-dir", default="training/runs/tfidf_lr")
    p_all.add_argument("--model-version", default="fraud-baseline-v1.0.0")
    p_all.add_argument("--max-features", type=int, default=50000)
    p_all.add_argument("--ngram-max", type=int, default=2)
    p_all.add_argument("--min-df", type=int, default=2)
    p_all.add_argument("--class-weight", default="balanced", help="balanced | none")
    p_all.add_argument("--max-iter", type=int, default=2000)
    p_all.add_argument("--threshold", type=float, default=0.5)

    p_all.add_argument("--export-dir", default=None)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
