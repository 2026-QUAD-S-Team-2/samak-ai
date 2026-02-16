from __future__ import annotations

"""
ML baseline (TF-IDF + LogisticRegression) 추론 모듈.

요구사항:
- models/fraud-baseline/ 아래 아티팩트 로딩
- threshold는 metadata.json에서 읽고 없으면 0.5
- OCR 텍스트는 고정 템플릿의 [DESCRIPTION]에만 삽입 (title/location 등 2차 추출은 하지 않음)
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from app.ml.text_utils import clean_posting_text


class ModelArtifactsError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineModel:
    vectorizer: Any
    classifier: Any
    threshold: float
    model_version: str
    model_dir: str

    @staticmethod
    def load_default() -> "BaselineModel":
        model_dir = os.environ.get("MODEL_DIR", os.path.join("models", "fraud-baseline"))
        model_version = os.environ.get("MODEL_VERSION", "fraud-baseline-v1.0.0")
        return BaselineModel.load(model_dir=model_dir, model_version=model_version)

    @staticmethod
    def load(*, model_dir: str, model_version: str) -> "BaselineModel":
        vec_path = os.path.join(model_dir, "vectorizer.joblib")
        model_path = os.path.join(model_dir, "model.joblib")
        meta_path = os.path.join(model_dir, "metadata.json")

        if not os.path.isdir(model_dir):
            raise ModelArtifactsError(f"MODEL_DIR does not exist: {model_dir}")
        if not os.path.isfile(vec_path):
            raise ModelArtifactsError(f"Missing vectorizer: {vec_path}")
        if not os.path.isfile(model_path):
            raise ModelArtifactsError(f"Missing model: {model_path}")

        try:
            vectorizer = joblib.load(vec_path)
            classifier = joblib.load(model_path)
        except Exception as e:  # noqa: BLE001
            raise ModelArtifactsError(f"Failed to load joblib artifacts: {e}") from e

        threshold = 0.5
        if os.path.isfile(meta_path):
            try:
                meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
                if isinstance(meta.get("threshold"), (int, float)):
                    threshold = float(meta["threshold"])
                if isinstance(meta.get("model_version"), str):
                    model_version = meta["model_version"]
            except Exception:
                # metadata는 없거나 깨져도 서비스는 계속 동작
                pass

        return BaselineModel(
            vectorizer=vectorizer,
            classifier=classifier,
            threshold=float(threshold),
            model_version=model_version,
            model_dir=model_dir,
        )

    def structure_ocr_text(self, ocr_text: str) -> str:
        # 고정 템플릿: OCR 전체 텍스트는 [DESCRIPTION]에만 삽입
        return (
            "[TITLE] \n"
            "[LOCATION] \n"
            "[EMPLOYMENT_TYPE] \n"
            "[INDUSTRY] \n"
            "[SALARY] \n"
            "[COMPANY_PROFILE] \n"
            f"[DESCRIPTION] {ocr_text}\n"
            "[REQUIREMENTS] \n"
            "[BENEFITS] \n"
        )

    def clean_text(self, structured_text: str, max_chars: int = 20000) -> str:
        return clean_posting_text(structured_text, max_chars=max_chars)

    def predict_proba(self, cleaned_text: str) -> float:
        x = self.vectorizer.transform([cleaned_text])  # type: ignore[attr-defined]
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(x)  # type: ignore[attr-defined]
            value = float(proba[0][1])
        elif hasattr(self.classifier, "decision_function"):
            score = float(self.classifier.decision_function(x)[0])  # type: ignore[attr-defined]
            value = float(1.0 / (1.0 + np.exp(-score)))
        else:
            raise RuntimeError("Loaded classifier has neither predict_proba nor decision_function")

        if not (0.0 <= value <= 1.0):
            value = float(min(1.0, max(0.0, value)))
        return value

    def predict_proba_from_ocr(self, ocr_text: str) -> float:
        structured = self.structure_ocr_text(ocr_text)
        cleaned = self.clean_text(structured, max_chars=20000)
        return self.predict_proba(cleaned)
