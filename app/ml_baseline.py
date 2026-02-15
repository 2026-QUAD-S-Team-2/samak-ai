from __future__ import annotations

"""
Baseline ML (TF-IDF + Logistic Regression) 추론 로직.

모델 아티팩트는 아래 파일 2개가 필수입니다.
- {MODEL_DIR}/vectorizer.joblib  (예: TfidfVectorizer)
- {MODEL_DIR}/model.joblib       (예: LogisticRegression)

metadata.json은 선택이며, notes 같은 문자열을 넣어 응답에 전달할 수 있습니다.
"""

import html as html_lib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np


class ModelLoadError(RuntimeError):
    pass


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}")
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\d{6,}")


def clean_posting_text(text: str, *, max_chars: int) -> str:
    # 학습/추론에서 동일한 방향으로 정규화:
    # 1) HTML 태그 제거 2) HTML 엔티티 unescape 3) URL/이메일 토큰 치환
    # 4) 너무 긴 숫자 정리 5) 소문자 6) 공백 정리 7) 길이 제한
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = html_lib.unescape(cleaned)
    cleaned = _URL_RE.sub(" <URL> ", cleaned)
    cleaned = _EMAIL_RE.sub(" <EMAIL> ", cleaned)
    cleaned = _PHONE_RE.sub(" <PHONE> ", cleaned)
    cleaned = _LONG_NUM_RE.sub(" <LONGNUM> ", cleaned)
    cleaned = cleaned.lower()
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def is_structured_posting_text(text: str) -> bool:
    """
    입력 텍스트가 이미 템플릿 형태로 구조화되어 있는지(대략) 판정합니다.

    예: [TITLE], [DESCRIPTION] 같은 섹션 헤더가 포함되어 있으면 True로 간주.
    (대소문자 차이는 무시)
    """
    t = text.lower()
    return ("[title]" in t) or ("[description]" in t)


def make_structured_posting_text(text: str) -> str:
    """
    백엔드가 "필드 분리" 대신 전체 텍스트만 주는 MVP 상황에서,
    학습과 동일한 템플릿 형태로 감싸서 문서를 만듭니다.

    이미 구조화된 입력이면 중복 감싸기를 방지하기 위해 그대로 반환합니다.
    """
    if is_structured_posting_text(text):
        return text
    return (
        "[TITLE] \n"
        "[LOCATION] \n"
        "[EMPLOYMENT_TYPE] \n"
        "[INDUSTRY] \n"
        "[SALARY] \n"
        "[COMPANY_PROFILE] \n"
        f"[DESCRIPTION] {text}\n"
        "[REQUIREMENTS] \n"
        "[BENEFITS] \n"
    )


@dataclass(frozen=True)
class BaselineArtifacts:
    vectorizer: Any
    classifier: Any
    metadata: dict[str, Any] | None


def load_baseline_artifacts(model_dir: str) -> BaselineArtifacts:
    # 디렉토리/파일 존재 여부를 먼저 확인해서 에러 원인을 명확히 합니다.
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    model_path = os.path.join(model_dir, "model.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.isdir(model_dir):
        raise ModelLoadError(f"MODEL_DIR does not exist: {model_dir}")
    if not os.path.isfile(vectorizer_path):
        raise ModelLoadError(f"Missing vectorizer: {vectorizer_path}")
    if not os.path.isfile(model_path):
        raise ModelLoadError(f"Missing model: {model_path}")

    try:
        # joblib로 저장한 sklearn 객체 로딩
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(model_path)
    except Exception as e:  # noqa: BLE001
        raise ModelLoadError(f"Failed to load joblib artifacts: {e}") from e

    metadata = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"Failed to read metadata.json: {e}") from e

    return BaselineArtifacts(vectorizer=vectorizer, classifier=classifier, metadata=metadata)


@dataclass(frozen=True)
class TfidfLogRegPredictor:
    vectorizer: object
    classifier: object
    notes: str | None = None
    threshold: float = 0.5
    high_precision_threshold: float | None = None

    @staticmethod
    def from_dir(model_dir: str) -> "TfidfLogRegPredictor":
        artifacts = load_baseline_artifacts(model_dir)
        notes = None
        threshold = 0.5
        high_precision_threshold = None
        if artifacts.metadata and "notes" in artifacts.metadata:
            notes = str(artifacts.metadata["notes"])
        if artifacts.metadata and isinstance(artifacts.metadata.get("threshold"), (int, float)):
            threshold = float(artifacts.metadata["threshold"])
        # 선택: evaluate에서 저장한 정책 임계치(precision 기반) 로딩
        if artifacts.metadata and isinstance(artifacts.metadata.get("threshold_policies"), dict):
            policies = artifacts.metadata["threshold_policies"]
            if isinstance(policies.get("threshold_precision_90"), (int, float)):
                high_precision_threshold = float(policies["threshold_precision_90"])
        return TfidfLogRegPredictor(
            vectorizer=artifacts.vectorizer,
            classifier=artifacts.classifier,
            notes=notes,
            threshold=threshold,
            high_precision_threshold=high_precision_threshold,
        )

    def predict_proba(self, text: str) -> float:
        # sklearn vectorizer는 "문서 리스트" 형태 입력을 기대합니다.
        x = self.vectorizer.transform([text])  # type: ignore[attr-defined]
        if hasattr(self.classifier, "predict_proba"):
            # LogisticRegression 등 대부분의 분류기가 제공
            proba = self.classifier.predict_proba(x)  # type: ignore[attr-defined]
            value = float(proba[0][1])
        elif hasattr(self.classifier, "decision_function"):
            # decision_function만 있는 경우는 sigmoid로 0~1 값으로 변환
            score = float(self.classifier.decision_function(x)[0])  # type: ignore[attr-defined]
            value = float(1.0 / (1.0 + np.exp(-score)))
        else:
            raise RuntimeError("Loaded classifier has neither predict_proba nor decision_function")

        # 안전장치: 범위를 벗어나면 clamp
        if not (0.0 <= value <= 1.0):
            value = float(min(1.0, max(0.0, value)))
        return value
