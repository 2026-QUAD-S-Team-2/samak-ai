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
import logging
import re
from typing import Any

import joblib
import numpy as np

from app.ml.message_risk_rules import extract_risk_signals
from app.ml.text_utils import clean_posting_text


class ModelArtifactsError(RuntimeError):
    pass


logger = logging.getLogger(__name__)

_TEMPLATE_TOKEN_RE = re.compile(r"[\[\]]")
_NUMERIC_ONLY_RE = re.compile(r"^\d+$")
_TEMPLATE_WORDS = {
    "title",
    "location",
    "employment_type",
    "industry",
    "salary",
    "company_profile",
    "description",
    "requirements",
    "benefits",
}

# 설명 품질 개선용 stopwords/blacklist
_BASE_STOPWORDS = {
    "and",
    "or",
    "to",
    "in",
    "for",
    "of",
    "the",
    "a",
    "an",
    "is",
    "are",
    "with",
    "as",
    "on",
    "at",
    "by",
    "be",
    "this",
    "that",
    "it",
    "we",
    "you",
    "your",
    "our",
    "from",
}

try:  # sklearn이 있으면 더 넓은 stopwords를 사용
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOP  # type: ignore

    _STOPWORDS = set(_BASE_STOPWORDS) | set(_SK_STOP)
except Exception:  # noqa: BLE001
    _STOPWORDS = set(_BASE_STOPWORDS)

_EXPLANATION_BLACKLIST = {
    "assistant",
    "administrative",
    "data",
    "entry",
    "role",
    "position",
    "job",
    "full-time",
    "part-time",
    "team",
    "work",
    "office",
}


def _is_valid_phrase(
    phrase: str,
    *,
    enforce_stopwords: bool = True,
    enforce_blacklist: bool = True,
) -> bool:
    p = (phrase or "").strip()
    # 너무 짧은 토큰은 제외(필터 완화)
    if len(p) < 2:
        return False
    if _NUMERIC_ONLY_RE.match(p):
        return False
    # 템플릿 토큰([TITLE] 등) 또는 템플릿 섹션 단어는 제외
    if _TEMPLATE_TOKEN_RE.search(p):
        return False
    tokens = p.split()
    if any(t in _TEMPLATE_WORDS for t in tokens):
        return False
    # 너무 단순한 토큰 제외
    if p in {"amp", "lt", "gt"}:
        return False

    ltokens = [t.lower() for t in tokens]
    # stopwords 제거 (설명에서는 의미가 거의 없음)
    if enforce_stopwords:
        if len(ltokens) == 1 and ltokens[0] in _STOPWORDS:
            return False
        if len(ltokens) >= 2 and all(t in _STOPWORDS for t in ltokens):
            return False

    # 설명용 blacklist (일반적인 직무/구조 단어는 제외)
    if enforce_blacklist:
        if any(t in _EXPLANATION_BLACKLIST for t in ltokens):
            return False

    return True


def explain_tfidf_lr(text: str, vectorizer: Any, model: Any, top_k: int = 3) -> dict[str, list[str]]:
    """
    TF-IDF + LogisticRegression(biary)에서 feature contribution 기반으로 top phrase를 추출합니다.

    반환:
    - topRiskPhrases: contribution > 0 상위 top_k
    - topTrustPhrases: contribution < 0 절댓값 상위 top_k
    """
    if top_k <= 0:
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    if not hasattr(vectorizer, "transform") or not hasattr(vectorizer, "get_feature_names_out"):
        return {"topRiskPhrases": [], "topTrustPhrases": []}
    if not hasattr(model, "coef_"):
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    try:
        coef = np.asarray(model.coef_)[0]
    except Exception:  # noqa: BLE001
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:  # noqa: BLE001
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    if len(feature_names) != int(len(coef)):
        logger.debug(
            "explain_tfidf_lr: shape mismatch len(feature_names)=%d len(coef)=%d",
            len(feature_names),
            len(coef),
        )
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    X = vectorizer.transform([text])
    if X.shape[1] != len(coef):
        logger.debug("explain_tfidf_lr: X.shape=%s len(coef)=%d", X.shape, len(coef))
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    # 계산 방식 확실히: sparse multiply → dense 1D로 변환
    try:
        X = X.tocsr()  # type: ignore[attr-defined]
        contrib_dense = X.multiply(coef).toarray()[0]  # type: ignore[operator]
    except Exception as e:  # noqa: BLE001
        logger.debug("explain_tfidf_lr: multiply/toarray failed: %s", e)
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    nz = getattr(X, "indices", None)
    if nz is None:
        nz = np.nonzero(contrib_dense)[0]

    nz = np.asarray(nz, dtype=int)
    if nz.size == 0:
        logger.debug("explain_tfidf_lr: X.nnz=0 (no vocab match)")
        return {"topRiskPhrases": [], "topTrustPhrases": []}

    # debug 로그(원인 규명)
    try:
        max_abs = float(np.max(np.abs(contrib_dense[nz])))
    except Exception:
        max_abs = 0.0
    logger.debug(
        "explain_tfidf_lr: X.shape=%s X.nnz=%s len(feature_names)=%d len(coef)=%d max_abs_contrib=%.6g",
        X.shape,
        getattr(X, "nnz", "n/a"),
        len(feature_names),
        len(coef),
        max_abs,
    )

    # 필터 적용 전 후보(상위 10개) 추출
    nz_vals = contrib_dense[nz]
    pos_mask = nz_vals > 0
    neg_mask = nz_vals < 0
    pos_idx = nz[pos_mask]
    neg_idx = nz[neg_mask]

    pos_sorted = pos_idx[np.argsort(contrib_dense[pos_idx])[::-1]] if pos_idx.size else np.array([], dtype=int)
    neg_sorted = neg_idx[np.argsort(np.abs(contrib_dense[neg_idx]))[::-1]] if neg_idx.size else np.array([], dtype=int)

    top_pos_raw = [str(feature_names[i]) for i in pos_sorted[:10]]
    top_neg_raw = [str(feature_names[i]) for i in neg_sorted[:10]]
    logger.debug("explain_tfidf_lr: top_pos_raw=%s", top_pos_raw)
    logger.debug("explain_tfidf_lr: top_neg_raw=%s", top_neg_raw)

    # 필터 적용 후 top_k 구성
    def _pick(
        sorted_indices: np.ndarray,
        *,
        top_k: int,
        mode: str,
        enforce_stopwords: bool,
        enforce_blacklist: bool,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        # 3) bigram 우선 정책: (2-gram) → (unigram) 순으로 채움
        def _iter_indices(prefer_bigrams: bool):
            for i in sorted_indices:
                phrase = str(feature_names[int(i)]).strip()
                if not phrase or phrase in seen:
                    continue
                tokens = phrase.split()
                is_bigram = len(tokens) == 2
                if prefer_bigrams and not is_bigram:
                    continue
                if (not prefer_bigrams) and is_bigram:
                    continue
                yield phrase

        for phrase in _iter_indices(prefer_bigrams=True):
            seen.add(phrase)
            if _is_valid_phrase(
                phrase,
                enforce_stopwords=enforce_stopwords,
                enforce_blacklist=enforce_blacklist,
            ):
                out.append(phrase)
            if len(out) >= top_k:
                return out

        for phrase in _iter_indices(prefer_bigrams=False):
            seen.add(phrase)
            if _is_valid_phrase(
                phrase,
                enforce_stopwords=enforce_stopwords,
                enforce_blacklist=enforce_blacklist,
            ):
                out.append(phrase)
            if len(out) >= top_k:
                return out

        logger.debug(
            "explain_tfidf_lr: pick(%s) produced %d (stopwords=%s blacklist=%s)",
            mode,
            len(out),
            enforce_stopwords,
            enforce_blacklist,
        )
        return out

    # 4) 후보가 비면 fallback: 필터 일부 완화해서 최소 1~3개를 시도
    def _pick_with_fallback(sorted_indices: np.ndarray, *, top_k: int, mode: str) -> list[str]:
        # strict
        out = _pick(
            sorted_indices,
            top_k=top_k,
            mode=mode,
            enforce_stopwords=True,
            enforce_blacklist=True,
        )
        if out:
            return out
        # relax: blacklist만 유지
        out = _pick(
            sorted_indices,
            top_k=top_k,
            mode=mode,
            enforce_stopwords=True,
            enforce_blacklist=False,
        )
        if out:
            return out
        # relax: stopwords만 유지
        out = _pick(
            sorted_indices,
            top_k=top_k,
            mode=mode,
            enforce_stopwords=False,
            enforce_blacklist=True,
        )
        if out:
            return out
        # relax: 둘 다 해제(템플릿/숫자 필터만 유지)
        out = _pick(
            sorted_indices,
            top_k=top_k,
            mode=mode,
            enforce_stopwords=False,
            enforce_blacklist=False,
        )
        return out

    top_risk = _pick_with_fallback(pos_sorted, top_k=top_k, mode="pos")
    top_trust = _pick_with_fallback(neg_sorted, top_k=top_k, mode="neg")
    logger.debug(
        "explain_tfidf_lr: after_filter risk=%d trust=%d",
        len(top_risk),
        len(top_trust),
    )

    return {"topRiskPhrases": top_risk[:top_k], "topTrustPhrases": top_trust[:top_k]}


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

    def get_cleaned_input_from_ocr(self, ocr_text: str) -> str:
        structured = self.structure_ocr_text(ocr_text)
        return self.clean_text(structured, max_chars=20000)

    def risk_signals_from_ocr(self, ocr_text: str, *, top_k: int = 3) -> list[str]:
        # 입력 텍스트는 "모델에 실제로 들어간 최종 텍스트(inputCleaned)" 기준
        cleaned = self.get_cleaned_input_from_ocr(ocr_text)
        return extract_risk_signals(cleaned, top_k=top_k)
