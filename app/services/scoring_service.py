from __future__ import annotations

"""
점수/레벨 계산 로직.

- ML 확률 → riskScore / riskLevel
- UI 노출용 레벨/라벨 계산
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionScores:
    risk_score: int
    # 모델 정책(평가용) riskLevel: threshold 기반
    model_risk_level: str  # LOW|MEDIUM|HIGH
    # UI 노출용 레벨/라벨: 확률 구간 기반
    ui_risk_level: str  # LOW|MEDIUM|HIGH|UNKNOWN
    trust_score: int
    ui_trust_label: str  # Good|Warning|Danger


def score_prediction(fraud_probability: float, threshold_used: float) -> PredictionScores:
    p = float(max(0.0, min(1.0, fraud_probability)))
    th = float(max(0.0, min(1.0, threshold_used)))

    # 반올림 규칙(정수):
    # - riskScore(=fraudPercent) = round(100 * p)
    # - trustScore = 100 - riskScore
    fraud_percent = int(round(100 * p))
    risk_score = fraud_percent
    # 1) 모델 정책(평가용): threshold 기반
    model_high_th = max(th, 0.7)
    if p >= model_high_th:
        model_risk_level = "HIGH"
    elif p >= th:
        model_risk_level = "MEDIUM"
    else:
        model_risk_level = "LOW"

    trust_score = int(max(0, min(100, 100 - fraud_percent)))

    # 2) UI 정책(고정): 확률 구간 기반 (riskLevel과 trustLabel이 모순되지 않게 함께 결정)
    if p < 0.35:
        ui_risk_level = "LOW"
        ui_trust_label = "Good"
    elif p < 0.65:
        ui_risk_level = "MEDIUM"
        ui_trust_label = "Warning"
    else:
        ui_risk_level = "HIGH"
        ui_trust_label = "Danger"

    return PredictionScores(
        risk_score=risk_score,
        model_risk_level=model_risk_level,
        ui_risk_level=ui_risk_level,
        trust_score=trust_score,
        ui_trust_label=ui_trust_label,
    )
