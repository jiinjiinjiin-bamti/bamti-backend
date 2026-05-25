from app.core.config import settings
from app.inference.risk_scoring import Driver4RiskScorer


def create_driver4_v7_risk_scorer() -> Driver4RiskScorer:
    return Driver4RiskScorer(
        alpha=settings.driver4_v7_alpha,
        activation_threshold=settings.driver4_v7_activation_threshold,
    )
