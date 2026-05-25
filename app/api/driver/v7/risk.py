from app.core.config import settings
from app.inference.risk_scoring import Driver4RiskScorer


def create_driver4_v7_risk_scorer() -> Driver4RiskScorer:
    return Driver4RiskScorer(
        activation_threshold=settings.driver4_v7_activation_threshold,
        decay=settings.driver4_v7_decay,
        recovery_decay=settings.driver4_v7_recovery_decay,
        score_scale=settings.driver4_v7_score_scale,
        weights={
            "body_touching": settings.driver4_v7_weight_body_touching,
            "distraction": settings.driver4_v7_weight_distraction,
            "phone_operation": settings.driver4_v7_weight_phone_operation,
            "steering_operation": settings.driver4_v7_weight_steering_operation,
        },
    )
