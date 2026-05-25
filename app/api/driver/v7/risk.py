from app.core.config import settings
from app.inference.risk_scoring import Driver4RiskScorer


def create_driver4_v7_risk_scorer() -> Driver4RiskScorer:
    return Driver4RiskScorer(
        activation_threshold=settings.driver4_v7_activation_threshold,
        charge_rate=settings.driver4_v7_charge_rate,
        decay_rate=settings.driver4_v7_decay_rate,
        warning_threshold=settings.driver4_v7_warning_threshold,
        max_delta_seconds=settings.driver4_v7_max_delta_seconds,
        reset_gap_seconds=settings.driver4_v7_reset_gap_seconds,
        warning_excluded_classes={"steering_operation"},
    )
