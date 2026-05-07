from app.alerts.engine import AlertEngine
from app.inference.schemas import InferenceResult


def test_alert_engine_returns_no_alert_for_attentive_result() -> None:
    result = InferenceResult(
        is_distracted=False,
        label="attentive",
        confidence=0.99,
    )

    assert AlertEngine().evaluate(result) == []


def test_alert_engine_returns_alert_for_distraction() -> None:
    result = InferenceResult(
        is_distracted=True,
        label="phone_usage",
        confidence=0.91,
    )

    alerts = AlertEngine().evaluate(result)

    assert len(alerts) == 1
    assert alerts[0].code == "distraction_detected"
    assert alerts[0].severity == "warning"
    assert alerts[0].message == "Driver distraction detected."
