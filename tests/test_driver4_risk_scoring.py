from app.inference.risk_scoring import Driver4RiskScorer


def test_driver4_risk_scorer_accumulates_class_scores_with_elapsed_time() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        base_interval_seconds=0.1,
        decay=0.95,
        recovery_decay=0.85,
        score_scale=10.0,
        weights={"phone_operation": 1.3},
    )

    first = scorer.update({"phone_operation": 0.8}, now=10.0)
    second = scorer.update({"phone_operation": 0.8}, now=10.1)
    third = scorer.update({"phone_operation": 0.8}, now=10.2)

    assert first["phone_operation"] == 10.4
    assert second["phone_operation"] == 20.28
    assert third["phone_operation"] == 29.67


def test_driver4_risk_scorer_recovers_when_confidence_is_below_activation() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        base_interval_seconds=0.1,
        decay=0.95,
        recovery_decay=0.85,
        score_scale=10.0,
        weights={"phone_operation": 1.3},
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    recovered = scorer.update({"phone_operation": 0.2}, now=10.1)

    assert recovered["phone_operation"] == 8.84


def test_driver4_risk_scorer_clamps_scores_to_100() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        base_interval_seconds=0.1,
        decay=0.95,
        recovery_decay=0.85,
        score_scale=200.0,
        weights={"phone_operation": 1.3},
    )

    scores = scorer.update({"phone_operation": 1.0}, now=10.0)

    assert scores["phone_operation"] == 100.0
