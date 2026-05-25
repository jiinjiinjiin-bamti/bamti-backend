from app.inference.risk_scoring import Driver4RiskScorer


def test_driver4_risk_scorer_smooths_normalized_risk_input_with_ewma() -> None:
    scorer = Driver4RiskScorer(
        alpha=0.08,
        activation_threshold=0.5,
        base_interval_seconds=0.1,
    )

    first = scorer.update({"phone_operation": 0.8}, now=10.0)
    second = scorer.update({"phone_operation": 0.8}, now=10.1)
    third = scorer.update({"phone_operation": 0.8}, now=10.2)

    assert first["phone_operation"] == 4.8
    assert second["phone_operation"] == 9.22
    assert third["phone_operation"] == 13.28


def test_driver4_risk_scorer_recovers_when_confidence_is_below_activation_threshold() -> None:
    scorer = Driver4RiskScorer(
        alpha=0.08,
        activation_threshold=0.5,
        base_interval_seconds=0.1,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    recovered = scorer.update({"phone_operation": 0.2}, now=10.1)

    assert recovered["phone_operation"] == 4.42


def test_driver4_risk_scorer_resets_when_frame_time_moves_backward() -> None:
    scorer = Driver4RiskScorer(
        alpha=0.08,
        activation_threshold=0.5,
        base_interval_seconds=0.1,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    scorer.update({"phone_operation": 0.8}, now=10.1)
    reset_score = scorer.update({"phone_operation": 0.8}, now=5.0)

    assert reset_score["phone_operation"] == 4.8


def test_driver4_risk_scorer_clamps_scores_to_100() -> None:
    scorer = Driver4RiskScorer(
        alpha=1.0,
        activation_threshold=0.5,
        base_interval_seconds=0.1,
    )

    scores = scorer.update({"phone_operation": 1.0}, now=10.0)

    assert scores["phone_operation"] == 100.0


def test_driver4_risk_scorer_excludes_non_risk_classes() -> None:
    scorer = Driver4RiskScorer(
        alpha=0.08,
        activation_threshold=0.5,
        excluded_classes={"steering_operation"},
    )

    scores = scorer.update({"phone_operation": 0.8, "steering_operation": 0.9}, now=10.0)

    assert scores == {"phone_operation": 4.8}
    assert scorer.metadata()["excludedClasses"] == ["steering_operation"]
