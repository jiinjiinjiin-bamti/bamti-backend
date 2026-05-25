from app.inference.risk_scoring import Driver4RiskScorer


def test_driver4_risk_scorer_charges_risk_score_from_sustained_evidence() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
    )

    first = scorer.update({"phone_operation": 0.8}, now=10.0)
    second = scorer.update({"phone_operation": 0.8}, now=10.1)
    third = scorer.update({"phone_operation": 0.8}, now=10.2)

    assert first["phone_operation"] == 2.1
    assert second["phone_operation"] == 4.2
    assert third["phone_operation"] == 6.3


def test_driver4_risk_scorer_decays_when_confidence_is_below_activation_threshold() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    recovered = scorer.update({"phone_operation": 0.2}, now=10.1)

    assert recovered["phone_operation"] == 0.1


def test_driver4_risk_scorer_resets_when_frame_time_moves_backward() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    scorer.update({"phone_operation": 0.8}, now=10.1)
    reset_score = scorer.update({"phone_operation": 0.8}, now=5.0)

    assert reset_score["phone_operation"] == 2.1


def test_driver4_risk_scorer_clamps_scores_to_100() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=1000.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
    )

    scores = scorer.update({"phone_operation": 1.0}, now=10.0)

    assert scores["phone_operation"] == 100.0


def test_driver4_risk_scorer_scores_steering_but_excludes_it_from_warning_candidate() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        warning_threshold=70.0,
        base_interval_seconds=1.0,
        warning_excluded_classes={"steering_operation"},
    )

    scores = scorer.update({"phone_operation": 0.8, "steering_operation": 1.0}, now=10.0)

    assert scores == {"phone_operation": 21.0, "steering_operation": 35.0}
    assert scorer.warning_state() == {
        "isWarning": False,
        "warningClass": "phone_operation",
        "warningScore": 21.0,
        "warningThreshold": 70.0,
    }
    assert scorer.metadata()["warningExcludedClasses"] == ["steering_operation"]


def test_driver4_risk_scorer_caps_delta_time_to_prevent_gap_spikes() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
        max_delta_seconds=0.5,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    scores = scorer.update({"phone_operation": 0.8}, now=11.5)

    assert scores["phone_operation"] == 12.6


def test_driver4_risk_scorer_resets_after_large_frame_gap() -> None:
    scorer = Driver4RiskScorer(
        activation_threshold=0.5,
        charge_rate=35.0,
        decay_rate=20.0,
        base_interval_seconds=0.1,
        reset_gap_seconds=2.0,
    )

    scorer.update({"phone_operation": 0.8}, now=10.0)
    scorer.update({"phone_operation": 0.8}, now=10.1)
    scores = scorer.update({"phone_operation": 0.8}, now=12.5)

    assert scores["phone_operation"] == 2.1
