from dataclasses import dataclass, field
from time import perf_counter


def _clamp_score(score: float) -> float:
    return min(100.0, max(0.0, score))


@dataclass
class Driver4RiskScorer:
    activation_threshold: float = 0.4
    charge_rate: float = 60.0
    decay_rate: float = 20.0
    warning_threshold: float = 70.0
    base_interval_seconds: float = 0.1
    max_delta_seconds: float = 0.5
    reset_gap_seconds: float = 2.0
    excluded_classes: set[str] = field(default_factory=set)
    warning_excluded_classes: set[str] = field(default_factory=set)
    scores: dict[str, float] = field(default_factory=dict)
    last_updated_at: float | None = None

    def update(self, confidences: dict[str, float], now: float | None = None) -> dict[str, float]:
        current_time = perf_counter() if now is None else now
        delta_seconds = self._delta_seconds(current_time)
        if self.last_updated_at is not None and (
            current_time < self.last_updated_at or current_time - self.last_updated_at >= self.reset_gap_seconds
        ):
            self.scores.clear()
            delta_seconds = self.base_interval_seconds
        self.last_updated_at = current_time

        variable_names = set(confidences) | set(self.scores)
        for variable_name in variable_names:
            if variable_name in self.excluded_classes:
                self.scores.pop(variable_name, None)
                continue

            confidence = confidences.get(variable_name, 0.0)
            previous_score = self.scores.get(variable_name, 0.0)
            evidence = self._evidence(confidence)
            if evidence > 0.0:
                next_score = previous_score + self.charge_rate * evidence * delta_seconds
            else:
                next_score = previous_score - self.decay_rate * delta_seconds

            self.scores[variable_name] = round(_clamp_score(next_score), 2)

        return dict(self.scores)

    def warning_state(self) -> dict:
        warning_candidates = {
            variable_name: score
            for variable_name, score in self.scores.items()
            if variable_name not in self.warning_excluded_classes
        }
        if not warning_candidates:
            return {
                "isWarning": False,
                "warningClass": None,
                "warningScore": 0.0,
                "warningThreshold": self.warning_threshold,
            }

        warning_class, warning_score = max(warning_candidates.items(), key=lambda item: item[1])
        return {
            "isWarning": warning_score >= self.warning_threshold,
            "warningClass": warning_class,
            "warningScore": warning_score,
            "warningThreshold": self.warning_threshold,
        }

    def metadata(self) -> dict:
        return {
            "algorithm": "time_buffer_charge_decay_risk_accumulation",
            "scoreRange": [0, 100],
            "activationThreshold": self.activation_threshold,
            "chargeRate": self.charge_rate,
            "decayRate": self.decay_rate,
            "warningThreshold": self.warning_threshold,
            "baseIntervalMs": round(self.base_interval_seconds * 1000),
            "maxDeltaMs": round(self.max_delta_seconds * 1000),
            "resetGapMs": round(self.reset_gap_seconds * 1000),
            "excludedClasses": sorted(self.excluded_classes),
            "warningExcludedClasses": sorted(self.warning_excluded_classes),
        }

    def _delta_seconds(self, now: float) -> float:
        if self.last_updated_at is None:
            return self.base_interval_seconds
        if now < self.last_updated_at:
            return self.base_interval_seconds
        elapsed_seconds = max(0.0, now - self.last_updated_at)
        return min(elapsed_seconds, self.max_delta_seconds)

    def _evidence(self, confidence: float) -> float:
        denominator = max(1.0 - self.activation_threshold, 0.01)
        return min(1.0, max(0.0, confidence - self.activation_threshold) / denominator)
