from dataclasses import dataclass, field
from time import perf_counter


def _clamp_score(score: float) -> float:
    return min(100.0, max(0.0, score))


@dataclass
class Driver4RiskScorer:
    activation_threshold: float = 0.5
    alpha: float = 0.2
    base_interval_seconds: float = 0.1
    excluded_classes: set[str] = field(default_factory=set)
    scores: dict[str, float] = field(default_factory=dict)
    last_updated_at: float | None = None

    def update(self, confidences: dict[str, float], now: float | None = None) -> dict[str, float]:
        current_time = perf_counter() if now is None else now
        if self.last_updated_at is not None and current_time < self.last_updated_at:
            self.scores.clear()
            self.last_updated_at = None
        frame_factor = self._frame_factor(current_time)
        self.last_updated_at = current_time

        for variable_name, confidence in confidences.items():
            if variable_name in self.excluded_classes:
                continue

            previous_score = self.scores.get(variable_name, 0.0)
            normalized_risk_input = self._normalized_risk_input(confidence)
            effective_alpha = self._effective_alpha(frame_factor)
            next_score = effective_alpha * normalized_risk_input + (1.0 - effective_alpha) * previous_score

            self.scores[variable_name] = round(_clamp_score(next_score), 2)

        return dict(self.scores)

    def metadata(self) -> dict:
        return {
            "algorithm": "ema_temporal_risk_score",
            "scoreRange": [0, 100],
            "activationThreshold": self.activation_threshold,
            "alpha": self.alpha,
            "baseIntervalMs": round(self.base_interval_seconds * 1000),
            "excludedClasses": sorted(self.excluded_classes),
        }

    def _frame_factor(self, now: float) -> float:
        if self.last_updated_at is None:
            return 1.0
        elapsed_seconds = max(0.0, now - self.last_updated_at)
        return max(elapsed_seconds / self.base_interval_seconds, 0.01)

    def _effective_alpha(self, frame_factor: float) -> float:
        return min(1.0, max(0.0, 1.0 - ((1.0 - self.alpha) ** frame_factor)))

    def _normalized_risk_input(self, confidence: float) -> float:
        denominator = max(1.0 - self.activation_threshold, 0.01)
        return _clamp_score((max(0.0, confidence - self.activation_threshold) / denominator) * 100.0)
