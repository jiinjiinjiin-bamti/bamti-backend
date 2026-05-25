from dataclasses import dataclass, field
from time import perf_counter


def _clamp_score(score: float) -> float:
    return min(100.0, max(0.0, score))


@dataclass
class Driver4RiskScorer:
    activation_threshold: float = 0.5
    base_interval_seconds: float = 0.1
    decay: float = 0.95
    recovery_decay: float = 0.85
    score_scale: float = 10.0
    weights: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    last_updated_at: float | None = None

    def update(self, confidences: dict[str, float], now: float | None = None) -> dict[str, float]:
        current_time = perf_counter() if now is None else now
        frame_factor = self._frame_factor(current_time)
        self.last_updated_at = current_time

        for variable_name, confidence in confidences.items():
            previous_score = self.scores.get(variable_name, 0.0)
            if confidence >= self.activation_threshold:
                next_score = previous_score * (self.decay**frame_factor)
                next_score += confidence * self.score_scale * self.weights.get(variable_name, 1.0) * frame_factor
            else:
                next_score = previous_score * (self.recovery_decay**frame_factor)

            self.scores[variable_name] = round(_clamp_score(next_score), 2)

        return dict(self.scores)

    def metadata(self) -> dict:
        return {
            "algorithm": "ema_temporal_risk_score",
            "scoreRange": [0, 100],
            "activationThreshold": self.activation_threshold,
            "baseIntervalMs": round(self.base_interval_seconds * 1000),
            "decay": self.decay,
            "recoveryDecay": self.recovery_decay,
            "scoreScale": self.score_scale,
            "weights": self.weights,
        }

    def _frame_factor(self, now: float) -> float:
        if self.last_updated_at is None:
            return 1.0
        elapsed_seconds = max(0.0, now - self.last_updated_at)
        return max(elapsed_seconds / self.base_interval_seconds, 0.01)
