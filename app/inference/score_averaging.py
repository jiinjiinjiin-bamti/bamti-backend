import time
from collections import deque

from app.inference.schemas import DetectionScore


class RollingScoreAverager:
    def __init__(self, window_seconds: float = 1.0) -> None:
        self.window_seconds = window_seconds
        self.samples: deque[tuple[float, dict[str, float]]] = deque()

    def average(self, detections: list[DetectionScore]) -> list[DetectionScore]:
        now = time.perf_counter()
        cutoff = now - self.window_seconds
        self.samples.append((now, {detection.variable_name: detection.score for detection in detections}))

        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _, score_by_variable in self.samples:
            for variable_name, score in score_by_variable.items():
                sums[variable_name] = sums.get(variable_name, 0.0) + score
                counts[variable_name] = counts.get(variable_name, 0) + 1

        return [
            detection.model_copy(
                update={
                    "score": round(sums.get(detection.variable_name, detection.score) / counts.get(detection.variable_name, 1), 4),
                },
            )
            for detection in detections
        ]
