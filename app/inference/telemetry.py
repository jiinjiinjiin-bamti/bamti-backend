import time
from collections import deque
from threading import Lock


class ProcessingFpsCounter:
    def __init__(self, window_seconds: float = 1.0) -> None:
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()
        self.lock = Lock()

    def mark_processed(self) -> float:
        now = time.perf_counter()
        cutoff = now - self.window_seconds

        with self.lock:
            self.timestamps.append(now)
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()
            return len(self.timestamps) / self.window_seconds


processing_fps_counter = ProcessingFpsCounter()
