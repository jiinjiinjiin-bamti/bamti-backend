from abc import ABC, abstractmethod

from app.inference.schemas import InferenceResult


class InferenceRunner(ABC):
    @abstractmethod
    async def infer(self, frame: bytes) -> InferenceResult:
        """Run inference for a single encoded frame."""
