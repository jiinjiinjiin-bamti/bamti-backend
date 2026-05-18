from abc import ABC, abstractmethod

from app.inference.schemas import InferenceResult, ModelManifest


class InferenceRunner(ABC):
    @abstractmethod
    async def infer(self, frame: bytes) -> InferenceResult:
        """Run inference for a single encoded frame."""

    @abstractmethod
    def manifest(self) -> ModelManifest:
        """Return the active model manifest."""
