from app.inference.runner import InferenceRunner
from app.inference.schemas import InferenceResult


class MockRunner(InferenceRunner):
    async def infer(self, frame: bytes) -> InferenceResult:
        return InferenceResult(
            is_distracted=False,
            label="attentive",
            confidence=0.99,
        )
