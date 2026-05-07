from app.inference.schemas import InferenceResult
from pydantic import BaseModel


class Alert(BaseModel):
    code: str
    severity: str
    message: str


class AlertEngine:
    def evaluate(self, result: InferenceResult) -> list[Alert]:
        if result.is_distracted:
            return [
                Alert(
                    code="distraction_detected",
                    severity="warning",
                    message="Driver distraction detected.",
                )
            ]
        return []
