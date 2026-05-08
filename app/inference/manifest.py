from app.inference.mock_runner import MockRunner
from app.inference.runner import InferenceRunner
from app.inference.schemas import DetectionClass, ModelManifest


MOCK_MODEL_MANIFEST = ModelManifest(
    model_version="mock-v1",
    classes=(
        DetectionClass(
            variable_name="attentive",
            class_id="attentive",
            display_name="Attentive",
            description="Normal attentive driving state.",
            threshold=0.5,
        ),
        DetectionClass(
            variable_name="distracted",
            class_id="distracted",
            display_name="Distracted",
            description="Driver distraction state.",
            threshold=0.5,
        ),
    ),
)


def get_runner(name: str = "mock") -> InferenceRunner:
    if name == "mock":
        return MockRunner()
    raise ValueError(f"Unsupported runner: {name}")


def get_model_manifest(name: str = "mock") -> ModelManifest:
    if name == "mock":
        return MOCK_MODEL_MANIFEST
    raise ValueError(f"Unsupported model manifest: {name}")
