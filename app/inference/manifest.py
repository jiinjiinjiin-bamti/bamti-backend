from app.inference.mock_runner import MockRunner
from app.inference.runner import InferenceRunner


def get_runner(name: str = "mock") -> InferenceRunner:
    if name == "mock":
        return MockRunner()
    raise ValueError(f"Unsupported runner: {name}")
