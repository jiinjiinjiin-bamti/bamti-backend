from app.inference.runner import InferenceRunner
from app.inference.schemas import ModelManifest


def get_runner(name: str = "bamti-torch") -> InferenceRunner:
    if name in {"bamti-torch", "torch"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner()
    raise ValueError(f"Unsupported runner: {name}")


def get_model_manifest(name: str = "bamti-torch") -> ModelManifest:
    return get_runner(name).manifest()
