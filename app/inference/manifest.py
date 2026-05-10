from app.inference.runner import InferenceRunner
from app.inference.schemas import ModelManifest


def get_runner(name: str = "bamti-torch") -> InferenceRunner:
    if name in {"bamti-torch", "torch"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner()
    if name in {"bamti-torch-compiled", "torch-compiled"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner(use_compiled_model=True)
    raise ValueError(f"Unsupported runner: {name}")


def get_model_manifest(name: str = "bamti-torch") -> ModelManifest:
    return get_runner(name).manifest()
