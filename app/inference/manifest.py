from app.core.config import settings
from app.inference.runner import InferenceRunner
from app.inference.schemas import ModelManifest


def get_runner(name: str = "bamti-torch") -> InferenceRunner:
    if name in {"bamti-torch", "torch"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner()
    if name in {"bamti-torch-compiled", "torch-compiled"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner(use_compiled_model=True)
    if name in {"bamti-torch-debug-raw", "torch-debug-raw"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner(include_debug_raw_detections=True)
    if name in {"aihub-torch", "torch-aihub"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner(model_path=settings.aihub_model_path)
    if name in {"aihub-torch-compiled", "torch-aihub-compiled"}:
        from app.inference.torch_runner import BamtiTorchRunner

        return BamtiTorchRunner(use_compiled_model=True, model_path=settings.aihub_model_path)
    raise ValueError(f"Unsupported runner: {name}")


def get_model_manifest(name: str = "bamti-torch") -> ModelManifest:
    return get_runner(name).manifest()
