from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import vit_b_16

from app.core.config import settings
from app.inference.class_mapping import ServiceDetectionClass, raw_action_class_names, service_detection_classes


class BamtiVisionModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = vit_b_16(weights=None, num_classes=num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class BamtiTimmVisionModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("The timm package is required for the custom ViT checkpoint.") from exc

        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    class_names: list[str]
    device: torch.device
    model_path: Path
    compiled: bool
    architecture: str
    service_classes: tuple[ServiceDetectionClass, ...] = ()
    raw_class_names: tuple[str, ...] = ()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _configure_torch_threads() -> None:
    if settings.torch_num_threads > 0:
        torch.set_num_threads(settings.torch_num_threads)


def _compile_model(model: nn.Module) -> nn.Module:
    return torch.compile(
        model,
        backend=settings.torch_compile_backend,
        mode=settings.torch_compile_mode,
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary or state_dict.")

    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint

    raise ValueError("Checkpoint does not include a valid model state_dict.")


def _strip_common_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized_state_dict = dict(state_dict)
    for prefix in ("module.", "model."):
        if any(key.startswith(prefix) for key in normalized_state_dict):
            normalized_state_dict = {
                key.removeprefix(prefix): value
                for key, value in normalized_state_dict.items()
            }
    return normalized_state_dict


def _is_timm_custom_vit_state_dict(state_dict: dict[str, Any]) -> bool:
    return any(key.startswith("backbone.patch_embed.") or key.startswith("backbone.blocks.") for key in state_dict)


def _load_timm_custom_vit_model(state_dict: dict[str, Any], compiled: bool, model_path: Path) -> LoadedModel:
    model = BamtiTimmVisionModel(num_classes=len(raw_action_class_names))
    model.load_state_dict(state_dict, strict=True)

    device = _resolve_device(settings.model_device)
    model.to(device)
    model.eval()
    if compiled:
        model = _compile_model(model)

    return LoadedModel(
        model=model,
        class_names=[detection_class.variable_name for detection_class in service_detection_classes],
        device=device,
        model_path=model_path,
        compiled=compiled,
        architecture="timm_vit_b_16_custom",
        service_classes=service_detection_classes,
        raw_class_names=raw_action_class_names,
    )


def _load_torchvision_vit_model(checkpoint: dict[str, Any], state_dict: dict[str, Any], compiled: bool, model_path: Path) -> LoadedModel:
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not all(isinstance(class_name, str) for class_name in class_names):
        raise ValueError("Checkpoint does not include a valid class_names list.")

    model = BamtiVisionModel(num_classes=len(class_names))
    model.load_state_dict(state_dict, strict=True)

    device = _resolve_device(settings.model_device)
    model.to(device)
    model.eval()
    if compiled:
        model = _compile_model(model)

    return LoadedModel(
        model=model,
        class_names=class_names,
        device=device,
        model_path=model_path,
        compiled=compiled,
        architecture="torchvision_vit_b_16",
    )


@lru_cache(maxsize=2)
def load_model(compiled: bool = False) -> LoadedModel:
    _configure_torch_threads()

    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    checkpoint: Any = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = _strip_common_prefixes(_extract_state_dict(checkpoint))

    if _is_timm_custom_vit_state_dict(state_dict):
        return _load_timm_custom_vit_model(state_dict, compiled, model_path)

    if not isinstance(checkpoint, dict):
        raise ValueError("Torchvision checkpoint must be a dictionary.")
    return _load_torchvision_vit_model(checkpoint, state_dict, compiled, model_path)
