from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import vit_b_16

from app.core.config import settings


class BamtiVisionModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = vit_b_16(weights=None, num_classes=num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class LoadedModel:
    def __init__(self, model: nn.Module, class_names: list[str], device: torch.device, model_path: Path) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model_path = model_path


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    model_path = settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    checkpoint: dict[str, Any] = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not all(isinstance(class_name, str) for class_name in class_names):
        raise ValueError("Checkpoint does not include a valid class_names list.")

    model = BamtiVisionModel(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = _resolve_device(settings.model_device)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        class_names=class_names,
        device=device,
        model_path=model_path,
    )
