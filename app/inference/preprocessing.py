from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import functional as transforms

from app.core.config import settings


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def image_bytes_to_tensor(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transforms.resize(image, [settings.model_input_size, settings.model_input_size])
    tensor = transforms.to_tensor(image)
    tensor = transforms.normalize(tensor, mean=imagenet_mean, std=imagenet_std)
    return tensor.unsqueeze(0).to(device)


def raw_rgb_bytes_to_tensor(raw_bytes: bytes, width: int, height: int, device: torch.device) -> torch.Tensor:
    expected_size = width * height * 3
    if len(raw_bytes) != expected_size:
        raise ValueError(f"Raw RGB frame size must be {expected_size} bytes for {width}x{height}.")
    if width != settings.model_input_size or height != settings.model_input_size:
        raise ValueError(f"Raw RGB frames must be {settings.model_input_size}x{settings.model_input_size}.")

    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    tensor = tensor.reshape(height, width, 3).permute(2, 0, 1).float().div(255.0)
    tensor = transforms.normalize(tensor, mean=imagenet_mean, std=imagenet_std)
    return tensor.unsqueeze(0).to(device)
