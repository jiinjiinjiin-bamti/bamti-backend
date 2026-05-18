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
