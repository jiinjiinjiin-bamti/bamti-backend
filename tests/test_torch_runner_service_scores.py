from pathlib import Path

import torch

from app.inference.class_mapping import raw_action_class_names, service_detection_classes
from app.inference.model_loader import LoadedModel
from app.inference.torch_runner import BamtiTorchRunner


def test_service_detection_scores_use_max_raw_class_score() -> None:
    loaded_model = LoadedModel(
        model=torch.nn.Identity(),
        class_names=[detection_class.variable_name for detection_class in service_detection_classes],
        device=torch.device("cpu"),
        model_path=Path("exp04_pseudo_ir_aug.pth"),
        compiled=False,
        architecture="timm_vit_b_16_custom",
        service_classes=service_detection_classes,
        raw_class_names=raw_action_class_names,
    )
    scores = torch.tensor(
        [
            0.10,  # A1
            0.20,  # A2
            0.31,  # A3
            0.32,  # A4
            0.11,  # A5
            0.12,  # A6
            0.91,  # A7
            0.13,  # A8
            0.14,  # A9
            0.41,  # A10
            0.42,  # A11
            0.50,  # A12
            0.21,  # A13
            0.22,  # A14
            0.60,  # A15
            0.23,  # A16
        ],
    )

    detections = BamtiTorchRunner()._detections_from_scores(loaded_model, scores)
    scores_by_variable = {detection.variable_name: detection.score for detection in detections}

    assert scores_by_variable["normal_driving"] == 0.10
    assert scores_by_variable["phone_use"] == 0.91
    assert scores_by_variable["vehicle_device_operation"] == 0.32
    assert scores_by_variable["face_action"] == 0.23
    assert scores_by_variable["distraction"] == 0.42
    assert scores_by_variable["drowsiness"] == 0.50
    assert scores_by_variable["rear_seat_interaction"] == 0.60
