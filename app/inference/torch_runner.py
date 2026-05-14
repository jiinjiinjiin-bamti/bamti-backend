import asyncio
import time
from pathlib import Path

import torch

from app.core.config import settings
from app.inference.class_mapping import ServiceDetectionClass
from app.inference.model_loader import LoadedModel, load_model
from app.inference.preprocessing import image_bytes_to_tensor
from app.inference.runner import InferenceRunner
from app.inference.schemas import DetectionClass, DetectionScore, InferenceResult, InferenceTelemetry, ModelManifest, ModelRuntimeInfo
from app.inference.telemetry import processing_fps_counter


class BamtiTorchRunner(InferenceRunner):
    def __init__(self, use_compiled_model: bool = False, model_path: Path | None = None) -> None:
        self.use_compiled_model = use_compiled_model
        self.model_path = model_path

    async def infer(self, frame: bytes) -> InferenceResult:
        return await asyncio.to_thread(self._infer_sync, frame)

    def _load_model(self) -> LoadedModel:
        if self.model_path is not None:
            from app.inference.model_loader import load_model_from_path

            return load_model_from_path(self.model_path, self.use_compiled_model)
        return load_model(self.use_compiled_model)

    def manifest(self) -> ModelManifest:
        loaded_model = self._load_model()
        return ModelManifest(
            model_version=loaded_model.model_path.stem,
            classes=tuple(
                DetectionClass(
                    variable_name=class_name,
                    class_id=class_name,
                    display_name=self._display_name_for_class(loaded_model, class_name),
                    description=self._description_for_class(loaded_model, class_name),
                    threshold=self._threshold_for_class(loaded_model, class_name),
                )
                for class_name in loaded_model.class_names
            ),
        )

    def _infer_sync(self, frame: bytes) -> InferenceResult:
        loaded_model = self._load_model()

        preprocess_started = time.perf_counter()
        input_tensor = image_bytes_to_tensor(frame, loaded_model.device)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = loaded_model.model(input_tensor)
            scores = self._scores_from_logits(logits).squeeze(0).detach().cpu()
        if loaded_model.device.type == "mps":
            torch.mps.synchronize()
        if loaded_model.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000

        postprocess_started = time.perf_counter()
        detections = self._detections_from_scores(loaded_model, scores)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        processing_fps = processing_fps_counter.mark_processed()

        return InferenceResult(
            detections=detections,
            model=ModelRuntimeInfo(
                name=loaded_model.model_path.name,
                architecture=f"{loaded_model.architecture}+torch_compile" if loaded_model.compiled else loaded_model.architecture,
                class_names=loaded_model.class_names,
                device=str(loaded_model.device),
                input_size=settings.model_input_size,
                score_activation=settings.model_score_activation,
            ),
            telemetry=InferenceTelemetry(
                processing_fps=round(processing_fps, 2),
                preprocess_ms=round(preprocess_ms, 2),
                inference_ms=round(inference_ms, 2),
                postprocess_ms=round(postprocess_ms, 2),
                server_total_ms=round(preprocess_ms + inference_ms + postprocess_ms, 2),
            ),
        )

    def _scores_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if settings.model_score_activation == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def _detections_from_scores(self, loaded_model: LoadedModel, scores: torch.Tensor) -> list[DetectionScore]:
        if loaded_model.service_classes:
            raw_score_by_class = {
                class_name: float(score)
                for class_name, score in zip(loaded_model.raw_class_names, scores, strict=True)
            }
            return [
                DetectionScore(
                    variable_name=service_class.variable_name,
                    class_id=service_class.variable_name,
                    display_name=service_class.display_name,
                    score=round(self._max_service_score(service_class, raw_score_by_class), 4),
                )
                for service_class in loaded_model.service_classes
            ]

        return [
            DetectionScore(
                variable_name=class_name,
                class_id=class_name,
                display_name=class_name,
                score=round(float(score), 4),
            )
            for class_name, score in zip(loaded_model.class_names, scores, strict=True)
        ]

    def _max_service_score(self, service_class: ServiceDetectionClass, raw_score_by_class: dict[str, float]) -> float:
        scores = [raw_score_by_class[class_name] for class_name in service_class.raw_class_names]
        return max(scores)

    def _service_class_by_name(self, loaded_model: LoadedModel, class_name: str) -> ServiceDetectionClass | None:
        return next((service_class for service_class in loaded_model.service_classes if service_class.variable_name == class_name), None)

    def _display_name_for_class(self, loaded_model: LoadedModel, class_name: str) -> str:
        service_class = self._service_class_by_name(loaded_model, class_name)
        return service_class.display_name if service_class else class_name

    def _description_for_class(self, loaded_model: LoadedModel, class_name: str) -> str:
        service_class = self._service_class_by_name(loaded_model, class_name)
        return service_class.description if service_class else f"BAMTI model class: {class_name}"

    def _threshold_for_class(self, loaded_model: LoadedModel, class_name: str) -> float:
        service_class = self._service_class_by_name(loaded_model, class_name)
        return service_class.threshold if service_class else 0.65
