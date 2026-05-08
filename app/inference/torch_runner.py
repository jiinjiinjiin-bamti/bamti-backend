import asyncio
import time

import torch

from app.core.config import settings
from app.inference.model_loader import load_model
from app.inference.preprocessing import image_bytes_to_tensor
from app.inference.runner import InferenceRunner
from app.inference.schemas import DetectionClass, DetectionScore, InferenceResult, InferenceTelemetry, ModelManifest, ModelRuntimeInfo
from app.inference.telemetry import processing_fps_counter


class BamtiTorchRunner(InferenceRunner):
    async def infer(self, frame: bytes) -> InferenceResult:
        return await asyncio.to_thread(self._infer_sync, frame)

    def manifest(self) -> ModelManifest:
        loaded_model = load_model()
        return ModelManifest(
            model_version=loaded_model.model_path.stem,
            classes=tuple(
                DetectionClass(
                    variable_name=class_name,
                    class_id=class_name,
                    display_name=class_name,
                    description=f"BAMTI model class: {class_name}",
                    threshold=0.65,
                )
                for class_name in loaded_model.class_names
            ),
        )

    def _infer_sync(self, frame: bytes) -> InferenceResult:
        loaded_model = load_model()

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
        detections = [
            DetectionScore(
                variable_name=class_name,
                class_id=class_name,
                display_name=class_name,
                score=round(float(score), 4),
            )
            for class_name, score in zip(loaded_model.class_names, scores, strict=True)
        ]
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
        processing_fps = processing_fps_counter.mark_processed()

        return InferenceResult(
            detections=detections,
            model=ModelRuntimeInfo(
                name=loaded_model.model_path.name,
                architecture="vit_b_16",
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
