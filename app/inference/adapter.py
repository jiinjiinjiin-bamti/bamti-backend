from app.inference.schemas import DetectionScore, InferenceResult, ModelManifest


def build_detection_scores(
    result: InferenceResult,
    manifest: ModelManifest,
) -> list[DetectionScore]:
    """Convert the runner's single-label MVP result into API detection scores."""
    detections: list[DetectionScore] = []
    for detection_class in manifest.classes:
        score = result.confidence if detection_class.class_id == result.label else 0.0
        detections.append(
            DetectionScore(
                variable_name=detection_class.variable_name,
                class_id=detection_class.class_id,
                display_name=detection_class.display_name,
                score=score,
            )
        )
    return detections
