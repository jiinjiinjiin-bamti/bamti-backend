from fastapi.testclient import TestClient
from httpx import Response

from app.inference.schemas import DetectionClass, DetectionScore, InferenceResult, InferenceTelemetry, ModelManifest, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0sample-jpeg\xff\xd9"


class FakeRunner:
    async def infer(self, frame: bytes) -> InferenceResult:
        return InferenceResult(
            detections=[
                DetectionScore(
                    variable_name="forward_inattention",
                    class_id="forward_inattention",
                    display_name="forward_inattention",
                    score=0.73,
                ),
                DetectionScore(
                    variable_name="surrounding_inattention",
                    class_id="surrounding_inattention",
                    display_name="surrounding_inattention",
                    score=0.19,
                ),
                DetectionScore(
                    variable_name="vehicle_interaction",
                    class_id="vehicle_interaction",
                    display_name="vehicle_interaction",
                    score=0.08,
                ),
            ],
            model=ModelRuntimeInfo(
                name="final_model.pth",
                architecture="vit_b_16",
                class_names=[
                    "forward_inattention",
                    "surrounding_inattention",
                    "vehicle_interaction",
                ],
                device="cpu",
                input_size=224,
                score_activation="softmax",
            ),
            telemetry=InferenceTelemetry(
                processing_fps=10.0,
                preprocess_ms=2.0,
                inference_ms=70.0,
                postprocess_ms=1.0,
                server_total_ms=73.0,
            ),
        )

    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_version="final_model",
            classes=(
                DetectionClass(
                    variable_name="forward_inattention",
                    class_id="forward_inattention",
                    display_name="forward_inattention",
                    description="BAMTI model class: forward_inattention",
                    threshold=0.65,
                ),
                DetectionClass(
                    variable_name="surrounding_inattention",
                    class_id="surrounding_inattention",
                    display_name="surrounding_inattention",
                    description="BAMTI model class: surrounding_inattention",
                    threshold=0.65,
                ),
                DetectionClass(
                    variable_name="vehicle_interaction",
                    class_id="vehicle_interaction",
                    display_name="vehicle_interaction",
                    description="BAMTI model class: vehicle_interaction",
                    threshold=0.65,
                ),
            ),
        )


def _post_inference_frame(
    client: TestClient,
    frame_bytes: bytes = JPEG_BYTES,
    content_type: str = "image/jpeg",
) -> Response:
    return client.post(
        "/api/v1/inference/frame",
        data={
            "frameId": "rest-frame-1",
            "clientSentAt": "12345.67",
        },
        files={
            "frame": ("frame.jpg", frame_bytes, content_type),
        },
    )


def test_detection_classes_returns_model_manifest(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.inference.get_runner", lambda _: FakeRunner())
    monkeypatch.setattr("app.api.v1.inference.get_model_manifest", lambda _: FakeRunner().manifest())
    client = TestClient(app)

    response = client.get("/api/v1/detection-classes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["modelVersion"] == "final_model"
    assert [item["variableName"] for item in payload["classes"]] == [
        "forward_inattention",
        "surrounding_inattention",
        "vehicle_interaction",
    ]


def test_inference_frame_accepts_jpeg_and_returns_detection_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.inference.get_runner", lambda _: FakeRunner())
    client = TestClient(app)

    response = _post_inference_frame(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["frameId"] == "rest-frame-1"
    assert payload["clientSentAt"] == "12345.67"
    assert payload["detections"][0]["variableName"] == "forward_inattention"
    assert payload["detections"][0]["score"] == 0.73
    assert payload["model"]["architecture"] == "vit_b_16"
    assert payload["telemetry"]["processingFps"] == 10.0
    assert payload["telemetry"]["serverTotalMs"] == 73.0


def test_inference_frame_rejects_non_jpeg() -> None:
    client = TestClient(app)

    response = _post_inference_frame(
        client,
        frame_bytes=b"not-a-jpeg",
        content_type="image/png",
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only image/jpeg frames are supported."


def test_inference_frame_rejects_empty_frame() -> None:
    client = TestClient(app)

    response = _post_inference_frame(client, frame_bytes=b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "Frame file must not be empty."


def test_inference_frame_rejects_oversized_frame(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.inference.settings.max_frame_bytes", 4)
    client = TestClient(app)

    response = _post_inference_frame(client, frame_bytes=b"12345")

    assert response.status_code == 413
    assert response.json()["detail"] == "Frame exceeds max_frame_bytes=4."


def test_v1_routes_are_not_double_prefixed(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.inference.get_model_manifest", lambda _: FakeRunner().manifest())
    client = TestClient(app)

    assert client.get("/api/v1/detection-classes").status_code == 200
    assert client.get("/api/api/v1/detection-classes").status_code == 404
