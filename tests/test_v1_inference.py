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
                    variable_name="normal_driving",
                    class_id="normal_driving",
                    display_name="정상 주행",
                    score=0.72,
                ),
                DetectionScore(
                    variable_name="phone_use",
                    class_id="phone_use",
                    display_name="휴대기기 조작",
                    score=0.18,
                ),
                DetectionScore(
                    variable_name="drowsiness",
                    class_id="drowsiness",
                    display_name="졸음",
                    score=0.04,
                ),
            ],
            model=ModelRuntimeInfo(
                name="exp04_pseudo_ir_aug_DayBest",
                architecture="timm_vit_b_16_custom",
                class_names=[
                    "normal_driving",
                    "phone_use",
                    "drowsiness",
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
            model_version="exp04_pseudo_ir_aug_DayBest",
            classes=(
                DetectionClass(
                    variable_name="normal_driving",
                    class_id="normal_driving",
                    display_name="정상 주행",
                    description="A1 - 정상 주행",
                    threshold=0.50,
                ),
                DetectionClass(
                    variable_name="phone_use",
                    class_id="phone_use",
                    display_name="휴대기기 조작",
                    description="A5, A6, A7, A8, A9 - 휴대기기 조작",
                    threshold=0.13,
                ),
                DetectionClass(
                    variable_name="drowsiness",
                    class_id="drowsiness",
                    display_name="졸음",
                    description="A12 - 졸음",
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
    assert payload["modelVersion"] == "exp04_pseudo_ir_aug_DayBest"
    assert [item["variableName"] for item in payload["classes"]] == [
        "normal_driving",
        "phone_use",
        "drowsiness",
    ]


def test_inference_frame_accepts_jpeg_and_returns_detection_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v1.inference.get_runner", lambda _: FakeRunner())
    client = TestClient(app)

    response = _post_inference_frame(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["frameId"] == "rest-frame-1"
    assert payload["clientSentAt"] == "12345.67"
    assert payload["detections"][0]["variableName"] == "normal_driving"
    assert payload["detections"][0]["score"] == 0.72
    assert payload["model"]["architecture"] == "timm_vit_b_16_custom"
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
