from fastapi.testclient import TestClient

from app.inference.schemas import DetectionClass, DetectionScore, InferenceResult, InferenceTelemetry, ModelManifest, ModelRuntimeInfo
from app.main import app


class FakeAihubRunner:
    async def infer(self, frame: bytes) -> InferenceResult:
        return InferenceResult(
            detections=[
                DetectionScore(variable_name="forward_inattention", class_id="forward_inattention", display_name="전방 주의 소홀", score=0.72),
                DetectionScore(variable_name="surrounding_inattention", class_id="surrounding_inattention", display_name="주변 주의 소홀", score=0.24),
                DetectionScore(variable_name="vehicle_interaction", class_id="vehicle_interaction", display_name="차량 간 상호작용", score=0.18),
            ],
            model=ModelRuntimeInfo(
                name="final_model.pth",
                architecture="torchvision_vit_b_16",
                class_names=["forward_inattention", "surrounding_inattention", "vehicle_interaction"],
                device="cpu",
                input_size=224,
                score_activation="softmax",
            ),
            telemetry=InferenceTelemetry(
                processing_fps=8.0,
                preprocess_ms=2.0,
                inference_ms=80.0,
                postprocess_ms=1.0,
                server_total_ms=83.0,
            ),
        )

    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_version="final_model",
            classes=(
                DetectionClass(
                    variable_name="forward_inattention",
                    class_id="forward_inattention",
                    display_name="전방 주의 소홀",
                    description="AIHub model class: forward_inattention",
                    threshold=0.65,
                ),
                DetectionClass(
                    variable_name="surrounding_inattention",
                    class_id="surrounding_inattention",
                    display_name="주변 주의 소홀",
                    description="AIHub model class: surrounding_inattention",
                    threshold=0.65,
                ),
                DetectionClass(
                    variable_name="vehicle_interaction",
                    class_id="vehicle_interaction",
                    display_name="차량 간 상호작용",
                    description="AIHub model class: vehicle_interaction",
                    threshold=0.65,
                ),
            ),
        )


def test_aihub_detection_classes_route_uses_aihub_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_model_manifest(name: str):
        requested_runner_names.append(name)
        return FakeAihubRunner().manifest()

    monkeypatch.setattr("app.api.aihub.inference.get_model_manifest", fake_get_model_manifest)
    client = TestClient(app)

    response = client.get("/api/aihub/detection-classes")

    assert response.status_code == 200
    payload = response.json()
    assert requested_runner_names == ["aihub-torch"]
    assert payload["modelVersion"] == "final_model"
    assert [detection["variableName"] for detection in payload["classes"]] == [
        "forward_inattention",
        "surrounding_inattention",
        "vehicle_interaction",
    ]

    response = client.get("/api/aihub/v4/detection-classes")
    assert response.status_code == 200
    response = client.get("/api/aihub/v6/detection-classes")
    assert response.status_code == 200


def test_aihub_inference_frame_returns_aihub_model_scores(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeAihubRunner()

    monkeypatch.setattr("app.api.aihub.inference.get_runner", fake_get_runner)
    client = TestClient(app)

    response = client.post(
        "/api/aihub/inference/frame",
        data={"frameId": "frame-1", "clientSentAt": "123.4"},
        files={"frame": ("frame.jpg", b"\xff\xd8\xff\xe0sample-jpeg\xff\xd9", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert requested_runner_names == ["aihub-torch"]
    assert payload["model"]["name"] == "final_model.pth"
    assert [detection["variableName"] for detection in payload["detections"]] == [
        "forward_inattention",
        "surrounding_inattention",
        "vehicle_interaction",
    ]


def test_aihub_mobile_session_routes_support_v4_and_v6_prefixes() -> None:
    client = TestClient(app)

    for api_version in ("aihub-v4", "aihub-v6"):
        version = api_version.removeprefix("aihub-")
        response = client.post(
            f"/api/aihub/{version}/mobile/sessions",
            json={"cameraUrlBase": f"https://example.test/camera?sessionId={{sessionId}}&apiVersion={api_version}"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["sessionId"].startswith("mobile-")
        assert payload["cameraUrl"].endswith(f"sessionId={payload['sessionId']}&apiVersion={api_version}")
