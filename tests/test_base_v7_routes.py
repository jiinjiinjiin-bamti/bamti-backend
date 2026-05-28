from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0base-v7-jpeg\xff\xd9"


class FakeBaseRunner:
    async def infer(self, frame: bytes) -> InferenceResult:
        assert frame == JPEG_BYTES
        return InferenceResult(
            detections=[
                DetectionScore(variable_name="body_touching", class_id="body_touching", display_name="신체 만짐", score=0.1),
                DetectionScore(variable_name="distraction", class_id="distraction", display_name="주의 분산", score=0.2),
                DetectionScore(variable_name="phone_operation", class_id="phone_operation", display_name="핸드폰 조작", score=0.8),
                DetectionScore(variable_name="steering_operation", class_id="steering_operation", display_name="핸들 조작", score=0.3),
            ],
            model=ModelRuntimeInfo(
                name="aihub_notuned.pth",
                architecture="timm_vit_b_16_base",
                class_names=["body_touching", "distraction", "phone_operation", "steering_operation"],
                device="cpu",
                input_size=224,
                score_activation="softmax",
            ),
            telemetry=InferenceTelemetry(
                processing_fps=7.0,
                preprocess_ms=1.0,
                inference_ms=90.0,
                postprocess_ms=1.0,
                server_total_ms=92.0,
            ),
        )


def test_base_v7_detection_classes_route_uses_base_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_model_manifest(name: str):
        requested_runner_names.append(name)
        return {
            "modelVersion": "aihub_notuned",
            "classes": [
                {
                    "variableName": "phone_operation",
                    "classId": "phone_operation",
                    "displayName": "핸드폰 조작",
                    "description": "Base model class: 핸드폰 조작",
                    "threshold": 0.65,
                },
            ],
        }

    monkeypatch.setattr("app.api.base.inference.get_model_manifest", fake_get_model_manifest)
    client = TestClient(app)

    response = client.get("/api/base/v7/detection-classes")

    assert response.status_code == 200
    assert response.json()["modelVersion"] == "aihub_notuned"
    assert requested_runner_names == ["base-torch"]


def test_base_v7_websocket_returns_temporal_risk_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.api.base.v7.websocket.get_runner", lambda _: FakeBaseRunner())
    client = TestClient(app)

    with client.websocket_connect("/api/base/v7/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "base-session-1",
                "targetTransmissionFps": 10,
                "transport": "websocket",
            },
        )
        started = websocket.receive_json()
        assert started["type"] == "session_started"
        assert started["riskScoring"]["algorithm"] == "time_buffer_charge_decay_risk_accumulation"

        websocket.send_json(
            {
                "type": "frame_meta",
                "sessionId": "base-session-1",
                "frameId": "frame-1",
                "clientSentAt": "12345.67",
                "contentType": "image/jpeg",
                "width": 224,
                "height": 224,
                "encodingMs": 8.0,
            },
        )
        websocket.send_bytes(JPEG_BYTES)

        result = websocket.receive_json()
        assert result["type"] == "inference_result"
        assert result["model"]["name"] == "aihub_notuned.pth"
        assert result["detections"][2]["variableName"] == "phone_operation"
        assert result["riskScores"]["phone_operation"] > result["riskScores"]["distraction"]
        assert result["riskScores"]["steering_operation"] == 0.0
