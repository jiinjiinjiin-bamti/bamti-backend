from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0sample-jpeg\xff\xd9"


class FakeRunner:
    async def infer(self, frame: bytes) -> InferenceResult:
        assert frame == JPEG_BYTES
        return InferenceResult(
            detections=[
                DetectionScore(
                    variable_name="forward_inattention",
                    class_id="forward_inattention",
                    display_name="forward_inattention",
                    score=0.73,
                ),
            ],
            model=ModelRuntimeInfo(
                name="final_model.pth",
                architecture="vit_b_16",
                class_names=["forward_inattention"],
                device="cpu",
                input_size=224,
                score_activation="softmax",
            ),
            telemetry=InferenceTelemetry(
                processing_fps=6.0,
                preprocess_ms=2.0,
                inference_ms=100.0,
                postprocess_ms=1.0,
                server_total_ms=103.0,
            ),
        )


def test_v2_websocket_session_and_frame_inference(monkeypatch) -> None:
    monkeypatch.setattr("app.api.v2.websocket.get_runner", lambda _: FakeRunner())
    client = TestClient(app)

    with client.websocket_connect("/api/v2/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "session-1",
                "targetTransmissionFps": 24,
                "transport": "websocket",
            },
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "sessionId": "session-1",
                "frameId": "frame-1",
                "clientSentAt": "12345.67",
                "contentType": "image/jpeg",
                "width": 640,
                "height": 480,
                "encodingMs": 12.3,
            },
        )
        websocket.send_bytes(JPEG_BYTES)

        result = websocket.receive_json()
        assert result["type"] == "inference_result"
        assert result["sessionId"] == "session-1"
        assert result["frameId"] == "frame-1"
        assert result["clientSentAt"] == "12345.67"
        assert result["detections"][0]["variableName"] == "forward_inattention"
        assert result["telemetry"]["serverTotalMs"] == 103.0

        websocket.send_json(
            {
                "type": "session_end",
                "sessionId": "session-1",
            },
        )
        assert websocket.receive_json()["type"] == "session_ended"


def test_v2_websocket_requires_session_start() -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/v2/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "frame_meta",
                "sessionId": "session-1",
                "frameId": "frame-1",
                "clientSentAt": "12345.67",
                "contentType": "image/jpeg",
            },
        )

        result = websocket.receive_json()
        assert result["type"] == "error"
        assert result["code"] == "session_not_started"
