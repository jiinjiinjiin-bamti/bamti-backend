from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0sample-v5-jpeg\xff\xd9"


class FakeCompiledRunner:
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
                architecture="vit_b_16+torch_compile",
                class_names=["forward_inattention"],
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


def test_v5_websocket_uses_compiled_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeCompiledRunner()

    monkeypatch.setattr("app.api.v3.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/v5/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "session-1",
                "targetTransmissionFps": 24,
                "transport": "websocket",
            },
        )
        started = websocket.receive_json()
        assert started["type"] == "session_started"
        assert started["runtime"] == "torch_compile"

        websocket.send_json(
            {
                "type": "frame_meta",
                "sessionId": "session-1",
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
        assert result["model"]["architecture"] == "vit_b_16+torch_compile"
        assert requested_runner_names == ["bamti-torch-compiled"]
