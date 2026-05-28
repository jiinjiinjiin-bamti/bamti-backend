from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0driver4-performance-jpeg\xff\xd9"
RAW_RGB_BYTES = bytes([128]) * (224 * 224 * 3)


class FakeDriver4Runner:
    def __init__(self, *, expected_frame: bytes = JPEG_BYTES, architecture: str = "timm_vit_b_16_driver4") -> None:
        self.expected_frame = expected_frame
        self.architecture = architecture

    async def infer(self, frame: bytes) -> InferenceResult:
        assert frame == self.expected_frame
        return self._result()

    async def infer_raw_rgb(self, frame: bytes, width: int, height: int) -> InferenceResult:
        assert frame == RAW_RGB_BYTES
        assert width == 224
        assert height == 224
        return self._result()

    def _result(self) -> InferenceResult:
        return InferenceResult(
            detections=[
                DetectionScore(variable_name="body_touching", class_id="body_touching", display_name="신체 만짐", score=0.1),
                DetectionScore(variable_name="distraction", class_id="distraction", display_name="주의 분산", score=0.2),
                DetectionScore(variable_name="phone_operation", class_id="phone_operation", display_name="핸드폰 조작", score=0.8),
                DetectionScore(variable_name="steering_operation", class_id="steering_operation", display_name="핸들 조작", score=0.3),
            ],
            model=ModelRuntimeInfo(
                name="final_model_0528.pth",
                architecture=self.architecture,
                class_names=["body_touching", "distraction", "phone_operation", "steering_operation"],
                device="cpu",
                input_size=224,
                score_activation="softmax",
            ),
            telemetry=InferenceTelemetry(
                processing_fps=8.0,
                preprocess_ms=1.0,
                inference_ms=120.0,
                postprocess_ms=1.0,
                server_total_ms=122.0,
            ),
        )


def _start_session(websocket, session_id: str = "session-1") -> dict:
    websocket.send_json(
        {
            "type": "session_start",
            "sessionId": session_id,
            "targetTransmissionFps": 24,
            "transport": "websocket",
        },
    )
    return websocket.receive_json()


def _send_jpeg_frame(websocket, *, session_id: str = "session-1", frame_id: str = "frame-1") -> dict:
    websocket.send_json(
        {
            "type": "frame_meta",
            "sessionId": session_id,
            "frameId": frame_id,
            "clientSentAt": "12345.67",
            "contentType": "image/jpeg",
            "width": 224,
            "height": 224,
            "encodingMs": 8.0,
        },
    )
    websocket.send_bytes(JPEG_BYTES)
    return websocket.receive_json()


def test_driver4_v1_rest_uses_driver4_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner()

    monkeypatch.setattr("app.api.driver.v1.inference.get_runner", fake_get_runner)
    client = TestClient(app)

    response = client.post(
        "/api/driver/v1/inference/frame",
        data={"frameId": "rest-frame-1", "clientSentAt": "12345.67"},
        files={"frame": ("frame.jpg", JPEG_BYTES, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["detections"][2]["variableName"] == "phone_operation"
    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v2_websocket_uses_driver4_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner()

    monkeypatch.setattr("app.api.driver.v2.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v2/inference/stream") as websocket:
        started = _start_session(websocket)
        assert started["type"] == "session_started"
        assert started["apiVersion"] == "v2"

        result = _send_jpeg_frame(websocket)
        assert result["type"] == "inference_result"
        assert result["detections"][2]["variableName"] == "phone_operation"

    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v3_websocket_uses_latest_pending_policy(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner()

    monkeypatch.setattr("app.api.v3.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v3/inference/stream") as websocket:
        started = _start_session(websocket)
        assert started["type"] == "session_started"
        assert started["apiVersion"] == "v3"
        assert started["queuePolicy"] == "latest_pending_only"

        result = _send_jpeg_frame(websocket)
        assert result["type"] == "inference_result"
        assert result["queue"]["policy"] == "latest_pending_only"

    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v4_websocket_accepts_raw_rgb_frames(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner(expected_frame=RAW_RGB_BYTES)

    monkeypatch.setattr("app.api.driver.v4.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v4/inference/stream") as websocket:
        started = _start_session(websocket)
        assert started["type"] == "session_started"
        assert started["apiVersion"] == "v4"
        assert started["frameEncoding"] == "raw_rgb_224"

        websocket.send_json(
            {
                "type": "frame_meta",
                "sessionId": "session-1",
                "frameId": "raw-frame-1",
                "clientSentAt": "12345.67",
                "contentType": "application/x-rgb24",
                "width": 224,
                "height": 224,
                "encodingMs": 4.0,
            },
        )
        websocket.send_bytes(RAW_RGB_BYTES)

        result = websocket.receive_json()
        assert result["type"] == "inference_result"
        assert result["detections"][2]["variableName"] == "phone_operation"

    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v4_1_websocket_uses_224_jpeg_driver4_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner()

    monkeypatch.setattr("app.api.v3.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v4-1/inference/stream") as websocket:
        started = _start_session(websocket)
        assert started["type"] == "session_started"
        assert started["apiVersion"] == "v4-1"
        assert started["frameEncoding"] == "jpeg_224_quality_0_7"

        result = _send_jpeg_frame(websocket)
        assert result["type"] == "inference_result"

    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v5_websocket_uses_driver4_compiled_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return FakeDriver4Runner(architecture="torchvision_vit_b_16_driver4+torch_compile")

    monkeypatch.setattr("app.api.v3.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v5/inference/stream") as websocket:
        started = _start_session(websocket)
        assert started["type"] == "session_started"
        assert started["apiVersion"] == "v5"
        assert started["runtime"] == "torch_compile"

        result = _send_jpeg_frame(websocket)
        assert result["type"] == "inference_result"
        assert result["model"]["architecture"] == "torchvision_vit_b_16_driver4+torch_compile"

    assert requested_runner_names == ["driver4-torch-compiled"]
