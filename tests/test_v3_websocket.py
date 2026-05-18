import asyncio
import threading

from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES_1 = b"\xff\xd8\xff\xe0frame-1\xff\xd9"
JPEG_BYTES_2 = b"\xff\xd8\xff\xe0frame-2\xff\xd9"
JPEG_BYTES_3 = b"\xff\xd8\xff\xe0frame-3\xff\xd9"


def inference_result(frame: bytes, call_index: int) -> InferenceResult:
    return InferenceResult(
        detections=[
            DetectionScore(
                variable_name="phone_use",
                class_id="phone_use",
                display_name="휴대기기 조작",
                score=0.18,
            ),
        ],
        debug_raw_detections=[
            DetectionScore(
                variable_name=f"A{index}",
                class_id=f"A{index}",
                display_name=f"A{index}",
                score=round(index / 100, 4),
            )
            for index in range(1, 17)
        ],
        model=ModelRuntimeInfo(
            name="exp04_pseudo_ir_aug_DayBest",
            architecture="timm_vit_b_16_custom",
            class_names=["phone_use"],
            device="cpu",
            input_size=224,
            score_activation="softmax",
        ),
        telemetry=InferenceTelemetry(
            processing_fps=float(call_index),
            preprocess_ms=2.0,
            inference_ms=100.0,
            postprocess_ms=1.0,
            server_total_ms=103.0,
        ),
    )


class BlockingRunner:
    def __init__(self, processing_started: threading.Event, release_processing: threading.Event) -> None:
        self.call_index = 0
        self.frames: list[bytes] = []
        self.processing_started = processing_started
        self.release_processing = release_processing

    async def infer(self, frame: bytes) -> InferenceResult:
        self.call_index += 1
        self.frames.append(frame)
        if self.call_index == 1:
            self.processing_started.set()
            while not self.release_processing.is_set():
                await asyncio.sleep(0.01)
        return inference_result(frame, self.call_index)


def send_frame(websocket, frame_id: str, frame: bytes) -> None:
    websocket.send_json(
        {
            "type": "frame_meta",
            "sessionId": "session-1",
            "frameId": frame_id,
            "clientSentAt": "12345.67",
            "contentType": "image/jpeg",
            "width": 640,
            "height": 480,
            "encodingMs": 12.3,
        },
    )
    websocket.send_bytes(frame)


def test_v3_websocket_replaces_pending_frame_with_latest(monkeypatch) -> None:
    processing_started = threading.Event()
    release_processing = threading.Event()
    runner = BlockingRunner(processing_started, release_processing)
    monkeypatch.setattr("app.api.v3.websocket.get_runner", lambda _: runner)
    client = TestClient(app)

    with client.websocket_connect("/api/v3/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "session-1",
                "targetTransmissionFps": 24,
                "transport": "websocket",
            },
        )
        assert websocket.receive_json()["type"] == "session_started"

        send_frame(websocket, "frame-1", JPEG_BYTES_1)
        assert processing_started.wait(timeout=2)

        send_frame(websocket, "frame-2", JPEG_BYTES_2)
        send_frame(websocket, "frame-3", JPEG_BYTES_3)

        dropped = websocket.receive_json()
        assert dropped["type"] == "frame_dropped"
        assert dropped["frameId"] == "frame-2"
        assert dropped["droppedFrames"] == 1

        release_processing.set()
        first_result = websocket.receive_json()
        second_result = websocket.receive_json()

        assert first_result["type"] == "inference_result"
        assert first_result["frameId"] == "frame-1"
        assert second_result["type"] == "inference_result"
        assert second_result["frameId"] == "frame-3"
        assert second_result["queue"]["droppedFrames"] == 1
        assert runner.frames == [JPEG_BYTES_1, JPEG_BYTES_3]

        websocket.send_json(
            {
                "type": "session_end",
                "sessionId": "session-1",
            },
        )
        ended = websocket.receive_json()
        assert ended["type"] == "session_ended"
        assert ended["droppedFrames"] == 1


def test_v3_websocket_requires_session_start() -> None:
    client = TestClient(app)

    with client.websocket_connect("/api/v3/inference/stream") as websocket:
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


def test_v4_debug_websocket_includes_raw_action_scores(monkeypatch) -> None:
    processing_started = threading.Event()
    release_processing = threading.Event()
    runner = BlockingRunner(processing_started, release_processing)
    requested_runner_names: list[str] = []

    def fake_get_runner(name: str):
        requested_runner_names.append(name)
        return runner

    monkeypatch.setattr("app.api.v3.websocket.get_runner", fake_get_runner)
    client = TestClient(app)

    with client.websocket_connect("/api/v4/debug/inference/stream") as websocket:
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
        assert started["apiVersion"] == "v4-debug"

        send_frame(websocket, "frame-1", JPEG_BYTES_1)
        assert processing_started.wait(timeout=2)
        release_processing.set()

        result = websocket.receive_json()
        assert result["type"] == "inference_result"
        assert requested_runner_names == ["bamti-torch-debug-raw"]
        assert [detection["variableName"] for detection in result["debugRawDetections"]] == [
            f"A{index}"
            for index in range(1, 17)
        ]
