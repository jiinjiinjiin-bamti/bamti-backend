from fastapi.testclient import TestClient

from app.inference.schemas import DetectionScore, InferenceResult, InferenceTelemetry, ModelRuntimeInfo
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0driver4-v7-jpeg\xff\xd9"


class FakeDriver4Runner:
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
                name="final_model_4cls",
                architecture="torchvision_vit_b_16_driver4",
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


def test_driver4_v7_detection_classes_route_uses_driver4_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_model_manifest(name: str):
        requested_runner_names.append(name)
        return {
            "modelVersion": "final_model_4cls",
            "classes": [
                {
                    "variableName": "phone_operation",
                    "classId": "phone_operation",
                    "displayName": "핸드폰 조작",
                    "description": "Driver4 model class: 핸드폰 조작",
                    "threshold": 0.65,
                },
            ],
        }

    monkeypatch.setattr("app.api.driver.inference.get_model_manifest", fake_get_model_manifest)
    client = TestClient(app)

    response = client.get("/api/driver/v7/detection-classes")

    assert response.status_code == 200
    assert response.json()["modelVersion"] == "final_model_4cls"
    assert requested_runner_names == ["driver4-torch"]


def test_driver4_v7_websocket_returns_temporal_risk_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.api.driver.v7.websocket.get_runner", lambda _: FakeDriver4Runner())
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v7/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "session-1",
                "targetTransmissionFps": 10,
                "transport": "websocket",
            },
        )
        started = websocket.receive_json()
        assert started["type"] == "session_started"
        assert started["riskScoring"]["algorithm"] == "ema_temporal_risk_score"

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
        assert result["detections"][2]["variableName"] == "phone_operation"
        assert result["riskScores"]["phone_operation"] > result["riskScores"]["distraction"]
        assert result["riskScores"]["steering_operation"] == 0.0
        assert result["riskScoring"]["excludedClasses"] == []
        assert result["riskScoring"]["scoreRange"] == [0, 100]


def test_driver4_v7_websocket_uses_frame_time_for_risk_scores(monkeypatch) -> None:
    monkeypatch.setattr("app.api.driver.v7.websocket.get_runner", lambda _: FakeDriver4Runner())
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v7/inference/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "session-1",
                "targetTransmissionFps": 10,
                "transport": "websocket",
            },
        )
        assert websocket.receive_json()["type"] == "session_started"

        for frame_id, frame_time_seconds in [("frame-1", 0.0), ("frame-2", 0.5)]:
            websocket.send_json(
                {
                    "type": "frame_meta",
                    "sessionId": "session-1",
                    "frameId": frame_id,
                    "clientSentAt": "12345.67",
                    "contentType": "image/jpeg",
                    "width": 224,
                    "height": 224,
                    "encodingMs": 8.0,
                    "frameTimeSeconds": frame_time_seconds,
                },
            )
            websocket.send_bytes(JPEG_BYTES)
            result = websocket.receive_json()
            assert result["type"] == "inference_result"

        assert result["riskScores"]["phone_operation"] == 23.62


def test_driver4_v7_websocket_resets_risk_scores_for_new_session_start(monkeypatch) -> None:
    monkeypatch.setattr("app.api.driver.v7.websocket.get_runner", lambda _: FakeDriver4Runner())
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v7/inference/stream") as websocket:
        for session_id, frame_id in [("session-1", "frame-1"), ("session-2", "frame-2")]:
            websocket.send_json(
                {
                    "type": "session_start",
                    "sessionId": session_id,
                    "targetTransmissionFps": 10,
                    "transport": "websocket",
                },
            )
            assert websocket.receive_json()["type"] == "session_started"

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

            result = websocket.receive_json()
            assert result["type"] == "inference_result"
            if session_id == "session-1":
                first_phone_score = result["riskScores"]["phone_operation"]
            else:
                assert result["riskScores"]["phone_operation"] == first_phone_score


def test_driver4_v7_pre_analysis_websocket_processes_all_frames_in_order(monkeypatch) -> None:
    monkeypatch.setattr("app.api.driver.v7.pre_analysis_websocket.get_runner", lambda _: FakeDriver4Runner())
    client = TestClient(app)

    with client.websocket_connect("/api/driver/v7/pre-analysis/stream") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "sessionId": "pre-analysis-session",
                "targetTransmissionFps": 8,
                "transport": "websocket",
            },
        )
        started = websocket.receive_json()
        assert started["type"] == "session_started"
        assert started["queuePolicy"] == "fifo_no_drop"

        for frame_id, frame_time_seconds in [("frame-1", 0.0), ("frame-2", 0.125), ("frame-3", 0.25)]:
            websocket.send_json(
                {
                    "type": "frame_meta",
                    "sessionId": "pre-analysis-session",
                    "frameId": frame_id,
                    "clientSentAt": "12345.67",
                    "contentType": "image/jpeg",
                    "width": 224,
                    "height": 224,
                    "encodingMs": 8.0,
                    "frameTimeSeconds": frame_time_seconds,
                },
            )
            websocket.send_bytes(JPEG_BYTES)

        results = [websocket.receive_json() for _ in range(3)]

        assert [result["frameId"] for result in results] == ["frame-1", "frame-2", "frame-3"]
        assert [result["queue"]["droppedFrames"] for result in results] == [0, 0, 0]
        assert all(result["queue"]["policy"] == "fifo_no_drop" for result in results)
        assert results[2]["riskScores"]["phone_operation"] > results[0]["riskScores"]["phone_operation"]


def test_driver4_v7_mobile_session_route_exists() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/driver/v7/mobile/sessions",
        json={"cameraUrlBase": "http://localhost:5173/camera?sessionId={sessionId}"},
    )

    assert response.status_code == 200
    assert response.json()["cameraUrl"].startswith("http://localhost:5173/camera?sessionId=")
