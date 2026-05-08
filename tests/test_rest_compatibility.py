from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import settings
from app.main import app


JPEG_BYTES = b"\xff\xd8\xff\xe0mock-jpeg\xff\xd9"


def _post_inference_frame(
    client: TestClient,
    frame_bytes: bytes = JPEG_BYTES,
    content_type: str = "image/jpeg",
) -> Response:
    return client.post(
        "/api/v1/inference/frame",
        data={
            "frameId": "rest-frame-1",
            "clientSentAt": "2026-05-08T12:00:00Z",
        },
        files={
            "frame": ("frame.jpg", frame_bytes, content_type),
        },
    )


def test_detection_classes_returns_model_manifest() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/detection-classes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["modelVersion"] == "mock-v1"
    assert "classes" in payload
    assert len(payload["classes"]) >= 1
    assert payload["classes"][0]["variableName"] == "attentive"
    assert payload["classes"][0]["classId"] == "attentive"


def test_inference_frame_accepts_jpeg_and_returns_detection_scores() -> None:
    client = TestClient(app)

    response = _post_inference_frame(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["frameId"] == "rest-frame-1"
    assert payload["clientSentAt"] == "2026-05-08T12:00:00Z"
    assert "detections" in payload
    assert "telemetry" in payload
    assert payload["detections"][0]["variableName"] == "attentive"
    assert payload["detections"][0]["classId"] == "attentive"
    assert payload["detections"][0]["score"] == 0.99
    assert payload["telemetry"]["processingMs"] >= 0.0
    assert payload["telemetry"]["processingFps"] >= 0.0


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
    monkeypatch.setattr(settings, "max_frame_bytes", 4)
    client = TestClient(app)

    response = _post_inference_frame(client, frame_bytes=b"12345")

    assert response.status_code == 413
    assert response.json()["detail"] == "Frame exceeds max_frame_bytes=4."


def test_telemetry_runs_returns_accepted_response() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/telemetry/runs",
        json={
            "durationMs": 60000,
            "samples": [{"variableName": "attentive", "score": 0.99}],
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["id"].startswith("temp-")
    assert payload["status"] == "accepted"
    assert "createdAt" in payload


def test_v1_routes_are_not_double_prefixed() -> None:
    client = TestClient(app)

    assert client.get("/api/v1/detection-classes").status_code == 200
    assert client.get("/api/api/v1/detection-classes").status_code == 404


def test_inference_detections_use_detection_class_manifest() -> None:
    client = TestClient(app)

    classes = client.get("/api/v1/detection-classes").json()["classes"]
    detections = _post_inference_frame(client).json()["detections"]

    assert [
        (item["variableName"], item["classId"], item["displayName"])
        for item in detections
    ] == [
        (item["variableName"], item["classId"], item["displayName"])
        for item in classes
    ]


def test_rest_inference_does_not_use_websocket_session_lifecycle(
    recording_session_lifecycle,
) -> None:
    client = TestClient(app)

    response = _post_inference_frame(client)

    assert response.status_code == 200
    assert recording_session_lifecycle.starts == []
    assert recording_session_lifecycle.ends == []
    assert recording_session_lifecycle.summaries == []
    assert recording_session_lifecycle.raw_frames == []
    assert recording_session_lifecycle.per_frame_results == []
