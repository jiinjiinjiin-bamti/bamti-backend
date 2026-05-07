import asyncio

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_websocket_inference_flow(recording_session_lifecycle) -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "session_id": "session-1",
                "driver_id": "driver-1",
                "started_at": "2026-05-07T12:00:00Z",
            }
        )
        assert websocket.receive_json() == {
            "type": "session_started",
            "session_id": "session-1",
        }
        assert len(recording_session_lifecycle.starts) == 1
        assert recording_session_lifecycle.starts[0].session_id == "session-1"
        assert recording_session_lifecycle.starts[0].driver_id == "driver-1"

        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-1",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )
        websocket.send_bytes(b"\xff\xd8mock-jpeg")

        result = websocket.receive_json()
        assert result["type"] == "inference_result"
        assert result["session_id"] == "session-1"
        assert result["frame_id"] == "frame-1"
        assert result["result"]["label"] == "attentive"

        websocket.send_json(
            {
                "type": "session_end",
                "session_id": "session-1",
                "ended_at": "2026-05-07T12:00:02Z",
            }
        )
        assert websocket.receive_json() == {
            "type": "session_ended",
            "session_id": "session-1",
        }
        assert len(recording_session_lifecycle.ends) == 1
        assert recording_session_lifecycle.ends[0].session_id == "session-1"
        assert recording_session_lifecycle.ends[0].close_reason == "session_end"
        assert recording_session_lifecycle.summaries[0].session_id == "session-1"
        assert recording_session_lifecycle.summaries[0].total_events == 0
        assert recording_session_lifecycle.summaries[0].distraction_seconds == 0.0
        assert recording_session_lifecycle.raw_frames == []
        assert recording_session_lifecycle.per_frame_results == []


def test_websocket_session_start_persists_driving_session(
    recording_session_lifecycle,
) -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "session_id": "session-persist-start",
                "driver_id": "driver-persist",
                "started_at": "2026-05-07T12:00:00Z",
            }
        )

        assert websocket.receive_json() == {
            "type": "session_started",
            "session_id": "session-persist-start",
        }

    assert len(recording_session_lifecycle.starts) == 1
    assert recording_session_lifecycle.starts[0].session_id == "session-persist-start"
    assert recording_session_lifecycle.starts[0].driver_id == "driver-persist"


def test_websocket_session_end_persists_end_and_summary(
    recording_session_lifecycle,
) -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {"type": "session_start", "session_id": "session-persist-end"}
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "session_end",
                "session_id": "session-persist-end",
                "ended_at": "2026-05-07T12:01:00Z",
            }
        )

        assert websocket.receive_json() == {
            "type": "session_ended",
            "session_id": "session-persist-end",
        }

    assert len(recording_session_lifecycle.ends) == 1
    assert recording_session_lifecycle.ends[0].session_id == "session-persist-end"
    assert recording_session_lifecycle.ends[0].close_reason == "session_end"
    assert len(recording_session_lifecycle.summaries) == 1
    assert recording_session_lifecycle.summaries[0].session_id == "session-persist-end"
    assert recording_session_lifecycle.summaries[0].total_events == 0
    assert recording_session_lifecycle.summaries[0].distraction_seconds == 0.0


def test_websocket_client_disconnect_cleans_open_session(
    recording_session_lifecycle,
) -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {"type": "session_start", "session_id": "session-disconnect"}
        )
        assert websocket.receive_json()["type"] == "session_started"

    assert len(recording_session_lifecycle.ends) == 1
    assert recording_session_lifecycle.ends[0].session_id == "session-disconnect"
    assert recording_session_lifecycle.ends[0].close_reason == "client_disconnect"


def test_websocket_idle_timeout_cleans_open_session(
    monkeypatch,
    recording_session_lifecycle,
) -> None:
    monkeypatch.setattr(settings, "websocket_idle_timeout_seconds", 0.01)
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json({"type": "session_start", "session_id": "session-idle"})
        assert websocket.receive_json()["type"] == "session_started"
        assert websocket.receive_json()["code"] == "idle_timeout"

    assert len(recording_session_lifecycle.ends) == 1
    assert recording_session_lifecycle.ends[0].session_id == "session-idle"
    assert recording_session_lifecycle.ends[0].close_reason == "idle_timeout"


def test_websocket_rejects_empty_frame() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "session_id": "session-empty",
            }
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-empty",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )
        websocket.send_bytes(b"")

        error = websocket.receive_json()
        assert error == {
            "type": "error",
            "code": "empty_frame",
            "message": "Binary frame must not be empty.",
        }


def test_websocket_rejects_unsupported_content_type() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "session_id": "session-content-type",
            }
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-png",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/png",
            }
        )

        error = websocket.receive_json()
        assert error == {
            "type": "error",
            "code": "unsupported_content_type",
            "message": "Only image/jpeg frames are supported.",
        }


def test_websocket_rejects_too_large_frame(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_frame_bytes", 4)
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "session_start",
                "session_id": "session-large",
            }
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-large",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )
        websocket.send_bytes(b"12345")

        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "frame_too_large"


def test_websocket_rejects_frame_meta_before_session_start() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-before-start",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )

        assert websocket.receive_json() == {
            "type": "error",
            "code": "session_not_started",
            "message": "Send session_start before frame_meta.",
        }


def test_websocket_rejects_binary_before_session_start() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_bytes(b"\xff\xd8mock-jpeg")

        assert websocket.receive_json() == {
            "type": "error",
            "code": "frame_meta_required",
            "message": "Send frame_meta before a binary frame.",
        }


def test_websocket_rejects_binary_without_frame_meta() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json({"type": "session_start", "session_id": "session-no-meta"})
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_bytes(b"\xff\xd8mock-jpeg")

        assert websocket.receive_json() == {
            "type": "error",
            "code": "frame_meta_required",
            "message": "Send frame_meta before a binary frame.",
        }


def test_websocket_rejects_consecutive_frame_meta() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json({"type": "session_start", "session_id": "session-meta"})
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-1",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )
        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-2",
                "captured_at": "2026-05-07T12:00:02Z",
                "content_type": "image/jpeg",
            }
        )

        assert websocket.receive_json()["code"] == "frame_already_pending"

        websocket.send_bytes(b"\xff\xd8mock-jpeg")
        assert websocket.receive_json()["code"] == "frame_meta_required"


def test_invalid_frame_meta_clears_pending_frame() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json(
            {"type": "session_start", "session_id": "session-invalid-meta"}
        )
        assert websocket.receive_json()["type"] == "session_started"

        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-png",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/png",
            }
        )
        assert websocket.receive_json()["code"] == "unsupported_content_type"

        websocket.send_bytes(b"\xff\xd8mock-jpeg")
        assert websocket.receive_json()["code"] == "frame_meta_required"


def test_websocket_idle_timeout_sends_error_and_closes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "websocket_idle_timeout_seconds", 0.01)
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        assert websocket.receive_json()["code"] == "idle_timeout"


def test_session_end_does_not_wait_forever_for_slow_worker(
    monkeypatch,
    recording_session_lifecycle,
) -> None:
    class SlowRunner:
        async def infer(self, frame: bytes):
            await asyncio.sleep(0.1)

    monkeypatch.setattr(settings, "websocket_drain_timeout_seconds", 0.01)
    monkeypatch.setattr("app.ws.inference.get_runner", lambda name: SlowRunner())
    client = TestClient(app)

    with client.websocket_connect("/ws/inference") as websocket:
        websocket.send_json({"type": "session_start", "session_id": "session-drain"})
        assert websocket.receive_json()["type"] == "session_started"
        websocket.send_json(
            {
                "type": "frame_meta",
                "frame_id": "frame-slow",
                "captured_at": "2026-05-07T12:00:01Z",
                "content_type": "image/jpeg",
            }
        )
        websocket.send_bytes(b"\xff\xd8mock-jpeg")
        websocket.send_json({"type": "session_end", "session_id": "session-drain"})

        assert websocket.receive_json()["code"] == "queue_drain_timeout"
        assert websocket.receive_json() == {
            "type": "session_ended",
            "session_id": "session-drain",
        }

    assert len(recording_session_lifecycle.ends) == 1
    assert recording_session_lifecycle.ends[0].session_id == "session-drain"
    assert recording_session_lifecycle.ends[0].close_reason == "queue_drain_timeout"
