# WebSocket Contract

Endpoint:

- `GET /ws/inference`

Message flow:

1. Client sends `session_start` JSON.
2. Server sends `session_started` JSON.
3. Client may send `ping` JSON at any time after connection.
4. Server replies with `pong` JSON.
5. Client sends `frame_meta` JSON.
6. Client sends binary JPEG frame.
7. Server sends `inference_result` JSON.
8. Client sends `session_end` JSON or disconnects.
9. Server sends `session_ended` JSON for a normal `session_end`.

The server prioritizes low latency. If the session queue is full, older pending
frames may be dropped.

If the server receives no frame or control message before the configured idle
timeout, it sends an `error` message with code `idle_timeout` and closes the
connection normally.

Clients that rely on heartbeat messages should send `ping` at an interval shorter
than `WEBSOCKET_IDLE_TIMEOUT_SECONDS`.

## Client Messages

### session_start

```json
{
  "type": "session_start",
  "session_id": "session-123",
  "driver_id": "driver-123",
  "started_at": "2026-05-07T12:00:00Z"
}
```

### frame_meta

```json
{
  "type": "frame_meta",
  "frame_id": "frame-1",
  "captured_at": "2026-05-07T12:00:01Z",
  "content_type": "image/jpeg"
}
```

`content_type` must be `image/jpeg`.

### binary JPEG frame

The binary message immediately following `frame_meta` is interpreted as the JPEG
frame for that metadata.

Validation rules:

- The binary frame must not be empty.
- The binary frame must not exceed the configured `MAX_FRAME_BYTES`.
- The preceding `frame_meta.content_type` must be `image/jpeg`.

Invalid frames are rejected with an `error` message and are not queued for
inference.

### ping

```json
{
  "type": "ping"
}
```

### session_end

```json
{
  "type": "session_end",
  "session_id": "session-123",
  "ended_at": "2026-05-07T12:30:00Z"
}
```

## Server Messages

### session_started

```json
{
  "type": "session_started",
  "session_id": "session-123"
}
```

### pong

```json
{
  "type": "pong"
}
```

### inference_result

```json
{
  "type": "inference_result",
  "session_id": "session-123",
  "frame_id": "frame-1",
  "captured_at": "2026-05-07T12:00:01Z",
  "result": {
    "is_distracted": false,
    "label": "attentive",
    "confidence": 0.98
  },
  "alerts": []
}
```

### session_ended

```json
{
  "type": "session_ended",
  "session_id": "session-123"
}
```

### error

```json
{
  "type": "error",
  "code": "empty_frame",
  "message": "Binary frame must not be empty."
}
```

Current error codes:

- `session_not_started`
- `frame_meta_required`
- `unsupported_content_type`
- `empty_frame`
- `frame_too_large`
- `frame_already_pending`
- `queue_drain_timeout`
- `unsupported_message`
- `invalid_message`
- `idle_timeout`
- `session_persistence_failed`
