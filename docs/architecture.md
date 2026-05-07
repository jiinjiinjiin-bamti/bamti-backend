# Architecture

This document describes the current repository structure and implemented runtime
shape of the DMS backend.

## Runtime Entry Point

- `app/main.py` creates the FastAPI application.
- HTTP routes are mounted under `/api`.
- WebSocket routes are mounted under `/ws`.

Current route modules:

- `app/api/routes.py` includes HTTP route modules.
- `app/api/health.py` implements `GET /api/health`.
- `app/ws/inference.py` implements `GET /ws/inference`.

## Module Boundaries

The code is split into four main application areas:

- `app/ws`
  - Owns the WebSocket endpoint and per-session queue handling.
  - `inference.py` handles the WebSocket message flow.
  - `manager.py` manages bounded queues for active sessions.
- `app/inference`
  - Owns inference schemas, runner interface, runner selection, and the mock runner.
  - `runner.py` defines `InferenceRunner`.
  - `mock_runner.py` implements `MockRunner`.
  - `manifest.py` selects a runner by name.
- `app/alerts`
  - Owns alert evaluation.
  - `engine.py` converts an `InferenceResult` into zero or more alerts.
- `app/storage`
  - Owns database setup, SQLAlchemy models, schema initialization, and repositories.
  - No raw frames or per-frame inference results are modeled for persistence.

## WebSocket Inference Flow

The implemented endpoint is:

```text
GET /ws/inference
```

Implemented flow:

1. Client sends `session_start` JSON.
2. Server creates a `driving_session` row, creates a session queue, starts a background result worker, and sends `session_started`.
3. Client may send `ping`; server replies with `pong`.
4. Client sends `frame_meta` JSON with `content_type` set to `image/jpeg`.
5. Client sends a binary JPEG frame.
6. Server validates the frame, queues it, runs the configured runner, evaluates alerts, and sends `inference_result`.
7. Client sends `session_end` JSON or disconnects.

The WebSocket implementation uses one `asyncio.Queue` per session. Queue size is
configured by `FRAME_QUEUE_SIZE` and defaults to `4`.

When a queue is full, `WebSocketSessionManager.put_latest()` removes one pending
frame before enqueueing the newest frame. This favors low latency over processing
every frame.

The WebSocket endpoint also rejects empty frames, frames larger than
`MAX_FRAME_BYTES`, and frame metadata whose `content_type` is not `image/jpeg`.
If no frame or control message arrives before `WEBSOCKET_IDLE_TIMEOUT_SECONDS`,
the server sends an `idle_timeout` error and closes the connection normally.
On `session_end`, the endpoint waits up to `WEBSOCKET_DRAIN_TIMEOUT_SECONDS` for
queued frames to finish, marks the `driving_session` ended, creates or updates
`session_summary`, and then sends `session_ended`.

If a client disconnects, idle timeout occurs, or queue drain timeout occurs while
a session is open, the endpoint attempts to mark the `driving_session` ended and
create or update `session_summary` during cleanup. The current database schema
does not distinguish normal and abnormal close reasons; close reasons are logged.

The WebSocket contract is documented in `docs/websocket-contract.md`.

## Inference

`InferenceRunner` is the current runner interface:

```python
async def infer(self, frame: bytes) -> InferenceResult
```

Current implemented runner:

- `MockRunner`
  - Ignores the frame bytes.
  - Always returns an attentive result with high confidence.

Runner selection is done by `get_runner()` in `app/inference/manifest.py`.
Currently only the name `mock` is supported.

## Alerts

`AlertEngine.evaluate()` accepts an `InferenceResult` and returns a list of
alerts.

Current behavior:

- If `is_distracted` is false, no alerts are returned.
- If `is_distracted` is true, one `distraction_detected` warning alert is returned.

The alert engine does not call model runners directly.

## Storage

Database settings are defined in `app/core/config.py`.

The async SQLAlchemy engine and session factory live in `app/storage/database.py`.
The schema can be created with:

```bash
python -m app.storage.init_db
```

Current SQLAlchemy models:

- `DrivingSession`
  - Table: `driving_session`
  - Tracks session id, optional driver id, status, start/end timestamps, and audit timestamps.
- `DistractionEvent`
  - Table: `distraction_event`
  - Tracks session-linked distraction events.
- `SessionSummary`
  - Table: `session_summary`
  - Tracks one summary row per session.

Current repositories:

- `DrivingSessionRepository`
- `DistractionEventRepository`
- `SessionSummaryRepository`

The WebSocket lifecycle writes `driving_session` and `session_summary` through a
small helper in `app/ws/lifecycle.py`. Raw frames and per-frame inference results
are not passed to persistence. `distraction_event` persistence is intentionally
not wired yet.

## Deployment Shape

The repository includes:

- `Dockerfile` for the FastAPI API container.
- `docker-compose.yml` with `api`, `mysql`, and `nginx` services.
- `nginx/conf.d/dms.conf` for HTTP reverse proxy.
- `nginx/conf.d/dms-https.conf.example` for HTTPS reverse proxy shape.

Current Nginx HTTP routing:

- `/` serves static files from `/usr/share/nginx/html`.
- `/api/` proxies to `api:8000/api/`.
- `/ws/` proxies WebSocket traffic to `api:8000/ws/`.

The Compose file mounts `./frontend-dist` as the Nginx static root, but this
repository currently does not contain a frontend build.
