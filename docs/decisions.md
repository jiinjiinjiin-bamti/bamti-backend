# Decisions

This document records architecture and implementation decisions that are visible
in the current codebase.

## Keep Runtime Modules Separate

Decision:

- Keep WebSocket handling, inference, alerting, and storage in separate modules.

Current implementation:

- `app/ws`
- `app/inference`
- `app/alerts`
- `app/storage`

Reason:

- The project needs to swap inference implementations later without entangling
  WebSocket transport, alert rules, or persistence.

## Use a Runner Interface

Decision:

- Model execution goes through `InferenceRunner`.

Current implementation:

- `app/inference/runner.py` defines the interface.
- `app/inference/mock_runner.py` implements `MockRunner`.
- `app/inference/manifest.py` exposes `get_runner()`.

Current limit:

- Only `mock` is supported.
- There is no manifest file parser yet.

## Start With MockRunner

Decision:

- Build the end-to-end backend flow first with `MockRunner`.

Current implementation:

- `MockRunner.infer()` ignores the input frame and returns an attentive result.

Reason:

- This lets WebSocket behavior, queueing, alerts, tests, and container setup exist
  before real model integration.

## Keep Alert Logic Separate From Model Logic

Decision:

- Runners return inference results.
- `AlertEngine` converts results into alerts.

Current implementation:

- `app/ws/inference.py` calls the runner and then calls `AlertEngine`.
- `AlertEngine` does not call runner code.

## Do Not Persist Frames or Per-Frame Results

Decision:

- The database should not store raw frame bytes.
- The database should not store per-frame inference results.

Current implementation:

- SQLAlchemy models only cover sessions, distraction events, and session summaries.
- No table exists for frames or per-frame inference results.

## Prefer Low Latency Over Processing Every Frame

Decision:

- Use a bounded queue per WebSocket session.
- If the queue is full, drop an older pending frame and keep the newest one.

Current implementation:

- `WebSocketSessionManager.create_queue()` creates `asyncio.Queue(maxsize=N)`.
- `WebSocketSessionManager.put_latest()` drops one queued frame when full.

Reason:

- For real-time driver monitoring, stale frames are less useful than recent frames.

## Validate Frames At The WebSocket Boundary

Decision:

- Reject empty frames, oversized frames, and frames whose declared content type is
  not `image/jpeg` before enqueueing.
- Keep JPEG byte-level validation out of the MVP.

Current implementation:

- `FrameMeta.content_type` is validated from the client-provided metadata.
- The binary frame is checked for non-empty bytes and `MAX_FRAME_BYTES`.
- JPEG magic bytes are not validated yet.

Reason:

- The MVP needs predictable transport behavior without adding image decoding or
  heavier content inspection to the WebSocket endpoint.

## Use FastAPI and Pydantic

Decision:

- Use FastAPI for HTTP and WebSocket endpoints.
- Use Pydantic models for request/result structures where present.

Current implementation:

- `HealthResponse`, `SessionStartMessage`, `SessionEndMessage`, `FrameMeta`,
  `InferenceResult`, `QueuedFrame`, and `Alert` are Pydantic models.

## Use Async SQLAlchemy With MySQL

Decision:

- Use SQLAlchemy async engine with MySQL.

Current implementation:

- `DATABASE_URL` defaults to an `asyncmy` MySQL URL.
- `docker-compose.yml` runs `mysql:8.4`.
- `app/storage/database.py` creates an async engine and session factory.

Current limit:

- The WebSocket flow does not yet persist sessions, events, or summaries.
- Migrations are not implemented.

## Use Nginx as Reverse Proxy

Decision:

- Expose one public domain and route by path.

Current implementation:

- `/` serves frontend static files from the Nginx container.
- `/api/` proxies to FastAPI HTTP routes.
- `/ws/` proxies to FastAPI WebSocket routes.

Current limit:

- The repository does not include frontend files.
- HTTPS config is an example, not an automated certificate setup.
