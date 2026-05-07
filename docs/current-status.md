# Current Status

This document captures the implementation state of the repository as it exists
now. It is meant to help future Codex sessions quickly recover project context.

## Implemented

- FastAPI application skeleton exists in `app/main.py`.
- HTTP router is mounted under `/api`.
- WebSocket router is mounted under `/ws`.
- `GET /api/health` is implemented.
- `GET /ws/inference` is implemented.
- WebSocket flow supports:
  - `session_start`
  - `driving_session` creation on `session_start`
  - `session_started`
  - app-level `ping` / `pong`
  - `frame_meta`
  - binary JPEG frame
  - frame validation for content type, empty frames, and oversized frames
  - `inference_result`
  - `session_end`
  - `session_ended`
  - `driving_session` end cleanup on `session_end`, disconnect, idle timeout,
    and queue drain timeout
  - `session_summary` creation on session cleanup
  - idle timeout
  - disconnect cleanup
- A per-session bounded `asyncio.Queue` is used.
- Queue overflow drops an older pending frame before enqueueing the newest frame.
- `session_end` waits briefly for queued frames to finish, bounded by
  `WEBSOCKET_DRAIN_TIMEOUT_SECONDS`.
- `InferenceRunner` interface exists.
- `MockRunner` exists and is the only working runner.
- `get_runner()` selects the mock runner when `INFERENCE_RUNNER=mock`.
- `AlertEngine` exists and is separate from inference runners.
- SQLAlchemy async database setup exists.
- SQLAlchemy models exist for:
  - `driving_session`
  - `distraction_event`
  - `session_summary`
- Repository classes exist for sessions, events, and summaries.
- WebSocket lifecycle persistence is isolated in `app/ws/lifecycle.py`.
- Dockerfile exists.
- Docker Compose defines `api`, `mysql`, and `nginx`.
- Nginx config proxies `/api/` and `/ws/`.
- HTTPS Nginx example config exists.
- README contains local, Docker, test, WebSocket, and HTTPS notes.
- GitHub Actions CI workflow exists.
- Pytest coverage exists for health, mock runner, alert engine, normal WebSocket flow,
  invalid WebSocket frames, and `session_end`.
- `scripts/ws_smoke_test.py` exists for optional Docker Compose MySQL verification
  of WebSocket session persistence.

## Verified Locally

The following commands should be run after WebSocket contract changes:

```bash
python -m compileall app
python -m pytest
docker compose config
```

Optional MySQL-backed WebSocket persistence smoke test:

```bash
docker compose up --build
docker compose exec api python -m app.storage.init_db
docker compose exec api python scripts/ws_smoke_test.py
```

## Not Implemented

- `PytorchRunner` is not implemented.
- `OnnxRunner` is not implemented.
- Model manifest loading from a file is not implemented.
- `distraction_event` writes are not integrated into the WebSocket inference flow.
- Alembic migrations are not present.
- Authentication and authorization are not present.
- A frontend is not included in this repository.
- Actual Let's Encrypt certificate issuance automation is not included.
- Deployment to a server from GitHub Actions is not implemented.
- Raw frame storage is not implemented, by design.
- Per-frame inference result storage is not implemented, by design.

## Important Files

- `AGENTS.md`: project rules for future agent sessions.
- `docs/websocket-contract.md`: current WebSocket contract.
- `app/main.py`: FastAPI app creation and router mounting.
- `app/ws/inference.py`: WebSocket inference endpoint.
- `app/ws/lifecycle.py`: WebSocket session persistence helper.
- `app/ws/manager.py`: per-session queue management.
- `app/inference/runner.py`: runner interface.
- `app/inference/mock_runner.py`: MVP runner.
- `app/inference/manifest.py`: runner selection.
- `app/alerts/engine.py`: alert logic.
- `app/storage/models.py`: SQLAlchemy models.
- `app/storage/repositories.py`: repository classes.
- `docker-compose.yml`: local container composition.
- `nginx/conf.d/dms.conf`: HTTP reverse proxy config.
- `.github/workflows/ci.yml`: CI workflow.

## Known Caveats

- WebSocket JSON parsing and transport-level validation are implemented inside the endpoint.
- Frame content type validation currently trusts the client-provided
  `frame_meta.content_type`; JPEG magic bytes are not validated yet.
- The database schema records only `active` or `ended` session status. Normal
  `session_end`, client disconnect, idle timeout, and queue drain timeout are
  distinguishable in logs but not in persisted session rows.
- `app/storage/init_db.py` uses `Base.metadata.create_all`; there is no migration
  history yet.
- The repository folder was not initialized as a Git repository during the previous
  session.
