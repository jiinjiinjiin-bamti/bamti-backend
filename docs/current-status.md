# Current Status

This document captures the current implementation state of the backend.

## Implemented

- FastAPI application entrypoint in `app/main.py`.
- HTTP router mounted under `/api`.
- Versioned v1 API package in `app/api/v1`.
- `GET /api/health`.
- `GET /api/v1/detection-classes`.
- `POST /api/v1/inference/frame`.
- `POST /api/v1/telemetry/runs`.
- `GET /api/v1/telemetry/runs`.
- Real PyTorch model runner selected by `INFERENCE_RUNNER=bamti-torch`.
- ViT-B/16 checkpoint loading from `MODEL_PATH`.
- JPEG bytes to RGB tensor preprocessing.
- Softmax/sigmoid score activation selection through `MODEL_SCORE_ACTIVATION`.
- Per-frame telemetry:
  - `processingFps`
  - `preprocessMs`
  - `inferenceMs`
  - `postprocessMs`
  - `serverTotalMs`
- One-minute performance telemetry JSON file persistence under `TELEMETRY_RUNS_DIR`.
- SQLAlchemy async MySQL storage modules remain present for later persistence work.
- MySQL Compose service is behind the `persistence` profile because v1 HTTP
  inference does not require database startup.
- Docker Compose mounts `../model` into the API container at `/models`.
- Nginx proxies `/api/` to FastAPI.
- Tests cover health, v1 inference contract, and v1 telemetry JSON persistence.

## Current API Surface

```text
GET  /api/health
GET  /api/v1/detection-classes
POST /api/v1/inference/frame
POST /api/v1/telemetry/runs
GET  /api/v1/telemetry/runs
```

## Not Implemented

- WebSocket inference is intentionally not implemented in the active backend.
- Runtime alert/event persistence is not connected to inference.
- `distraction_event` write policy is not implemented.
- Alembic migrations are not present.
- Authentication and authorization are not present.
- ONNX Runtime runner is not implemented.
- Deployment from GitHub Actions is not implemented.

## Important Files

- `app/main.py`: FastAPI app creation and router mounting.
- `app/api/routes.py`: root HTTP router.
- `app/api/v1/routes.py`: v1 router.
- `app/api/v1/inference.py`: v1 HTTP frame inference endpoints.
- `app/api/v1/telemetry.py`: v1 telemetry run save/list endpoints.
- `app/inference/runner.py`: runner interface.
- `app/inference/torch_runner.py`: real PyTorch runner.
- `app/inference/model_loader.py`: checkpoint loading.
- `app/inference/preprocessing.py`: JPEG preprocessing.
- `app/inference/telemetry.py`: processing FPS counter.
- `app/storage/models.py`: SQLAlchemy models retained for later persistence work.
- `docker-compose.yml`: local API/MySQL/Nginx composition.
- `nginx/conf.d/dms.conf`: HTTP reverse proxy config.

## Verification Commands

```bash
python -m compileall app
python -m pytest
docker compose config
```

## Known Caveats

- `GET /api/v1/detection-classes` loads the configured model to read checkpoint
  class names. If `MODEL_PATH` is missing, the endpoint fails.
- Docker Compose expects the workspace-level `model/final_model.pth` to exist and
  mounts `../model` into the API container.
- `POST /api/v1/inference/frame` validates the uploaded content type and byte
  size, but does not yet perform JPEG magic-byte validation before preprocessing.
- The active transport is HTTP. WebSocket work should be reintroduced later as a
  separate versioned design, not mixed into the current v1 HTTP integration.
