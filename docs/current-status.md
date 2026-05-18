# Current Status

This document captures the current backend implementation state.

## Implemented

- FastAPI application entrypoint in `app/main.py`.
- Router mounting under `/api`.
- `GET /api/health`.
- v1 REST frame inference and telemetry run persistence.
- v2/v3 WebSocket inference experiments.
- v4 REST and WebSocket inference.
- v4 raw score debug WebSocket stream.
- v5 WebSocket torch compile experiment.
- v6 REST and WebSocket inference with one-second score averaging.
- AIHub 3-class REST and WebSocket routes.
- AIHub v4 and v6 route prefixes.
- Mobile session routes for BAMTI and AIHub variants.
- Real PyTorch runner selected through `app/inference/manifest.py`.
- `timm` custom ViT-B/16 checkpoint support for BAMTI 7-class.
- `torchvision` ViT-B/16 checkpoint support for AIHub 3-class.
- JPEG bytes to RGB tensor preprocessing.
- ImageNet mean/std normalization.
- Softmax/sigmoid score activation selection through `MODEL_SCORE_ACTIVATION`.
- BAMTI raw `A1`-`A16` to service detection mapping.
- Max-score aggregation for grouped BAMTI service classes.
- Per-frame telemetry:
  - `processingFps`
  - `preprocessMs`
  - `inferenceMs`
  - `postprocessMs`
  - `serverTotalMs`
- One-minute performance telemetry JSON file persistence under `TELEMETRY_RUNS_DIR`.
- CPU Docker configuration.
- CUDA Docker configuration.
- Tests for health, inference contracts, WebSocket behavior, AIHub routes, model cache, and service score mapping.

## Current API Surface

```text
GET  /api/health

GET  /api/v1/detection-classes
POST /api/v1/inference/frame
POST /api/v1/telemetry/runs
GET  /api/v1/telemetry/runs

WS   /api/v2/inference/stream
WS   /api/v3/inference/stream

GET  /api/v4/detection-classes
POST /api/v4/inference/frame
WS   /api/v4/inference/stream
WS   /api/v4/debug/inference/stream

WS   /api/v5/inference/stream

GET  /api/v6/detection-classes
POST /api/v6/inference/frame
WS   /api/v6/inference/stream

GET  /api/aihub/detection-classes
POST /api/aihub/inference/frame
WS   /api/aihub/inference/stream

GET  /api/aihub/v4/detection-classes
POST /api/aihub/v4/inference/frame
WS   /api/aihub/v4/inference/stream

GET  /api/aihub/v6/detection-classes
POST /api/aihub/v6/inference/frame
WS   /api/aihub/v6/inference/stream
```

## Important Files

- `app/main.py`: FastAPI app creation and router mounting.
- `app/api/routes.py`: root API router.
- `app/api/v4/`: BAMTI v4 REST, WebSocket, debug stream, mobile routes.
- `app/api/v6/`: BAMTI v6 REST, WebSocket, mobile routes.
- `app/api/aihub/`: AIHub 3-class routes.
- `app/inference/class_mapping.py`: BAMTI 7-class service mapping.
- `app/inference/manifest.py`: runner and model manifest selection.
- `app/inference/model_loader.py`: checkpoint loading and active model cache.
- `app/inference/preprocessing.py`: JPEG preprocessing.
- `app/inference/score_averaging.py`: v6 rolling score averaging.
- `app/inference/telemetry.py`: processing FPS and latency metrics.
- `app/inference/torch_runner.py`: real PyTorch inference runner.
- `docker-compose.yml`: base Docker Compose.
- `docker-compose.cuda.yml`: CUDA override.
- `Dockerfile.cuda`: CUDA image.

## Not Implemented

- Authentication and authorization are not present.
- Alembic migrations are not present.
- ONNX Runtime runner is not implemented.
- Deployment from GitHub Actions is not implemented.
- Runtime database persistence for detected events is not connected.

## Verification Commands

```bash
python -m compileall app
python -m pytest
docker compose config
```

CUDA compose config:

```bash
docker compose --env-file .env.cuda -f docker-compose.yml -f docker-compose.cuda.yml config
```

## Known Caveats

- Docker model paths must be container paths such as `/models/exp04_pseudo_ir_aug.pth`.
- Missing model files fail at model load time with `FileNotFoundError`.
- The active model cache keeps one loaded model at a time. Alternating between BAMTI and AIHub profiles reloads models.
- WebSocket latest-pending streams may report dropped pending frames when the client sends faster than inference can complete.
- REST frame endpoints validate content type and size, then rely on PIL decode during preprocessing.
