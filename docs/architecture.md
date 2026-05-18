# Architecture

This backend exposes FastAPI inference APIs for BAMTI DMS.

The active architecture supports both HTTP frame upload and WebSocket stream inference. It also supports two model profiles: BAMTI 7-class and AIHub 3-class.

## Runtime Entry Point

- `app/main.py` creates the FastAPI application.
- `app/api/routes.py` mounts all API routers under `/api`.
- `/api/health` remains unversioned.
- Versioned inference APIs live under `/api/v*`.
- AIHub inference APIs live under `/api/aihub/*`.

## API Shape

```text
GET /api/health

POST /api/v1/inference/frame
POST /api/v1/telemetry/runs
GET  /api/v1/telemetry/runs

WS   /api/v2/inference/stream
WS   /api/v3/inference/stream

POST /api/v4/inference/frame
WS   /api/v4/inference/stream
WS   /api/v4/debug/inference/stream

WS   /api/v5/inference/stream

POST /api/v6/inference/frame
WS   /api/v6/inference/stream

POST /api/aihub/v4/inference/frame
WS   /api/aihub/v4/inference/stream
POST /api/aihub/v6/inference/frame
WS   /api/aihub/v6/inference/stream
```

## Module Boundaries

- `app/api`
  - Owns routing, request parsing, response formatting, and WebSocket protocol handling.
- `app/inference`
  - Owns model loading, preprocessing, runner selection, score mapping, score averaging, and telemetry.
- `app/core`
  - Owns environment-driven settings.
- `app/storage`
  - Keeps SQLAlchemy database models and repository code for later persistence work.

## Inference Flow

REST frame endpoints accept one JPEG frame using `multipart/form-data`.

Expected form fields:

- `frame`: uploaded JPEG file.
- `frameId`: optional client frame id.
- `clientSentAt`: optional client timestamp string.
- `sessionId`: optional smoothing key for v6 frame endpoints.

Processing steps:

1. Validate `Content-Type` is `image/jpeg`.
2. Read up to `MAX_FRAME_BYTES`.
3. Reject empty or oversized frames.
4. Select the model runner from `app/inference/manifest.py`.
5. Decode JPEG bytes.
6. Resize to `MODEL_INPUT_SIZE`.
7. Normalize with ImageNet mean/std.
8. Run the model.
9. Convert logits using configured activation.
10. Map raw model outputs into service detections.
11. Apply v6 score averaging when the route requires it.
12. Return detections, model metadata, and telemetry.

## WebSocket Flow

WebSocket stream endpoints use a latest-pending policy.

The frontend sends a session start message, then alternates frame metadata and binary JPEG payloads. If a new frame arrives while inference is still processing, the backend keeps only the latest pending frame and drops older pending frames.

This protects the server from unbounded queue growth while allowing the frontend to send at a fixed target FPS.

## Model Runner

Runner selection is handled by `app/inference/manifest.py`.

Important runners:

- `bamti-torch`: BAMTI 7-class model.
- `bamti-torch-debug-raw`: BAMTI 7-class model with raw score debug output.
- `aihub-torch`: AIHub 3-class model.

Model loading is handled by `app/inference/model_loader.py`.

The loader supports:

- `timm` `vit_base_patch16_224` custom checkpoint for BAMTI 7-class.
- `torchvision.models.vit_b_16` checkpoint for AIHub 3-class.

The active model cache keeps one active loaded model. Switching model paths releases the previous cached model to reduce memory pressure.

## Score Mapping

BAMTI 7-class mapping is defined in `app/inference/class_mapping.py`.

Raw classes:

```text
A1, A2, ..., A16
```

Service detections:

| Variable | Raw classes |
|---|---|
| `normal_driving` | A1 |
| `phone_use` | A5, A6, A7, A8, A9 |
| `vehicle_device_operation` | A3, A4 |
| `face_action` | A2, A13, A14, A16 |
| `distraction` | A10, A11 |
| `drowsiness` | A12 |
| `rear_seat_interaction` | A15 |

Grouped service detections use max raw score.

## Score Smoothing

v4 returns immediate model scores.

v6 applies a one-second rolling average using `app/inference/score_averaging.py`.

Smoothing is keyed by session where available.

## Telemetry Runs

`POST /api/v1/telemetry/runs` stores frontend one-minute performance measurement payloads as JSON files.

Default path:

```text
backend/telemetry_runs/
```

The directory is ignored by Git.

## Deployment Shape

In production, Nginx runs natively and proxies `/api/` to FastAPI.

```text
public domain
  |
  v
native Nginx
  |-- /      -> frontend process/static serving
  `-- /api/* -> FastAPI backend at 127.0.0.1:8000
```

Docker Compose mounts the workspace model directory into the API container:

```text
../model -> /models
MODEL_PATH=/models/exp04_pseudo_ir_aug.pth
AIHUB_MODEL_PATH=/models/final_model.pth
```
