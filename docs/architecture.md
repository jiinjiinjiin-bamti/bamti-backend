# Architecture

This backend currently exposes a versioned HTTP API for frontend integration.
WebSocket inference is intentionally deferred.

## Runtime Entry Point

- `app/main.py` creates the FastAPI application.
- HTTP routes are mounted under `/api`.
- `app/api/routes.py` includes `GET /api/health` and the versioned v1 router.
- `app/api/v1/routes.py` mounts v1 inference and telemetry endpoints.

## API Shape

```text
GET  /api/health
GET  /api/v1/detection-classes
POST /api/v1/inference/frame
POST /api/v1/telemetry/runs
GET  /api/v1/telemetry/runs
```

`/api/health` is not versioned because it describes the API process itself.
Model inference and performance telemetry are versioned under `/api/v1`.

## Module Boundaries

- `app/api`
  - Owns HTTP routing and request/response schemas.
- `app/inference`
  - Owns model loading, preprocessing, runner selection, inference execution, and
    per-frame telemetry.
- `app/storage`
  - Keeps SQLAlchemy database models and repository code for later persistence
    work. The current v1 inference endpoint does not write frame results to the
    database.
- `app/core`
  - Owns environment-driven settings.

## Inference Flow

`POST /api/v1/inference/frame` accepts one JPEG frame using
`multipart/form-data`.

Expected form fields:

- `frame`: uploaded JPEG file.
- `frameId`: optional client frame id.
- `clientSentAt`: optional client timestamp string.

Processing steps:

1. Validate `Content-Type` is `image/jpeg`.
2. Read up to `MAX_FRAME_BYTES`.
3. Reject empty or oversized frames.
4. Run the configured `InferenceRunner`.
5. Decode JPEG bytes and resize to `MODEL_INPUT_SIZE`.
6. Normalize with ImageNet mean/std.
7. Run the ViT-B/16 model checkpoint.
8. Convert logits with `softmax` or `sigmoid`.
9. Return class scores and telemetry.

Response fields include:

- `frameId`
- `clientSentAt`
- `serverReceivedAt`
- `serverRespondedAt`
- `detections`
- `model`
- `telemetry`

## Model Runner

The active runner is `BamtiTorchRunner`.

Runner selection:

```text
INFERENCE_RUNNER=bamti-torch
```

The runner loads `MODEL_PATH`, expects a checkpoint with `model_state_dict` and
`class_names`, and builds a `torchvision.models.vit_b_16` model with the number
of checkpoint classes.

The current model classes are read from the checkpoint and surfaced through
`GET /api/v1/detection-classes`.

## Telemetry Runs

`POST /api/v1/telemetry/runs` stores the frontend's one-minute performance
measurement payload as a JSON file.

Default path:

```text
backend/telemetry_runs/
```

The directory is ignored by Git. The matching `GET /api/v1/telemetry/runs`
endpoint lists saved JSON files.

## Deployment Shape

Nginx serves frontend static files and proxies only API requests:

```text
public domain
  |
  v
Nginx
  |-- /      -> frontend static files
  `-- /api/* -> FastAPI HTTP
```

Docker Compose mounts the workspace model directory into the API container:

```text
../model -> /models
MODEL_PATH=/models/final_model.pth
```

The MySQL service is available behind the `persistence` Compose profile. It is
not required for the active v1 HTTP inference flow.
