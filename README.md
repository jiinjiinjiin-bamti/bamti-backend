# DMS Backend

Driver Monitoring System backend for BAMTI.

The backend exposes FastAPI endpoints for real model inference. It supports both the current BAMTI 7-class model profile and the restored AIHub 3-class profile.

## Current Capabilities

- Health check API
- REST single-frame inference
- WebSocket latest-pending inference
- BAMTI 7-class model inference
- AIHub 3-class model inference
- v4 realtime score responses
- v6 one-second rolling average score responses
- v4 raw score debug stream
- One-minute telemetry JSON persistence
- CPU Docker execution
- CUDA Docker execution

## API Surface

All routes are mounted under `/api`.

### Common

```text
GET /api/health
```

### BAMTI 7-class

```text
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
```

### AIHub 3-class

```text
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

## API Version Behavior

v4:

- Returns model score responses immediately.
- Used for realtime score checks and debugging.

v6:

- Aggregates detection scores in a one-second rolling window.
- Returns averaged scores per session.
- Intended for more stable threshold-based UI updates.

v4 debug:

- `WS /api/v4/debug/inference/stream`
- Exposes raw BAMTI `A1`-`A16` scores for debugging.
- Not intended as the default production route.

## Project Structure

```text
app/
|-- api/
|   |-- health.py
|   |-- routes.py
|   |-- v1/
|   |-- v2/
|   |-- v3/
|   |-- v4/
|   |-- v5/
|   |-- v6/
|   `-- aihub/
|-- core/
|   `-- config.py
|-- inference/
|   |-- class_mapping.py
|   |-- manifest.py
|   |-- model_loader.py
|   |-- preprocessing.py
|   |-- runner.py
|   |-- schemas.py
|   |-- score_averaging.py
|   |-- telemetry.py
|   `-- torch_runner.py
`-- storage/
```

## Model Profiles

### BAMTI 7-class

The current BAMTI model uses a custom `timm` ViT-B/16 checkpoint and emits raw action classes `A1` through `A16`.

Service-level mapping:

| Variable | Display | Raw classes |
|---|---|---|
| `normal_driving` | 정상 주행 | A1 |
| `phone_use` | 휴대기기 조작 | A5, A6, A7, A8, A9 |
| `vehicle_device_operation` | 차량 장치 조작 | A3, A4 |
| `face_action` | 얼굴 행동 | A2, A13, A14, A16 |
| `distraction` | 주의 분산 | A10, A11 |
| `drowsiness` | 졸음 | A12 |
| `rear_seat_interaction` | 뒷좌석 상호작용 | A15 |

When multiple raw classes map to one service class, the backend uses the maximum raw score.

### AIHub 3-class

The AIHub profile uses the legacy model and the following service classes:

| Variable | Display |
|---|---|
| `forward_inattention` | 전방 주의 소홀 |
| `surrounding_inattention` | 주변 주의 소홀 |
| `vehicle_interaction` | 차량 간 상호작용 |

## Preprocessing

Frames are expected as JPEG images.

The backend preprocessing is:

```python
Image.open(...).convert("RGB")
Resize((224, 224))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

This matches the ImageNet mean/std transform used during training.

## Environment Variables

```text
APP_NAME="DMS Backend"
ENVIRONMENT=local
INFERENCE_RUNNER=bamti-torch
MODEL_PATH=/models/exp04_pseudo_ir_aug.pth
AIHUB_MODEL_PATH=/models/final_model.pth
MODEL_DEVICE=cpu
MODEL_INPUT_SIZE=224
MODEL_SCORE_ACTIVATION=softmax
TORCH_NUM_THREADS=5
TORCH_COMPILE_BACKEND=inductor
TORCH_COMPILE_MODE=reduce-overhead
TELEMETRY_RUNS_DIR=./telemetry_runs
MAX_FRAME_BYTES=1048576
DATABASE_URL=mysql+aiomysql://dms_user:change-me@mysql:3306/dms
```

`MODEL_DEVICE` can be `cpu`, `mps`, or `cuda`. If the requested device is not available, the runner falls back to CPU.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Detection classes:

```bash
curl http://127.0.0.1:8000/api/v4/detection-classes
curl http://127.0.0.1:8000/api/aihub/v4/detection-classes
```

## Docker Compose

CPU:

```bash
docker compose --env-file .env -f docker-compose.yml up -d --build
```

CUDA:

```bash
docker compose --env-file .env.cuda -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

Compose mounts the workspace model directory into the API container at `/models`. Model paths must be container paths, not host paths.

Recommended CUDA model paths:

```env
MODEL_PATH=/models/exp04_pseudo_ir_aug.pth
AIHUB_MODEL_PATH=/models/final_model.pth
MODEL_DEVICE=cuda
```

## Telemetry Runs

The frontend can save one-minute performance measurement payloads through:

```text
POST /api/v1/telemetry/runs
```

Saved files are written under `TELEMETRY_RUNS_DIR`, which is ignored by Git.

List saved runs:

```text
GET /api/v1/telemetry/runs
```

## Verification

```bash
python -m compileall app
python -m pytest
docker compose config
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/current-status.md`](docs/current-status.md)
- [`docs/decisions.md`](docs/decisions.md)
- [`docs/roadmap.md`](docs/roadmap.md)
