# DMS Backend

Driver Monitoring System(DMS) 백엔드입니다.

현재 백엔드는 프론트엔드 연동을 위해 HTTP 기반 `/api/v1` 추론 API를
제공합니다. WebSocket은 아직 활성 구현 범위가 아니며, 실제 모델 연동과
1분 성능 측정 기록을 먼저 안정화합니다.

## 현재 API

```text
GET  /api/health
GET  /api/v1/detection-classes
POST /api/v1/inference/frame
POST /api/v1/telemetry/runs
GET  /api/v1/telemetry/runs
```

## 주요 구조

```text
app/
|-- api/
|   |-- health.py
|   |-- routes.py
|   `-- v1/
|       |-- inference.py
|       |-- routes.py
|       |-- schemas.py
|       `-- telemetry.py
|-- core/
|   `-- config.py
|-- inference/
|   |-- manifest.py
|   |-- model_loader.py
|   |-- preprocessing.py
|   |-- runner.py
|   |-- schemas.py
|   |-- telemetry.py
|   `-- torch_runner.py
`-- storage/
```

## 모델 연동

기본 runner는 실제 PyTorch 모델을 사용하는 `bamti-torch`입니다.

기본 모델 경로:

```text
../model/final_model.pth
```

checkpoint는 다음 값을 포함해야 합니다.

- `model_state_dict`
- `class_names`

현재 프론트엔드는 checkpoint의 class name을 기준으로 다음 클래스들을 기대합니다.

- `forward_inattention`
- `surrounding_inattention`
- `vehicle_interaction`

## 환경 변수

```text
APP_NAME="DMS Backend"
ENVIRONMENT=local
INFERENCE_RUNNER=bamti-torch
MODEL_PATH=../model/final_model.pth
MODEL_DEVICE=cpu
MODEL_INPUT_SIZE=224
MODEL_SCORE_ACTIVATION=softmax
TELEMETRY_RUNS_DIR=./telemetry_runs
MAX_FRAME_BYTES=1048576
DATABASE_URL=mysql+aiomysql://dms_user:dms_password@mysql:3306/dms
```

`MODEL_DEVICE`는 사용 가능한 환경에 따라 `cpu`, `mps`, `cuda`를 사용할 수 있습니다.
요청한 device가 사용할 수 없으면 runner는 CPU로 fallback합니다.

## 로컬 실행

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

모델 클래스 확인:

```bash
curl http://127.0.0.1:8000/api/v1/detection-classes
```

## Docker Compose

```bash
docker compose up --build
```

Compose는 workspace의 `../model` 디렉터리를 API 컨테이너의 `/models`로
마운트하고, 기본 `MODEL_PATH`를 `/models/final_model.pth`로 설정합니다.

Nginx는 `/api/` 요청만 FastAPI로 proxy합니다.

MySQL은 현재 v1 HTTP 추론 경로에서 사용하지 않으므로 기본 Compose 실행에는
포함하지 않습니다. 나중에 persistence 검증이 필요하면 profile로 실행합니다.

```bash
docker compose --profile persistence up --build
```

## 1분 성능 측정 기록

프론트엔드는 1분 성능 측정 종료 시 다음 API로 payload를 보냅니다.

```text
POST /api/v1/telemetry/runs
```

백엔드는 payload를 `TELEMETRY_RUNS_DIR` 아래 JSON 파일로 저장합니다.
기본 디렉터리인 `telemetry_runs/`는 Git에 포함하지 않습니다.

저장된 파일 목록은 다음 API로 조회합니다.

```text
GET /api/v1/telemetry/runs
```

## 검증

```bash
python -m compileall app
python -m pytest
docker compose config
```

## 문서

- [`docs/architecture.md`](docs/architecture.md): 현재 HTTP v1 중심 구조
- [`docs/current-status.md`](docs/current-status.md): 구현 현황
- [`docs/decisions.md`](docs/decisions.md): 설계 결정
- [`docs/roadmap.md`](docs/roadmap.md): 다음 작업
