# DMS Backend

Driver Monitoring System(DMS) 백엔드 레포지토리입니다.

이 프로젝트는 카메라로 촬영되는 운전자 프레임을 WebSocket으로 받아 모델 추론을 수행하고, 추론 결과와 알림 상태를 프론트엔드에 실시간으로 전달하기 위한 FastAPI 기반 백엔드입니다. 현재 단계는 실제 AI 모델을 붙이기 전, `MockRunner`로 HTTP, WebSocket, queue, persistence, Docker, Nginx 흐름을 먼저 검증하는 MVP입니다.

## 프로젝트 개요

DMS는 운전자 상태를 실시간으로 판단해야 하므로 일반 HTTP polling보다 WebSocket 연결이 더 적합합니다. 프론트엔드는 카메라 프레임을 백엔드로 보내고, 백엔드는 runner를 통해 추론한 뒤 `inference_result`와 `alerts`를 같은 WebSocket 연결로 돌려줍니다.

현재 백엔드의 목적은 다음입니다.

- FastAPI로 HTTP API와 WebSocket API를 제공한다.
- `/ws/inference`에서 운전자 프레임을 받아 runner 기반 추론 흐름을 실행한다.
- 모델 구현체는 `InferenceRunner` interface 뒤에 숨겨 교체 가능하게 둔다.
- 알림 판단은 모델 실행 코드와 분리한다.
- DB에는 세션 단위 데이터만 저장하고, raw frame과 per-frame inference result는 저장하지 않는다.
- Docker Compose와 Nginx로 로컬 reverse proxy 구조를 검증한다.

## 현재 MVP 범위

현재 동작하는 범위입니다.

- FastAPI app skeleton
- `GET /api/health`
- `GET /ws/inference`
- `MockRunner` 기반 inference flow
- `InferenceRunner` interface
- `AlertEngine` 분리
- WebSocket session별 `asyncio.Queue(maxsize=N)`
- queue가 꽉 찼을 때 오래된 frame을 drop하는 low-latency 정책
- app-level heartbeat
  - client: `{"type": "ping"}`
  - server: `{"type": "pong"}`
- WebSocket idle timeout
  - `WEBSOCKET_IDLE_TIMEOUT_SECONDS`
- `session_end` queue drain timeout
  - `WEBSOCKET_DRAIN_TIMEOUT_SECONDS`
- frame validation
  - empty binary frame reject
  - unsupported `content_type` reject
  - oversized frame reject
  - repeated `frame_meta` reject
  - `frame_meta` without session reject
  - binary without `frame_meta` reject
- `MAX_FRAME_BYTES`
- `frame_meta.content_type`
  - 현재 `image/jpeg`만 허용
- error response schema
  - `{"type": "error", "code": "...", "message": "..."}`
- SQLAlchemy async MySQL models/repositories
  - `driving_session`
  - `distraction_event`
  - `session_summary`
- WebSocket lifecycle persistence helper
  - `app/ws/lifecycle.py`
- `session_start` 시 `driving_session` 생성
- `session_end` 시 `driving_session` 종료
- `session_end` 시 `session_summary` 생성/저장
- client disconnect, `idle_timeout`, `queue_drain_timeout` 시 열린 session cleanup
- Dockerfile
- `docker-compose.yml`
- Nginx reverse proxy config
- GitHub Actions CI 기본 파일
- WebSocket 비정상 입력 테스트
- MySQL persistence smoke test script

현재 MVP에서 의도적으로 하지 않는 일입니다.

- raw frame 저장
- per-frame inference result 저장
- `PytorchRunner` 연결
- ONNX Runtime 연결
- `distraction_event` 저장 정책 구현
- `session_summary` 실제 통계 계산

## 기술 스택

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic / pydantic-settings
- SQLAlchemy async
- asyncmy
- MySQL 8.4
- Docker / Docker Compose
- Nginx
- GitHub Actions
- pytest

## 전체 아키텍처 요약

배포 목표 구조는 단일 공개 도메인을 Nginx가 받고, path 기반 reverse proxy로 프론트엔드와 백엔드를 나누는 방식입니다.

```text
public domain
  |
  v
Nginx
  |-- /      -> frontend static files
  |-- /api/* -> FastAPI HTTP
  |-- /ws/*  -> FastAPI WebSocket
```

현재 Docker Compose 기준 서비스 구성은 다음입니다.

```text
nginx:80
  |-- /api/ -> api:8000/api/
  |-- /ws/  -> api:8000/ws/
  |-- /     -> /usr/share/nginx/html

api:8000
  |-- FastAPI app
  |-- MockRunner
  |-- AlertEngine
  |-- WebSocket session queues

mysql:3306
  |-- dms database
  |-- driving_session
  |-- distraction_event
  |-- session_summary
```

모듈 경계는 다음 원칙을 따릅니다.

- `app/ws`: WebSocket transport, message validation, session queue 관리
- `app/inference`: runner interface, runner selection, inference schema, `MockRunner`
- `app/alerts`: inference result를 alert로 변환하는 독립 로직
- `app/storage`: SQLAlchemy database, models, repositories, schema initialization

이 경계를 유지하는 이유는 이후 `MockRunner`를 `PytorchRunner`, ONNX Runtime runner로 바꾸더라도 WebSocket contract, alert logic, DB persistence가 runner 구현 세부사항에 묶이지 않게 하기 위해서입니다.

## 요청 흐름

`/ws/inference`의 기본 흐름은 다음입니다.

1. Client가 WebSocket에 연결한다.
2. Client가 `session_start` JSON을 보낸다.
3. Server가 `driving_session` row를 생성한다.
4. Server가 session별 `asyncio.Queue(maxsize=N)`를 생성하고 background worker를 시작한다.
5. Server가 `session_started` JSON을 보낸다.
6. Client가 필요하면 `ping`을 보내고, server는 `pong`으로 응답한다.
7. Client가 `frame_meta` JSON을 보낸다.
8. Client가 바로 이어서 binary JPEG frame을 보낸다.
9. Server가 frame을 검증하고 queue에 넣는다.
10. Queue가 가득 차 있으면 오래된 pending frame을 하나 버리고 최신 frame을 넣는다.
11. Worker가 runner를 호출한다.
12. `AlertEngine`이 추론 결과를 alert로 변환한다.
13. Server가 `inference_result` JSON을 보낸다.
14. Client가 `session_end` JSON을 보내거나 연결을 종료한다.
15. 정상 `session_end`에서는 queue drain을 기다린 뒤 `driving_session` 종료와 `session_summary` 저장을 수행한다.
16. client disconnect, `idle_timeout`, `queue_drain_timeout`에서도 열린 session cleanup을 시도한다.

이 흐름에서 중요한 점은 실시간성을 우선한다는 것입니다. 운전자 모니터링에서는 오래된 frame을 모두 처리하는 것보다 최신 frame을 빠르게 처리하는 편이 더 유용하므로, queue overflow 시 오래된 frame을 drop합니다.

## 디렉터리 구조

```text
.
|-- app/
|   |-- api/
|   |   |-- health.py              # GET /api/health
|   |   `-- routes.py              # HTTP router mount
|   |-- alerts/
|   |   `-- engine.py              # AlertEngine
|   |-- core/
|   |   `-- config.py              # environment settings
|   |-- inference/
|   |   |-- manifest.py            # runner selection
|   |   |-- mock_runner.py         # MockRunner
|   |   |-- runner.py              # InferenceRunner interface
|   |   `-- schemas.py             # inference/frame schemas
|   |-- storage/
|   |   |-- database.py            # async SQLAlchemy engine/session
|   |   |-- init_db.py             # schema create helper
|   |   |-- models.py              # SQLAlchemy models
|   |   `-- repositories.py        # repository classes
|   |-- ws/
|   |   |-- inference.py           # GET /ws/inference
|   |   |-- lifecycle.py           # WebSocket persistence helper
|   |   `-- manager.py             # per-session queue manager
|   `-- main.py                    # FastAPI app entrypoint
|-- docs/
|   |-- architecture.md
|   |-- current-status.md
|   |-- decisions.md
|   |-- roadmap.md
|   `-- websocket-contract.md
|-- nginx/
|   `-- conf.d/
|       |-- dms.conf               # local HTTP reverse proxy
|       `-- dms-https.conf.example # HTTPS example shape
|-- scripts/
|   `-- ws_smoke_test.py           # WebSocket + MySQL smoke test
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- pytest.ini
`-- AGENTS.md
```

## 실행 방법

처음 실행할 때는 Docker Compose 기준 흐름을 권장합니다. 이 레포는 API, MySQL, Nginx가 함께 맞물려야 의미 있는 검증이 가능하므로, 컨테이너를 먼저 올린 뒤 DB schema를 만들고, HTTP와 WebSocket을 차례로 확인합니다.

### 1. 컨테이너 시작

```bash
docker compose up --build -d
```

이 명령은 `api`, `mysql`, `nginx` 서비스를 빌드하고 백그라운드로 실행합니다. `api`는 FastAPI 서버이고, `mysql`은 persistence 확인용 DB이며, `nginx`는 `/api/*`, `/ws/*` reverse proxy 동작을 확인하기 위해 필요합니다.

### 2. 컨테이너 상태 확인

```bash
docker compose ps
```

`mysql`이 healthy 상태가 된 뒤 다음 단계로 넘어가는 것이 좋습니다. DB가 준비되기 전에 schema 초기화나 smoke test를 실행하면 연결 실패가 날 수 있습니다.

## DB 초기화 방법

현재는 Alembic migration이 아니라 SQLAlchemy `Base.metadata.create_all` 기반의 로컬 MVP 초기화 방식을 사용합니다.

```bash
docker compose exec api sh -c "PYTHONPATH=/app python -m app.storage.init_db"
```

이 명령을 `api` 컨테이너 안에서 실행하는 이유는 Compose 네트워크 안에서는 MySQL host가 `mysql`이고, 기본 `DATABASE_URL`도 `mysql+asyncmy://dms_user:dms_password@mysql:3306/dms`를 바라보기 때문입니다.

테이블 생성 확인:

```bash
docker compose exec mysql mysql -udms_user -pdms_password dms -e "SHOW TABLES;"
```

현재 확인되어야 하는 테이블은 다음입니다.

- `driving_session`
- `distraction_event`
- `session_summary`

## Health check 방법

Nginx reverse proxy를 경유해 FastAPI health endpoint를 확인합니다.

```bash
curl http://localhost/api/health
```

이 확인이 중요한 이유는 단순히 FastAPI만 떠 있는지 보는 것이 아니라, 로컬 공개 진입점인 Nginx가 `/api/` 요청을 `api:8000`으로 제대로 전달하는지도 함께 확인하기 위해서입니다.

FastAPI를 직접 확인할 때는 다음 endpoint를 사용할 수 있습니다.

```text
http://localhost:8000/api/health
```

## FastAPI 직접 WebSocket smoke test 방법

FastAPI API 컨테이너에 직접 붙는 WebSocket 경로를 확인합니다.

```bash
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py"
```

기본 `ws_smoke_test.py`는 `ws://localhost:8000/ws/inference`로 연결합니다. 이 명령은 `api` 컨테이너 안에서 실행되므로 `localhost:8000`은 API 컨테이너 자신의 Uvicorn 서버를 의미합니다.

Smoke test가 확인하는 흐름은 다음입니다.

- `session_start`
- `frame_meta`
- binary JPEG frame
- `inference_result`
- `session_end`
- MySQL에 `driving_session` row 생성 확인
- MySQL에 `session_summary` row 생성 확인
- `MockRunner` 기준 `distraction_event` count가 `0`인지 확인

## Nginx 경유 WebSocket smoke test 방법

Nginx reverse proxy를 통해 WebSocket upgrade와 `/ws/` routing을 확인합니다.

```bash
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py --ws-url ws://nginx/ws/inference"
```

이 검증은 실제 배포 구조와 더 가깝습니다. 프론트엔드는 최종적으로 같은 공개 도메인에서 `/ws/*`로 WebSocket을 연결하게 되므로, FastAPI 직접 연결뿐 아니라 Nginx 경유 연결도 확인해야 합니다.

## 로컬 Python 실행 시 PYTHONPATH 주의사항

컨테이너 안에서 script를 실행할 때는 다음처럼 `PYTHONPATH=/app`을 명시합니다.

```bash
docker compose exec api sh -c "PYTHONPATH=/app python -m app.storage.init_db"
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py"
```

이유는 `scripts/ws_smoke_test.py`가 `app.core.config`, `app.storage.models` 등을 import하기 때문입니다. 컨테이너의 작업 디렉터리는 `/app`이지만, 실행 방식이나 환경에 따라 module path가 달라질 수 있으므로 README의 검증 명령은 `PYTHONPATH=/app`을 명시한 형태로 통일합니다.

호스트에서 직접 Python을 실행하는 경우에는 레포지토리 루트에서 실행해야 하며, MySQL 접속 주소도 컨테이너 내부 주소인 `mysql`이 아니라 host에서 접근 가능한 주소를 사용해야 합니다.

예:

```bash
python scripts/ws_smoke_test.py --database-url mysql+asyncmy://dms_user:dms_password@localhost:3306/dms
```

## WebSocket 연동 요약

Endpoint:

```text
GET /ws/inference
```

로컬 개발 중 직접 연결:

```text
ws://localhost:8000/ws/inference
```

Nginx 경유 연결:

```text
ws://localhost/ws/inference
```

Compose 내부에서 Nginx 경유 smoke test:

```text
ws://nginx/ws/inference
```

Client message 흐름:

```text
session_start -> frame_meta -> binary JPEG frame -> session_end
```

Heartbeat:

```json
{"type": "ping"}
```

Server response:

```json
{"type": "pong"}
```

`frame_meta` 예시:

```json
{
  "type": "frame_meta",
  "frame_id": "frame-1",
  "captured_at": "2026-05-07T12:00:01Z",
  "content_type": "image/jpeg"
}
```

`inference_result` 예시:

```json
{
  "type": "inference_result",
  "session_id": "session-123",
  "frame_id": "frame-1",
  "captured_at": "2026-05-07T12:00:01+00:00",
  "result": {
    "is_distracted": false,
    "label": "attentive",
    "confidence": 0.98
  },
  "alerts": []
}
```

Error response schema:

```json
{
  "type": "error",
  "code": "empty_frame",
  "message": "Binary frame must not be empty."
}
```

현재 error code는 `docs/websocket-contract.md`를 기준으로 확인합니다. 프론트엔드 연동 시 message type과 payload 구조는 README보다 `docs/websocket-contract.md`를 우선합니다.

## DB 저장 정책

DB에는 세션 단위로 의미 있는 데이터만 저장합니다.

저장하는 테이블:

- `driving_session`
- `distraction_event`
- `session_summary`

저장하지 않는 데이터:

- raw frame bytes
- per-frame inference result

이 정책의 이유는 다음입니다.

- raw frame은 용량과 개인정보 부담이 크다.
- per-frame inference result는 실시간 화면 표시에는 필요하지만, DB에 모두 저장하면 저장량이 빠르게 증가한다.
- DMS 백엔드의 MVP persistence 목표는 세션 시작/종료와 요약, 그리고 향후 event-level distraction 기록이다.

현재 persistence 상태:

- `session_start` 시 `driving_session` 생성
- `session_end` 시 `driving_session` 종료
- `session_end` 시 `session_summary` 생성/저장
- client disconnect, `idle_timeout`, `queue_drain_timeout` 시 열린 session cleanup
- `distraction_event` 저장은 아직 WebSocket inference flow에 연결하지 않음
- `session_summary`의 실제 통계 계산은 아직 구현하지 않음

## 성공한 테스트 목록

최근 검증된 항목입니다.

```bash
python -m compileall app
python -m py_compile scripts/ws_smoke_test.py
python -m pytest
docker compose config
docker compose up --build -d
docker compose exec api sh -c "PYTHONPATH=/app python -m app.storage.init_db"
docker compose exec mysql mysql -udms_user -pdms_password dms -e "SHOW TABLES;"
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py"
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py --ws-url ws://nginx/ws/inference"
curl http://localhost/api/health
```

`python -m pytest` 결과:

```text
19 passed
```

확인된 runtime 결과:

- Docker Compose clean start 검증
- MySQL 테이블 생성 확인
  - `distraction_event`
  - `driving_session`
  - `session_summary`
- FastAPI 직접 WebSocket smoke test 성공
  - `ws://localhost:8000/ws/inference`
- Nginx 경유 WebSocket smoke test 성공
  - `ws://nginx/ws/inference`
- Nginx HTTP reverse proxy health check 성공
  - `curl http://localhost/api/health`
  - `200 OK`
  - `Server: nginx`

## 미완료 작업

현재 남아 있는 작업입니다.

- `distraction_event` 저장 정책 구현
- `session_summary` 실제 통계 계산
- 프론트엔드 WebSocket 연동
- 프론트엔드 + 백엔드 + Nginx 로컬 통합
- 실제 배포 검증
- GitHub Actions CD 자동화
- `PytorchRunner` 연결
- ONNX Runtime 기반 runner 교체
- manifest file 기반 runner selection
- Alembic migration 도입
- authentication / authorization
- 배포 환경용 Let's Encrypt 발급 자동화

## 주요 docs 문서 링크

- [`docs/architecture.md`](docs/architecture.md): 현재 runtime 구조, module boundary, WebSocket inference flow, storage/deployment 구조
- [`docs/current-status.md`](docs/current-status.md): 구현된 것, 검증된 것, 미구현 항목, 주요 파일, caveat
- [`docs/decisions.md`](docs/decisions.md): runner interface, `MockRunner` 우선, low-latency queue, persistence 제한 같은 설계 결정
- [`docs/roadmap.md`](docs/roadmap.md): 다음 구현 후보와 runner/deployment/observability/API 개선 방향
- [`docs/websocket-contract.md`](docs/websocket-contract.md): 프론트엔드가 따라야 하는 WebSocket message contract

README는 온보딩용 요약이고, 상세 contract나 설계 판단은 위 docs를 기준으로 확인합니다.

## 팀원 작업 시 주의사항

- WebSocket contract를 바꿀 때는 반드시 `docs/websocket-contract.md`와 테스트를 함께 갱신합니다.
- `ws`, `inference`, `alerts`, `storage` module boundary를 유지합니다.
- runner 구현체 세부사항을 WebSocket endpoint나 alert logic에 직접 넣지 않습니다.
- `AlertEngine`은 model runner를 직접 호출하지 않습니다.
- raw frame을 DB에 저장하지 않습니다.
- per-frame inference result를 DB에 저장하지 않습니다.
- DB persistence는 `driving_session`, `distraction_event`, `session_summary` 중심으로 유지합니다.
- 실시간 처리에서는 모든 frame 처리보다 낮은 latency를 우선합니다.
- WebSocket session마다 bounded queue를 사용하고, overflow 시 오래된 frame을 drop하는 정책을 유지합니다.
- 현재 MVP에서는 `MockRunner`가 기본 runner입니다.
- `PytorchRunner`와 ONNX Runtime runner는 `InferenceRunner` interface 뒤에 추가해야 합니다.
- Docker Compose 환경에서는 DB host가 `mysql`입니다.
- host에서 DB에 직접 접근할 때는 `localhost:3306`을 사용합니다.
- 로컬 script 실행 시 import 문제가 나면 레포지토리 루트와 `PYTHONPATH`를 먼저 확인합니다.
- 기능 구현 전 현재 상태는 `docs/current-status.md`, 다음 작업은 `docs/roadmap.md`를 확인합니다.

## 커밋 전 권장 확인 순서

팀원이 처음 레포를 받은 뒤 실행해야 할 기본 순서입니다.

1. Docker Compose stack 시작

```bash
docker compose up --build -d
```

2. 컨테이너 상태 확인

```bash
docker compose ps
```

3. DB schema 초기화

```bash
docker compose exec api sh -c "PYTHONPATH=/app python -m app.storage.init_db"
```

4. 테이블 확인

```bash
docker compose exec mysql mysql -udms_user -pdms_password dms -e "SHOW TABLES;"
```

5. Nginx 경유 HTTP health check

```bash
curl http://localhost/api/health
```

6. FastAPI 직접 WebSocket smoke test

```bash
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py"
```

7. Nginx 경유 WebSocket smoke test

```bash
docker compose exec api sh -c "PYTHONPATH=/app python scripts/ws_smoke_test.py --ws-url ws://nginx/ws/inference"
```

커밋 전 다시 실행하면 좋은 테스트 명령입니다.

```bash
python -m compileall app
python -m py_compile scripts/ws_smoke_test.py
python -m pytest
docker compose config
```
