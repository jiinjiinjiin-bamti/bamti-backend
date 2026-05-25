# Driver4 Default Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `final_model_4cls.pth` the default desktop and mobile dashboard profile, while moving the existing BAMTI 7-class experience to `/dad` and preserving AIHub on `/aihub`.

**Architecture:** Add `driver4` as a third explicit model profile with `/api/driver/v4` and `/api/driver/v6` endpoints. Its checkpoint uses a torchvision ViT backbone with a separate `classifier.*` output head, so the inference layer gets an explicit `Driver4VisionModel` load path and four-class mapping rather than reusing the AIHub head loader unchanged. Extend the existing React profile/transport switching so the URL chooses `driver4`, `bamti7`, or `aihub` without changing the meaning of existing BAMTI or AIHub API contracts. Work on top of current uncommitted route cleanup and video-WebSocket changes; do not replace or revert them.

**Tech Stack:** Python 3, FastAPI, PyTorch/torchvision, pytest, React 19, TypeScript, Vite, WebSocket/WebRTC.

**Commit policy:** The design and plan documents remain uncommitted by user request. Implementation changes also remain uncommitted unless the user later requests a source-code commit.

---

## Current Constraints

- `backend` already contains uncommitted changes to v4/v6 and AIHub route behavior and related tests/docs. Driver4 changes must integrate with those file contents.
- `frontend` already contains uncommitted work in `package-lock.json`, `src/pages/Home.tsx`, and `src/hooks/useVideoFileInference.tsx`. Driver4 changes must preserve the active video-WebSocket implementation.
- `frontend` is currently on `main` with user changes. Do not create a clean worktree that omits those changes; make scoped edits only after user authorization to work in the existing dirty branch.

## File Map

Backend additions/modifications:

- `app/core/config.py`: expose `driver4_model_path`.
- `app/inference/class_mapping.py`: define four direct driver4 service classes.
- `app/inference/model_loader.py`: load the new `classifier.*` four-head checkpoint with its direct service classes.
- `app/inference/manifest.py`: select `driver4-torch`.
- `app/api/driver/`: route package for v4/v6 manifests, WebSocket streams, and mobile session APIs.
- `app/api/routes.py`: mount the driver API prefix.
- `tests/test_driver4_routes.py`: new route contract coverage.
- `tests/test_torch_runner_service_scores.py`: new direct driver4 score/manifest coverage.
- `.env.example` and `docker-compose.yml`: document/mount the new model path.
- `AGENTS.md`, `README.md`, `docs/architecture.md`, `docs/current-status.md`: align explicit third-profile documentation without discarding existing edits.

Frontend additions/modifications:

- `src/constants/driver4DashboardData.ts`: four-class initial dashboard rows.
- `src/types/dashboard.ts`: add driver4 live transport values when transport is typed there.
- `src/App.tsx`: map `/`, `/mobile`, `/dad`, `/dad/mobile`, and existing AIHub paths to profiles.
- `src/pages/Home.tsx`: add `driver4` profile selection and driver4 transports.
- `src/hooks/useInferenceWebSocket.ts`: generate driver4 stream URLs.
- `src/hooks/useVideoFileInference.tsx`: extend the in-progress WebSocket endpoint selection to driver4.
- `src/services/mobileCamera/base.ts`: create driver4 mobile API prefixes and camera URL version tokens.

### Task 1: Backend Driver4 Class And Runner Contract

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/backend/app/core/config.py`
- Modify: `/Users/anjeonghyeon/web/dms/backend/app/inference/class_mapping.py`
- Modify: `/Users/anjeonghyeon/web/dms/backend/app/inference/model_loader.py`
- Modify: `/Users/anjeonghyeon/web/dms/backend/app/inference/manifest.py`
- Modify: `/Users/anjeonghyeon/web/dms/backend/tests/test_torch_runner_service_scores.py`

- [ ] **Step 1: Write failing runner-selection and direct-class tests**

Add tests that require a new runner name and the checkpoint-derived service identities:

```python
def test_driver4_runner_uses_configured_model_path(monkeypatch) -> None:
    monkeypatch.setattr(settings, "driver4_model_path", Path("final_model_4cls.pth"))
    runner = get_runner("driver4-torch")
    assert runner.model_path == Path("final_model_4cls.pth")


def test_driver4_service_classes_match_checkpoint_metadata() -> None:
    assert [
        (item.variable_name, item.display_name)
        for item in driver4_service_detection_classes
    ] == [
        ("body_touching", "신체 만짐"),
        ("distraction", "주의 분산"),
        ("phone_operation", "핸드폰 조작"),
        ("steering_operation", "핸들 조작"),
    ]
```

- [ ] **Step 2: Run focused backend tests and confirm RED**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/backend
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m pytest tests/test_torch_runner_service_scores.py -q
```

Expected: failure because `driver4_service_detection_classes`, `settings.driver4_model_path`, or `driver4-torch` does not yet exist.

- [ ] **Step 3: Implement model profile primitives**

Add direct raw labels and service mappings in `class_mapping.py`:

```python
driver4_raw_class_names = ("신체만짐", "주의분산", "핸드폰 조작", "핸들 조작")

driver4_service_detection_classes = (
    ServiceDetectionClass("body_touching", "신체 만짐", ("신체만짐",)),
    ServiceDetectionClass("distraction", "주의 분산", ("주의분산",)),
    ServiceDetectionClass("phone_operation", "핸드폰 조작", ("핸드폰 조작",)),
    ServiceDetectionClass("steering_operation", "핸들 조작", ("핸들 조작",)),
)
```

Add the setting in `config.py`:

```python
driver4_model_path: Path = Field(
    default=workspace_root / "model" / "final_model_4cls.pth",
    validation_alias=AliasChoices("DRIVER4_MODEL_PATH", "BAMTI_DRIVER4_MODEL_PATH"),
)
```

Add a driver4-specific torchvision wrapper because the checkpoint stores
`classifier.*`, not the AIHub model's `backbone.heads.head.*`:

```python
class Driver4VisionModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(image))
```

Detect the `classifier.weight` checkpoint signature inside
`_load_model_from_path_uncached`, load `Driver4VisionModel`, and return:

```python
LoadedModel(
    model=model,
    class_names=[item.variable_name for item in driver4_service_detection_classes],
    device=device,
    model_path=model_path,
    compiled=compiled,
    architecture="torchvision_vit_b_16_driver4",
    service_classes=driver4_service_detection_classes,
    raw_class_names=driver4_raw_class_names,
)
```

Register the named runner in `manifest.py`:

```python
if name in {"driver4-torch", "torch-driver4"}:
    return BamtiTorchRunner(model_path=settings.driver4_model_path)
```

The signature-based driver4 branch must run before the existing generic
torchvision branch and must retain BAMTI raw-class mapping and AIHub
direct-class behavior.

- [ ] **Step 4: Re-run focused tests and confirm GREEN**

Run the same `pytest tests/test_torch_runner_service_scores.py -q` command.
Expected: the new driver4 contract tests pass alongside existing score tests.

### Task 2: Backend Driver4 API Prefix And Mobile Streams

**Files:**
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/__init__.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/inference.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v4/__init__.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v4/routes.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v4/websocket.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v4/mobile/routes.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v6/__init__.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v6/routes.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v6/websocket.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/app/api/driver/v6/mobile/routes.py`
- Modify: `/Users/anjeonghyeon/web/dms/backend/app/api/routes.py`
- Create: `/Users/anjeonghyeon/web/dms/backend/tests/test_driver4_routes.py`

- [ ] **Step 1: Write failing route tests**

Create tests using a fake driver4 runner, following `tests/test_aihub_routes.py`:

```python
class FakeDriver4Runner:
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_version="final_model_4cls",
            classes=(
                DetectionClass(variable_name="body_touching", class_id="body_touching", display_name="신체 만짐", description="Driver4 model class: 신체만짐", threshold=0.65),
                DetectionClass(variable_name="distraction", class_id="distraction", display_name="주의 분산", description="Driver4 model class: 주의분산", threshold=0.65),
                DetectionClass(variable_name="phone_operation", class_id="phone_operation", display_name="핸드폰 조작", description="Driver4 model class: 핸드폰 조작", threshold=0.65),
                DetectionClass(variable_name="steering_operation", class_id="steering_operation", display_name="핸들 조작", description="Driver4 model class: 핸들 조작", threshold=0.65),
            ),
        )


def test_driver4_v4_detection_classes_route(monkeypatch) -> None:
    monkeypatch.setattr("app.api.driver.inference.get_model_manifest", lambda _: FakeDriver4Runner().manifest())
    response = TestClient(app).get("/api/driver/v4/detection-classes")
    assert response.status_code == 200
    assert [row["variableName"] for row in response.json()["classes"]] == [
        "body_touching", "distraction", "phone_operation", "steering_operation"
    ]
```

Add equivalent v6 and mobile-session runner-selection coverage modeled on the
existing v4/v6 mobile tests.

- [ ] **Step 2: Run new route tests and confirm RED**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/backend
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m pytest tests/test_driver4_routes.py -q
```

Expected: import or 404 failure because `/api/driver` is not mounted.

- [ ] **Step 3: Implement driver route package**

Create an explicit route hierarchy:

```python
# app/api/driver/routes.py (or app/api/driver/__init__.py export)
router = APIRouter(prefix="/driver")
router.include_router(v4_router)
router.include_router(v6_router)
```

Implement `inference.py` manifest handlers using:

```python
return get_model_manifest("driver4-torch")
```

Implement v4 stream/mobile route variants from the current BAMTI v4 shape and
v6 stream/mobile route variants from the current BAMTI v6 shape, changing only
the runner selection to `get_runner("driver4-torch")`. Mount the driver router
from `app/api/routes.py`.

- [ ] **Step 4: Verify new route tests and relevant existing routes**

Run:

```bash
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m pytest \
  tests/test_driver4_routes.py tests/test_aihub_routes.py \
  tests/test_v2_websocket.py tests/test_v3_websocket.py tests/test_v5_websocket.py -q
```

Expected: all selected tests pass and existing AIHub/BAMTI stream contracts
remain valid.

### Task 3: Backend Environment And Documentation Alignment

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/backend/.env.example`
- Modify: `/Users/anjeonghyeon/web/dms/backend/docker-compose.yml`
- Modify: `/Users/anjeonghyeon/web/dms/backend/AGENTS.md`
- Modify: `/Users/anjeonghyeon/web/dms/backend/README.md`
- Modify: `/Users/anjeonghyeon/web/dms/backend/docs/architecture.md`
- Modify: `/Users/anjeonghyeon/web/dms/backend/docs/current-status.md`

- [ ] **Step 1: Add deployment configuration for the new weight**

Add:

```dotenv
DRIVER4_MODEL_PATH=/models/final_model_4cls.pth
```

and include the same API service environment default in Compose:

```yaml
DRIVER4_MODEL_PATH: ${DRIVER4_MODEL_PATH:-/models/final_model_4cls.pth}
```

- [ ] **Step 2: Update docs on top of existing local modifications**

Document `Driver4 4-class`, `/api/driver/v4`, `/api/driver/v6`, and
`DRIVER4_MODEL_PATH`, while retaining the already edited removal of v4/v6 REST
frame claims. Extend `AGENTS.md` to permit explicit `/api/driver/v*` model
routes and preserve the raw-frame persistence rules.

- [ ] **Step 3: Verify backend configuration and full suite**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/backend
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m compileall app
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m pytest -q
docker compose config --quiet
```

Expected: compile succeeds, pytest has zero failures, and Compose validates.

### Task 4: Frontend Driver4 Route And Class Mapping

**Files:**
- Create: `/Users/anjeonghyeon/web/dms/frontend/src/constants/driver4DashboardData.ts`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/App.tsx`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/types/dashboard.ts`

- [ ] **Step 1: Establish frontend baseline before modification**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run lint
npm run build
```

Record any pre-existing failure before applying driver4 changes.

- [ ] **Step 2: Add four-class dashboard constants**

Create:

```typescript
export const driver4BehaviorDetections: BehaviorDetection[] = [
  { label: '신체 만짐', variableName: 'body_touching', score: '0.00', threshold: '0.65', duration: '0.0s', active: false, tone: 'danger' },
  { label: '주의 분산', variableName: 'distraction', score: '0.00', threshold: '0.65', duration: '0.0s', active: false, tone: 'danger' },
  { label: '핸드폰 조작', variableName: 'phone_operation', score: '0.00', threshold: '0.65', duration: '0.0s', active: false, tone: 'danger' },
  { label: '핸들 조작', variableName: 'steering_operation', score: '0.00', threshold: '0.65', duration: '0.0s', active: false, tone: 'danger' },
]
```

Add matching status and initial event-log constants using existing AIHub
constant shapes.

- [ ] **Step 3: Make route/profile selection explicit**

Change `DetectionProfile` to:

```typescript
type DetectionProfile = 'driver4' | 'bamti7' | 'aihub'
```

Set `Home`'s default to `driver4`, then map routes:

```tsx
if (pathname.startsWith('/dad/mobile')) return <Home cameraMode="mobile" detectionProfile="bamti7" />
if (pathname.startsWith('/dad')) return <Home detectionProfile="bamti7" />
if (pathname.startsWith('/mobile')) return <Home cameraMode="mobile" detectionProfile="driver4" />
return <Home detectionProfile="driver4" />
```

Keep existing `/camera`, `/aihub/mobile`, and `/aihub` checks ahead of the
generic default.

- [ ] **Step 4: Add driver4 profile data/transport options**

Extend `Home.tsx` profile branches so `driver4` selects the new four-class
constant data and transport values:

```typescript
const driver4WebcamTransportOptions: SourceTransportOption[] = [
  { description: '4-class latest-pending websocket stream', label: 'Driver4 v4', value: 'websocket-driver4-v4' },
  { description: '4-class 1-second smoothed websocket stream', label: 'Driver4 v6', value: 'websocket-driver4-v6' },
]
```

Add `websocket-driver4-v4` and `websocket-driver4-v6` to `LiveTransport` and
choose `driver4-v4`/`driver4-v6` mobile API version tokens for mobile mode.

### Task 5: Frontend Endpoint Selection And Mobile Sessions

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useInferenceWebSocket.ts`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useVideoFileInference.tsx`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/services/mobileCamera/base.ts`

- [ ] **Step 1: Extend WebSocket URL selection**

In the webcam hook and the currently modified video-file WebSocket helper,
route driver4 profile transports to:

```typescript
'/api/driver/v4/inference/stream'
'/api/driver/v6/inference/stream'
```

Retain existing AIHub and BAMTI branches, including the active video-file
WebSocket session implementation.

- [ ] **Step 2: Extend mobile endpoint tokens**

Change:

```typescript
export type MobileApiVersion =
  'driver4-v4' | 'driver4-v6' | 'v4' | 'v6' | 'aihub-v4' | 'aihub-v6'
```

and map:

```typescript
if (apiVersion === 'driver4-v4') return '/api/driver/v4/mobile'
if (apiVersion === 'driver4-v6') return '/api/driver/v6/mobile'
```

The `/camera` query keeps the selected token so phone streams join the same
profile as the dashboard.

- [ ] **Step 3: Verify frontend behavior**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run lint
npm run build
```

Expected: both commands succeed after driver4 additions.

### Task 6: End-To-End Route Verification

**Files:**
- No new source files; verify prior modifications.

- [ ] **Step 1: Run complete backend evidence**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/backend
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m pytest -q
/Users/anjeonghyeon/web/dms/backend-test/.venv/bin/python -m compileall app
docker compose config --quiet
```

- [ ] **Step 2: Run complete frontend evidence**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run lint
npm run build
```

- [ ] **Step 3: Inspect frontend routes in a local browser**

With the frontend served locally, verify these mappings:

```text
/             -> driver4 /api/driver/*
/mobile       -> driver4 /api/driver/*/mobile
/dad          -> bamti7 /api/v*
/dad/mobile   -> bamti7 /api/v*/mobile
/aihub        -> aihub /api/aihub/*
/aihub/mobile -> aihub /api/aihub/*/mobile
```

No implementation commit or push is performed without a new user request.
