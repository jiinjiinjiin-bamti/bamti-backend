# Driver4 Default Profile Integration Design

## Goal

Make the new four-class `final_model_4cls.pth` profile the default dashboard and
mobile-camera experience while keeping the existing BAMTI 7-class and AIHub
3-class profiles available on explicit paths.

## Confirmed User Routes

| Frontend route | Model profile | Purpose |
| --- | --- | --- |
| `/` | `driver4` | Default desktop dashboard using the new four-class model |
| `/mobile` | `driver4` | Default mobile-camera dashboard using the new four-class model |
| `/dad` | `bamti7` | Existing BAMTI 7-class desktop dashboard |
| `/dad/mobile` | `bamti7` | Existing BAMTI 7-class mobile-camera dashboard |
| `/aihub` | `aihub` | Existing AIHub 3-class desktop dashboard |
| `/aihub/mobile` | `aihub` | Existing AIHub 3-class mobile-camera dashboard |

`/camera` remains the phone sender page. The QR/session URL continues to pass
an API-version value so that the phone frame stream joins the selected
dashboard model profile.

## Model Analysis Result

The file `/Users/anjeonghyeon/web/dms/model/final_model_4cls.pth` was loaded
with PyTorch safe weight loading. Its checkpoint metadata contains:

```text
class_names = ['신체만짐', '주의분산', '핸드폰 조작', '핸들 조작']
classifier.weight shape = (4, 768)
classifier.bias shape = (4,)
```

The checkpoint has a torchvision ViT-style backbone comparable to the existing
AIHub model, but it stores a distinct four-output `classifier.*` head instead
of AIHub's `backbone.heads.head.*` output head. Its loader must therefore
construct the matching driver4 head explicitly; it is a separate model profile
rather than an AIHub replacement.

## Service Class Contract

The `driver4` profile exposes one service detection per model output. It does
not apply the grouped raw-class aggregation required by the BAMTI 7-class
profile.

| Model output label | API `variableName` / `classId` | Display name |
| --- | --- | --- |
| `신체만짐` | `body_touching` | `신체 만짐` |
| `주의분산` | `distraction` | `주의 분산` |
| `핸드폰 조작` | `phone_operation` | `핸드폰 조작` |
| `핸들 조작` | `steering_operation` | `핸들 조작` |

The initial threshold follows the existing AIHub direct-class profile
threshold of `0.65`; threshold judgment remains in the frontend as established
by the backend decisions document.

## Backend Architecture

### Profile Selection

Add a new explicit runner profile named `driver4-torch`.

- Add `DRIVER4_MODEL_PATH`, defaulting to
  `workspace_root / "model" / "final_model_4cls.pth"` for local execution.
- For Docker documentation and Compose configuration, mount/use
  `/models/final_model_4cls.pth`.
- Add a driver4-specific torchvision ViT load path whose backbone feeds the
  checkpoint's separate `classifier.*` head, while preserving separate model
  path and four-class manifest data.
- Keep runner selection and manifest generation in `app/inference`; routes
  must not encode model loading details.

### API Surface

Add an explicit third model prefix rather than changing existing meanings:

| Behavior | Driver4 4-class | BAMTI 7-class, preserved | AIHub 3-class, preserved |
| --- | --- | --- | --- |
| Immediate detection classes | `GET /api/driver/v4/detection-classes` | `GET /api/v4/detection-classes` | `GET /api/aihub/v4/detection-classes` |
| Immediate WebSocket stream | `WS /api/driver/v4/inference/stream` | `WS /api/v4/inference/stream` | `WS /api/aihub/v4/inference/stream` |
| Smoothed detection classes | `GET /api/driver/v6/detection-classes` | `GET /api/v6/detection-classes` | `GET /api/aihub/v6/detection-classes` |
| Smoothed WebSocket stream | `WS /api/driver/v6/inference/stream` | `WS /api/v6/inference/stream` | `WS /api/aihub/v6/inference/stream` |
| Immediate mobile sessions | `/api/driver/v4/mobile/*` | `/api/v4/mobile/*` | `/api/aihub/v4/mobile/*` |
| Smoothed mobile sessions | `/api/driver/v6/mobile/*` | `/api/v6/mobile/*` | `/api/aihub/v6/mobile/*` |

`/api/health` remains unversioned. Existing BAMTI debug routes remain BAMTI
only; no driver4 raw-debug endpoint is introduced because its API outputs
already correspond one-to-one with model classes.

Update the backend architecture guidance and API documentation to list
`/api/driver/v*` as the explicit third profile prefix alongside the preserved
BAMTI and AIHub prefixes.

### Reuse Boundaries

Driver4 route modules follow the established AIHub v4/v6 and mobile route
patterns, selecting `driver4-torch` instead of `aihub-torch`. Duplication may
be reduced only where an existing small helper already supports profile
selection cleanly; this change does not introduce a broad route refactor.

## Frontend Architecture

### Profile and Routes

Extend the dashboard profile type from `bamti7 | aihub` to
`driver4 | bamti7 | aihub`, with `driver4` as the `Home` default.

- `/` renders `Home` with `driver4`.
- `/mobile` renders `Home` with `driver4` and mobile camera mode.
- `/dad` and `/dad/mobile` explicitly render `bamti7`.
- `/aihub` and `/aihub/mobile` remain explicitly `aihub`.
- Route checks must handle `/dad/mobile` before `/dad`, matching the existing
  prefix-based routing style.

### Dashboard Data and Transport

Add driver4 dashboard constants using the four service classes above. Extend
transport/API-version selection so both webcam/video-file inference and mobile
sessions target `/api/driver/v4` or `/api/driver/v6` for the new profile.

The currently modified frontend files, including `src/pages/Home.tsx` and
`src/hooks/useVideoFileInference.tsx`, contain ongoing user work. Integration
must preserve those edits and layer the driver4 behavior onto the current file
contents rather than replacing them.

## Data Flow

### Desktop Webcam or Video File

1. The selected page determines `driver4`, `bamti7`, or `aihub`.
2. The profile supplies UI class mappings and eligible v4/v6 transports.
3. The selected transport opens the matching WebSocket endpoint.
4. The backend runner loads the profile model, infers JPEG frames, and returns
   profile-specific detection scores.
5. The frontend applies thresholds and session reporting against the matching
   four-, seven-, or three-class mapping.

### Mobile Camera

1. The dashboard creates a session under the selected profile/version mobile
   API prefix.
2. The QR camera URL includes the session id and selected API-version token.
3. The phone sends JPEG frames to its profile-specific
   `phone-frame-stream` WebSocket.
4. The backend runs the selected model and emits inference results on the
   session dashboard channel.
5. WebRTC signaling continues to provide preview video independently of the
   selected inference model.

## Failure Handling

- If `final_model_4cls.pth` is absent or incompatible, driver4 model loading
  fails through the same backend failure behavior used for current real-model
  profiles; runtime mock inference is not added.
- Invalid or oversized JPEG frame handling remains aligned with current
  WebSocket/mobile route validation.
- Alternating among model profiles may reload the active model under the
  existing cache policy; optimizing multi-profile caching is out of scope.
- Authentication, event database persistence, and migrations remain out of
  scope for this profile integration.

## Testing And Verification

### Backend

- Add manifest/runner tests confirming driver4 exposes exactly the four
  checkpoint-derived service classes.
- Add route contract tests for driver4 v4 and v6 detection/stream selection.
- Add mobile route contract tests confirming driver4 sessions select the new
  runner for v4 and v6.
- Run the backend test suite and compile validation used by the repository.

### Frontend

- The frontend currently has no configured test script or discovered unit-test
  harness, so this change does not create a new testing stack solely for route
  mapping.
- Run `npm run lint` and `npm run build`.
- Verify the six dashboard routes select the intended profile and that mobile
  QR/session paths preserve the same profile.

## Implementation Order

1. Add failing backend tests for the driver4 model manifest and API prefix.
2. Add backend settings, runner/manifest support, routes, and Docker/docs
   updates needed to pass those tests.
3. Establish the frontend baseline with its existing `lint` and `build`
   verification commands before changing profile routing and endpoint mapping.
4. Add frontend driver4 constants, transport mapping, and `/dad` route moves
   without discarding existing uncommitted frontend edits.
5. Run backend and frontend verification, then inspect route behavior in the
   browser where the local runtime is available.
