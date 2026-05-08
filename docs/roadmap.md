# Roadmap

Items here are planned work and are not implemented unless also listed in
`docs/current-status.md`.

## Near Term

1. Verify frontend integration against the real v1 HTTP model API.
   - Confirm `/api/v1/detection-classes` returns the three checkpoint classes.
   - Confirm `/api/v1/inference/frame` drives Detection Status from model output.
   - Confirm 1-minute measurement JSON files are written through
     `/api/v1/telemetry/runs`.

2. Add model startup diagnostics.
   - Surface model path, device, class names, and load status in a lightweight
     status endpoint if needed.
   - Keep `/api/health` lightweight.

3. Improve frame validation.
   - Add JPEG magic-byte validation or decode-time error handling.
   - Keep validation cheap enough for repeated HTTP frame uploads.

4. Add persisted performance-run retrieval.
   - `GET /api/v1/telemetry/runs` currently lists files.
   - Add a detail endpoint only if the frontend needs to load historical JSON.

## Model Runner Evolution

1. Measure CPU, MPS, and CUDA behavior.
   - Compare `MODEL_DEVICE=cpu`, `mps`, and `cuda` where available.
   - Record results through the one-minute measurement flow.

2. Add ONNX Runtime runner if needed.
   - Keep it behind `InferenceRunner`.
   - Do not change the v1 route contract for runner-specific details.

3. Add runner warmup if cold-start latency becomes visible.

## Persistence

1. Revisit database usage after inference integration stabilizes.
   - Define session/event persistence separately from raw frame handling.
   - Avoid storing raw frames by default.

2. Add migrations.
   - Introduce Alembic or another explicit migration workflow before production
     database changes.

## Transport

1. Reintroduce WebSocket only after HTTP v1 baseline measurements are complete.
   - Treat it as a separate design step.
   - Preserve the ability to compare HTTP and WebSocket results using the same
     one-minute telemetry payload shape.

## Deployment

1. Harden Docker Compose for the actual model.
   - Document model volume mounting.
   - Confirm the container has the required torch/torchvision runtime.

2. Complete HTTPS setup.
   - Keep `/api/` routing stable.
   - Add certificate automation only after the target deployment environment is
     fixed.
