# Decisions

This document records architecture and implementation decisions visible in the
current backend.

## Use Versioned HTTP APIs First

Decision:

- Implement the frontend integration through `/api/v1`.
- Keep health outside versioning as `/api/health`.
- Defer WebSocket inference until the HTTP path is stable.

Reason:

- The current frontend already sends frames through HTTP.
- The team needs stable integration and one-minute performance measurements
  before optimizing transport.

## Remove Runtime Mock Inference

Decision:

- Do not ship a runtime mock runner in the backend.
- The default runner is the real PyTorch runner, selected with
  `INFERENCE_RUNNER=bamti-torch`.

Reason:

- The frontend should now connect to the real model response shape.
- Tests can still use test-local fake runners through monkeypatching without
  keeping a mock implementation in application code.

## Keep Model Execution Behind a Runner Interface

Decision:

- Model execution still goes through `InferenceRunner`.

Current implementation:

- `app/inference/runner.py` defines the interface.
- `app/inference/torch_runner.py` implements `BamtiTorchRunner`.
- `app/inference/manifest.py` selects the runner.

Reason:

- ONNX or another backend can be added later without changing the v1 HTTP route
  contract.

## Persist One-Minute Measurements as JSON Files

Decision:

- Store one-minute measurement payloads as JSON files under `TELEMETRY_RUNS_DIR`.
- Do not require database persistence for performance experiment logs.

Reason:

- The team wants to compare optimization attempts over time.
- JSON files are easy to inspect, commit-exclude, and archive separately.

## Do Not Store Raw Frames

Decision:

- Do not store raw frame bytes.
- Do not store per-frame inference results in the database during v1 HTTP
  integration.

Reason:

- The current goal is real-time inference and performance measurement, not data
  collection.
- Avoiding raw-frame persistence reduces privacy and storage risk.

## Keep SQLAlchemy Storage Modules for Later

Decision:

- Keep `app/storage` in the repository even though v1 HTTP inference does not
  use it yet.

Reason:

- Session/event persistence is likely to return later, but it should be designed
  after the model integration path is stable.
- The Compose MySQL service is kept behind the `persistence` profile so the
  active HTTP model API can start without waiting for an unused database.

## Use Nginx as API Reverse Proxy

Decision:

- Nginx proxies `/api/` to FastAPI.
- WebSocket proxying is removed from the active config until WebSocket inference
  is intentionally reintroduced.
