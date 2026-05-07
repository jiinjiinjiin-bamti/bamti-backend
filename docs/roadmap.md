# Roadmap

This roadmap lists next work based on the current repository state. Items here
are not implemented unless they are also listed in `docs/current-status.md`.

## Near Term

1. Define and implement `distraction_event` persistence policy.
   - Do not store one row per frame.
   - Treat alerts as state transitions: open an event when
     `AlertEngine.evaluate()` first returns `distraction_detected` after an
     attentive period, and close the event when alerts clear or the session ends.
   - Store event-level fields only: label/code, severity, representative
     confidence, message, started_at, and ended_at.
   - Update `session_summary` from stored event rows once event persistence exists.

2. Add database tests.
   - Add repository tests using a test database strategy.
   - Keep tests focused on the three persisted concepts:
     `driving_session`, `distraction_event`, and `session_summary`.

3. Add migrations.
   - Introduce Alembic or another explicit migration workflow.
   - Stop relying on `Base.metadata.create_all` for environments beyond local MVP.

4. Consider stronger JPEG validation.
   - Decide whether to validate JPEG magic bytes before enqueueing.
   - Keep any validation lightweight enough for the WebSocket hot path.

## Model Runner Evolution

1. Expand runner selection.
   - Replace hardcoded `get_runner()` branching with a manifest-backed loader.
   - Keep `MockRunner` as the default local runner.

2. Add `PytorchRunner`.
   - Keep it behind `InferenceRunner`.
   - Do not change WebSocket or alert code for runner-specific details.

3. Add `OnnxRunner`.
   - Keep it behind `InferenceRunner`.
   - Make it selectable by manifest/config once available.

## Deployment

1. Harden Docker Compose for deployment.
   - Add production environment documentation.
   - Decide how frontend artifacts are produced and mounted into Nginx.

2. Complete HTTPS setup.
   - Add the operational steps or scripts for Let's Encrypt certificate issuance.
   - Keep `/api/` and `/ws/` routing behavior unchanged.

3. Extend GitHub Actions.
   - The current workflow runs compile and tests.
   - Add deployment only after target server, secrets, and release process are defined.

## Observability And Operations

1. Add structured logging.
   - Log session start/end.
   - Log runner selection.
   - Log WebSocket disconnects and validation errors.

2. Add health/readiness distinction.
   - Keep `/api/health` lightweight.
   - Add database readiness only if needed by deployment.

3. Add configuration documentation.
   - Document `APP_NAME`, `ENVIRONMENT`, `INFERENCE_RUNNER`, `FRAME_QUEUE_SIZE`,
     `WEBSOCKET_IDLE_TIMEOUT_SECONDS`, `WEBSOCKET_DRAIN_TIMEOUT_SECONDS`,
     `MAX_FRAME_BYTES`, and `DATABASE_URL`.

## API And Contract Quality

1. Add more WebSocket tests.
   - Invalid JSON.
   - Queue overflow behavior.

2. Add API schemas as the HTTP surface grows.
   - Keep schemas explicit and small.
   - Avoid adding broad abstractions before more endpoints exist.
