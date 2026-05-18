# Decisions

This document records current backend architecture and implementation decisions.

## Keep Versioned APIs

Decision:

- Keep `/api/health` unversioned.
- Keep inference variants under explicit version prefixes.
- Use `/api/v*` for BAMTI 7-class.
- Use `/api/aihub/v*` for AIHub 3-class.

Reason:

- The frontend needs to select API behavior explicitly.
- v4 and v6 have different score semantics.
- AIHub should not be hidden behind the BAMTI model profile.

## Preserve v4 and v6 Semantics

Decision:

- v4 returns immediate model score responses.
- v6 returns one-second rolling average score responses.

Reason:

- v4 is useful for realtime debugging and raw behavior checks.
- v6 is more stable for threshold-based user-facing warnings.

## Add a Dedicated v4 Debug Stream

Decision:

- Expose raw BAMTI `A1`-`A16` scores through `/api/v4/debug/inference/stream`.

Reason:

- The service UI uses grouped detections.
- Model debugging sometimes requires raw class visibility.
- Keeping this under a debug route prevents confusing it with normal production inference.

## Keep Model Execution Behind Runner Selection

Decision:

- Model execution goes through `InferenceRunner` and `app/inference/manifest.py`.

Current implementation:

- `bamti-torch`
- `bamti-torch-debug-raw`
- `aihub-torch`

Reason:

- Model profile differences should stay out of route handlers as much as possible.
- Future runners can be added without changing every route.

## Use Max Score for Grouped BAMTI Classes

Decision:

- When several raw action classes map to one service class, use the maximum score.

Reason:

- A grouped event should activate when any member raw class is strongly detected.
- Averaging can hide high-confidence raw detections inside a group.

## Keep Threshold Judgment on the Frontend

Decision:

- Backend may provide threshold metadata.
- Frontend performs final active/inactive threshold judgment based on user settings.

Reason:

- The UI allows threshold changes.
- Backend fallback thresholds must not block live score updates.

## Do Not Store Raw Frames

Decision:

- Do not store raw frame bytes.
- Do not store per-frame inference results by default.

Reason:

- The current goal is live inference and demo analysis, not dataset collection.
- Avoiding raw-frame persistence reduces privacy and storage risk.

## Keep One Active Model Cached

Decision:

- The model loader keeps one active loaded model cache.
- Loading another model path clears the existing cache.

Reason:

- Running BAMTI and AIHub models in one process can otherwise consume unnecessary VRAM.
- The current deployment target favors stability over simultaneous multi-model residency.

## Keep JSON Telemetry Runs

Decision:

- Store one-minute measurement payloads as JSON files under `TELEMETRY_RUNS_DIR`.

Reason:

- Performance experiments are easy to inspect and compare.
- Database persistence is not required for this measurement flow.

## Native Nginx in Production

Decision:

- Production Nginx runs outside Docker.
- Backend container/process listens on `127.0.0.1:8000`.
- Nginx proxies `/api/` and preserves WebSocket upgrade headers.

Reason:

- This matches the current deployment server setup.
- The backend repo should not assume Nginx is part of the Docker deployment.
