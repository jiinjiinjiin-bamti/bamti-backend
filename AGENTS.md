# DMS Backend Agent Guide

## Project Direction

This repository is the backend for a Driver Monitoring System (DMS).
Build the project in small, reviewable steps:

1. Skeleton
2. Health and versioned HTTP APIs
3. Real v1 model inference
4. One-minute telemetry JSON persistence
5. Docker, Nginx, tests, and README

## Architecture Rules

- Keep `api`, `inference`, and `storage` modules separated.
- Model implementations must be replaceable through a runner interface.
- Do not store raw frames or per-frame inference results in the database.
- Store only `driving_session`, `distraction_event`, and `session_summary`.
- Version inference and telemetry APIs under `/api/v1`.
- Keep `/api/health` unversioned.
- Select runners through `app/inference/manifest.py`.
- Runtime mock inference should not be reintroduced; tests may use test-local fakes.
- WebSocket inference is deferred until the HTTP v1 baseline is stable.

## Coding Style

- Use explicit type hints.
- Keep Pydantic and SQLAlchemy schemas clear and small.
- Avoid broad abstractions until the code needs them.
- Keep changes scoped to the active implementation step.
