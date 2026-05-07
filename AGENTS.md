# DMS Backend Agent Guide

## Project Direction

This repository is the backend for a Driver Monitoring System (DMS).
Build the project in small, reviewable steps:

1. Skeleton
2. Health, WebSocket, and MockRunner
3. MySQL models and repositories
4. Docker and Nginx
5. Tests and README

## Architecture Rules

- Keep `ws`, `inference`, `alerts`, and `storage` modules separated.
- Model implementations must be replaceable through a runner interface.
- Alert logic must be independent from model runner implementations.
- Do not store raw frames or per-frame inference results in the database.
- Store only `driving_session`, `distraction_event`, and `session_summary`.
- Use one `asyncio.Queue(maxsize=N)` per WebSocket session.
- Prefer low latency over processing every frame.
- Match the WebSocket contract in `docs/websocket-contract.md`.
- Select runners through a model manifest.
- In MVP, only `MockRunner` needs to work.

## Coding Style

- Use explicit type hints.
- Keep Pydantic and SQLAlchemy schemas clear and small.
- Avoid broad abstractions until the code needs them.
- Keep changes scoped to the active implementation step.
