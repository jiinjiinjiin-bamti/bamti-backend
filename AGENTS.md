# DMS Backend Agent Guide

## Project Direction

This repository is the FastAPI backend for BAMTI DMS.

The active backend is no longer a v1-only HTTP baseline. It supports real model inference across multiple API variants:

- BAMTI 7-class model profile
- AIHub 3-class model profile
- REST frame inference
- WebSocket stream inference
- v4 realtime score responses
- v6 one-second rolling average score responses
- v4 raw A-score debug stream
- CPU and CUDA Docker execution

## Architecture Rules

- Keep `api`, `inference`, `core`, and `storage` module boundaries clear.
- Keep model execution behind the runner/manifest layer.
- Keep model profile selection explicit. Do not hide BAMTI 7-class and AIHub 3-class differences in route code.
- Keep `/api/health` unversioned.
- Keep inference APIs versioned under `/api/v*` or `/api/aihub/v*`.
- Do not store raw frames.
- Do not persist per-frame inference results unless a separate persistence design is requested.
- Do not reintroduce runtime mock inference. Tests may use test-local fakes or monkeypatching.
- Treat `/api/v4/debug/inference/stream` as a debug-only endpoint.

## Model Profiles

### BAMTI 7-class

The latest BAMTI model emits raw `A1`-`A16` scores and maps them to service-level detections in `app/inference/class_mapping.py`.

Grouped classes use the maximum raw score, not an average.

### AIHub 3-class

AIHub routes use the legacy 3-class model through `AIHUB_MODEL_PATH`.

## API Version Rules

- v4 returns model scores immediately.
- v6 returns one-second rolling average scores per session.
- v4 debug exposes raw `A1`-`A16` scores for diagnostics.

## Coding Style

- Use explicit type hints.
- Keep Pydantic schemas clear and small.
- Avoid broad abstractions until the code needs them.
- Keep changes scoped to the requested API/model profile.
- Keep deployment paths container-relative when documenting Docker behavior.
