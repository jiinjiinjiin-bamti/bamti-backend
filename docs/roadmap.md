# Roadmap

Items here are planned or potential work. Implemented work is tracked in `docs/current-status.md`.

## Near Term

1. Stabilize deployment model paths.
   - Keep `.env.cuda` using container paths.
   - Confirm `/models/exp04_pseudo_ir_aug.pth` and `/models/final_model.pth` are mounted before startup.

2. Improve startup diagnostics.
   - Add an optional model status endpoint if needed.
   - Surface selected device, model path, loaded architecture, and class count.
   - Keep `/api/health` lightweight.

3. Improve frame validation.
   - Add cheap JPEG magic-byte validation or clearer decode errors.
   - Keep validation cheap enough for repeated frame uploads.

4. Measure v4 vs v6 behavior.
   - Compare immediate scores against one-second rolling average scores.
   - Confirm threshold behavior in the frontend for both BAMTI and AIHub profiles.

## Model Runtime

1. Measure CPU, MPS, and CUDA behavior.
   - Compare `MODEL_DEVICE=cpu`, `mps`, and `cuda` where available.
   - Record results through the one-minute measurement flow.

2. Revisit model cache policy if simultaneous users require different profiles.
   - Current behavior keeps one active loaded model.
   - If concurrent BAMTI and AIHub usage becomes common, consider a bounded multi-model cache or separate backend processes.

3. Add runner warmup if cold-start latency remains visible.
   - Warmup must not block health check.
   - Warmup should target the selected model profile.

4. Add ONNX Runtime runner only if needed.
   - Keep it behind `InferenceRunner`.
   - Do not change route contracts for runner-specific details.

## API Cleanup

1. Decide whether v2/v3/v5 experimental WebSocket routes should remain public.
   - v4 and v6 are the active frontend-facing paths.
   - Older experimental versions can stay during comparison but should be documented as non-primary.

2. Keep v4 debug isolated.
   - Do not make raw `A1`-`A16` debug payloads part of the normal v4 route unless explicitly requested.

3. Review mobile route duplication.
   - BAMTI and AIHub mobile v4/v6 routes are intentionally separate now.
   - If maintenance cost grows, extract shared code while preserving route behavior.

## Persistence

1. Revisit database usage after inference behavior stabilizes.
   - Define session/event persistence separately from raw frame handling.
   - Avoid storing raw frames by default.

2. Add migrations before production database changes.
   - Introduce Alembic or another explicit migration workflow.

## Deployment

1. Keep Nginx deployment notes aligned with production.
   - Production Nginx is native, not part of the backend compose stack.
   - `/api/` proxying must preserve WebSocket upgrade headers.

2. Add deployment checklist.
   - Verify model files.
   - Verify `.env.cuda`.
   - Verify `docker compose ... config`.
   - Verify `/api/health`.
   - Verify one v4 and one v6 inference path.
