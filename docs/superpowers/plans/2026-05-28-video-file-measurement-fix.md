# Video File Measurement Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Driver4 video file 1-minute performance measurement so saved JSON files contain valid transmission FPS and model inference latency data.

**Architecture:** Keep backend telemetry persistence as-is, and fix the frontend measurement pipeline. The video analysis hook should own live frame telemetry updates, while `Home.tsx` should save a fresh immutable snapshot at the end of the 60-second window. For WebSocket V3 and later, separate frame transmission from response handling so queue/drop behavior is measured correctly.

**Tech Stack:** React 19, TypeScript, Vite, FastAPI telemetry save endpoint, JSON telemetry files.

---

## Problem Summary

Current saved files in `/Users/anjeonghyeon/web/dms/test_result` show:

```txt
requestedFrameCount: 0
successfulFrameCount: 0
failedFrameCount: 0
roundTripLatencyMs.samples: 0
inferenceLatencyMs.samples: 0
video.measuredFromSec: 0
analysisDurationMs: ~60000
```

This means the 60-second measurement timer runs and saves a file, but the save callback reads a stale empty `videoFileAnalysis` snapshot.

Main issues:

1. `Home.tsx` saves `videoFileAnalysis.telemetrySamples` from an old closure.
2. `Home.tsx` also saves stale `videoFileAnalysis.currentTime`.
3. WebSocket video measurement currently awaits every response before sending the next frame, so V3+ queue/drop policy is not exercised.
4. Video file JSON does not explicitly store `transmissionFps`, `responseFps`, `droppedFrames`, or `frameSkipRate`.
5. Saving accepts zero-sample runs without marking them invalid.

---

## File Structure

- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`
  - Keep latest video measurement telemetry in refs.
  - Save from refs, not stale hook state.
  - Add validity metadata to the payload.

- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useVideoFileInference.tsx`
  - Add video telemetry fields for transmission/response/drop metrics.
  - Separate WebSocket send loop from response handling for V3+ transports.
  - Expose a stable `measurementSnapshot` or latest telemetry values.

- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/types/dashboard.ts`
  - Extend `VideoFileTelemetrySamples` fields if shared types are moved there.

- Optional Modify: `/Users/anjeonghyeon/web/dms/frontend/src/lib/videoPerformanceMeasurement.ts`
  - Add helper functions for `fpsFromCount`, `frameSkipRate`, and `measurementValid`.

- Optional Test: add lightweight TypeScript helper tests only if the project test runner is introduced later. Current frontend has no test script, so verification is `npm run build` plus browser measurement.

---

### Task 1: Fix Stale Video Measurement Save Snapshot

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`

- [ ] **Step 1: Add refs for latest video measurement data**

Add refs near the existing `inferenceTelemetryRef`:

```ts
const videoFileTelemetryRef = useRef(videoFileAnalysis.telemetrySamples)
const videoFileSnapshotRef = useRef({
  currentTime: videoFileAnalysis.currentTime,
  duration: videoFileAnalysis.duration,
  fileName: videoFileAnalysis.fileName,
  resolution: videoFileAnalysis.resolution,
})
```

- [ ] **Step 2: Keep refs synced with latest hook state**

Add effects after `videoFileAnalysis` is created:

```ts
useEffect(() => {
  videoFileTelemetryRef.current = videoFileAnalysis.telemetrySamples
}, [videoFileAnalysis.telemetrySamples])

useEffect(() => {
  videoFileSnapshotRef.current = {
    currentTime: videoFileAnalysis.currentTime,
    duration: videoFileAnalysis.duration,
    fileName: videoFileAnalysis.fileName,
    resolution: videoFileAnalysis.resolution,
  }
}, [
  videoFileAnalysis.currentTime,
  videoFileAnalysis.duration,
  videoFileAnalysis.fileName,
  videoFileAnalysis.resolution,
])
```

- [ ] **Step 3: Save from refs**

In `saveMeasurementRun`, replace:

```ts
const videoSamples = videoFileAnalysis.telemetrySamples
```

with:

```ts
const videoSamples = videoFileTelemetryRef.current
const videoSnapshot = videoFileSnapshotRef.current
```

Then replace video fields:

```ts
video: {
  fileName: videoSnapshot.fileName,
  durationSec: videoSnapshot.duration,
  measuredFromSec: videoSnapshot.currentTime,
  resolution: videoSnapshot.resolution,
},
```

- [ ] **Step 4: Verify build**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run build
```

Expected: `tsc -b` and `vite build` complete with exit code 0.

---

### Task 2: Add Explicit Transmission And Model Metrics

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useVideoFileInference.tsx`

- [ ] **Step 1: Add explicit computed FPS fields when saving**

In `saveMeasurementRun`, compute:

```ts
const elapsedSeconds = elapsedMs / 1000
const transmissionFps = elapsedSeconds > 0
  ? Number((videoSamples.requestedFrameCount / elapsedSeconds).toFixed(2))
  : 0
const responseFps = elapsedSeconds > 0
  ? Number((videoSamples.successfulFrameCount / elapsedSeconds).toFixed(2))
  : 0
```

Add to `metrics`:

```ts
transmissionFps,
responseFps,
```

- [ ] **Step 2: Keep model inference latency summary as the main model metric**

Keep the existing field:

```ts
inferenceLatencyMs,
```

This is the model forward-pass time from backend telemetry:

```ts
const inferenceMs = toFiniteMetric(result.telemetry?.inferenceMs)
```

- [ ] **Step 3: Add measurement validity fields**

Add:

```ts
const measurementValid = videoSamples.requestedFrameCount > 0 && videoSamples.successfulFrameCount > 0
const invalidReason = measurementValid ? null : 'No video inference samples were recorded.'
```

Add to payload:

```ts
measurementValid,
invalidReason,
```

- [ ] **Step 4: Verify build**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run build
```

Expected: build succeeds.

---

### Task 3: Fix WebSocket V3+ Queue/Drop Measurement Shape

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useVideoFileInference.tsx`

- [ ] **Step 1: Add transport capability helper**

Add near transport helpers:

```ts
const usesLatestPendingQueueTransport = (apiVersion: LiveTransport) =>
  apiVersion === 'websocket-driver4-v3'
  || apiVersion === 'websocket-driver4-v4-raw'
  || apiVersion === 'websocket-driver4-v4-jpeg'
  || apiVersion === 'websocket-driver4-v5'
  || apiVersion === 'websocket-driver4-v6'
  || apiVersion === 'websocket-driver4-v7'
```

- [ ] **Step 2: Do not await each WebSocket response for latest-pending transports**

In the main playback loop, keep sequential behavior for REST and V2:

```ts
if (usesRestTransport || !usesLatestPendingQueueTransport(apiVersion)) {
  const { encodingMs, latencyMs, result } = usesRestTransport
    ? await sendRestFrame(video, frameTime, 'file-frame')
    : await sendFrame(video, frameTime, 'file-frame', await (streamPromiseRef.current || startStream(sessionIdRef.current)))
  if (abortRef.current || requestRevision !== seekRevisionRef.current) continue
  recordFrameSuccess(latencyMs, encodingMs, result)
  updateDetections(result, frameTime)
  setProcessedFrames((currentValue) => currentValue + 1)
  continue
}
```

For V3+ use fire-and-handle:

```ts
void sendFrame(
  video,
  frameTime,
  'file-frame',
  await (streamPromiseRef.current || startStream(sessionIdRef.current)),
).then(({ encodingMs, latencyMs, result }) => {
  if (abortRef.current || requestRevision !== seekRevisionRef.current) return
  recordFrameSuccess(latencyMs, encodingMs, result)
  updateDetections(result, frameTime)
  setProcessedFrames((currentValue) => currentValue + 1)
}).catch(() => {
  if (abortRef.current || requestRevision !== seekRevisionRef.current) return
  recordFrameFailure()
  setFailedFrames((currentValue) => currentValue + 1)
})
```

- [ ] **Step 3: Keep the 24 FPS pacing**

Do not remove:

```ts
const frameIntervalSec = 1 / Math.max(samplingFps, 1)
```

The loop must keep using:

```ts
if (frameTime - lastCapturedTime >= frameIntervalSec - 0.01) {
  lastCapturedTime = frameTime
  ...
}
```

- [ ] **Step 4: Verify V3+ can record drops**

After implementation, run a one-minute V3 measurement and confirm JSON has:

```txt
requestedFrameCount > successfulFrameCount
failedFrameCount or droppedFrames > 0
transmissionFps near 24
inferenceLatencyMs.samples.length > 0
```

---

### Task 4: Add Dropped Frame And Skip Rate Metrics

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/hooks/useVideoFileInference.tsx`
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`

- [ ] **Step 1: Extend video telemetry sample type**

Add fields:

```ts
droppedFrameCount: number
frameSkipRateSamples: number[]
```

Initialize in `emptyVideoFileTelemetrySamples()`:

```ts
droppedFrameCount: 0,
frameSkipRateSamples: [],
```

- [ ] **Step 2: Count `frame_dropped` messages**

In WebSocket `onmessage`, inside `message.type === 'frame_dropped'`, update telemetry:

```ts
updateTelemetrySamples((currentSamples) => {
  const droppedFrameCount = currentSamples.droppedFrameCount + 1
  const requestedFrameCount = Math.max(currentSamples.requestedFrameCount, 1)
  return {
    ...currentSamples,
    droppedFrameCount,
    frameSkipRateSamples: [
      ...currentSamples.frameSkipRateSamples,
      Number(((droppedFrameCount / requestedFrameCount) * 100).toFixed(2)),
    ],
  }
})
```

- [ ] **Step 3: Save drop metrics**

In `saveMeasurementRun`, add:

```ts
const frameSkipRate = metricSummary(videoSamples.frameSkipRateSamples)
```

Add to `metrics`:

```ts
droppedFrameCount: videoSamples.droppedFrameCount,
frameSkipRate,
```

- [ ] **Step 4: Verify build**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run build
```

Expected: build succeeds.

---

### Task 5: Add Measurement Result Sanity Check In UI

**Files:**
- Modify: `/Users/anjeonghyeon/web/dms/frontend/src/pages/Home.tsx`

- [ ] **Step 1: Warn on invalid measurement**

After creating the payload, before POST:

```ts
if (!measurementValid) {
  console.warn('Video measurement did not record inference samples.', payload)
}
```

- [ ] **Step 2: Surface invalid result in summary**

If `measurementValid` is false, set a user-facing error:

```ts
if (!measurementValid) {
  setError('No video inference samples were recorded during the measurement.')
}
```

Use the existing error surface in the dashboard. Do not block file saving, because invalid files are useful for debugging.

- [ ] **Step 3: Verify**

Run:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
npm run build
```

Expected: build succeeds.

---

### Task 6: End-To-End Manual Verification

**Files:**
- No code files.

- [ ] **Step 1: Rebuild frontend and backend**

Run on DGX deployment:

```bash
cd /path/to/backend
docker compose -f docker-compose.yml -f docker-compose.dgx.yml up -d --build api
```

Deploy the frontend build according to the current deployment path.

- [ ] **Step 2: Run V1 and V3 measurements**

Run one-minute video file measurements for:

```txt
V1 REST
V3 Queue / Drop
```

- [ ] **Step 3: Inspect JSON**

Expected V1:

```txt
environment.apiVersion = driver4-v1
metrics.requestedFrameCount > 0
metrics.successfulFrameCount > 0
metrics.transmissionFps > 0
metrics.inferenceLatencyMs.samples.length > 0
metrics.measurementValid = true
```

Expected V3:

```txt
environment.apiVersion = driver4-v3
metrics.requestedFrameCount > metrics.successfulFrameCount
metrics.transmissionFps close to 24
metrics.inferenceLatencyMs.samples.length > 0
metrics.measurementValid = true
```

- [ ] **Step 4: Commit frontend changes**

Use the required commit format:

```bash
cd /Users/anjeonghyeon/web/dms/frontend
git add src/pages/Home.tsx src/hooks/useVideoFileInference.tsx src/types/dashboard.ts src/lib/videoPerformanceMeasurement.ts
git commit -m "✨ Feat: 비디오 성능 측정 지표 저장 개선"
git push origin main
```

---

## Self-Review

Spec coverage:
- Fixes zero-count JSON files.
- Preserves first-inference-ready warm-up behavior.
- Adds explicit transmission FPS and model inference latency fields for presentation.
- Makes V3+ queue/drop measurement meaningful.
- Adds invalid measurement detection.

Placeholder scan:
- No `TBD`, `TODO`, or unspecified test steps remain.

Type consistency:
- `VideoFileTelemetrySamples` fields must be updated wherever initialized and saved.
- If `src/types/dashboard.ts` does not own this type, keep the changes local to `useVideoFileInference.tsx`.

