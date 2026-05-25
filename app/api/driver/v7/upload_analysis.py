import os
import tempfile
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.driver.v7.risk import create_driver4_v7_risk_scorer
from app.inference.manifest import get_runner


router = APIRouter(tags=["driver4-v7-pre-analysis"])


def _server_time_ms() -> float:
    return round(time.time() * 1000, 3)


def extract_video_frames(video_path: str, sampling_fps: float) -> list[tuple[float, bytes]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for backend video upload analysis.") from exc

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Video file could not be opened.")

    frames: list[tuple[float, bytes]] = []
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / sampling_fps
    next_sample_time = 0.0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            frame_time = position_ms / 1000.0 if position_ms > 0 else frame_index / source_fps if source_fps > 0 else next_sample_time
            frame_index += 1

            if frame_time + 1e-6 < next_sample_time:
                continue

            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                continue

            frames.append((round(next_sample_time, 6), buffer.tobytes()))
            next_sample_time += frame_interval
    finally:
        capture.release()

    return frames


@router.post("/pre-analysis/upload")
async def upload_pre_analysis(
    video: Annotated[UploadFile, File()],
    sampling_fps: Annotated[float, Form(alias="samplingFps", gt=0.0, le=60.0)] = 8.0,
) -> dict:
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only video uploads are supported.")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while chunk := await video.read(1024 * 1024):
                temp_file.write(chunk)

        try:
            frames = extract_video_frames(temp_path, sampling_fps)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        if not frames:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Video does not contain analyzable frames.")

        runner = get_runner("driver4-torch")
        risk_scorer = create_driver4_v7_risk_scorer()
        session_id = f"upload-{round(time.time() * 1000)}"
        analyzed_frames = []

        for index, (frame_time_seconds, frame_bytes) in enumerate(frames):
            server_received_at = _server_time_ms()
            result = await runner.infer(frame_bytes)
            risk_scores = risk_scorer.update(
                {detection.variable_name: detection.score for detection in result.detections},
                now=frame_time_seconds,
            )
            analyzed_frames.append(
                {
                    "type": "inference_result",
                    "sessionId": session_id,
                    "frameId": f"upload-frame-{index}",
                    "frameTimeSeconds": frame_time_seconds,
                    "clientSentAt": None,
                    "serverReceivedAt": server_received_at,
                    "serverRespondedAt": _server_time_ms(),
                    "detections": [detection.model_dump(by_alias=True) for detection in result.detections],
                    "riskScores": risk_scores,
                    "riskScoring": risk_scorer.metadata(),
                    "model": result.model.model_dump(by_alias=True),
                    "queue": {
                        "policy": "backend_video_upload",
                        "droppedFrames": 0,
                        "pendingFrames": max(len(frames) - index - 1, 0),
                    },
                    "telemetry": result.telemetry.model_dump(by_alias=True),
                },
            )

        return {
            "type": "pre_analysis_complete",
            "sessionId": session_id,
            "samplingFps": sampling_fps,
            "frameCount": len(analyzed_frames),
            "frames": analyzed_frames,
            "riskScoring": risk_scorer.metadata(),
        }
    finally:
        await video.close()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
