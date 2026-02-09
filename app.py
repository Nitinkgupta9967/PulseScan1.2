# app.py

import time
import uuid
import logging
import tempfile
import requests
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline_runner import run_pipeline

# ---------------- CONFIG ---------------- #

MAX_PROCESS_SECONDS = 15      # HARD LIMIT
VIDEO_DOWNLOAD_TIMEOUT = 40

# ---------------- Logging Setup ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PulseScan")

# ---------------- FastAPI App ---------------- #

app = FastAPI(
    title="PulseScan API",
    description="Fast rPPG Vital Signs Extraction (Optimized)",
    version="1.1"
)

# ---------------- Request Model ---------------- #

class VideoRequest(BaseModel):
    video_url: str

# ---------------- Root ---------------- #

@app.get("/")
def root():
    return {
        "status": "PulseScan API running",
        "docs": "/docs",
        "endpoints": {
            "analyze": "/analyze_video",
            "quick_test": "/test_video_quick"
        }
    }

# ---------------- MAIN ANALYSIS ---------------- #

@app.post("/analyze_video")
def analyze_video(request: VideoRequest):
    request_id = str(uuid.uuid4())[:8]
    request_start = time.time()

    logger.info(f"[{request_id}] Request received")
    logger.info(f"[{request_id}] Video URL: {request.video_url}")

    temp_video_path = None

    try:
        # ---------- Download Video ----------
        logger.info(f"[{request_id}] Downloading video")
        download_start = time.time()

        response = requests.get(
            request.video_url,
            stream=True,
            timeout=VIDEO_DOWNLOAD_TIMEOUT
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Failed to download video"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            temp_video_path = tmp.name

        logger.info(
            f"[{request_id}] Download completed in "
            f"{round(time.time() - download_start, 2)} sec"
        )

        # ---------- Run Pipeline (TIME-BOUND) ----------
        logger.info(f"[{request_id}] Starting rPPG pipeline")

        result = run_pipeline(
            video_path=temp_video_path,
            timeout=MAX_PROCESS_SECONDS
        )

        total_time = round(time.time() - request_start, 2)
        logger.info(f"[{request_id}] SUCCESS | Total time {total_time}s")

        return {
            "request_id": request_id,
            "processing_time_sec": total_time,
            **result
        }

    except TimeoutError:
        logger.error(f"[{request_id}] Processing timeout")
        raise HTTPException(
            status_code=408,
            detail="Processing exceeded time limit"
        )

    except Exception as e:
        logger.error(f"[{request_id}] ERROR | {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"[{request_id}] Temporary file removed")

# ---------------- QUICK VIDEO TEST ---------------- #

@app.post("/test_video_quick")
def test_video_quick(request: VideoRequest):
    import cv2

    try:
        response = requests.get(
            request.video_url,
            stream=True,
            timeout=20
        )

        if response.status_code != 200:
            return {"status": "FAILED", "reason": "Download failed"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            path = tmp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.remove(path)
            return {"status": "FAILED", "reason": "Invalid video"}

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, frame = cap.read()
        cap.release()
        os.remove(path)

        return {
            "status": "OK",
            "total_frames": frames,
            "fps": fps,
            "duration_sec": round(frames / fps, 2) if fps else 0,
            "first_frame_ok": ret,
            "recommended": "≤ 15 seconds video"
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e)
        }
