# app.py

import time
import uuid
import logging
import tempfile
import requests
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline_runner import run_pipeline

# ---------------- Logging Setup ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PulseScan")

# ---------------- FastAPI App ---------------- #

app = FastAPI(
    title="PulseScan API",
    description="Video-based rPPG Vital Signs Extraction",
    version="1.0"
)

# ---------------- Request Model ---------------- #

class VideoRequest(BaseModel):
    video_url: str

# ---------------- Root Endpoint ---------------- #

@app.get("/")
def root():
    return {
        "status": "PulseScan API running",
        "docs": "/docs"
    }

# ---------------- Analyze Endpoint ---------------- #

@app.post("/analyze")
def analyze_video(request: VideoRequest):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] Request received")
    logger.info(f"[{request_id}] Video URL: {request.video_url}")

    try:
        # ---------- Download Video ----------
        logger.info(f"[{request_id}] Starting video download")
        download_start = time.time()

        response = requests.get(request.video_url, stream=True, timeout=60)

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download video")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_video.write(chunk)
            video_path = temp_video.name

        download_end = time.time()
        logger.info(
            f"[{request_id}] Video download completed "
            f"in {round(download_end - download_start, 2)} sec"
        )

        # ---------- Processing ----------
        logger.info(f"[{request_id}] Starting rPPG processing")
        processing_start = time.time()

        result = run_pipeline(video_path)

        processing_end = time.time()
        logger.info(
            f"[{request_id}] rPPG processing completed "
            f"in {round(processing_end - processing_start, 2)} sec"
        )

        # ---------- Cleanup ----------
        if os.path.exists(video_path):
            os.remove(video_path)

        total_time = time.time() - start_time
        logger.info(
            f"[{request_id}] TOTAL request time: {round(total_time, 2)} sec"
        )

        return result

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"[{request_id}] ERROR after {round(total_time, 2)} sec | {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
