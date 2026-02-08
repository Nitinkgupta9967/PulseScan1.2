import os
import uuid
import time
import logging
import requests
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("PulseScan")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="PulseScan API",
    description="Remote rPPG-based vital extraction",
    version="1.0"
)

# =========================
# REQUEST MODEL
# =========================
class VideoRequest(BaseModel):
    video_url: str

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def root():
    return {"status": "OK", "message": "PulseScan API is running"}

# =========================
# MAIN ANALYSIS ENDPOINT
# =========================
@app.post("/analyze-video")
def analyze_video(payload: VideoRequest):
    request_id = str(uuid.uuid4())[:8]
    api_start = time.time()

    logger.info(f"[{request_id}] Request received")
    logger.info(f"[{request_id}] Video URL: {payload.video_url}")

    # -------------------------
    # 1. DOWNLOAD VIDEO
    # -------------------------
    try:
        dl_start = time.time()
        logger.info(f"[{request_id}] Starting video download")

        response = requests.get(payload.video_url, stream=True, timeout=30)
        if response.status_code != 200:
            raise Exception("Video download failed")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            video_path = tmp.name

        logger.info(
            f"[{request_id}] Video download completed in "
            f"{round(time.time() - dl_start, 2)} sec"
        )

    except Exception as e:
        logger.error(f"[{request_id}] Download error: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to download video")

    # -------------------------
    # 2. PROCESS VIDEO
    # -------------------------
    try:
        logger.info(f"[{request_id}] Starting rPPG processing")

        PROCESSING_TIMEOUT = 120  # seconds (adjust if needed)
        process_start = time.time()

        # 👉 IMPORT HERE to avoid startup delays
        from pipeline_runner import run_pipeline


        result = run_pipeline(
            video_path=video_path,
            request_id=request_id,
            start_time=process_start,
            timeout=PROCESSING_TIMEOUT
        )

        logger.info(
            f"[{request_id}] Processing completed in "
            f"{round(time.time() - process_start, 2)} sec"
        )

    except TimeoutError:
        logger.error(f"[{request_id}] Processing timeout")
        raise HTTPException(
            status_code=408,
            detail="Processing timeout – video quality unstable or too long"
        )

    except Exception as e:
        logger.exception(f"[{request_id}] Processing failed")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error"
        )

    finally:
        # -------------------------
        # CLEANUP
        # -------------------------
        try:
            os.remove(video_path)
            logger.info(f"[{request_id}] Temporary file deleted")
        except Exception:
            pass

    # -------------------------
    # 3. RESPONSE
    # -------------------------
    logger.info(
        f"[{request_id}] Total request time "
        f"{round(time.time() - api_start, 2)} sec"
    )

    return result
