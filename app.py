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
        "docs": "/docs",
        "test_endpoints": {
            "cascade": "/test_cascade",
            "dependencies": "/test_dependencies",
            "video_quick_test": "/test_video_quick"
        }
    }

# ---------------- Analyze Endpoint ---------------- #

@app.post("/analyze_video")
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


# ---------------- Test Endpoints ---------------- #

@app.get("/test_cascade")
def test_cascade():
    """Test if face cascade classifier can be loaded"""
    import cv2
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    exists = os.path.exists(cascade_path)
    
    fallback_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    fallback_exists = os.path.exists(fallback_path)
    
    cascade = cv2.CascadeClassifier(cascade_path)
    loaded = not cascade.empty()
    
    return {
        "opencv_version": cv2.__version__,
        "primary_cascade": {
            "path": cascade_path,
            "exists": exists,
            "loaded": loaded
        },
        "fallback_cascade": {
            "path": fallback_path,
            "exists": fallback_exists
        },
        "status": "OK" if loaded else "FAILED"
    }


@app.get("/test_dependencies")
def test_dependencies():
    """Test if all dependencies are installed"""
    deps = {}
    
    # Test each dependency
    for dep_name in ['cv2', 'numpy', 'scipy', 'requests']:
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            deps[dep_name] = {
                "installed": True,
                "version": version
            }
        except ImportError as e:
            deps[dep_name] = {
                "installed": False,
                "error": str(e)
            }
    
    return {
        "python_version": sys.version,
        "dependencies": deps
    }


@app.post("/test_video_quick")
def test_video_quick(request: VideoRequest):
    """Quick test to see if video can be downloaded and opened"""
    import cv2
    
    try:
        # Download
        logger.info(f"Testing video: {request.video_url}")
        response = requests.get(request.video_url, stream=True, timeout=30)
        if response.status_code != 200:
            return {"status": "FAILED", "reason": "Download failed", "status_code": response.status_code}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_video.write(chunk)
            video_path = temp_video.name
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return {"status": "FAILED", "reason": "Video file corrupted or invalid format"}
        
        # Get info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try to read first frame
        ret, frame = cap.read()
        first_frame_ok = ret
        
        # Test luminance on first frame
        lum_ok = False
        mean_lum = None
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(gray.mean())
            lum_ok = 50 <= mean_lum <= 200
        
        # Test face detection on first frame
        faces_detected = 0
        cascade_loaded = False
        if ret:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            cascade_loaded = not cascade.empty()
            
            if cascade_loaded:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.2, 5)
                faces_detected = len(faces)
        
        cap.release()
        os.remove(video_path)
        
        return {
            "status": "SUCCESS",
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "duration_seconds": round(total_frames / fps if fps > 0 else 0, 2)
            },
            "first_frame": {
                "readable": first_frame_ok,
                "mean_luminance": mean_lum,
                "luminance_ok": lum_ok
            },
            "face_detection": {
                "cascade_loaded": cascade_loaded,
                "faces_in_first_frame": faces_detected,
                "detection_ok": faces_detected == 1
            },
            "ready_for_processing": first_frame_ok and lum_ok and faces_detected == 1
        }
        
    except Exception as e:
        logger.error(f"Test video error: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e)
        }