from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from video_downloader import download_video
from pipeline_runner import run_pipeline

app = FastAPI(title="PulseScan API")

class VideoRequest(BaseModel):
    cloudinary_url: str

@app.post("/analyze-video")
def analyze_video(request: VideoRequest):
    try:
        video_path = download_video(request.cloudinary_url)
        result = run_pipeline(video_path)
        return result

    except ValueError as ve:
        # Client-side error (bad video)
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )

    except Exception as e:
        # Server-side error
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
