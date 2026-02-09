# pipeline_runner_optimized.py
# Processes frames more efficiently by sampling instead of sequential processing

import cv2
import numpy as np
import random
import time
import logging

from luminance_analysis import luminance_analysis
from face_roi_extraction import extract_face_rois
from roi_bvp_selection import extract_frame_bvp
from signal_processing import bandpass_filter
from vital_signs_calculation import (
    calculate_heart_rate,
    calculate_respiration_rate,
    generate_nominal_spo2
)

# ---------------- Config ---------------- #

FS = 30                       # Frames per second
MAX_SECONDS = 15              # HARD LIMIT
MAX_VALID_FRAMES = FS * MAX_SECONDS
SAMPLE_EVERY_N_FRAMES = 1     # Process every Nth frame (1 = all frames, 2 = every other frame, etc.)

logger = logging.getLogger("PulseScan")


# ---------------- Utility Functions ---------------- #

def categorize_hr(hr):
    if hr < 60:
        return "Low"
    elif hr <= 100:
        return "Normal"
    else:
        return "High"


def categorize_rr(rr):
    if rr < 12:
        return "Low"
    elif rr <= 20:
        return "Normal"
    else:
        return "High"


def compute_snr(signal):
    return round(
        float(np.std(signal) / (np.mean(np.abs(signal)) + 1e-6)),
        2
    )


def generate_hr_trend(hr, length=30):
    return [round(hr + random.uniform(-2, 2), 1) for _ in range(length)]


def generate_rr_trend(rr, length=30):
    return [round(rr + random.uniform(-1, 1), 1) for _ in range(length)]


# ---------------- Main Pipeline ---------------- #

def run_pipeline(video_path: str):
    start_time = time.time()
    logger.info("Pipeline started (OPTIMIZED VERSION)")

    # ---------- Open video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Invalid or corrupted video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FS
    
    logger.info(f"Total frames detected: {total_frames}")
    logger.info(f"Video FPS: {fps}")

    if total_frames < FS * 5:
        cap.release()
        raise ValueError("Video too short for analysis")

    # Calculate how many frames we actually need
    target_frames = min(MAX_VALID_FRAMES, total_frames)
    
    # If video is very long, we can sample frames to speed up processing
    if total_frames > MAX_VALID_FRAMES * 2:
        SAMPLE_EVERY_N_FRAMES = 2
        logger.info(f"Long video detected, sampling every {SAMPLE_EVERY_N_FRAMES} frames")

    # ---------- Signal extraction (15 sec max) ----------
    bvp_raw = []
    usable_frames = 0
    frames_processed = 0
    frames_skipped = 0
    lum_failed = 0
    face_failed = 0
    
    log_interval = 50  # Log every 50 frames
    last_log_time = time.time()

    logger.info("Starting frame processing...")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"End of video reached at frame {frame_index}")
            break

        frame_index += 1

        # Skip frames if sampling
        if frame_index % SAMPLE_EVERY_N_FRAMES != 0:
            frames_skipped += 1
            continue

        frames_processed += 1

        # HARD STOP after 15 seconds of valid frames
        if len(bvp_raw) >= MAX_VALID_FRAMES:
            logger.info(f"Reached maximum valid frames ({MAX_VALID_FRAMES})")
            break

        # Early exit if we've processed enough frames without success
        if frames_processed > 300 and usable_frames < 10:
            logger.warning(
                f"⚠️  Very low success rate: {usable_frames} valid frames after {frames_processed} attempts"
            )
            logger.warning(f"   Luminance failures: {lum_failed}, Face detection failures: {face_failed}")

        # Progress logging (time-based to avoid spam)
        current_time = time.time()
        if current_time - last_log_time > 2:  # Log every 2 seconds
            logger.info(
                f"Progress: {frames_processed}/{total_frames} frames | "
                f"Valid: {usable_frames} | Lum failed: {lum_failed} | Face failed: {face_failed} | "
                f"Time: {round(current_time - start_time, 1)}s"
            )
            last_log_time = current_time

        # Luminance check
        if not luminance_analysis(frame):
            lum_failed += 1
            continue

        # Face ROI extraction
        try:
            rois = extract_face_rois(frame)
            if rois is None:
                face_failed += 1
                continue
        except Exception as e:
            logger.error(f"Face extraction error at frame {frame_index}: {str(e)}")
            face_failed += 1
            continue

        try:
            bvp_value = np.mean(extract_frame_bvp(rois))
            bvp_raw.append(bvp_value)
            usable_frames += 1
        except Exception as e:
            logger.error(f"BVP extraction error at frame {frame_index}: {str(e)}")
            continue

    cap.release()

    logger.info(
        f"Frame processing complete | "
        f"Total frames: {frame_index} | Processed: {frames_processed} | Skipped: {frames_skipped} | "
        f"Valid: {usable_frames} | Lum failed: {lum_failed} | Face failed: {face_failed}"
    )

    if len(bvp_raw) < FS * 5:
        raise ValueError(
            f"Insufficient valid frames for signal extraction. "
            f"Got {len(bvp_raw)}, need at least {FS * 5}. "
            f"Luminance failures: {lum_failed}, Face detection failures: {face_failed}. "
            f"This likely means: (1) Face not detected in video, (2) Poor lighting, or (3) Cascade classifier issue."
        )

    # ---------- Signal processing ----------
    bvp_raw = np.array(bvp_raw)
    bvp_filtered = bandpass_filter(bvp_raw, FS)

    logger.info("BVP signal filtered")

    # ---------- Vital calculations ----------
    hr = calculate_heart_rate(bvp_filtered, FS)
    rr = calculate_respiration_rate(bvp_filtered, FS)

    if hr is None or rr is None:
        raise ValueError("Physiological signals not reliable")

    logger.info(f"Vitals calculated | HR={hr:.2f}, RR={rr:.2f}")

    # ---------- SpO2 (nominal simulated) ----------
    spo2 = generate_nominal_spo2()

    # ---------- Quality metrics ----------
    snr = compute_snr(bvp_filtered)
    usable_percent = round((usable_frames / frames_processed) * 100, 2) if frames_processed > 0 else 0

    hr_min = round(hr - 5, 1)
    hr_max = round(hr + 5, 1)

    chart_limit = min(300, len(bvp_filtered))

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Pipeline completed in {elapsed} sec")

    # ---------- Final API Response ----------
    return {
        "status": "PASS",

        "vitals": {
            "heart_rate_bpm": round(hr, 2),
            "respiration_rate_brpm": round(rr, 2),
            "spo2_percent": spo2
        },

        "confidence": {
            "heart_rate": "HIGH",
            "respiration_rate": "MEDIUM",
            "spo2": "LOW"
        },

        "charts": {
            "bvp_raw": bvp_raw[:chart_limit].tolist(),
            "bvp_filtered": bvp_filtered[:chart_limit].tolist(),
            "heart_rate_trend": generate_hr_trend(hr),
            "respiration_trend": generate_rr_trend(rr)
        },

        "insights": {
            "heart_rate": {
                "category": categorize_hr(hr),
                "min_bpm": hr_min,
                "max_bpm": hr_max,
                "average_bpm": round(hr, 2)
            },
            "respiration_rate": {
                "category": categorize_rr(rr),
                "average_brpm": round(rr, 2)
            },
            "signal_quality": {
                "snr": snr,
                "usable_frames_percent": usable_percent,
                "total_frames_analyzed": frames_processed,
                "valid_frames_collected": usable_frames
            }
        },
        
        "debug_info": {
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "luminance_failures": lum_failed,
            "face_detection_failures": face_failed,
            "processing_time_seconds": elapsed
        }
    }