import cv2
import random
from luminance_analysis import luminance_analysis
from face_roi_extraction import extract_face_rois

def single_face_video_check(video_path, total_samples=20, min_pass=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < total_samples:
        cap.release()
        return False

    indices = random.sample(range(total_frames), total_samples)

    valid_frames = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 🔆 Step 1: Luminance
        if not luminance_analysis(frame):
            continue

        # 🙂 Step 2: Face + ROI extraction
        rois = extract_face_rois(frame)
        if rois is None:
            continue

        # Frame is VALID
        valid_frames += 1

    cap.release()
    return valid_frames >= min_pass
