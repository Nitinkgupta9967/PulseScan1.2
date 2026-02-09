import cv2
import os

# Load cascade classifier once at module level
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    # Fallback path for some installations
    cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

# Verify cascade loaded successfully
if face_cascade.empty():
    raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")

def extract_face_rois(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    # Define ROIs
    forehead = frame[y:y + int(0.25*h), x + int(0.3*w):x + int(0.7*w)]
    left_cheek = frame[y + int(0.4*h):y + int(0.7*h), x:x + int(0.3*w)]
    right_cheek = frame[y + int(0.4*h):y + int(0.7*h), x + int(0.7*w):x + w]

    # Validate ROI sizes
    if forehead.size == 0 or left_cheek.size == 0 or right_cheek.size == 0:
        return None

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek
    }