import cv2
import numpy as np

def luminance_analysis(frame, min_lum=50, max_lum=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luminance = np.mean(gray)

    if mean_luminance < min_lum:
        return False
    if mean_luminance > max_lum:
        return False

    return True
