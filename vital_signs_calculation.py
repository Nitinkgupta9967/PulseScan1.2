import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

def calculate_heart_rate(bvp_signal, fs=30):
    """
    bvp_signal : 1D filtered BVP signal
    fs         : sampling rate (FPS)
    """

    # Peak detection (min 0.4s between beats → max 150 BPM)
    peaks, _ = find_peaks(
        bvp_signal,
        distance=int(0.4 * fs),
        prominence=0.2 * np.std(bvp_signal)
    )

    if len(peaks) < 2:
        return None

    # RR intervals (seconds)
    rr_intervals = np.diff(peaks) / fs

    mean_rr = np.mean(rr_intervals)
    heart_rate = 60 / mean_rr

    return round(heart_rate, 2)

def calculate_respiration_rate(bvp_signal, fs=30):
    """
    Estimate respiration rate from BVP signal
    """

    # Band-pass filter for respiration (0.1–0.5 Hz)
    nyq = 0.5 * fs
    b, a = butter(2, [0.1 / nyq, 0.5 / nyq], btype="band")
    resp_signal = filtfilt(b, a, bvp_signal)

    # Peak detection (min 2s between breaths → max 30 BPM)
    peaks, _ = find_peaks(
        resp_signal,
        distance=int(2 * fs),
        prominence=0.1 * np.std(resp_signal)
    )

    if len(peaks) < 2:
        return None

    duration_sec = len(resp_signal) / fs
    respiration_rate = (len(peaks) / duration_sec) * 60

    return round(respiration_rate, 2)

def calculate_spo2(red_signal, green_signal):
    """
    Estimate SpO2 using ratio-of-ratios method
    """

    # Remove DC component
    red_ac = red_signal - np.mean(red_signal)
    green_ac = green_signal - np.mean(green_signal)

    red_dc = np.mean(red_signal)
    green_dc = np.mean(green_signal)

    if red_dc == 0 or green_dc == 0:
        return None

    # AC/DC ratios
    r_red = np.std(red_ac) / red_dc
    r_green = np.std(green_ac) / green_dc

    R = r_red / r_green

    spo2 = 110 - 25 * R
    spo2 = np.clip(spo2, 85, 100)

    return round(float(spo2), 2)
import numpy as np

def estimate_confidence(signal, min_len=300):
    """
    Simple signal quality confidence estimation
    """

    if signal is None or len(signal) < min_len:
        return "LOW"

    snr = np.std(signal) / (np.mean(np.abs(signal)) + 1e-6)

    if snr > 0.6:
        return "HIGH"
    elif snr > 0.3:
        return "MEDIUM"
    else:
        return "LOW"
import random

def generate_nominal_spo2():
    """
    Generate realistic nominal SpO2 for demo purposes
    """
    return round(random.uniform(95.0, 99.0), 1)
