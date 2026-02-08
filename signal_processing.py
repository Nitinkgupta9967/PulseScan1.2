import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=30, low=0.7, high=4.0):
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)
