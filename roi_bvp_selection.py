import numpy as np

def extract_raw_bvp_from_roi(roi):
    # Green channel
    green = roi[:, :, 1]
    return np.mean(green)

def extract_frame_bvp(rois):
    bvp_signals = []

    for roi in rois.values():
        bvp = extract_raw_bvp_from_roi(roi)
        bvp_signals.append(bvp)

    return bvp_signals

def select_best_bvp(bvp_history):
    variances = [np.var(sig) for sig in bvp_history]
    best_index = variances.index(max(variances))
    return bvp_history[best_index]
def extract_rgb_signals(roi):
    red = np.mean(roi[:, :, 2])
    green = np.mean(roi[:, :, 1])
    return red, green

