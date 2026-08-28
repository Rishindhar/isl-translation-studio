import json
import os
import numpy as np
from src.config import FEATURE_DIM, LABELS_PATH, IDLE_VELOCITY_THRESHOLD

def normalize_landmarks(raw_landmarks):
    """
    Applies Spatial Normalization (Wrist-Relative Displacement).
    Subtracts wrist landmark coordinates (Landmark 0: X_0, Y_0, Z_0) from all 21 joint coordinates,
    converting absolute frame positions into relative displacement vectors from the wrist.

    Also applies Max-abs scaling to keep coordinates within [-1.0, 1.0] range for scale-invariance.

    :param raw_landmarks: NumPy array of shape (63,) or list of 21 (x, y, z) objects/tuples.
    :return: Normalized 1D NumPy array of shape (63,)
    """
    if isinstance(raw_landmarks, np.ndarray):
        arr = raw_landmarks.reshape(21, 3).copy()
    else:
        arr = np.array(raw_landmarks, dtype=np.float32).reshape(21, 3)

    wrist_coord = arr[0].copy()  # Landmark 0: Wrist
    arr = arr - wrist_coord      # Spatial normalization: relative displacement from wrist

    # Max-abs scaling to keep coordinates within [-1.0, 1.0] range
    max_val = np.max(np.abs(arr))
    if max_val > 1e-6:
        arr = arr / max_val

    return arr.flatten()

def calculate_hand_velocity(prev_frame, curr_frame):
    """
    Computes average landmark displacement velocity between consecutive frames.
    Used for idle state optimization to skip GRU inference when hands are stationary.

    :param prev_frame: (63,) float array
    :param curr_frame: (63,) float array
    :return: float (average Euclidean velocity)
    """
    if prev_frame is None or curr_frame is None:
        return 0.0

    diff = curr_frame - prev_frame
    velocity = float(np.mean(np.abs(diff)))
    return velocity

def is_hand_idle(prev_frame, curr_frame, threshold=IDLE_VELOCITY_THRESHOLD):
    """Returns True if the hand velocity is below the idle threshold."""
    return calculate_hand_velocity(prev_frame, curr_frame) < threshold

def save_gesture_labels(labels_list, path=LABELS_PATH):
    """Saves gesture label list to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, indent=2, ensure_ascii=False)

def load_gesture_labels(path=LABELS_PATH):
    """Loads gesture label list from JSON."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
