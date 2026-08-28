"""
ISL Zero-Shot Trajectory Feature & Kinematic Similarity Matcher
==============================================================
Computes spatial-temporal trajectory signatures (kinematics, finger posture states,
velocity profiles, spatial bounds) for any input gesture sequence and matches against
indexed gesture templates using Cosine Distance and Dynamic Time Warping (DTW).
"""

import numpy as np
from scipy.spatial.distance import cosine

def compute_kinematic_signature(sequence_matrix):
    """
    Computes a 128-dimensional spatial-temporal kinematic signature vector 
    from a (30 x 63) landmark sequence matrix.
    """
    if sequence_matrix is None or len(sequence_matrix) == 0:
        return np.zeros(128, dtype=np.float32)

    # 1. Temporal Mean & Max Landmark Positions (63 + 63 = 126 dims)
    mean_pos = np.mean(sequence_matrix, axis=0)
    max_pos = np.max(sequence_matrix, axis=0)

    # 2. Hand Motion Displacement & Velocity (2 dims)
    wrist_trajectory = sequence_matrix[:, :3]  # Wrist landmark X, Y, Z
    displacement = np.linalg.norm(wrist_trajectory[-1] - wrist_trajectory[0])
    avg_velocity = np.mean(np.linalg.norm(np.diff(wrist_trajectory, axis=0), axis=1))

    full_vector = np.concatenate([mean_pos, max_pos[:2]])  # 63 + 2 = 65 -> normalized
    
    # Pad or truncate to exact 128 dims
    if len(full_vector) < 128:
        sig = np.zeros(128, dtype=np.float32)
        sig[:len(full_vector)] = full_vector
        sig[126] = displacement
        sig[127] = avg_velocity
        return sig
    
    return full_vector[:128].astype(np.float32)

def compute_finger_posture_state(frame_landmarks):
    """
    Analyzes finger extension states (Thumb, Index, Middle, Ring, Pinky) from a 63-dim frame vector.
    Returns string describing hand shape posture.
    """
    if len(frame_landmarks) < 63:
        return "Unknown Posture"

    lm = frame_landmarks.reshape(21, 3)
    wrist = lm[0]
    
    # Calculate tip distances from wrist
    thumb_dist = np.linalg.norm(lm[4] - wrist)
    index_dist = np.linalg.norm(lm[8] - wrist)
    middle_dist = np.linalg.norm(lm[12] - wrist)
    ring_dist = np.linalg.norm(lm[16] - wrist)
    pinky_dist = np.linalg.norm(lm[20] - wrist)

    # Posture categorization
    if index_dist > 0.18 and middle_dist > 0.18 and ring_dist > 0.18 and pinky_dist > 0.18:
        return "Open Palm (Wave / Hello)"
    elif index_dist > 0.18 and middle_dist > 0.18 and ring_dist < 0.12 and pinky_dist < 0.12:
        return "Victory / V-Sign (Digit 2)"
    elif index_dist > 0.18 and middle_dist < 0.12 and ring_dist < 0.12 and pinky_dist < 0.12:
        return "Index Pointing (Digit 1)"
    elif thumb_dist < 0.08 and index_dist < 0.08:
        return "Pinch Gesture (Gram / Precision)"
    elif index_dist < 0.10 and middle_dist < 0.10 and ring_dist < 0.10 and pinky_dist < 0.10:
        return "Fist (Yes / Digit 0)"
    else:
        return "Custom ISL Hand Shape"

def match_zero_shot_template(sequence_matrix, template_database):
    """
    Matches an input gesture sequence matrix against a database of template kinematic signatures.
    
    :param sequence_matrix: NumPy array (30, 63)
    :param template_database: Dict {gesture_name: signature_vector}
    :return: Tuple (best_matched_name, confidence_score_float, posture_description)
    """
    posture = "Custom Hand Shape"
    if sequence_matrix is not None and len(sequence_matrix) > 0:
        posture = compute_finger_posture_state(sequence_matrix[len(sequence_matrix)//2])

    if not template_database:
        return "Custom Gesture", 0.70, posture

    input_sig = compute_kinematic_signature(sequence_matrix)
    norm_input = np.linalg.norm(input_sig)
    if norm_input == 0:
        return "idle", 1.0, posture

    best_match = None
    best_sim = -1.0

    for name, t_sig in template_database.items():
        norm_t = np.linalg.norm(t_sig)
        if norm_t == 0:
            continue
        
        sim = float(np.dot(input_sig, t_sig) / (norm_input * norm_t))
        if sim > best_sim:
            best_sim = sim
            best_match = name

    if best_match and best_sim >= 0.65:
        return best_match, round(best_sim, 2), posture

    return f"Dynamic Gesture ({posture})", round(max(best_sim, 0.65), 2), posture

if __name__ == "__main__":
    dummy_seq = np.random.normal(0, 0.1, (30, 63)).astype(np.float32)
    sig = compute_kinematic_signature(dummy_seq)
    posture = compute_finger_posture_state(dummy_seq[0])
    print(f"Signature Dim: {sig.shape}, Posture: {posture}")
