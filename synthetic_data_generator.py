import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM, DEFAULT_GESTURES

def generate_gesture_kinematics(gesture_name, sequence_length=30):
    """
    Generates 3D landmark sequence feature matrices (30 x 63) using 
    kinematic joint posture transformations tailored to specific ISL signs.
    """
    seq = np.zeros((sequence_length, 21, 3), dtype=np.float32)
    t = np.linspace(0, 1, sequence_length)

    # Base hand skeleton template (wrist at 0,0,0)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    
    # Palm joints (0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky)
    base_hand[0] = [0.0, 0.0, 0.0]
    
    # Finger bases
    base_hand[1] = [-0.05, 0.05, 0.0]
    base_hand[5] = [-0.03, 0.12, 0.0]
    base_hand[9] = [0.0, 0.13, 0.0]
    base_hand[13] = [0.03, 0.12, 0.0]
    base_hand[17] = [0.05, 0.10, 0.0]

    # Kinematic trajectories for specific signs
    if gesture_name == "hello" or gesture_name == "namaste":
        # Waving open palm side to side
        for i in range(sequence_length):
            h = base_hand.copy()
            wave = 0.08 * np.sin(t[i] * np.pi * 4)
            # Extended fingers
            h[4] = [-0.12 + wave, 0.10, 0.02]
            h[8] = [-0.04 + wave, 0.22, 0.0]
            h[12] = [0.0 + wave, 0.24, 0.0]
            h[16] = [0.04 + wave, 0.22, 0.0]
            h[20] = [0.08 + wave, 0.19, 0.0]
            seq[i] = h

    elif gesture_name == "thank_you":
        # Hand moving forward from chin
        for i in range(sequence_length):
            h = base_hand.copy()
            fwd = 0.15 * t[i]
            h[4] = [-0.08, 0.10 + fwd, 0.05]
            h[8] = [-0.03, 0.20 + fwd, 0.05]
            h[12] = [0.0, 0.22 + fwd, 0.05]
            h[16] = [0.03, 0.20 + fwd, 0.05]
            h[20] = [0.07, 0.18 + fwd, 0.05]
            seq[i] = h

    elif gesture_name == "yes":
        # Fist nodding up and down
        for i in range(sequence_length):
            h = base_hand.copy()
            nod = 0.05 * np.sin(t[i] * np.pi * 6)
            # Curled fingers (fist)
            h[4] = [-0.06, 0.06 + nod, 0.02]
            h[8] = [-0.02, 0.08 + nod, 0.02]
            h[12] = [0.0, 0.08 + nod, 0.02]
            h[16] = [0.02, 0.08 + nod, 0.02]
            h[20] = [0.05, 0.07 + nod, 0.02]
            seq[i] = h

    elif gesture_name == "no":
        # Index and middle finger snapping shut
        for i in range(sequence_length):
            h = base_hand.copy()
            snap = 0.08 * (1.0 - np.cos(t[i] * np.pi * 4))
            h[4] = [-0.08, 0.08, 0.02]
            h[8] = [-0.03, 0.20 - snap, 0.0]
            h[12] = [0.0, 0.22 - snap, 0.0]
            h[16] = [0.03, 0.08, 0.02]
            h[20] = [0.06, 0.07, 0.02]
            seq[i] = h

    elif gesture_name == "loud_speaker":
        # Funnel shape expanding near mouth
        for i in range(sequence_length):
            h = base_hand.copy()
            expand = 0.12 * t[i]
            h[4] = [-0.10 - expand, 0.12 + expand, 0.05]
            h[8] = [-0.04 - expand, 0.24 + expand, 0.05]
            h[12] = [0.0, 0.26 + expand, 0.05]
            h[16] = [0.04 + expand, 0.24 + expand, 0.05]
            h[20] = [0.09 + expand, 0.20 + expand, 0.05]
            seq[i] = h

    elif gesture_name == "gram":
        # Pinching thumb and index moving downwards
        for i in range(sequence_length):
            h = base_hand.copy()
            down = -0.10 * t[i]
            pinch = 0.02 * np.sin(t[i] * np.pi * 2)
            h[4] = [-0.03 + pinch, 0.12 + down, 0.02]
            h[8] = [-0.03 - pinch, 0.12 + down, 0.02]
            h[12] = [0.0, 0.08 + down, 0.01]
            h[16] = [0.03, 0.07 + down, 0.01]
            h[20] = [0.06, 0.06 + down, 0.01]
            seq[i] = h

    elif gesture_name.startswith("number_"):
        # Digit 0-9 posture
        num = int(gesture_name.split("_")[1])
        for i in range(sequence_length):
            h = base_hand.copy()
            # Finger extension according to digit count
            h[4] = [-0.10, 0.12, 0.0] if num >= 1 else [-0.04, 0.06, 0.0]
            h[8] = [-0.03, 0.22, 0.0] if num >= 2 or num == 1 else [-0.03, 0.08, 0.0]
            h[12] = [0.0, 0.24, 0.0] if num >= 3 or num == 2 else [0.0, 0.08, 0.0]
            h[16] = [0.03, 0.22, 0.0] if num >= 4 or num == 3 else [0.03, 0.08, 0.0]
            h[20] = [0.07, 0.19, 0.0] if num >= 5 or num == 4 else [0.07, 0.07, 0.0]
            seq[i] = h

    elif gesture_name.startswith("letter_"):
        # Alphabet A-Z posture
        char_idx = ord(gesture_name.split("_")[1]) - ord('a')
        for i in range(sequence_length):
            h = base_hand.copy()
            angle = (char_idx / 26.0) * np.pi * 2
            h[4] = [-0.08 * np.cos(angle), 0.12 * np.sin(angle), 0.02]
            h[8] = [-0.03 * np.sin(angle), 0.20 * np.cos(angle), 0.02]
            h[12] = [0.02 * np.cos(angle), 0.22 * np.sin(angle), 0.02]
            h[16] = [0.04 * np.sin(angle), 0.18 * np.cos(angle), 0.02]
            h[20] = [0.06 * np.cos(angle), 0.15 * np.sin(angle), 0.02]
            seq[i] = h

    else:
        # Generic ISL word movement
        for i in range(sequence_length):
            h = base_hand.copy()
            move = 0.06 * np.sin(t[i] * np.pi * 3)
            h[4] = [-0.08 + move, 0.10, 0.02]
            h[8] = [-0.03 + move, 0.20, 0.0]
            h[12] = [0.0 + move, 0.22, 0.0]
            h[16] = [0.03 + move, 0.20, 0.0]
            h[20] = [0.07 + move, 0.18, 0.0]
            seq[i] = h

    return seq.reshape(sequence_length, FEATURE_DIM)

def generate_full_isl_dataset(samples_per_gesture=30):
    """Generates synthetic dataset for all ISL gestures in DEFAULT_GESTURES."""
    total_generated = 0
    for gesture in DEFAULT_GESTURES:
        gesture_dir = os.path.join(DATA_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        
        for sample_idx in range(samples_per_gesture):
            seq = generate_gesture_kinematics(gesture)
            
            # Add Gaussian noise and micro-scaling for dataset variance
            noise = np.random.normal(0, 0.003, seq.shape).astype(np.float32)
            scale = np.random.uniform(0.95, 1.05)
            aug_seq = (seq * scale + noise).astype(np.float32)
            
            file_path = os.path.join(gesture_dir, f"synthetic_{sample_idx + 1}.npy")
            np.save(file_path, aug_seq)
            total_generated += 1
            
    print(f"SUCCESS: Generated {total_generated} sequence samples across {len(DEFAULT_GESTURES)} ISL gesture classes.")
    return total_generated

if __name__ == "__main__":
    generate_full_isl_dataset()
