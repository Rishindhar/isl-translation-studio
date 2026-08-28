import os
import glob
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM
from src.utils import normalize_landmarks

def scale_sequence(seq, scale_range=(0.9, 1.1)):
    """Applies random uniform scaling to landmark coordinates."""
    scale = np.random.uniform(*scale_range)
    return seq * scale

def add_gaussian_noise(seq, std=0.005):
    """Adds small Gaussian noise to simulate sensor jitter."""
    noise = np.random.normal(0, std, seq.shape)
    return seq + noise

def rotate_sequence_2d(seq, angle_deg_range=(-10, 10)):
    """Rotates landmark coordinates around wrist in XY plane."""
    angle_rad = np.radians(np.random.uniform(*angle_deg_range))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    augmented = seq.copy()
    for t in range(SEQUENCE_LENGTH):
        coords = augmented[t].reshape(21, 3)
        xy = coords[:, :2]
        coords[:, :2] = np.dot(xy, rot_matrix.T)
        augmented[t] = normalize_landmarks(coords)
    return augmented

def time_warp_sequence(seq):
    """Simulates faster or slower gesture execution using 1D interpolation."""
    warp_factor = np.random.uniform(0.8, 1.2)
    orig_steps = np.linspace(0, 1, SEQUENCE_LENGTH)
    new_steps = np.clip(np.linspace(0, 1, SEQUENCE_LENGTH) ** warp_factor, 0, 1)

    warped = np.zeros_like(seq)
    for f in range(FEATURE_DIM):
        warped[:, f] = np.interp(new_steps, orig_steps, seq[:, f])
    return warped

def augment_dataset(augmentation_factor=2):
    """
    Augments existing .npy sequence files in data/ to improve model robustness.
    
    :param augmentation_factor: Number of augmented samples to generate per original file
    """
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    gesture_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    total_generated = 0

    for folder in gesture_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        npy_files = glob.glob(os.path.join(folder_path, "sample_*.npy"))

        for file_path in npy_files:
            if "_aug_" in file_path:
                continue

            arr = np.load(file_path)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                continue

            base_name = os.path.splitext(os.path.basename(file_path))[0]

            for aug_idx in range(augmentation_factor):
                # Apply combination of scale, noise, rotation, and time warp
                aug = scale_sequence(arr)
                aug = add_gaussian_noise(aug)
                aug = rotate_sequence_2d(aug)
                if np.random.rand() > 0.5:
                    aug = time_warp_sequence(aug)

                aug_filename = f"{base_name}_aug_{aug_idx + 1}.npy"
                aug_filepath = os.path.join(folder_path, aug_filename)
                np.save(aug_filepath, aug.astype(np.float32))
                total_generated += 1

    print(f"Successfully generated {total_generated} augmented sequence samples across {len(gesture_folders)} gesture classes.")

if __name__ == "__main__":
    augment_dataset(augmentation_factor=2)
