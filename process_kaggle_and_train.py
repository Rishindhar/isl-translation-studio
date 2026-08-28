"""
ULTRA-FAST ISL Video Processor
===============================
Single MediaPipe init, sample only 10 key frames per video,
build sequences, retrain. Target: <10 minutes for 984 videos.
"""
import os, sys, glob, re, time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM, BASE_DIR

KAGGLE_DIR = os.path.join(BASE_DIR, "kaggle_isl_data")
FRAMES_TO_SAMPLE = 10  # Only read 10 evenly-spaced frames per video

def clean_label(filename):
    base = os.path.splitext(filename)[0]
    base = re.sub(r'[-_]\d+$', '', base)
    base = re.sub(r'_\d+_[A-F0-9]+$', '', base, flags=re.IGNORECASE)
    base = base.strip().lower().replace(' ', '_').replace('-', '_')
    base = re.sub(r'[^a-z0-9_]', '', base)
    base = re.sub(r'_+', '_', base).strip('_')
    return base if base and len(base) >= 2 else None

def init_mediapipe():
    """Initialize MediaPipe HandLandmarker ONCE."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    model_path = None
    for p in [
        os.path.join(BASE_DIR, "models", "hand_landmarker.task"),
        os.path.join(BASE_DIR, "hand_landmarker.task"),
    ]:
        if os.path.exists(p):
            model_path = p
            break

    if not model_path:
        # Download if missing
        import urllib.request
        model_path = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        print("Downloading hand_landmarker.task...", flush=True)
        urllib.request.urlretrieve(url, model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)

def extract_from_video(detector, video_path):
    """Extract landmarks from FRAMES_TO_SAMPLE evenly-spaced frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 3:
        cap.release()
        return []

    # Pick evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)

    import mediapipe as mp
    features = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = detector.detect(mp_image)
        except:
            continue

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            feat = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            # Normalize to wrist
            wrist = feat[0].copy()
            feat = feat - wrist
            features.append(feat.flatten())

    cap.release()
    return features

def make_sequence(features):
    """Pad/tile features to create a single 30x63 sequence."""
    n = len(features)
    if n < 3:
        return None

    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    arr = np.array(features, dtype=np.float32)

    if n >= SEQUENCE_LENGTH:
        # Downsample to 30
        indices = np.linspace(0, n - 1, SEQUENCE_LENGTH, dtype=int)
        seq = arr[indices]
    else:
        # Place features and interpolate/repeat
        seq[:n] = arr
        for i in range(n, SEQUENCE_LENGTH):
            seq[i] = arr[i % n]  # Cycle through

    return seq

def augment_sequence(seq):
    """Create augmented versions of a sequence."""
    augmented = []
    # Original
    augmented.append(seq.copy())
    # Noise
    noise = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)
    augmented.append(noise)
    # Scale
    scale = seq * np.random.uniform(0.9, 1.1)
    augmented.append(scale.astype(np.float32))
    # Time shift (roll)
    shift = np.roll(seq, np.random.randint(1, 5), axis=0)
    augmented.append(shift)
    return augmented

def main():
    print("=" * 60)
    print("ULTRA-FAST ISL Kaggle Video Processor")
    print("=" * 60, flush=True)

    mp4_files = sorted(glob.glob(os.path.join(KAGGLE_DIR, "*.mp4")))
    mp4_files = [f for f in mp4_files if os.path.getsize(f) < 5_000_000]
    print(f"Videos to process: {len(mp4_files)}", flush=True)

    # Group by label
    label_files = {}
    for f in mp4_files:
        label = clean_label(os.path.basename(f))
        if label:
            label_files.setdefault(label, []).append(f)

    print(f"Unique gesture labels: {len(label_files)}", flush=True)

    # Init MediaPipe ONCE
    print("Initializing MediaPipe HandLandmarker...", flush=True)
    detector = init_mediapipe()
    print("MediaPipe ready!", flush=True)

    total_saved = 0
    processed_videos = 0
    start = time.time()

    for label, files in sorted(label_files.items()):
        label_dir = os.path.join(DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        existing = len(glob.glob(os.path.join(label_dir, "kaggle_*.npy")))

        saved_for_label = 0
        for fpath in files[:3]:  # Max 3 videos per label
            features = extract_from_video(detector, fpath)
            seq = make_sequence(features)
            processed_videos += 1

            if seq is not None:
                aug_seqs = augment_sequence(seq)
                for i, s in enumerate(aug_seqs):
                    fname = f"kaggle_{existing + saved_for_label + 1}.npy"
                    np.save(os.path.join(label_dir, fname), s)
                    saved_for_label += 1

        total_saved += saved_for_label
        elapsed = time.time() - start
        rate = processed_videos / max(elapsed, 0.1)
        remaining = len(mp4_files) - processed_videos
        eta_sec = remaining / max(rate, 0.01)

        if processed_videos % 20 == 0 or saved_for_label > 0:
            print(f"  [{processed_videos:4d}/{len(mp4_files)}] {label:30s} | +{saved_for_label:2d} seqs | Total: {total_saved:5d} | ETA: {eta_sec/60:.1f}m", flush=True)

    detector.close()

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE in {(time.time()-start)/60:.1f} min")
    print(f"  Sequences saved: {total_saved}")
    print(f"  Gesture classes: {len(label_files)}")
    print(f"{'=' * 60}\n", flush=True)

    # RETRAIN
    if total_saved > 0:
        retrain()

def retrain():
    from src.train import load_dataset
    from src.model import ISLGestureClassifier
    from sklearn.model_selection import train_test_split
    from collections import Counter

    print("Loading full dataset for retraining...", flush=True)
    X, y, class_names = load_dataset()
    print(f"Raw: {len(X)} sequences, {len(class_names)} classes", flush=True)

    # Filter classes with <3 samples
    counts = Counter(y)
    valid = {c for c, n in counts.items() if n >= 3}
    mask = [i for i, yi in enumerate(y) if yi in valid]
    X, y = X[mask], y[mask]

    remap = {}
    new_names = []
    for old in sorted(valid):
        remap[old] = len(new_names)
        new_names.append(class_names[old])
    y = np.array([remap[yi] for yi in y])

    print(f"After filter: {len(X)} sequences, {len(new_names)} classes", flush=True)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    classifier = ISLGestureClassifier(labels_list=new_names)
    classifier.train_model(X_tr, y_tr, X_val, y_val, epochs=25, batch_size=64, lr=0.001)

    correct = sum(1 for i in range(len(X_val))
                  if classifier.label_to_idx.get(classifier.predict(X_val[i])[0], -1) == y_val[i])
    acc = correct / max(len(X_val), 1) * 100

    print(f"\n{'=' * 60}")
    print(f"MODEL RETRAINED SUCCESSFULLY")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Classes: {len(new_names)}")
    print(f"  Model: {classifier.model_path}")
    print(f"{'=' * 60}", flush=True)

if __name__ == "__main__":
    main()
