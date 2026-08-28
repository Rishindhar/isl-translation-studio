import os
import sys
import glob
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASE_DIR, DATA_DIR, UPLOADS_DIR, SEQUENCE_LENGTH
from src.feature_extractor import HandFeatureExtractor

def extract_landmarks_from_video(video_path, label_name=None):
    """
    Reads an ISL video file, extracts MediaPipe 21-joint 3D landmark features,
    segments into 30-frame sequence matrices, and saves .npy files to data/<label_name>/.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return 0

    if label_name is None:
        filename = os.path.basename(video_path)
        base = os.path.splitext(filename)[0]
        # Clean label name
        label_name = base.lower().replace("whatsapp_video_", "").replace("sample_", "").replace("isl_", "").strip()
        if not label_name:
            label_name = "unknown_sign"

    target_dir = os.path.join(DATA_DIR, label_name)
    os.makedirs(target_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    extractor = HandFeatureExtractor(max_num_hands=1)
    
    extracted_features = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        feat, _, detected = extractor.process_frame(frame)
        if detected:
            extracted_features.append(feat)
        else:
            # If hand not detected in frame, pad with zeros or last valid feature
            if len(extracted_features) > 0:
                extracted_features.append(extracted_features[-1].copy())
            else:
                extracted_features.append(np.zeros(63, dtype=np.float32))

    cap.release()
    total_frames = len(extracted_features)
    
    if total_frames < 5:
        print(f"Warning: Insufficient hand frames detected in '{video_path}'.")
        return 0

    # Segment into sliding window 30-frame sequences
    sequences = []
    step = 5  # Sliding window step
    
    if total_frames <= SEQUENCE_LENGTH:
        # Pad sequence to 30 frames
        seq = np.zeros((SEQUENCE_LENGTH, 63), dtype=np.float32)
        seq[:total_frames] = np.array(extracted_features[:total_frames])
        for i in range(total_frames, SEQUENCE_LENGTH):
            seq[i] = extracted_features[-1]
        sequences.append(seq)
    else:
        for start_idx in range(0, total_frames - SEQUENCE_LENGTH + 1, step):
            seq = np.array(extracted_features[start_idx : start_idx + SEQUENCE_LENGTH], dtype=np.float32)
            sequences.append(seq)

    # Save sequence .npy files
    existing_count = len(glob.glob(os.path.join(target_dir, "*.npy")))
    saved_count = 0
    for idx, seq in enumerate(sequences):
        seq_filename = f"sample_video_{existing_count + idx + 1}.npy"
        np.save(os.path.join(target_dir, seq_filename), seq)
        saved_count += 1

    print(f"SUCCESS: Extracted {saved_count} sequence samples for label '{label_name}' from '{os.path.basename(video_path)}'.")
    return saved_count

def process_all_uploaded_videos():
    """Scans uploads/ directory and extracts training sequences for all videos."""
    video_files = glob.glob(os.path.join(UPLOADS_DIR, "*.mp4")) + glob.glob(os.path.join(BASE_DIR, "*.mp4"))
    total_extracted = 0
    
    for vid in video_files:
        filename = os.path.basename(vid)
        # Skip output/translated files
        if "translated" in filename or "out.mp4" in filename:
            continue
        
        # Derive label name
        if "loud_speaker" in filename.lower():
            label = "loud_speaker"
        elif "whatsapp" in filename.lower():
            label = "hello"  # Or custom gesture from user video
        else:
            label = os.path.splitext(filename)[0].lower()
            
        count = extract_landmarks_from_video(vid, label_name=label)
        total_extracted += count
        
    print(f"Completed processing videos. Total extracted: {total_extracted} samples.")
    return total_extracted

if __name__ == "__main__":
    process_all_uploaded_videos()
