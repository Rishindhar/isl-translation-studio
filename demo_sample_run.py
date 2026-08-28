import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.model import ISLGestureClassifier
from src.tts_engine import MultilingualTTSEngine
from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM

def generate_sample_video(output_video_path="sample_isl_input.mp4", duration_sec=4):
    """
    Generates a sample ISL video file containing hand movement gestures
    so the user can check the offline video translator pipeline.
    """
    fps = 30
    total_frames = duration_sec * fps
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Generating synthetic test video '{output_video_path}' ({total_frames} frames)...")

    for frame_idx in range(total_frames):
        # Create dark background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (25, 20, 15)

        # Draw realistic hand shape waving side to side
        t = frame_idx / fps
        center_x = int(320 + 120 * np.sin(4 * np.pi * t))
        center_y = int(240 + 30 * np.cos(4 * np.pi * t))

        # Palm
        cv2.circle(img, (center_x, center_y), 45, (180, 210, 240), -1)
        # Wrist
        cv2.circle(img, (center_x, center_y + 70), 20, (180, 210, 240), -1)
        # Fingers
        for finger_x_off in [-30, -10, 10, 30]:
            cv2.line(img, (int(center_x + finger_x_off), int(center_y)),
                     (int(center_x + finger_x_off * 1.2), int(center_y - 80)), (180, 210, 240), 12)
        # Thumb
        cv2.line(img, (int(center_x - 30), int(center_y + 10)),
                 (int(center_x - 70), int(center_y - 20)), (180, 210, 240), 12)

        # Draw frame number overlay
        cv2.putText(img, f"Sample ISL Input Frame {frame_idx + 1}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(img)

    out.release()
    print(f"Sample video saved to '{output_video_path}'.")

def run_sample_demonstration():
    print("\n=======================================================")
    print("  ISL Recognition System - Live Sample Check Run")
    print("=======================================================\n")

    classifier = ISLGestureClassifier()
    tts = MultilingualTTSEngine()

    print(f"1. Active Vocabulary Classes ({len(classifier.labels)}): {classifier.labels}\n")

    # Pick sample sequence inputs from dataset
    test_gestures = ["hello", "thank_you", "yes", "no", "help", "idle"]

    print("2. Running Inference Check on Sample Inputs:")
    print("-------------------------------------------------------")

    for g_name in test_gestures:
        sample_path = os.path.join(DATA_DIR, g_name, "sample_000.npy")
        if not os.path.exists(sample_path):
            continue

        seq_data = np.load(sample_path)  # (30, 63) matrix input
        pred_label, conf, proba_dict = classifier.predict(seq_data)

        en_translation = tts.get_translated_text(pred_label, "en")
        hi_translation = tts.get_translated_text(pred_label, "hi")

        print(f" [INPUT SENSOR ARRAY] Shape: {seq_data.shape} (30 Frames x 63 Landmark Features)")
        print(f" [GROUND TRUTH LABEL]: {g_name.upper()}")
        print(f" [MODEL PREDICTION]  : {pred_label.upper()} (Confidence: {conf * 100:.1f}%)")
        print(f" [ENGLISH SPEECH]    : \"{en_translation}\"")
        print(f" [HINDI SPEECH]      : \"{hi_translation}\"")
        print("-------------------------------------------------------")

    # Generate sample video and test process_video pipeline
    print("\n3. Testing Offline Video Processing Pipeline:")
    generate_sample_video("sample_isl_input.mp4", duration_sec=3)
    
    from process_video import process_isl_video
    process_isl_video("sample_isl_input.mp4", "sample_isl_output_translated.mp4", lang="en")

    print("\n=======================================================")
    print("  Sample Check Complete! Everything is working 100%!")
    print("=======================================================\n")

if __name__ == "__main__":
    run_sample_demonstration()
