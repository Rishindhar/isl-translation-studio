import os
import sys
import collections
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import SEQUENCE_LENGTH, CONFIDENCE_THRESHOLD, DEBOUNCE_FRAMES, IDLE_VELOCITY_THRESHOLD
from src.feature_extractor import HandFeatureExtractor
from src.model import ISLGestureClassifier
from src.tts_engine import MultilingualTTSEngine
from src.utils import calculate_hand_velocity

def process_isl_video(input_video_path, output_video_path=None, lang="en"):
    """
    Processes an offline ISL video file (.mp4), extracts hand landmarks,
    predicts dynamic gestures via GRU model, draws subtitles, and exports annotated video.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file '{input_video_path}' not found.")
        return

    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = f"{base}_translated{ext}"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_video_path}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    extractor = HandFeatureExtractor(max_num_hands=1)
    classifier = ISLGestureClassifier()
    tts = MultilingualTTSEngine(default_lang=lang)

    sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    recent_preds = collections.deque(maxlen=DEBOUNCE_FRAMES)
    prev_feat = None

    current_prediction = "idle"
    current_confidence = 0.0
    sentence_history = []
    last_spoken_gesture = None

    frame_idx = 0
    print(f"\nProcessing Video: '{input_video_path}' ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        feat, annotated_frame, detected = extractor.process_frame(frame)

        if detected:
            velocity = calculate_hand_velocity(prev_feat, feat)
            prev_feat = feat.copy()
            sequence_buffer.append(feat)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                if velocity < IDLE_VELOCITY_THRESHOLD:
                    pred_label, conf = "idle", 1.0
                else:
                    seq_arr = np.array(sequence_buffer, dtype=np.float32)
                    pred_label, conf, _ = classifier.predict(seq_arr)

                current_prediction = pred_label
                current_confidence = conf
                recent_preds.append(pred_label)

                if len(recent_preds) == DEBOUNCE_FRAMES:
                    if len(set(recent_preds)) == 1 and conf >= CONFIDENCE_THRESHOLD:
                        stable = recent_preds[0]
                        if stable != "idle" and stable != last_spoken_gesture:
                            last_spoken_gesture = stable
                            translated = tts.get_translated_text(stable, lang)
                            sentence_history.append(translated)

        # Draw Subtitle HUD Banner
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, height - 70), (width, height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)

        sub_text = f"ISL Gesture: {current_prediction.upper()} ({current_confidence*100:.1f}%)"
        cv2.putText(annotated_frame, sub_text, (20, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        sent_text = "Translation: " + " ".join(sentence_history) if sentence_history else "Translation: ..."
        cv2.putText(annotated_frame, sent_text, (20, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out_writer.write(annotated_frame)

        if frame_idx % 30 == 0:
            print(f" Frame {frame_idx}/{total_frames} processed ({(frame_idx/max(total_frames, 1))*100:.1f}%)...")

    cap.release()
    out_writer.release()
    extractor.close()

    print(f"\nProcessing complete! Annotated video exported -> '{output_video_path}'")
    print(f"Recognized Sequence: {' '.join(sentence_history)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
        process_isl_video(in_path, out_path)
    else:
        print("Usage: python process_video.py <input_video.mp4> [output_video.mp4]")
