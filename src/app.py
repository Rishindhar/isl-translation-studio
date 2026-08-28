import sys
import os
import time
import collections
import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SEQUENCE_LENGTH, FEATURE_DIM, CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES, SUPPORTED_LANGUAGES, IDLE_VELOCITY_THRESHOLD
)
from src.feature_extractor import HandFeatureExtractor
from src.model import ISLGestureClassifier
from src.tts_engine import MultilingualTTSEngine
from src.utils import calculate_hand_velocity
from src.dataset_collector import collect_gesture_samples
from src.train import train_and_evaluate

class ISLRealTimeApp:
    """
    Main Real-Time Indian Sign Language Recognition Application.
    Integrates MediaPipe tracking, rolling 30-frame FIFO buffer, GRU model inference,
    5-frame debouncing, idle state optimization, and multilingual TTS speech generation.
    """
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.extractor = HandFeatureExtractor(max_num_hands=1)
        self.classifier = ISLGestureClassifier()
        self.tts = MultilingualTTSEngine(default_lang="en")

        self.sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.prev_frame_feat = None

        # Prediction & Debouncing State
        self.current_prediction = "idle"
        self.current_confidence = 0.0
        self.recent_predictions = collections.deque(maxlen=DEBOUNCE_FRAMES)
        self.last_spoken_gesture = None
        self.sentence_history = []
        self.last_speech_time = 0

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Unable to access webcam camera ID {self.camera_id}.")
            return

        print("\n=======================================================")
        print("  AI-Based Real-Time ISL Recognition System Active")
        print("-------------------------------------------------------")
        print(" Controls:")
        print("  'q' - Quit application")
        print("  'l' - Toggle Speech Language (English / Hindi)")
        print("  'c' - Capture & Record New Custom Gesture")
        print("  't' - Retrain Model with Transfer Learning")
        print("  'r' - Reset Sentence Buffer")
        print("=======================================================\n")

        fps_start = time.time()
        frame_count = 0
        fps = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            frame = cv2.flip(frame, 1)
            feat, annotated_frame, hand_detected = self.extractor.process_frame(frame)

            if hand_detected:
                # Check idle velocity optimization
                velocity = calculate_hand_velocity(self.prev_frame_feat, feat)
                self.prev_frame_feat = feat.copy()
                self.sequence_buffer.append(feat)

                # Run GRU inference if sequence buffer is full (30 frames) and velocity > threshold
                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    if velocity < IDLE_VELOCITY_THRESHOLD:
                        pred_label, conf = "idle", 1.0
                    else:
                        seq_arr = np.array(self.sequence_buffer, dtype=np.float32)
                        pred_label, conf, _ = self.classifier.predict(seq_arr)

                    self.current_prediction = pred_label
                    self.current_confidence = conf
                    self.recent_predictions.append(pred_label)

                    # 5-Frame Debouncing Logic & Confidence Thresholding
                    if len(self.recent_predictions) == DEBOUNCE_FRAMES:
                        # Check if all recent predictions match and exceed confidence threshold
                        if len(set(self.recent_predictions)) == 1 and conf >= CONFIDENCE_THRESHOLD:
                            stable_gesture = self.recent_predictions[0]

                            if stable_gesture != "idle" and stable_gesture != self.last_spoken_gesture:
                                now = time.time()
                                if now - self.last_speech_time > 2.0:
                                    self.last_spoken_gesture = stable_gesture
                                    self.last_speech_time = now

                                    # Append to continuous sentence formulation
                                    translated = self.tts.get_translated_text(stable_gesture)
                                    self.sentence_history.append(translated)
                                    if len(self.sentence_history) > 6:
                                        self.sentence_history.pop(0)

                                    # Trigger asynchronous TTS audio output
                                    self.tts.speak(translated)
            else:
                self.sequence_buffer.clear()
                self.recent_predictions.clear()
                self.current_prediction = "idle"
                self.current_confidence = 0.0

            # Render HUD Overlay on Live OpenCV Frame
            self._render_hud(annotated_frame, fps)

            cv2.imshow("ISL Real-Time Recognition & Speech System", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                new_lang = "hi" if self.tts.current_lang == "en" else "en"
                self.tts.set_language(new_lang)
                print(f"Switched Audio Language to: {SUPPORTED_LANGUAGES[new_lang]}")
            elif key == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                print("\nOpening Custom Gesture Collector...")
                g_name = input("Enter name for new gesture: ").strip()
                if g_name:
                    collect_gesture_samples(g_name, num_samples=15, camera_id=self.camera_id)
                cap = cv2.VideoCapture(self.camera_id)
            elif key == ord('t'):
                cap.release()
                cv2.destroyAllWindows()
                print("\nStarting Model Retraining / Fine-Tuning...")
                train_and_evaluate()
                self.classifier.load_model()
                cap = cv2.VideoCapture(self.camera_id)
            elif key == ord('r'):
                self.sentence_history.clear()
                self.last_spoken_gesture = None
                print("Reset sentence buffer.")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _render_hud(self, frame, fps):
        """Draws semi-transparent HUD banner, predictions, confidence meter, and language indicator."""
        h, w, _ = frame.shape

        # Top Overlay Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Active Gesture Prediction & Confidence Meter
        pred_text = f"Gesture: {self.current_prediction.upper()}"
        conf_text = f"Conf: {self.current_confidence * 100:.1f}%"
        cv2.putText(frame, pred_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Draw Confidence Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, 55, 300, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * min(self.current_confidence, 1.0))
        color = (0, 255, 0) if self.current_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # Multilingual Indicator & FPS
        lang_str = f"Lang: {SUPPORTED_LANGUAGES[self.tts.current_lang]} (Press 'L')"
        cv2.putText(frame, lang_str, (w - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 100), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Bottom Sentence Buffer Display
        sentence_str = "Sentence: " + " ".join(self.sentence_history) if self.sentence_history else "Sentence: [Waiting for gestures...]"
        cv2.putText(frame, sentence_str, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

if __name__ == "__main__":
    app = ISLRealTimeApp()
    app.run()
