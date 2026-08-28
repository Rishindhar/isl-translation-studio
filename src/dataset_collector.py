import os
import time
import cv2
import numpy as np
from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM
from src.feature_extractor import HandFeatureExtractor

def collect_gesture_samples(gesture_name, num_samples=15, camera_id=0):
    """
    Interactive live webcam dataset collection for custom ISL gestures.
    
    :param gesture_name: String label for the gesture
    :param num_samples: Number of 30-frame sequence repetitions to collect
    :param camera_id: Webcam device index
    """
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    extractor = HandFeatureExtractor(max_num_hands=1)
    print(f"\n=======================================================")
    print(f" Recording Custom Gesture: '{gesture_name}'")
    print(f" Collecting {num_samples} repetitions of {SEQUENCE_LENGTH} frames each.")
    print(f" Press SPACEBAR to start recording, or 'q' to cancel.")
    print(f"=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        _, annotated_frame, _ = extractor.process_frame(frame)

        cv2.putText(annotated_frame, f"Gesture Target: {gesture_name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "Press SPACE to Begin Recording", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("ISL Gesture Collector", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            extractor.close()
            return

    # Record sequences
    for sample_idx in range(num_samples):
        # 3-Second Countdown prompt before each repetition
        for countdown in range(3, 0, -1):
            t_start = time.time()
            while time.time() - t_start < 1.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                _, annotated_frame, _ = extractor.process_frame(frame)

                cv2.putText(annotated_frame, f"Repetition {sample_idx + 1} of {num_samples}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Get Ready! {countdown}", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                cv2.imshow("ISL Gesture Collector", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    extractor.close()
                    return

        # Record 30 continuous frames
        sequence = []
        recorded_frames = 0

        while recorded_frames < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            feat, annotated_frame, detected = extractor.process_frame(frame)

            sequence.append(feat)
            recorded_frames += 1

            cv2.putText(annotated_frame, f"RECORDING... Frame {recorded_frames}/{SEQUENCE_LENGTH}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.circle(annotated_frame, (600, 40), 15, (0, 0, 255), -1)
            cv2.imshow("ISL Gesture Collector", annotated_frame)
            cv2.waitKey(1)

        # Save numpy sequence array
        seq_array = np.array(sequence, dtype=np.float32)
        sample_path = os.path.join(gesture_dir, f"sample_{int(time.time()*1000)}.npy")
        np.save(sample_path, seq_array)
        print(f"Saved repetition {sample_idx + 1}/{num_samples} -> {sample_path}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\nCompleted recording {num_samples} samples for '{gesture_name}'.")

if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "custom_gesture"
    collect_gesture_samples(name, num_samples=15)
