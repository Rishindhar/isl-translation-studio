import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.model import ISLGestureClassifier
from src.tts_engine import MultilingualTTSEngine
from src.config import DATA_DIR, GESTURE_TRANSLATIONS

def run_system_verification():
    print("\n=======================================================")
    print("  ISL System Verification & Model Evaluation Test")
    print("=======================================================\n")

    classifier = ISLGestureClassifier()
    tts = MultilingualTTSEngine()

    print(f"Loaded Gesture Vocabulary ({len(classifier.labels)} classes): {classifier.labels}\n")

    # Test prediction on each saved sample
    correct = 0
    total = 0

    for gesture in classifier.labels:
        gesture_dir = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(gesture_dir):
            continue

        sample_file = os.path.join(gesture_dir, "sample_000.npy")
        if not os.path.exists(sample_file):
            continue

        seq_arr = np.load(sample_file)
        pred, conf, probs = classifier.predict(seq_arr)
        
        en_text = tts.get_translated_text(pred, "en")
        hi_text = tts.get_translated_text(pred, "hi")

        match_status = "PASS" if pred == gesture else "FAIL"
        if pred == gesture:
            correct += 1
        total += 1

        print(f"Class: {gesture:<12} | Pred: {pred:<12} | Conf: {conf*100:5.1f}% | EN: '{en_text}' | HI: '{hi_text}' -> [{match_status}]")

    accuracy = (correct / max(total, 1)) * 100
    print(f"\nSystem Test Verification Accuracy: {accuracy:.1f}% ({correct}/{total} passed)\n")
    return accuracy == 100.0

if __name__ == "__main__":
    success = run_system_verification()
    sys.exit(0 if success else 1)
