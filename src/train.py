import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM
from src.model import ISLGestureClassifier

def load_dataset():
    """
    Scans data/ directory for gesture class folders and loads .npy sequence files.
    
    :return: Tuple (X_data array [N, 30, 63], y_labels integer list, class_names list)
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist.")

    gesture_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    gesture_folders.sort()

    if not gesture_folders:
        raise ValueError(f"No gesture folders found in {DATA_DIR}.")

    X_list = []
    y_list = []
    class_names = []

    for idx, folder in enumerate(gesture_folders):
        class_names.append(folder)
        folder_path = os.path.join(DATA_DIR, folder)
        npy_files = glob.glob(os.path.join(folder_path, "*.npy"))

        # If a class has only 1 sample, duplicate it to satisfy train_test_split stratification
        if len(npy_files) == 1:
            npy_files = npy_files * 2

        for file_path in npy_files:
            arr = np.load(file_path)
            if arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                X_list.append(arr)
                y_list.append(idx)

    if not X_list:
        raise ValueError("No valid .npy files found in dataset folders.")

    X_data = np.array(X_list, dtype=np.float32)
    y_data = np.array(y_list, dtype=np.int64)

    return X_data, y_data, class_names

def train_and_evaluate():
    """Loads dataset, trains GRU model, prints classification metrics, and saves model checkpoint."""
    print("Loading ISL dataset from data/...", flush=True)
    X_data, y_data, class_names = load_dataset()
    print(f"Loaded {len(X_data)} sequences across {len(class_names)} gesture classes.", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    classifier = ISLGestureClassifier(labels_list=class_names)
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=15, batch_size=32, lr=0.001)

    # Evaluate validation predictions
    val_preds = []
    for i in range(len(X_val)):
        label, conf, _ = classifier.predict(X_val[i])
        val_preds.append(classifier.label_to_idx.get(label, 0))

    val_preds = np.array(val_preds)
    val_accuracy = np.mean(val_preds == y_val) * 100
    print(f"\nValidation Accuracy: {val_accuracy:.2f}%", flush=True)

    try:
        report = classification_report(y_val, val_preds, target_names=class_names, labels=np.unique(y_val))
        print("\nTraining & Validation Summary")
        print("===============================")
        print(report, flush=True)
    except Exception as e:
        print(f"\nCould not generate full classification report: {e}")
        
    print(f"\nModel successfully saved to {classifier.model_path}", flush=True)
    print(f"Labels saved to {classifier.labels_path}", flush=True)

if __name__ == "__main__":
    train_and_evaluate()
