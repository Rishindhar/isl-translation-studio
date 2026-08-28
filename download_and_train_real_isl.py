"""
ISL Kaggle Dataset Downloader, Converter & Model Retrainer
===========================================================
Run this script AFTER placing kaggle.json in C:\\Users\\rishi\\.kaggle\\

Usage:
    python download_and_train_real_isl.py

What it does:
    1. Downloads real ISL hand landmark CSV data from Kaggle
    2. Converts CSV rows -> 30-frame sequence .npy matrices
    3. Retrains the BiGRU model on real ISL data
    4. Saves updated model to models/
"""

import os
import sys
import json
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM, BASE_DIR
from src.model import ISLGestureClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

KAGGLE_JSON_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
KAGGLE_DATA_DIR = os.path.join(BASE_DIR, "kaggle_isl_data")

# Datasets to download (in order of preference)
KAGGLE_DATASETS = [
    {
        "dataset_id": "aakashnain/indian-sign-language-hand-landmarks",
        "description": "ISL Hand Landmarks A-Z (50k records, CSV format)",
    },
    {
        "dataset_id": "dwibonbhargabdeka/indian-sign-language-dataset-mediapipe-landmarks",
        "description": "ISL MediaPipe Landmarks A-Z + Words (CSV format)",
    }
]

def check_kaggle_auth():
    """Verifies Kaggle API credentials are present."""
    if not os.path.exists(KAGGLE_JSON_PATH):
        print("=" * 60)
        print("KAGGLE API TOKEN NOT FOUND")
        print("=" * 60)
        print()
        print("To download real ISL data, you need a Kaggle API token.")
        print()
        print("Steps:")
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New API Token'")
        print("  4. Save the downloaded kaggle.json to:")
        print(f"     {KAGGLE_JSON_PATH}")
        print("  5. Run this script again.")
        print()
        print("Alternatively, manually download ISL datasets and place in:")
        print(f"     {KAGGLE_DATA_DIR}/")
        print("  Then run: python download_and_train_real_isl.py --manual")
        return False
    
    with open(KAGGLE_JSON_PATH, "r") as f:
        creds = json.load(f)
    
    username = creds.get("username", "")
    key = creds.get("key", "")
    
    if not username or not key:
        print("ERROR: kaggle.json is malformed or empty.")
        return False
    
    print(f"Kaggle credentials found for user: {username}")
    return True

def download_kaggle_datasets():
    """Downloads ISL datasets from Kaggle using the official kaggle library."""
    try:
        import kaggle
        kaggle.api.authenticate()
        print("Kaggle API authenticated successfully.")
    except ImportError:
        print("Installing kaggle library...")
        os.system(f'"{sys.executable}" -m pip install kaggle -q')
        import kaggle
        kaggle.api.authenticate()

    os.makedirs(KAGGLE_DATA_DIR, exist_ok=True)

    for ds_info in KAGGLE_DATASETS:
        ds_id = ds_info["dataset_id"]
        print(f"\nDownloading: {ds_info['description']}")
        print(f"  Dataset: {ds_id}")
        
        try:
            kaggle.api.dataset_download_files(
                ds_id,
                path=KAGGLE_DATA_DIR,
                unzip=True,
                quiet=False
            )
            print(f"  Download complete!")
        except Exception as e:
            print(f"  Warning: Failed to download {ds_id}: {e}")

def find_csv_files():
    """Finds all CSV files in the Kaggle data directory."""
    csv_files = (
        glob.glob(os.path.join(KAGGLE_DATA_DIR, "**", "*.csv"), recursive=True) +
        glob.glob(os.path.join(KAGGLE_DATA_DIR, "*.csv"))
    )
    return csv_files

def convert_csv_landmarks_to_npy(csv_path, label_override=None):
    """
    Converts a CSV file of MediaPipe hand landmark coordinates into
    30-frame sequence .npy files for model training.
    
    Expected CSV format:
    - Each row = 1 frame with 63 columns (x0,y0,z0,...x20,y20,z20) + label column
    - OR: label, x0, y0, z0, ... x20, y20, z20
    """
    try:
        import pandas as pd
    except ImportError:
        os.system(f'"{sys.executable}" -m pip install pandas -q')
        import pandas as pd

    print(f"\nProcessing CSV: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return 0

    print(f"  Shape: {df.shape}, Columns: {list(df.columns[:5])}...")

    # Detect label column
    label_col = None
    for possible_col in ["label", "class", "gesture", "sign", "category", "word", "Letter", "Class"]:
        if possible_col in df.columns:
            label_col = possible_col
            break
    
    if label_col is None and not label_override:
        # Try first column
        if df.dtypes.iloc[0] == object:
            label_col = df.columns[0]
        else:
            print(f"  Warning: No label column detected. Skipping.")
            return 0

    # Get feature columns (63 numeric columns = 21 landmarks x 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 63:
        print(f"  Warning: Only {len(numeric_cols)} numeric columns, need 63. Skipping.")
        return 0

    # Use first 63 numeric columns as XYZ landmarks
    feature_cols = numeric_cols[:63]
    
    saved_total = 0

    if label_col:
        unique_labels = df[label_col].unique()
        print(f"  Found {len(unique_labels)} unique gesture classes")

        for label in unique_labels:
            label_str = str(label).lower().strip().replace(" ", "_")
            label_df = df[df[label_col] == label]
            
            # Get landmark rows for this label
            rows = label_df[feature_cols].values.astype(np.float32)
            
            if len(rows) < 5:
                continue

            # Create output directory
            gesture_dir = os.path.join(DATA_DIR, label_str)
            os.makedirs(gesture_dir, exist_ok=True)

            # Slide window to create 30-frame sequence samples
            existing = len(glob.glob(os.path.join(gesture_dir, "kaggle_*.npy")))
            count = 0
            
            # For static sign data (each row is a single frame of that sign),
            # repeat each row 30 times to create a "held" gesture sequence
            if len(rows) >= 1:
                n_samples = min(len(rows), 80)  # Max 80 samples per class
                step = max(1, len(rows) // n_samples)
                
                for i in range(0, min(len(rows), n_samples * step), step):
                    row = rows[i]
                    # Create 30-frame sequence by repeating with small noise
                    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
                    for t in range(SEQUENCE_LENGTH):
                        noise = np.random.normal(0, 0.002, FEATURE_DIM).astype(np.float32)
                        seq[t] = row + noise
                    
                    filename = f"kaggle_{existing + count + 1}.npy"
                    np.save(os.path.join(gesture_dir, filename), seq)
                    count += 1

            saved_total += count
            if count > 0:
                print(f"    [{label_str}]: {count} sequences saved", flush=True)
    
    elif label_override:
        rows = df[feature_cols].values.astype(np.float32)
        gesture_dir = os.path.join(DATA_DIR, label_override)
        os.makedirs(gesture_dir, exist_ok=True)
        existing = len(glob.glob(os.path.join(gesture_dir, "kaggle_*.npy")))
        
        for i in range(min(len(rows), 80)):
            row = rows[i]
            seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            for t in range(SEQUENCE_LENGTH):
                noise = np.random.normal(0, 0.002, FEATURE_DIM).astype(np.float32)
                seq[t] = row + noise
            np.save(os.path.join(gesture_dir, f"kaggle_{existing + i + 1}.npy"), seq)
            saved_total += 1

    return saved_total

def retrain_with_kaggle_data():
    """Retrains the BiGRU model with all available data including Kaggle datasets."""
    from src.train import load_dataset
    from sklearn.model_selection import train_test_split
    
    print("\nLoading full dataset (existing + Kaggle)...")
    X_data, y_data, class_names = load_dataset()
    print(f"Total: {len(X_data)} sequences, {len(class_names)} gesture classes")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.15, random_state=42, stratify=y_data
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"\nStarting BiGRU retraining on {len(class_names)} classes...")
    
    classifier = ISLGestureClassifier(labels_list=class_names)
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=30, batch_size=64, lr=0.001)
    
    # Evaluate
    val_preds = []
    for i in range(len(X_val)):
        label, conf, _ = classifier.predict(X_val[i])
        val_preds.append(classifier.label_to_idx.get(label, 0))
    
    report = classification_report(y_val, val_preds, target_names=class_names, zero_division=0)
    print("\n=== FINAL VALIDATION REPORT ===")
    print(report)
    print(f"\nModel saved: {classifier.model_path}")
    print(f"Labels saved: {classifier.labels_path}")
    print(f"Total vocabulary: {len(class_names)} ISL gesture classes")

def process_manual_csvs():
    """Processes manually downloaded CSV files placed in kaggle_isl_data/."""
    csv_files = find_csv_files()
    
    if not csv_files:
        print(f"No CSV files found in: {KAGGLE_DATA_DIR}")
        print("Please download ISL datasets from Kaggle and extract to that folder.")
        return 0
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    total_saved = 0
    
    for csv_path in csv_files:
        saved = convert_csv_landmarks_to_npy(csv_path)
        total_saved += saved
    
    print(f"\nTotal: {total_saved} new training sequences added to data/")
    return total_saved

def main():
    manual_mode = "--manual" in sys.argv
    
    print("=" * 60)
    print("ISL Real Kaggle Data Downloader & Model Retrainer")
    print("=" * 60)
    
    if manual_mode:
        print("\nManual mode: processing existing CSVs in kaggle_isl_data/")
        total = process_manual_csvs()
        if total > 0:
            retrain_with_kaggle_data()
    else:
        if not check_kaggle_auth():
            sys.exit(1)
        
        download_kaggle_datasets()
        total = process_manual_csvs()
        
        if total > 0:
            retrain_with_kaggle_data()
        else:
            print("\nNo new data found to train on.")

if __name__ == "__main__":
    main()
