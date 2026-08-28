"""
Multi-Word Combination Gesture Dataset Generator & Retrainer
============================================================
Generates landmark sequences for multi-word gesture combinations
and fine-tunes the BiGRU model for continuous multi-word ISL sentence recognition.
"""

import os
import sys
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, SEQUENCE_LENGTH, FEATURE_DIM
from src.sentence_translator import ISL_SENTENCE_DICTIONARY

def generate_multi_word_sequences(samples_per_combo=25):
    """
    Creates multi-gesture sequence matrices by concatenating single-gesture 
    landmark patterns from existing data/ subdirectories.
    """
    total_generated = 0
    print("Generating multi-word gesture combination sequences...", flush=True)

    for combo_tuple in ISL_SENTENCE_DICTIONARY.keys():
        if len(combo_tuple) < 2:
            continue

        combo_name = "_".join(combo_tuple)
        combo_dir = os.path.join(DATA_DIR, combo_name)
        os.makedirs(combo_dir, exist_ok=True)

        # Load available single gesture sequence files for each word in the combo
        word_sequences = []
        valid_combo = True

        for word in combo_tuple:
            word_dir = os.path.join(DATA_DIR, word)
            npy_files = glob.glob(os.path.join(word_dir, "*.npy"))
            if not npy_files:
                valid_combo = False
                break
            # Load up to 10 sequences for this word
            arrs = [np.load(f) for f in npy_files[:10] if np.load(f).shape == (SEQUENCE_LENGTH, FEATURE_DIM)]
            if not arrs:
                valid_combo = False
                break
            word_sequences.append(arrs)

        if not valid_combo:
            continue

        # Generate combined sequences
        for s in range(samples_per_combo):
            combined_frames = []
            for word_arrs in word_sequences:
                # Pick a random sequence for this word
                seq = word_arrs[np.random.randint(0, len(word_arrs))]
                # Take 10-15 key frames per word to form smooth transition
                frames_take = np.linspace(0, len(seq) - 1, 10, dtype=int)
                combined_frames.extend(seq[frames_take])

            # Convert to numpy array
            combined_arr = np.array(combined_frames, dtype=np.float32)
            n_frames = len(combined_arr)

            # Resample / interpolate to standard 30-frame matrix
            indices = np.linspace(0, n_frames - 1, SEQUENCE_LENGTH, dtype=int)
            final_seq = combined_arr[indices]

            # Add micro-variation
            noise = np.random.normal(0, 0.002, final_seq.shape).astype(np.float32)
            final_seq += noise

            filename = f"combo_synth_{s+1}.npy"
            np.save(os.path.join(combo_dir, filename), final_seq)
            total_generated += 1

        print(f"  [Combination]: '{combo_name}' -> {samples_per_combo} sequences created", flush=True)

    print(f"\nCompleted multi-word combination dataset generation: {total_generated} sequences total.", flush=True)
    return total_generated

def retrain_with_combos():
    """Retrains the model with all gesture classes including multi-word combinations."""
    from src.train import load_dataset
    from src.model import ISLGestureClassifier
    from sklearn.model_selection import train_test_split
    from collections import Counter

    print("\nLoading dataset including multi-word combinations...", flush=True)
    X, y, class_names = load_dataset()
    print(f"Total dataset: {len(X)} sequences across {len(class_names)} gesture classes.", flush=True)

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

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    print(f"Training: {len(X_tr)} | Validation: {len(X_val)}", flush=True)

    classifier = ISLGestureClassifier(labels_list=new_names)
    classifier.train_model(X_tr, y_tr, X_val, y_val, epochs=25, batch_size=64, lr=0.001)

    # Quick eval
    correct = sum(1 for i in range(len(X_val))
                  if classifier.label_to_idx.get(classifier.predict(X_val[i])[0], -1) == y_val[i])
    acc = correct / max(len(X_val), 1) * 100

    print(f"\n============================================================")
    print(f"MULTI-WORD COMBINATION RETRAINING COMPLETE")
    print(f"  Validation accuracy: {acc:.1f}%")
    print(f"  Total gesture classes: {len(new_names)}")
    print(f"  Model saved to: {classifier.model_path}")
    print(f"============================================================", flush=True)

if __name__ == "__main__":
    generate_multi_word_sequences()
    retrain_with_combos()
