"""
ISL Language Model Builder
===========================
Auto-generates a training corpus from:
  1. Multi-word folder names in data/ (e.g. father_go_work → ["father","go","work"])
  2. All entries in ISL_SENTENCE_DICTIONARY
  3. Hardcoded common ISL sentence templates

Then trains the ISLSentenceLanguageModel and saves it to models/isl_sentence_lm.json.
Run this script once before starting web_app.py.
"""

import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, MODELS_DIR
from src.isl_sentence_lm import ISLSentenceLanguageModel

LM_SAVE_PATH = os.path.join(MODELS_DIR, "isl_sentence_lm.json")

# ─────────────────────────────────────────────────────────────
# HARDCODED SENTENCE TRAINING CORPUS
# These are natural ISL sentence patterns — each inner list = one sentence
# ─────────────────────────────────────────────────────────────
HARDCODED_CORPUS = [
    # Greetings
    ["hello"],
    ["hello", "how", "you"],
    ["hello", "thank_you"],
    ["hello", "thank_you", "yes"],
    ["good", "morning"],
    ["good", "night"],
    ["good", "morning", "how", "you"],
    ["namaste"],
    ["welcome"],
    ["welcome", "home"],
    ["goodbye"],
    ["please", "help"],
    ["please", "stop"],
    ["sorry"],
    ["sorry", "late"],
    ["sorry", "mistake"],

    # Family
    ["father", "go", "work"],
    ["father", "home"],
    ["father", "go", "school"],
    ["mother", "food", "cook"],
    ["mother", "food"],
    ["mother", "home"],
    ["mother", "go", "work"],
    ["family", "home"],
    ["family", "love"],
    ["i", "love", "family"],
    ["friend", "good"],
    ["friend", "home"],
    ["friend", "school"],
    ["grandfather", "home"],
    ["grandmother", "home"],

    # Needs & Requests
    ["need", "water"],
    ["need", "food"],
    ["need", "help"],
    ["need", "medicine"],
    ["need", "doctor"],
    ["water", "drink"],
    ["food", "eat"],
    ["please", "water"],
    ["please", "food"],
    ["please", "help"],
    ["help", "me"],
    ["stop", "go"],
    ["go", "home"],
    ["go", "school"],
    ["go", "work"],
    ["come", "home"],
    ["come", "school"],
    ["read", "write"],
    ["work", "time"],
    ["study", "time"],

    # Questions
    ["what", "name"],
    ["where", "home"],
    ["where", "school"],
    ["where", "work"],
    ["time", "what"],
    ["how", "much"],
    ["how", "many"],
    ["how", "you"],
    ["what", "you", "do"],
    ["where", "you", "go"],

    # Responses
    ["yes"],
    ["no"],
    ["thank_you"],
    ["yes", "thank_you"],
    ["no", "thank_you"],
    ["yes", "i", "know"],
    ["no", "i", "know"],

    # School / Education
    ["school", "go"],
    ["student", "school"],
    ["teacher", "school"],
    ["read", "book"],
    ["write", "work"],
    ["learn", "school"],
    ["learn", "sign", "language"],

    # Medical
    ["doctor", "help"],
    ["medicine", "need"],
    ["hospital", "go"],
    ["pain", "help"],
    ["doctor", "hospital"],

    # Daily
    ["morning", "work"],
    ["night", "sleep"],
    ["day", "work"],
    ["lunch", "time"],
    ["food", "time"],
    ["work", "home"],
    ["home", "come"],
    ["water", "need"],
    ["food", "cook"],

    # Nature & Environment
    ["water", "river"],
    ["tree", "green"],
    ["mountain", "high"],
    ["sun", "hot"],
    ["rain", "water"],

    # Finance / Work
    ["money", "need"],
    ["salary", "work"],
    ["job", "work"],
    ["office", "go"],
    ["bank", "money"],

    # Transport
    ["car", "go"],
    ["train", "go"],
    ["bus", "school"],
    ["plane", "fly"],

    # Emotions
    ["happy", "good"],
    ["sad", "help"],
    ["angry", "stop"],
    ["love", "family"],
    ["proud", "work"],

    # Colors / Numbers
    ["number_1", "number_2", "number_3"],
    ["letter_a", "letter_b", "letter_c"],

    # Complex sentences
    ["father", "go", "work", "morning"],
    ["mother", "food", "cook", "home"],
    ["friend", "school", "go"],
    ["i", "go", "school", "morning"],
    ["i", "need", "water", "please"],
    ["please", "help", "me"],
    ["doctor", "medicine", "need"],
    ["family", "home", "happy"],
    ["work", "morning", "go"],
    ["school", "learn", "happy"],
    ["hello", "how", "you", "today"],
    ["good", "morning", "how", "you"],
    ["thank_you", "help", "me"],
    ["sorry", "late", "work"],
    ["need", "help", "please"],
    ["water", "need", "please"],
    ["food", "need", "hungry"],
    ["go", "home", "now"],
    ["come", "here", "please"],
    ["stop", "now", "please"],
]

def extract_multiword_sequences_from_data():
    """
    Scans the data/ directory for multi-word folder names and generates sentence sequences.
    E.g. folder 'father_go_work' → sentence ['father', 'go', 'work']
    """
    sentences = []
    if not os.path.exists(DATA_DIR):
        print(f"[Corpus] data/ directory not found at {DATA_DIR}")
        return sentences

    subdirs = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]

    print(f"[Corpus] Scanning {len(subdirs)} gesture class folders...")

    for folder_name in subdirs:
        parts = folder_name.lower().split("_")
        # Multi-word sequence: folder names with 2+ meaningful parts
        if len(parts) >= 2:
            # Filter out very long sequences (compound terms like 'leave_compensory')
            # by checking if individual parts exist as single-word classes
            sentences.append(parts)

        # Always add the full folder as a unigram (single word class)
        sentences.append([folder_name.lower()])

    print(f"[Corpus] Extracted {len(sentences)} sequences from data folder names")
    return sentences

def extract_from_sentence_dictionary():
    """Extract all sentence keys from ISL_SENTENCE_DICTIONARY as training examples."""
    try:
        from src.sentence_translator import ISL_SENTENCE_DICTIONARY
        sentences = [list(key) for key in ISL_SENTENCE_DICTIONARY.keys()]
        print(f"[Corpus] Extracted {len(sentences)} sequences from ISL_SENTENCE_DICTIONARY")
        return sentences
    except ImportError as e:
        print(f"[Corpus] Could not import ISL_SENTENCE_DICTIONARY: {e}")
        return []

def build_and_save_lm():
    """Main function: Build corpus, train LM, save."""
    print("\n" + "="*60)
    print("  ISL Sentence Language Model — Build & Train")
    print("="*60)

    # 1. Collect all training sentences from multiple sources
    corpus = []
    corpus.extend(HARDCODED_CORPUS)
    corpus.extend(extract_from_sentence_dictionary())
    corpus.extend(extract_multiword_sequences_from_data())

    # Remove duplicates
    seen = set()
    unique_corpus = []
    for sent in corpus:
        key = tuple(sent)
        if key not in seen:
            seen.add(key)
            unique_corpus.append(sent)

    print(f"\n[Corpus] Total unique training sentences: {len(unique_corpus)}")

    # 2. Train the language model
    lm = ISLSentenceLanguageModel()
    lm.train(unique_corpus)

    # 3. Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    lm.save(LM_SAVE_PATH)

    # 4. Quick sanity tests
    print("\n[Test] Predicting next words after 'hello':")
    preds = lm.predict_next(["hello"], top_k=5)
    for word, prob in preds:
        print(f"  {word}: {prob:.4f}")

    print("\n[Test] Predicting next words after 'father go':")
    preds = lm.predict_next(["father", "go"], top_k=5)
    for word, prob in preds:
        print(f"  {word}: {prob:.4f}")

    print("\n[Test] Rescoring candidates after 'need':")
    rescored = lm.rescore_candidates(["need"], ["food", "water", "help", "medicine"])
    print(f"  Best candidate: {rescored}")

    print("\n" + "="*60)
    print(f"  LM BUILT SUCCESSFULLY => {LM_SAVE_PATH}")
    print("="*60 + "\n")
    return lm

if __name__ == "__main__":
    build_and_save_lm()
