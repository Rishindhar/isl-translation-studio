"""
ISL Sentence N-gram Language Model
====================================
Builds a statistical bigram + trigram language model from ISL gesture sequences.
Used to:
  - Re-score ambiguous gesture predictions based on context
  - Predict next likely word in a sequence
  - Score how "natural" a sequence of signs is

Training corpus is auto-generated from:
  1. Multi-word folder names in data/ directory (e.g. father_go_work)
  2. All entries in ISL_SENTENCE_DICTIONARY
  3. Hardcoded common ISL sentence patterns
"""

import os
import json
import math
from collections import defaultdict, Counter

# Tokens
START = "<S>"
END   = "</S>"
UNK   = "<UNK>"

class ISLSentenceLanguageModel:
    """
    N-gram Language Model (Bigram + Trigram) with Laplace smoothing.
    Learns P(word | context) from a corpus of ISL word sequences.
    """

    def __init__(self):
        self.unigram_counts  = Counter()
        self.bigram_counts   = defaultdict(Counter)
        self.trigram_counts  = defaultdict(Counter)
        self.vocab           = set()
        self.total_tokens    = 0
        self._trained        = False

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────
    def train(self, sentence_list):
        """
        Train on a list of sentences, where each sentence is a list of words.
        E.g.: [["hello", "how", "you"], ["father", "go", "work"], ...]
        """
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.vocab = set()

        for sentence in sentence_list:
            tokens = [START, START] + [w.lower() for w in sentence] + [END]
            for tok in tokens:
                self.vocab.add(tok)

            for i, tok in enumerate(tokens):
                self.unigram_counts[tok] += 1
                self.total_tokens += 1

                if i >= 1:
                    self.bigram_counts[tokens[i-1]][tok] += 1

                if i >= 2:
                    self.trigram_counts[(tokens[i-2], tokens[i-1])][tok] += 1

        self._trained = True
        print(f"[ISL-LM] Trained on {len(sentence_list)} sentences | Vocab: {len(self.vocab)} tokens")

    # ──────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────
    def log_prob_sentence(self, words):
        """
        Compute log-probability of a word sequence using trigram LM with
        Laplace smoothing (k=0.1).  Returns a float (higher = more likely).
        """
        if not self._trained:
            return 0.0

        tokens = [START, START] + [w.lower() for w in words] + [END]
        V = len(self.vocab) + 1
        k = 0.1  # Laplace smoothing
        log_prob = 0.0

        for i in range(2, len(tokens)):
            tri_ctx = (tokens[i-2], tokens[i-1])
            bi_ctx  = tokens[i-1]
            word    = tokens[i]

            tri_count = self.trigram_counts[tri_ctx][word]
            tri_total = sum(self.trigram_counts[tri_ctx].values())

            bi_count  = self.bigram_counts[bi_ctx][word]
            bi_total  = sum(self.bigram_counts[bi_ctx].values())

            # Linear interpolation: 0.7 * trigram + 0.3 * bigram
            if tri_total > 0:
                p_tri = (tri_count + k) / (tri_total + k * V)
            else:
                p_tri = k / (k * V)

            if bi_total > 0:
                p_bi = (bi_count + k) / (bi_total + k * V)
            else:
                p_bi = k / (k * V)

            p = 0.7 * p_tri + 0.3 * p_bi
            log_prob += math.log(max(p, 1e-10))

        return log_prob

    def predict_next(self, context_words, top_k=5):
        """
        Given recent context words, return top-k most likely next words.
        Returns list of (word, probability) tuples.
        """
        if not self._trained or not context_words:
            return []

        ctx = [w.lower() for w in context_words]
        V = len(self.vocab)
        k = 0.1

        # Build candidate scores from trigram + bigram
        candidates = defaultdict(float)

        # Trigram context
        if len(ctx) >= 2:
            tri_ctx = (ctx[-2], ctx[-1])
            tri_total = sum(self.trigram_counts[tri_ctx].values())
            if tri_total > 0:
                for word, count in self.trigram_counts[tri_ctx].items():
                    if word not in (START, END, UNK):
                        candidates[word] += 0.7 * (count + k) / (tri_total + k * V)

        # Bigram context
        bi_ctx = ctx[-1]
        bi_total = sum(self.bigram_counts[bi_ctx].values())
        if bi_total > 0:
            for word, count in self.bigram_counts[bi_ctx].items():
                if word not in (START, END, UNK):
                    candidates[word] += 0.3 * (count + k) / (bi_total + k * V)

        # Sort and return top-k
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        return sorted_cands[:top_k]

    def rescore_candidates(self, context_words, candidate_labels, top_k=1):
        """
        Given a list of candidate prediction labels (from the GRU's top-k),
        re-rank them using language model probability.
        Returns the best label string.
        """
        if not self._trained or not candidate_labels:
            return candidate_labels[0] if candidate_labels else None

        best_label = candidate_labels[0]
        best_score = -1e18

        for label in candidate_labels:
            seq = list(context_words) + [label]
            score = self.log_prob_sentence(seq)
            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab":         list(self.vocab),
            "total_tokens":  self.total_tokens,
            "unigram":       dict(self.unigram_counts),
            "bigram":        {k: dict(v) for k, v in self.bigram_counts.items()},
            "trigram":       {f"{k[0]}|||{k[1]}": dict(v) for k, v in self.trigram_counts.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[ISL-LM] Saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"[ISL-LM] Model file not found: {path}")
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab          = set(data["vocab"])
        self.total_tokens   = data["total_tokens"]
        self.unigram_counts = Counter(data["unigram"])
        self.bigram_counts  = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)

        for k, v in data["bigram"].items():
            self.bigram_counts[k] = Counter(v)

        for k, v in data["trigram"].items():
            parts = k.split("|||")
            if len(parts) == 2:
                self.trigram_counts[(parts[0], parts[1])] = Counter(v)

        self._trained = True
        print(f"[ISL-LM] Loaded from {path} | Vocab: {len(self.vocab)}")
        return True
