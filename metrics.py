"""
Metrics module (Step 5: Define and measure performance metrics).

Provides:
  - Synthetic test-set generators (since no labeled dataset exists,
    we generate ground truth from the corpus itself: mask words for
    autocomplete, inject typos for autocorrect).
  - Evaluation functions returning standard NLP metrics:
      autocomplete -> Top-K accuracy, Mean Reciprocal Rank (MRR)
      autocorrect  -> Word accuracy, precision on misspelled subset
  - Simple latency measurement to proxy "user experience" responsiveness.
"""

import random
import time
import string
from typing import List, Callable, Dict, Tuple


# ---------------------------------------------------------------------
# Test-set generation
# ---------------------------------------------------------------------

def generate_autocomplete_testset(tokens: List[str], n: int = 200,
                                   min_prefix: int = 2, seed: int = 1) -> List[Tuple[str, str]]:
    """
    Build (prefix, true_word) pairs by sampling real words from the
    corpus and truncating them to a random prefix length. The model is
    then judged on whether it recovers the true full word.
    """
    rng = random.Random(seed)
    candidates = [t for t in tokens if len(t) > min_prefix]
    sample = rng.sample(candidates, min(n, len(candidates)))
    pairs = []
    for word in sample:
        cut = rng.randint(min_prefix, len(word) - 1)
        pairs.append((word[:cut], word))
    return pairs


def generate_autocorrect_testset(tokens: List[str], n: int = 200,
                                  seed: int = 2) -> List[Tuple[str, str]]:
    """
    Build (misspelled_word, correct_word) pairs by injecting a single
    random edit (insert/delete/replace/swap) into real corpus words.
    """
    rng = random.Random(seed)
    candidates = [t for t in tokens if len(t) > 2]
    sample = rng.sample(candidates, min(n, len(candidates)))
    pairs = []
    for word in sample:
        pairs.append((_inject_typo(word, rng), word))
    return pairs


def _inject_typo(word: str, rng: random.Random) -> str:
    op = rng.choice(["delete", "insert", "replace", "swap"])
    i = rng.randint(0, len(word) - 1)
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        ch = rng.choice(string.ascii_lowercase)
        return word[:i] + ch + word[i:]
    if op == "replace":
        ch = rng.choice(string.ascii_lowercase)
        return word[:i] + ch + word[i + 1:]
    if op == "swap" and len(word) > 1:
        j = min(i + 1, len(word) - 1)
        chars = list(word)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    return word


# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------

def evaluate_autocomplete(predict_fn: Callable[[str], List[str]],
                           testset: List[Tuple[str, str]], top_k: int = 5) -> Dict:
    """
    Top-K accuracy: fraction of cases where the true word appears
    anywhere in the top-K predictions.
    Mean Reciprocal Rank (MRR): average of 1/rank of the true word
    (0 if not found), rewarding predictions that rank the right
    answer higher.
    """
    hits, reciprocal_ranks = 0, []
    start = time.perf_counter()
    for prefix, true_word in testset:
        preds = predict_fn(prefix)[:top_k]
        if true_word in preds:
            hits += 1
            reciprocal_ranks.append(1.0 / (preds.index(true_word) + 1))
        else:
            reciprocal_ranks.append(0.0)
    elapsed = time.perf_counter() - start

    n = max(len(testset), 1)
    return {
        "top_k_accuracy": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_latency_ms": (elapsed / n) * 1000,
        "n": n,
    }


def evaluate_autocorrect(correct_fn: Callable[[str], str],
                          testset: List[Tuple[str, str]]) -> Dict:
    """
    Word accuracy: fraction of misspelled words restored to the exact
    correct word.
    """
    correct_count = 0
    start = time.perf_counter()
    for misspelled, true_word in testset:
        if correct_fn(misspelled) == true_word:
            correct_count += 1
    elapsed = time.perf_counter() - start

    n = max(len(testset), 1)
    return {
        "accuracy": correct_count / n,
        "avg_latency_ms": (elapsed / n) * 1000,
        "n": n,
    }
