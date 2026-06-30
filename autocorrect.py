"""
Autocorrect module (Step 4 + part of Step 7).

Implements two spelling-correction algorithms so they can be compared:
  1. EditDistanceAutocorrect - classic noisy-channel approach (Norvig-style):
     generate candidate edits (insert/delete/replace/transpose) within
     edit distance 1-2, rank surviving real words by corpus frequency.
  2. BaselineAutocorrect     - naive approach that only fixes a word if an
     edit-distance-1 candidate exists, with no frequency weighting
     (first match wins) -- used as a weaker baseline for comparison.
"""

import string
from collections import Counter
from typing import List, Optional


_ALPHABET = string.ascii_lowercase


def _edits1(word: str) -> set:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _ALPHABET]
    inserts = [a + c + b for a, b in splits for c in _ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def _edits2(word: str) -> set:
    return {e2 for e1 in _edits1(word) for e2 in _edits1(e1)}


class EditDistanceAutocorrect:
    """Norvig-style spelling corrector: candidate generation by edit
    distance, ranked by word frequency in the corpus (the noisy-channel
    approximation P(word) * P(typo|word), simplified to using only the
    word-frequency prior since real typo-likelihood data isn't available)."""

    def __init__(self, vocab: Counter):
        self.vocab = vocab
        self.vocab_set = set(vocab.keys())
        self.total = sum(vocab.values())

    def _known(self, words) -> set:
        return {w for w in words if w in self.vocab_set}

    def correct(self, word: str) -> str:
        word = word.lower()
        if word in self.vocab_set:
            return word

        candidates = (
            self._known([word])
            or self._known(_edits1(word))
            or self._known(_edits2(word))
            or [word]
        )
        return max(candidates, key=lambda w: self.vocab.get(w, 0))

    def correct_sentence(self, sentence: List[str]) -> List[str]:
        return [self.correct(w) for w in sentence]


class BaselineAutocorrect:
    """Weaker baseline: only attempts edit-distance-1 corrections and
    picks the first valid candidate found, ignoring frequency. Used to
    benchmark how much the frequency-ranked approach improves accuracy."""

    def __init__(self, vocab: Counter):
        self.vocab_set = set(vocab.keys())

    def correct(self, word: str) -> str:
        word = word.lower()
        if word in self.vocab_set:
            return word
        for cand in _edits1(word):
            if cand in self.vocab_set:
                return cand
        return word

    def correct_sentence(self, sentence: List[str]) -> List[str]:
        return [self.correct(w) for w in sentence]
