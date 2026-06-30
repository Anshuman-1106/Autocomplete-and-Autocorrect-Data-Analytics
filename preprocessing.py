"""
NLP Preprocessing module.

Handles step 1 (dataset collection helpers) and step 2 (cleaning /
preparation) of the project: tokenization, normalization, vocabulary
building, word-frequency counts, and n-gram extraction used by both
the autocomplete and autocorrect engines.
"""

import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple


class DatasetCollector:
    """Step 1: Dataset Collection -- gather diverse text data."""

    @staticmethod
    def load_from_file(path: str) -> str:
        """Load raw text from a file the user provides (.txt)."""
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def load_from_string(text: str) -> str:
        return text

    @staticmethod
    def combine(*texts: str) -> str:
        """Combine multiple text sources into one diverse corpus."""
        return " ".join(texts)


class Preprocessor:
    """Step 2: NLP Preprocessing -- clean and prepare data for analysis."""

    WORD_RE = re.compile(r"[a-zA-Z']+")

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """Remove non-alphabetic noise, normalize whitespace."""
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize cleaned text into a list of word tokens."""
        text = self.clean_text(text)
        if self.lowercase:
            text = text.lower()
        return self.WORD_RE.findall(text)

    def sentences(self, text: str) -> List[List[str]]:
        """Split into sentences (rough heuristic) and tokenize each."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [self.tokenize(s) for s in raw_sentences if s.strip()]

    def build_vocabulary(self, tokens: List[str]) -> Counter:
        """Word-frequency table used by autocomplete ranking and the
        autocorrect noisy-channel model."""
        return Counter(tokens)

    def build_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Build n-gram frequency counts (n=2 bigrams, n=3 trigrams, ...)."""
        grams = Counter()
        for i in range(len(tokens) - n + 1):
            grams[tuple(tokens[i:i + n])] += 1
        return grams

    def prepare(self, text: str, max_n: int = 3) -> Dict:
        """
        Full preprocessing pipeline. Returns a dict with tokens,
        vocabulary frequencies, and n-gram tables up to max_n -- the
        artifacts both downstream engines consume.
        """
        tokens = self.tokenize(text)
        vocab = self.build_vocabulary(tokens)
        ngram_tables = {n: self.build_ngrams(tokens, n) for n in range(2, max_n + 1)}
        return {
            "tokens": tokens,
            "vocab": vocab,
            "total_words": len(tokens),
            "unique_words": len(vocab),
            "ngrams": ngram_tables,
        }
