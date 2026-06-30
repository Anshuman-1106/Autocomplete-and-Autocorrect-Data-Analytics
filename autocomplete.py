"""
Autocomplete module (Step 3 + part of Step 7).

Implements two autocomplete algorithms so they can be compared:
  1. TrieAutocomplete   - prefix tree + frequency ranking (classic approach,
                           O(prefix length) lookup, good for word completion)
  2. NgramAutocomplete   - bigram/trigram based next-WORD prediction
                           (predicts the next word given prior context,
                           not just a prefix)
"""

from collections import Counter, defaultdict
from typing import List, Tuple, Dict


class TrieNode:
    __slots__ = ("children", "is_word", "freq")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word = False
        self.freq = 0


class TrieAutocomplete:
    """Prefix-based word completion ranked by corpus frequency."""

    def __init__(self, vocab: Counter):
        self.root = TrieNode()
        self.vocab = vocab
        for word, freq in vocab.items():
            self._insert(word, freq)

    def _insert(self, word: str, freq: int):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True
        node.freq = freq

    def _collect(self, node: TrieNode, prefix: str, results: List[Tuple[str, int]]):
        if node.is_word:
            results.append((prefix, node.freq))
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, results)

    def predict(self, prefix: str, top_k: int = 5) -> List[str]:
        """Return up to top_k word completions for a given prefix,
        ranked by frequency (most common words first)."""
        prefix = prefix.lower()
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: List[Tuple[str, int]] = []
        self._collect(node, prefix, results)
        results.sort(key=lambda x: -x[1])
        return [w for w, _ in results[:top_k]]


class NgramAutocomplete:
    """Predicts the next word given the previous 1-2 words of context,
    using bigram/trigram frequency tables built during preprocessing."""

    def __init__(self, ngrams: Dict[int, Counter], vocab: Counter):
        self.bigrams = ngrams.get(2, Counter())
        self.trigrams = ngrams.get(3, Counter())
        self.vocab = vocab
        self._bigram_index = self._index(self.bigrams, context_len=1)
        self._trigram_index = self._index(self.trigrams, context_len=2)

    @staticmethod
    def _index(ngram_counter: Counter, context_len: int) -> Dict[Tuple[str, ...], Counter]:
        index = defaultdict(Counter)
        for gram, count in ngram_counter.items():
            context, next_word = gram[:context_len], gram[-1]
            index[context][next_word] += count
        return index

    def predict(self, context: List[str], top_k: int = 5) -> List[str]:
        """Given preceding word(s), predict likely next words.
        Falls back: trigram context -> bigram context -> overall frequency."""
        context = [w.lower() for w in context]

        if len(context) >= 2:
            key = tuple(context[-2:])
            if key in self._trigram_index:
                return [w for w, _ in self._trigram_index[key].most_common(top_k)]

        if len(context) >= 1:
            key = (context[-1],)
            if key in self._bigram_index:
                return [w for w, _ in self._bigram_index[key].most_common(top_k)]

        # Fallback: most frequent words overall
        return [w for w, _ in self.vocab.most_common(top_k)]
