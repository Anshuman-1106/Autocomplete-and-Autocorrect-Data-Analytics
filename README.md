# Autocomplete & Autocorrect NLP Project

A self-contained Python implementation covering all stages of an
autocomplete/autocorrect analytics project: dataset collection,
preprocessing, two autocomplete algorithms, two autocorrect algorithms,
evaluation metrics, a simulated UX/satisfaction report, side-by-side
algorithm comparison, and chart visualization.

## Files
- `corpus.py` — Step 1: built-in diverse sample text corpus (swap for your own data)
- `preprocessing.py` — Step 2: cleaning, tokenization, vocabulary, n-grams
- `autocomplete.py` — Step 3: `TrieAutocomplete` (prefix completion) and `NgramAutocomplete` (next-word prediction)
- `autocorrect.py` — Step 4: `EditDistanceAutocorrect` (Norvig-style, frequency-ranked) and `BaselineAutocorrect` (naive first-match)
- `metrics.py` — Step 5: synthetic test-set generation + Top-K accuracy, MRR, word accuracy, latency
- `user_experience.py` — Step 6: simulated satisfaction scoring from accuracy + latency
- `visualize_charts.py` — Step 8: matplotlib charts (frequency, comparisons, latency)
- `main.py` — runs the full pipeline (Step 7: prints comparison + picks best engine)

## Usage
```bash
# Run with the built-in sample corpus
python main.py

# Run with your own text file instead
python main.py --file mydata.txt --out my_output
```

Outputs go to `output/` by default:
- `word_frequency.png`
- `autocomplete_comparison.png`
- `autocorrect_comparison.png`
- `latency_comparison.png`
- `summary.json` (all metrics + UX scores in one file)

## Notes / how to extend
- **Real dataset**: replace `corpus.build_corpus()` with `DatasetCollector.load_from_file("your_text.txt")` for a real, large text source (chat logs, books, support tickets, etc.). Larger and more diverse text will improve both engines' accuracy.
- **Real user feedback**: `user_experience.py` currently *simulates* a satisfaction score from accuracy/latency. Replace `simulate_survey_score` with real collected feedback (thumbs up/down, suggestion-acceptance rate, survey responses) once you have live users.
- **More algorithms**: the structure makes it easy to drop in a neural model (e.g. a small LSTM/transformer next-word predictor) alongside the Trie/n-gram baselines for a deeper Step 7 comparison.
