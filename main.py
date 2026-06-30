"""
main.py - End-to-end pipeline for the Autocomplete & Autocorrect project.

Runs all 8 project steps in order:
  1. Dataset Collection
  2. NLP Preprocessing
  3. Autocomplete (Trie + N-gram)
  4. Autocorrect (Edit-distance + Baseline)
  5. Metrics (Top-K accuracy, MRR, word accuracy, latency)
  6. User Experience (simulated satisfaction scores)
  7. Algorithm Comparison (printed + charted side by side)
  8. Visualization (PNG charts saved to ./output)

Usage:
    python main.py                  # uses built-in sample corpus
    python main.py --file mytext.txt   # uses your own text file instead
"""

import argparse
import json
from pathlib import Path

from corpus import build_corpus
from preprocessing import DatasetCollector, Preprocessor
from autocomplete import TrieAutocomplete, NgramAutocomplete
from autocorrect import EditDistanceAutocorrect, BaselineAutocorrect
import metrics as M
import visualize_charts as V
from user_experience import build_ux_report


def run_pipeline(text: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 2: Preprocessing ----
    pre = Preprocessor()
    prepared = pre.prepare(text, max_n=3)
    tokens, vocab, ngrams = prepared["tokens"], prepared["vocab"], prepared["ngrams"]
    print(f"[Preprocessing] {prepared['total_words']} tokens, "
          f"{prepared['unique_words']} unique words")

    # ---- Step 3: Autocomplete engines ----
    trie_engine = TrieAutocomplete(vocab)
    ngram_engine = NgramAutocomplete(ngrams, vocab)

    # ---- Step 4: Autocorrect engines ----
    edit_engine = EditDistanceAutocorrect(vocab)
    baseline_engine = BaselineAutocorrect(vocab)

    # ---- Demo predictions ----
    print("\n--- Autocomplete demo ---")
    for prefix in ["pr", "tra", "wea", "comp"]:
        print(f"  '{prefix}' -> Trie: {trie_engine.predict(prefix, top_k=3)} "
              f"| N-gram(no context, falls back to frequency): "
              f"{ngram_engine.predict([], top_k=3)}")

    print("\n--- Next-word prediction demo (n-gram, context-aware) ---")
    for context in [["the", "weather"], ["she", "walked"], ["the", "company"]]:
        print(f"  context={context} -> {ngram_engine.predict(context, top_k=3)}")

    print("\n--- Autocorrect demo ---")
    for typo in ["wether", "computer", "recieved", "marathno"]:
        print(f"  '{typo}' -> EditDistance: '{edit_engine.correct(typo)}' "
              f"| Baseline: '{baseline_engine.correct(typo)}'")

    # ---- Step 5: Metrics ----
    ac_testset = M.generate_autocomplete_testset(tokens, n=150)
    corr_testset = M.generate_autocorrect_testset(tokens, n=150)

    # Note: the n-gram model predicts a *next word* from prior context, not
    # a prefix completion, so on this prefix-completion benchmark it can
    # only fall back to overall word frequency -- this intentionally shows
    # why prefix-based (Trie) and context-based (n-gram) models solve
    # different sub-problems and are best used together in a real product.
    autocomplete_results = {
        "Trie (prefix-ranked)": M.evaluate_autocomplete(
            lambda p: trie_engine.predict(p, top_k=5), ac_testset),
        "N-gram (no-prefix fallback)": M.evaluate_autocomplete(
            lambda p: ngram_engine.predict([], top_k=5), ac_testset),
    }
    autocorrect_results = {
        "Edit-distance (freq-ranked)": M.evaluate_autocorrect(
            edit_engine.correct, corr_testset),
        "Baseline (first-match)": M.evaluate_autocorrect(
            baseline_engine.correct, corr_testset),
    }

    print("\n--- Step 5: Metrics ---")
    print(json.dumps({
        "autocomplete": autocomplete_results,
        "autocorrect": autocorrect_results,
    }, indent=2))

    # ---- Step 6: User Experience (simulated survey) ----
    ux_report = build_ux_report(autocomplete_results, autocorrect_results)
    print("\n--- Step 6: Simulated User Satisfaction (out of 5) ---")
    for name, score in ux_report.items():
        print(f"  {name}: {score}/5")

    # ---- Step 7: Algorithm comparison summary ----
    print("\n--- Step 7: Algorithm Comparison Summary ---")
    best_ac = max(autocomplete_results, key=lambda k: autocomplete_results[k]["top_k_accuracy"])
    best_corr = max(autocorrect_results, key=lambda k: autocorrect_results[k]["accuracy"])
    print(f"  Best autocomplete engine: {best_ac}")
    print(f"  Best autocorrect engine:  {best_corr}")

    # ---- Step 8: Visualization ----
    V.plot_word_frequency(vocab, str(out_dir / "word_frequency.png"))
    V.plot_autocomplete_comparison(autocomplete_results, str(out_dir / "autocomplete_comparison.png"))
    V.plot_autocorrect_comparison(autocorrect_results, str(out_dir / "autocorrect_comparison.png"))

    combined_latency = {**autocomplete_results, **autocorrect_results}
    V.plot_latency_comparison(combined_latency, str(out_dir / "latency_comparison.png"))

    print(f"\n[Visualization] Charts saved to {out_dir}/")

    # Save full metrics + UX report as JSON for the report/record
    summary = {
        "preprocessing": {
            "total_words": prepared["total_words"],
            "unique_words": prepared["unique_words"],
        },
        "autocomplete_metrics": autocomplete_results,
        "autocorrect_metrics": autocorrect_results,
        "user_experience_scores": ux_report,
        "best_autocomplete": best_ac,
        "best_autocorrect": best_corr,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] Full results saved to {out_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Autocomplete & Autocorrect NLP project")
    parser.add_argument("--file", type=str, default=None,
                         help="Path to a .txt file to use instead of the sample corpus")
    parser.add_argument("--out", type=str, default="output",
                         help="Output directory for charts and summary.json")
    args = parser.parse_args()

    if args.file:
        text = DatasetCollector.load_from_file(args.file)
        print(f"[Dataset Collection] Loaded text from {args.file}")
    else:
        text = build_corpus()
        print("[Dataset Collection] Using built-in diverse sample corpus "
              "(pass --file yourdata.txt to use your own dataset)")

    run_pipeline(text, Path(args.out))


if __name__ == "__main__":
    main()
