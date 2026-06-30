"""
Visualization module (Step 8: Use tools for data visualization).

Generates matplotlib charts saved as PNG files:
  1. Top word-frequency bar chart (corpus overview)
  2. Autocomplete algorithm comparison (Top-K accuracy & MRR)
  3. Autocorrect algorithm comparison (accuracy)
  4. Latency comparison across all four engines
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from typing import Dict


def plot_word_frequency(vocab: Counter, out_path: str, top_n: int = 15):
    common = vocab.most_common(top_n)
    words, counts = zip(*common)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words[::-1], counts[::-1], color="#4C72B0")
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} Most Frequent Words in Corpus")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_autocomplete_comparison(results: Dict[str, Dict], out_path: str):
    names = list(results.keys())
    acc = [results[n]["top_k_accuracy"] for n in names]
    mrr = [results[n]["mrr"] for n in names]

    x = range(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], acc, width, label="Top-K Accuracy", color="#55A868")
    ax.bar([i + width / 2 for i in x], mrr, width, label="MRR", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Autocomplete Algorithm Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_autocorrect_comparison(results: Dict[str, Dict], out_path: str):
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(names, acc, color="#8172B2")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Autocorrect Algorithm Comparison")
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_comparison(all_results: Dict[str, Dict], out_path: str):
    names = list(all_results.keys())
    latency = [all_results[n]["avg_latency_ms"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, latency, color="#CCB974")
    ax.set_ylabel("Avg. Latency per Query (ms)")
    ax.set_title("Response Time Comparison (lower = better UX)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
