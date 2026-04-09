#!/usr/bin/env python3
"""
Generate result plots and sample query output for the README.

Produces:
    results/singular_value_decay.png
    results/explained_variance.png
    results/sample_queries.txt

Usage:
    python results.py
    python results.py --weights weights/ --papers processing/frontend_papers.csv
"""

import argparse
import json
import csv
import re
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found. Install with: pip install matplotlib")
    print("       Text output will still be generated.\n")


# ── load (mirrors demo.py) ────────────────────────────────────────────────────

def load_weights(weights_dir: Path):
    P     = np.load(weights_dir / "P.npy")
    idf   = np.load(weights_dir / "idf.npy")
    sigma = np.load(weights_dir / "sigma.npy")
    V     = np.load(weights_dir / "V.npy")
    with open(weights_dir / "meta.json") as f:
        meta = json.load(f)
    with open(weights_dir / "vocab.json") as f:
        vocab = {k: int(v) for k, v in json.load(f).items()}
    doc_vecs = V * sigma[None, :]
    return P, idf, sigma, vocab, doc_vecs, meta


def load_titles(csv_path: Path, n_docs: int):
    titles = [f"Paper #{i}" for i in range(n_docs)]
    if not csv_path.exists():
        return titles
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_docs:
                break
            t = row.get("title") or row.get("Title") or f"Paper #{i}"
            titles[i] = t.strip().replace("\n", " ")
    return titles


# ── plots ─────────────────────────────────────────────────────────────────────

BG    = "#0d1117"
FG    = "#c9d1d9"
BLUE  = "#58a6ff"
GREEN = "#3fb950"
MUTED = "#8b949e"
ORG   = "#f0883e"
RED   = "#ff7b72"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def plot_singular_values(sigma: np.ndarray, out: Path):
    k = len(sigma)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.plot(range(1, k+1), sigma, color=BLUE, linewidth=1.6)
    ax.fill_between(range(1, k+1), sigma, alpha=0.12, color=BLUE)
    _style(ax, f"Singular Value Decay  (k={k})", "Rank", "σ")
    ax.set_xlim(1, k)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved: {out}")


def plot_explained_variance(sigma: np.ndarray, out: Path):
    var = sigma**2 / (sigma**2).sum()
    cum = np.cumsum(var) * 100
    k   = len(sigma)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.plot(range(1, k+1), cum, color=GREEN, linewidth=1.6)
    ax.fill_between(range(1, k+1), cum, alpha=0.10, color=GREEN)

    for pct, col in [(80, ORG), (90, RED)]:
        idx = np.searchsorted(cum, pct)
        if idx < k:
            ax.axhline(pct, color=col, linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axvline(idx+1, color=col, linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(idx+2, pct+0.8, f"{pct}% @ k={idx+1}",
                    color=col, fontsize=8)

    _style(ax, "Cumulative Explained Variance",
           "k (latent dimensions)", "Variance explained (%)")
    ax.set_xlim(1, k)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved: {out}")


# ── sample queries ─────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "neural network image classification",
    "graph neural networks node embedding",
    "natural language processing transformers attention",
    "reinforcement learning policy gradient",
    "randomized algorithms matrix approximation",
]


def encode_query(query, vocab, idf, P):
    tokens = re.findall(r"[a-z]+", query.lower())
    tf = {}
    for tok in tokens:
        if tok in vocab:
            idx = vocab[tok]
            tf[idx] = tf.get(idx, 0) + 1
    if not tf:
        return None
    n = sum(tf.values())
    q_vec = np.zeros(len(idf))
    for idx, count in tf.items():
        q_vec[idx] = (count / n) * idf[idx]
    q_embed = P @ q_vec
    norm = np.linalg.norm(q_embed)
    return q_embed / norm if norm > 1e-10 else None


def run_sample_queries(P, idf, sigma, vocab, doc_vecs, titles, out: Path):
    norms  = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
    d_norm = doc_vecs / norms

    SEP  = "═" * 70
    lines = [
        SEP,
        "  SAMPLE QUERY RESULTS — ArXiv Semantic Search",
        f"  {doc_vecs.shape[0]:,} papers · k={doc_vecs.shape[1]} latent dims",
        SEP,
    ]

    for query in SAMPLE_QUERIES:
        q_embed = encode_query(query, vocab, idf, P)
        lines.append(f"\n  Query: \"{query}\"")
        lines.append("  " + "─" * 66)

        if q_embed is None:
            lines.append("  [no vocabulary overlap — skipped]")
            continue

        scores  = d_norm @ q_embed
        top_idx = np.argsort(scores)[::-1][:5]

        for rank, idx in enumerate(top_idx, 1):
            bar   = "█" * int(scores[idx] * 24)
            title = titles[idx]
            if len(title) > 50:
                title = title[:47] + "..."
            lines.append(
                f"  #{rank}  {scores[idx]:.4f}  [{bar:<24}]  {title}"
            )

    lines.append("\n" + SEP)
    text = "\n".join(lines)

    print(text)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate result plots and sample output")
    parser.add_argument("--weights", default="weights",
                        help="Weights directory (default: weights/)")
    parser.add_argument("--papers",  default="processing/frontend_papers.csv")
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    out_dir     = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"Loading model from '{weights_dir}' ...")
    P, idf, sigma, vocab, doc_vecs, meta = load_weights(weights_dir)
    titles = load_titles(Path(args.papers), meta["n_docs"])
    print(f"  {meta['n_docs']:,} docs · vocab={meta['vocab_size']:,} · k={meta['k']}\n")

    if HAS_MPL:
        print("Generating plots...")
        plot_singular_values(sigma, out_dir / "singular_value_decay.png")
        plot_explained_variance(sigma, out_dir / "explained_variance.png")
    else:
        print("Skipping plots (matplotlib not installed).")

    print("\nRunning sample queries...")
    run_sample_queries(P, idf, sigma, vocab, doc_vecs, titles,
                       out_dir / "sample_queries.txt")

    print(f"\nDone. Results written to {out_dir}/")


if __name__ == "__main__":
    main()