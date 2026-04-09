#!/usr/bin/env python3
"""
ArXiv Semantic Search — Demo
Fold a query into LSA latent space and rank papers by cosine similarity.

Usage:
    python demo.py --weights weights/
    python demo.py --weights weights/ --query "attention mechanism"
    python demo.py --weights weights/ --query "graph neural networks" --top 10
"""

import argparse
import json
import csv
import re
import time
import numpy as np
from pathlib import Path

def load_weights(weights_dir: Path):
    """Load pre-trained LSA model."""
    required = ["P.npy", "idf.npy", "sigma.npy", "meta.json", "vocab.json", "V.npy"]
    for f in required:
        if not (weights_dir / f).exists():
            raise FileNotFoundError(
                f"Missing '{f}' in '{weights_dir}'. Run train.py first."
            )

    P     = np.load(weights_dir / "P.npy")       # k × vocab  (fold-in operator)
    idf   = np.load(weights_dir / "idf.npy")     # vocab
    sigma = np.load(weights_dir / "sigma.npy")   # k
    V     = np.load(weights_dir / "V.npy")        # n_docs × k

    with open(weights_dir / "meta.json") as f:
        meta = json.load(f)
    with open(weights_dir / "vocab.json") as f:
        vocab = {k: int(v) for k, v in json.load(f).items()}

    # document embeddings: scale V columns by singular values
    # V[i] is the right singular vector for doc i → scaled by sigma gives
    # the doc's coordinates in latent space
    doc_vecs = V * sigma[None, :]    # n_docs × k

    return P, idf, sigma, vocab, doc_vecs, meta


def load_titles(csv_path: Path, n_docs: int):
    """Load paper titles from frontend_papers.csv (columns: id, title, abstract)."""
    titles = [f"Paper #{i}" for i in range(n_docs)]
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found — titles will show as 'Paper #N'")
        return titles
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n_docs:
                break
            t = row.get("title") or row.get("Title") or f"Paper #{i}"
            titles[i] = t.strip().replace("\n", " ")
    return titles



def encode_query(query: str, vocab: dict, idf: np.ndarray, P: np.ndarray):
    """
    Fold query into latent space.

    1. Tokenize (lowercase alphabetic tokens — vocab keys are Porter stems,
       so exact stem matching; common words will still hit the index)
    2. TF vector normalized by query length
    3. Weight by IDF  →  q ∈ R^vocab
    4. Fold in:  q_embed = P @ q   (P = (U/σ)ᵀ is k × vocab)
    5. L2-normalize for cosine similarity
    """
    tokens = re.findall(r"[a-z]+", query.lower())
    tf = {}
    for tok in tokens:
        if tok in vocab:
            idx = vocab[tok]
            tf[idx] = tf.get(idx, 0) + 1

    matched_words = [w for w in tokens if w in vocab]

    if not tf:
        return None, matched_words

    n = sum(tf.values())
    q_vec = np.zeros(len(idf))
    for idx, count in tf.items():
        q_vec[idx] = (count / n) * idf[idx]

    q_embed = P @ q_vec      # k-dimensional
    norm = np.linalg.norm(q_embed)
    if norm < 1e-10:
        return None, matched_words

    return q_embed / norm, matched_words


def search(query: str, vocab: dict, idf: np.ndarray, P: np.ndarray,
           doc_vecs: np.ndarray, titles: list, top_n: int = 5):
    """Return top_n results for query string."""
    q_embed, matched = encode_query(query, vocab, idf, P)
    if q_embed is None:
        return [], matched

    norms  = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
    scores = (doc_vecs / norms) @ q_embed
    top_idx = np.argsort(scores)[::-1][:top_n]

    results = [
        {
            "rank":   r + 1,
            "doc_id": int(i),
            "score":  float(scores[i]),
            "title":  titles[i],
        }
        for r, i in enumerate(top_idx)
    ]
    return results, matched


def print_results(query: str, results: list, matched: list, elapsed: float):
    W = 70
    print("\n" + "─" * W)
    print(f"  Query : \"{query}\"")
    if matched:
        display = matched[:8] + (["..."] if len(matched) > 8 else [])
        print(f"  Terms : {', '.join(display)}")
    print(f"  Time  : {elapsed * 1000:.1f} ms")
    print("─" * W)

    if not results:
        print("  No results — no query terms matched the vocabulary.")
        print("  Tip: vocab uses NLTK Porter stems "
              "(e.g. 'running'→'run', 'neural'→'neural')")
    for r in results:
        bar = "█" * int(r["score"] * 24)
        print(f"\n  #{r['rank']}  score={r['score']:.4f}  [{bar:<24}]")
        # truncate long titles cleanly
        title = r["title"]
        if len(title) > 65:
            title = title[:62] + "..."
        print(f"      {title}")
        print(f"      doc_id={r['doc_id']}")
    print("─" * W + "\n")


def main():
    parser = argparse.ArgumentParser(description="ArXiv Semantic Search")
    parser.add_argument("--weights", default="weights",
                        help="Weights directory from train.py (default: weights/)")
    parser.add_argument("--papers",  default="processing/frontend_papers.csv",
                        help="CSV with id,title,abstract columns")
    parser.add_argument("--query",   default=None,
                        help="Search query (omit for interactive mode)")
    parser.add_argument("--top",     type=int, default=5,
                        help="Number of results (default: 5)")
    args = parser.parse_args()

    weights_dir = Path(args.weights)

    print(f"Loading model from '{weights_dir}' ...", end=" ", flush=True)
    t0 = time.time()
    P, idf, sigma, vocab, doc_vecs, meta = load_weights(weights_dir)
    titles = load_titles(Path(args.papers), meta["n_docs"])
    print(f"done in {time.time()-t0:.2f}s  "
          f"({meta['n_docs']:,} docs · vocab={meta['vocab_size']:,} · k={meta['k']})")

    if args.query:
        t0 = time.time()
        results, matched = search(
            args.query, vocab, idf, P, doc_vecs, titles, top_n=args.top
        )
        print_results(args.query, results, matched, time.time() - t0)

    else:
        print("\nArXiv Semantic Search — interactive mode")
        print("Type a query and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                query = input("  query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("Bye.")
                break
            t0 = time.time()
            results, matched = search(
                query, vocab, idf, P, doc_vecs, titles, top_n=args.top
            )
            print_results(query, results, matched, time.time() - t0)


if __name__ == "__main__":
    main()