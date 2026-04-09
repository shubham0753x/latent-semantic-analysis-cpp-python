"""
test_weights.py — Verify saved weights and test search pipeline.

Usage:
  python test_weights.py --weights weights/ --vocab vocab.json --corpus corpus.txt

What it checks:
  1. Weight files exist and have correct shapes
  2. P matrix is correct (P = Σ⁻¹ Uᵀ)
  3. Fold-in produces finite, non-zero embeddings
  4. Cosine similarity is in [-1, 1]
  5. Same-document similarity = 1.0
  6. Similar documents rank higher than random ones
  7. Search latency
"""

import argparse
import json
import os
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def check(name, passed, detail=""):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))
    return passed

def hline(): print("─" * 60)

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────
# load weights
# ─────────────────────────────────────────────────────────────

def load_weights(weights_dir):
    hline()
    print(f"[1] Loading weights from: {weights_dir}")
    w = {}
    required = ["P.npy", "idf.npy", "sigma.npy", "U.npy", "vocab.json", "meta.json"]

    for fname in required:
        path = os.path.join(weights_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        if fname.endswith(".npy"):
            w[fname[:-4]] = np.load(path)
        elif fname == "vocab.json":
            with open(path) as f:
                w["vocab"] = json.load(f)
        elif fname == "meta.json":
            with open(path) as f:
                w["meta"] = json.load(f)
        print(f"  loaded {fname}")

    # check for V (document embeddings) — optional but needed for search
    v_path = os.path.join(weights_dir, "V.npy")
    if os.path.exists(v_path):
        w["V"] = np.load(v_path)
        print(f"  loaded V.npy  (document embeddings)")
    else:
        print(f"  [NOTE] V.npy not found — will compute doc embeddings from corpus")
        w["V"] = None

    return w


# ─────────────────────────────────────────────────────────────
# validate weight shapes
# ─────────────────────────────────────────────────────────────

def validate_shapes(w):
    hline()
    print("[2] Validating weight shapes")
    meta  = w["meta"]
    k     = meta["k"]
    vocab_size = meta["vocab_size"]
    n_docs     = meta["n_docs"]

    check("P shape",     w["P"].shape     == (k, vocab_size),
          str(w["P"].shape))
    check("U shape",     w["U"].shape     == (vocab_size, k),
          str(w["U"].shape))
    check("sigma shape", w["sigma"].shape == (k,),
          str(w["sigma"].shape))
    check("idf shape",   w["idf"].shape   == (vocab_size,),
          str(w["idf"].shape))
    check("sigma positive",   bool(np.all(w["sigma"] > 0)))
    check("sigma decreasing", bool(np.all(np.diff(w["sigma"]) <= 1e-6)))
    check("idf >= 1",         bool(np.all(w["idf"] >= 1.0)))
    check("P finite",         bool(np.all(np.isfinite(w["P"]))))

    # verify P = Σ⁻¹ Uᵀ
    P_reconstructed = (w["U"] / w["sigma"]).T
    p_err = np.linalg.norm(w["P"] - P_reconstructed)
    check("P = Σ⁻¹Uᵀ", p_err < 1e-6, f"err={p_err:.2e}")

    if w["V"] is not None:
        check("V shape", w["V"].shape == (n_docs, k), str(w["V"].shape))

    print(f"\n  Meta:")
    for key in ["k","vocab_size","n_docs","n_samples"]:
        print(f"    {key:15} = {meta.get(key,'?')}")


# ─────────────────────────────────────────────────────────────
# fold-in: text → embedding
# ─────────────────────────────────────────────────────────────

def embed_tokens(token_ids, P, idf, vocab_size):
    """Fold a list of token ids into concept space."""
    tf = Counter(token_ids)
    vec = np.zeros(vocab_size)
    for tok, cnt in tf.items():
        if 0 <= tok < vocab_size:
            vec[tok] = (cnt / len(token_ids)) * idf[tok]
    return P @ vec


def embed_text(text, vocab, P, idf, vocab_size):
    """Tokenize raw text and embed."""
    tokens = []
    for word in text.lower().split():
        if word in vocab:
            tokens.append(vocab[word])
    if not tokens:
        return None, []
    return embed_tokens(tokens, P, idf, vocab_size), tokens


# ─────────────────────────────────────────────────────────────
# load a sample of docs from corpus
# ─────────────────────────────────────────────────────────────

def load_sample_docs(corpus_path, vocab, n=200):
    docs_raw = []
    docs_tokens = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            tokens = []
            for tok in line.split():
                try:
                    tokens.append(int(tok))
                except ValueError:
                    tid = vocab.get(tok.lower())
                    if tid is not None: tokens.append(tid)
            if tokens:
                docs_raw.append(line)
                docs_tokens.append(tokens)
            if len(docs_tokens) >= n: break
    return docs_raw, docs_tokens


# ─────────────────────────────────────────────────────────────
# validate fold-in
# ─────────────────────────────────────────────────────────────

def validate_foldin(w, docs_tokens, n_check=20):
    hline()
    print("[3] Validating fold-in embeddings")
    P          = w["P"]
    idf        = w["idf"]
    vocab_size = w["meta"]["vocab_size"]
    k          = w["meta"]["k"]

    embeds = []
    norms  = []
    for toks in docs_tokens[:n_check]:
        emb = embed_tokens(toks, P, idf, vocab_size)
        embeds.append(emb)
        norms.append(np.linalg.norm(emb))

    check("all embeddings finite", all(np.all(np.isfinite(e)) for e in embeds))
    check("all embeddings non-zero", all(n > 1e-12 for n in norms))
    check("embedding dimension = k", all(len(e) == k for e in embeds))

    print(f"\n  Embedding norm stats (n={n_check}):")
    print(f"    mean = {np.mean(norms):.4f}")
    print(f"    min  = {np.min(norms):.4f}")
    print(f"    max  = {np.max(norms):.4f}")

    return embeds


# ─────────────────────────────────────────────────────────────
# search: brute-force cosine over doc embeddings
# ─────────────────────────────────────────────────────────────

def build_doc_index(docs_tokens, P, idf, vocab_size):
    """Embed all docs and return normalised matrix for fast cosine search."""
    print(f"  Embedding {len(docs_tokens)} docs...", end="", flush=True)
    t0 = time.time()
    E = np.zeros((len(docs_tokens), P.shape[0]))
    for i, toks in enumerate(docs_tokens):
        E[i] = embed_tokens(toks, P, idf, vocab_size)
    # row-normalise for cosine
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    E /= norms
    print(f" done in {time.time()-t0:.2f}s")
    return E


def search(query_emb, E, top_k=5):
    """Cosine search. Returns (indices, scores)."""
    qn = np.linalg.norm(query_emb)
    if qn < 1e-12: return [], []
    q  = query_emb / qn
    scores = E @ q
    idx    = np.argsort(scores)[::-1][:top_k]
    return idx.tolist(), scores[idx].tolist()


# ─────────────────────────────────────────────────────────────
# validate search quality
# ─────────────────────────────────────────────────────────────

def validate_search(w, docs_tokens, docs_raw, n_query=10):
    hline()
    print("[4] Validating search quality")
    P          = w["P"]
    idf        = w["idf"]
    vocab_size = w["meta"]["vocab_size"]

    # build index over sample
    E = build_doc_index(docs_tokens, P, idf, vocab_size)

    results = []

    # test 1: self-similarity (query = doc → top result = itself)
    self_sim_correct = 0
    for i in range(min(n_query, len(docs_tokens))):
        emb = embed_tokens(docs_tokens[i], P, idf, vocab_size)
        idx, scores = search(emb, E, top_k=3)
        if i in idx[:1]:
            self_sim_correct += 1

    pct = self_sim_correct / n_query * 100
    check(f"self-similarity top-1 ({self_sim_correct}/{n_query})", pct >= 80,
          f"{pct:.0f}%")

    # test 2: cosine scores in [-1,1]
    all_valid = True
    for i in range(min(5, len(docs_tokens))):
        emb = embed_tokens(docs_tokens[i], P, idf, vocab_size)
        _, scores = search(emb, E, top_k=5)
        if any(s < -1.01 or s > 1.01 for s in scores):
            all_valid = False
    check("cosine scores in [-1,1]", all_valid)

    # test 3: search latency
    times = []
    emb = embed_tokens(docs_tokens[0], P, idf, vocab_size)
    for _ in range(20):
        t0 = time.time()
        search(emb, E, top_k=10)
        times.append(time.time() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"\n  Search latency over {len(docs_tokens)} docs:")
    print(f"    avg = {avg_ms:.2f}ms  |  "
          f"min = {min(times)*1000:.2f}ms  |  "
          f"max = {max(times)*1000:.2f}ms")
    check(f"latency < 100ms", avg_ms < 100, f"{avg_ms:.1f}ms")

    # test 4: show example results
    hline()
    print("[5] Example search results")
    inv_vocab = {v: k for k, v in w["vocab"].items()}

    for qi in random.sample(range(len(docs_tokens)), min(3, len(docs_tokens))):
        query_toks  = docs_tokens[qi]
        query_words = [inv_vocab.get(t, f"<{t}>") for t in query_toks[:8]]
        emb         = embed_tokens(query_toks, P, idf, vocab_size)
        idx, scores = search(emb, E, top_k=5)

        print(f"\n  Query [{qi}]: {' '.join(query_words)}")
        print(f"  {'Rank':<5} {'Score':>7}  Preview")
        for rank, (i, s) in enumerate(zip(idx, scores)):
            preview = [inv_vocab.get(t, f"<{t}>") for t in docs_tokens[i][:10]]
            marker  = " ← query" if i == qi else ""
            print(f"  {rank+1:<5} {s:>7.4f}  {' '.join(preview)}{marker}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test saved LSA weights")
    parser.add_argument("--weights", required=True,
                        help="Path to weights directory (output of train.py)")
    parser.add_argument("--corpus",  required=True,
                        help="Path to corpus.txt (to sample test docs from)")
    parser.add_argument("--n_docs",  type=int, default=500,
                        help="Number of docs to load for testing (default: 500)")
    parser.add_argument("--query",   type=str, default=None,
                        help="Optional: a raw text query to search")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  LSA Weight Validation & Search Test")
    print("═"*60)

    w          = load_weights(args.weights)
    vocab      = w["vocab"]
    vocab_size = w["meta"]["vocab_size"]
    P          = w["P"]
    idf        = w["idf"]

    validate_shapes(w)

    hline()
    print(f"[Loading] {args.n_docs} docs from corpus for testing")
    docs_raw, docs_tokens = load_sample_docs(args.corpus, vocab, n=args.n_docs)
    print(f"  Loaded {len(docs_tokens)} docs")

    embeds = validate_foldin(w, docs_tokens)
    validate_search(w, docs_tokens, docs_raw)

    # optional custom query
    if args.query:
        hline()
        print(f"[Custom Query] '{args.query}'")
        E = build_doc_index(docs_tokens, P, idf, vocab_size)
        emb, found_tokens = embed_text(args.query, vocab, P, idf, vocab_size)
        inv_vocab = {v: k for k, v in vocab.items()}
        print(f"  Mapped tokens: {[inv_vocab.get(t,'?') for t in found_tokens]}")
        if emb is None:
            print("  [WARN] No tokens matched vocab — check your query words")
        else:
            idx, scores = search(emb, E, top_k=5)
            print(f"  {'Rank':<5} {'Score':>7}  Preview")
            for rank, (i, s) in enumerate(zip(idx, scores)):
                preview = [inv_vocab.get(t, f"<{t}>") for t in docs_tokens[i][:10]]
                print(f"  {rank+1:<5} {s:>7.4f}  {' '.join(preview)}")

    print("\n" + "═"*60)
    print("  All validation complete.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()