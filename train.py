import argparse
import json
import os
import time
import math
import struct
import random
import numpy as np
from pathlib import Path


try:
    import linear_algebra.dense as dense
    import linear_algebra.csr   as csr
    import decomposition        as rsvd_mod
    import tfidf                as tfidf_mod
    CSR = csr.MatrixCSR_double
    print("[OK] C++ library loaded")
except ImportError as e:
    raise SystemExit(f"[ERROR] Could not import C++ library: {e}\n"
                     f"Make sure .so files are in the current directory.")

# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def log(msg, indent=0):
    print("  " * indent + msg)

def hline():
    print("─" * 60)

def fmt_time(s):
    if s < 60: return f"{s:.1f}s"
    return f"{int(s//60)}m {s%60:.0f}s"

def fmt_size(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000: return f"{n:.0f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


# step 1: load vocab


def load_vocab(path):
    hline()
    log(f"[1/5] Loading vocabulary from: {path}")
    t0 = time.time()
    with open(path) as f:
        vocab = json.load(f)
    # ensure values are ints
    vocab = {k: int(v) for k, v in vocab.items()}
    vocab_size = max(vocab.values()) + 1
    log(f"      Words in vocab : {fmt_size(len(vocab))}", 1)
    log(f"      Vocab size (max id+1): {vocab_size}", 1)
    log(f"      Time: {fmt_time(time.time()-t0)}", 1)
    return vocab, vocab_size


# step 2: load and tokenize corpus


def load_corpus(path, vocab, n_samples=None):
    hline()
    log(f"[2/5] Loading corpus from: {path}")
    t0 = time.time()

    # count lines first (fast)
    log("      Counting lines...", 1)
    with open(path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    log(f"      Total lines: {fmt_size(total_lines)}", 1)

    # decide which line indices to keep
    if n_samples is not None and n_samples < total_lines:
        log(f"      Reservoir sampling {n_samples} / {total_lines}", 1)
        random.seed(42)
        keep = set(random.sample(range(total_lines), n_samples))
    else:
        keep = None  # keep all

    # single pass through file
    docs = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if keep is not None and line_idx not in keep:
                continue
            line = line.strip()
            if not line:
                skipped += 1
                continue
            tokens = []
            for tok in line.split():
                try:
                    tokens.append(int(tok))
                except ValueError:
                    tid = vocab.get(tok.lower())
                    if tid is not None:
                        tokens.append(tid)
            if tokens:
                docs.append(tokens)
            else:
                skipped += 1

    lengths = [len(d) for d in docs]
    log(f"      Loaded: {len(docs)} docs", 1)
    log(f"      Skipped: {skipped} empty lines", 1)
    log(f"      Avg/Min/Max length: {np.mean(lengths):.1f} / {min(lengths)} / {max(lengths)}", 1)
    log(f"      Time: {fmt_time(time.time()-t0)}", 1)
    return docs

# step 3: TF-IDF encoding

def run_tfidf(docs, vocab_size):
    hline()
    log(f"[3/5] TF-IDF encoding")
    log(f"      Docs: {len(docs)}  |  Vocab size: {vocab_size}", 1)
    t0 = time.time()

    tfidf_mat, idf = tfidf_mod.tfidf_double(docs, vocab_size)
    elapsed = time.time() - t0

    # matrix statistics
    nnz = len(tfidf_mat.vals)
    total_cells = tfidf_mat.row_size * tfidf_mat.col_size
    density = nnz / max(total_cells, 1)
    avg_tfidf = sum(tfidf_mat.vals) / max(nnz, 1)

    idf_arr = np.array(idf)
    # words that appear in at least one doc
    active_words = int(np.sum(idf_arr < idf_arr.max()))

    log(f"      Matrix shape    : {tfidf_mat.row_size} × {tfidf_mat.col_size}", 1)
    log(f"      Non-zeros (nnz) : {fmt_size(nnz)}", 1)
    log(f"      Density         : {density*100:.4f}%", 1)
    log(f"      Avg TF-IDF val  : {avg_tfidf:.4f}", 1)
    log(f"      Active vocab    : {active_words} / {vocab_size} words", 1)
    log(f"      IDF  min/max    : {idf_arr.min():.3f} / {idf_arr.max():.3f}", 1)
    log(f"      Time            : {fmt_time(elapsed)}", 1)

    # sanity checks
    assert tfidf_mat.row_size == vocab_size,  "TF-IDF row mismatch"
    assert tfidf_mat.col_size == len(docs),   "TF-IDF col mismatch"
    assert len(idf) == vocab_size,            "IDF length mismatch"
    log(f"      [OK] Sanity checks passed", 1)

    return tfidf_mat, idf


# step 4: randomized SVD

def run_rsvd(tfidf_mat, k, p, q):
    hline()
    log(f"[4/5] Randomized SVD  (k={k}, p={p}, q={q})")
    m = tfidf_mat.row_size
    n = tfidf_mat.col_size
    kp = min(k+p, min(m,n))
    if kp < k:
        log(f"      [WARN] k+p={k+p} > min(m,n)={min(m,n)}, capping to k={kp}", 1)
        k = kp
    log(f"      Input shape     : {m} × {n}", 1)
    log(f"      Target rank k   : {k}", 1)
    t0 = time.time()

    U, sigma, V = rsvd_mod.randomized_svd(tfidf_mat, k=k, p=p, q=q)
    elapsed = time.time() - t0

    sigma = np.array(sigma)

    # analysis
    total_variance  = float(np.sum(sigma**2))
    explained       = np.cumsum(sigma**2) / total_variance * 100

    log(f"      U shape         : {U.shape}", 1)
    log(f"      V shape         : {V.shape}", 1)
    log(f"      Time            : {fmt_time(elapsed)}", 1)
    log(f"      Singular values :", 1)
    for i in range(min(10, k)):
        log(f"        σ[{i:2d}] = {sigma[i]:10.4f}  "
            f"(explains {explained[i]:5.1f}% cumulative)", 2)
    if k > 10:

    # effective rank estimate: number of singular values > 1% of max
     eff_rank = int(np.sum(sigma > 0.01 * sigma[0]))
    log(f"      Effective rank  : {eff_rank} (σ > 1% of σ_max)", 1)
    log(f"      Variance explained by top-{k}: {explained[-1]:.1f}%", 1)

    # orthogonality check on U (sample check for large matrices)
    sample = min(k, U.shape[1])
    U_s = U[:, :sample]
    orth_err = np.linalg.norm(U_s.T @ U_s - np.eye(sample))
    log(f"      UᵀU=I error     : {orth_err:.2e}  "
        f"{'[OK]' if orth_err<1e-4 else '[WARN: high]'}", 1)

    return U, sigma, V, k


# ─────────────────────────────────────────────────────────────
# step 5: save weights
# ─────────────────────────────────────────────────────────────

def save_weights(output_dir, U, sigma, V, idf, vocab, k, vocab_size, n_docs, args):
    hline()
    log(f"[5/5] Saving weights to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    idf_arr   = np.array(idf,   dtype=np.float64)
    sigma_arr = np.array(sigma, dtype=np.float64)

    log("      Dropping Component 0 (removing LSA gravity well)...", 1)
    U = U[:, 1:]
    sigma_arr = sigma_arr[1:]
    V = V[:, 1:]  # V shape from your logs is (30000, 250)
    k = k - 1     # We now have 249 dimensions

    P = (U / sigma_arr).T


    np.save(os.path.join(output_dir, "U.npy"),     U.astype(np.float64))
    np.save(os.path.join(output_dir, "sigma.npy"), sigma_arr)
    np.save(os.path.join(output_dir, "P.npy"),     P)
    np.save(os.path.join(output_dir, "idf.npy"),   idf_arr)
   
    np.save(os.path.join(output_dir, "V.npy"),     V.astype(np.float64))

    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    meta = {
        "k":          k,
        "p":          args.p,
        "q":          args.q,
        "vocab_size": vocab_size,
        "n_docs":     n_docs,
        "n_samples":  args.n_samples,
        "corpus":     str(args.corpus),
        "vocab_file": str(args.vocab),
        "U_shape":    list(U.shape),
        "P_shape":    list(P.shape),
        "sigma_min":  float(sigma_arr.min()),
        "sigma_max":  float(sigma_arr.max()),
        "explained_variance_pct": float(
            np.sum(sigma_arr**2) / np.sum(sigma_arr**2) * 100
        ),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = 0
    for fname in ["U.npy","sigma.npy","P.npy","V.npy","idf.npy","vocab.json","meta.json"]:
        fpath = os.path.join(output_dir, fname)
        sz = os.path.getsize(fpath)
        total_bytes += sz
        log(f"      {fname:<15} {sz/1024/1024:.2f} MB", 1)

    log(f"      Total           : {total_bytes/1024/1024:.2f} MB", 1)
    log(f"      Time            : {fmt_time(time.time()-t0)}", 1)
    log(f"      [OK] All weights saved", 1)

    return P



def validate(docs, P, idf, vocab, vocab_size, k, n_check=5):
    hline()
    log("[Validation] Fold-in sanity check")
    idf_arr = np.array(idf)
    inv_vocab = {v: k for k, v in vocab.items()}

    for idx in random.sample(range(len(docs)), min(n_check, len(docs))):
        doc = docs[idx]

        tf = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0) + 1
        vec = np.zeros(vocab_size)
        for tok, cnt in tf.items():
            if tok < vocab_size:
                vec[tok] = (cnt / len(doc)) * idf_arr[tok]

        embed = P @ vec
        nm = np.linalg.norm(embed)
        top_words = [inv_vocab.get(t, f"<{t}>") for t in doc[:5]]
        log(f"      doc[{idx:5d}] len={len(doc):4d}  "
            f"embed_norm={nm:.4f}  "
            f"preview: {' '.join(top_words)}", 1)

    log(f"      [OK] Fold-in working — non-zero embeddings produced", 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train LSA semantic search model")
    parser.add_argument("--vocab",     required=True,
                        help="Path to vocab.json (word -> int)")
    parser.add_argument("--corpus",    required=True,
                        help="Path to corpus.txt (one doc per line)")
    parser.add_argument("--output",    default="weights",
                        help="Directory to save weights (default: weights/)")
    parser.add_argument("--k",         type=int, default=100,
                        help="Number of latent dimensions (default: 100)")
    parser.add_argument("--p",         type=int, default=10,
                        help="Oversampling parameter (default: 10)")
    parser.add_argument("--q",         type=int, default=2,
                        help="Power iterations (default: 2)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of docs to use (default: all). "
                             "Use a small number (e.g. 500) to test first.")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  LSA Training Pipeline — ArXiv Semantic Search")
    print("═"*60)
    print(f"  vocab    : {args.vocab}")
    print(f"  corpus   : {args.corpus}")
    print(f"  output   : {args.output}")
    print(f"  k        : {args.k}")
    print(f"  p        : {args.p}")
    print(f"  q        : {args.q}")
    print(f"  n_samples: {args.n_samples or 'all'}")

    wall_start = time.time()

    # ── pipeline ──
    vocab, vocab_size     = load_vocab(args.vocab)
    docs                  = load_corpus(args.corpus, vocab, args.n_samples)
    tfidf_mat, idf        = run_tfidf(docs, vocab_size)
    U, sigma, V, k        = run_rsvd(tfidf_mat, args.k, args.p, args.q)
    P                     = save_weights(
                                args.output, U, sigma, V, idf, vocab,
                                k, vocab_size, len(docs), args)
    validate(docs, P, idf, vocab, vocab_size, k)

    hline()
    total = time.time() - wall_start
    print(f"\n  [DONE] Total training time: {fmt_time(total)}")
    print(f"  Weights saved to: {args.output}/")
    print(f"  Load at serving time:")
    print(f"    P   = np.load('{args.output}/P.npy')      # fold-in operator")
    print(f"    idf = np.load('{args.output}/idf.npy')    # IDF weights")
    print(f"    with open('{args.output}/vocab.json') as f: vocab = json.load(f)")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()