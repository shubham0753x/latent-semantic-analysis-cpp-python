# Semantic Latent-Space Analysis Engine in C++

A high-performance **Latent Semantic Analysis (LSA)** engine built from scratch in C++ with Python bindings, capable of embedding academic text into a low-dimensional semantic space and retrieving conceptually similar documents via approximate nearest-neighbor search.

---

## Motivation

I wanted to understand how semantic similarity actually works at the mathematical level — not just call a library. That meant implementing the full pipeline myself: sparse matrix storage, TF-IDF vectorization, randomized SVD with power iteration, and an IVF index over the resulting embeddings.

The secondary motivation was practical: the fractional-order neural network stability literature spans control theory, functional analysis, and neuromorphic hardware. A semantic search engine over ArXiv abstracts makes navigating that intersection significantly easier than keyword search.

---

## Mathematical Foundation

The pipeline is grounded in three key results:

**1. Latent Semantic Analysis** — Deerwester et al. (1990)
TF-IDF weighted term-document matrices are decomposed via truncated SVD. The resulting singular vectors span a semantic subspace where geometric proximity corresponds to conceptual similarity.

**2. Randomized SVD** — Halko, Martinsson & Tropp (2011)
For large sparse matrices, exact SVD is prohibitively expensive. The HMT algorithm constructs a near-optimal low-rank approximation via random projections with power iteration, running in O(mn log k) time. This is the core computational engine of the project.

**3. Optimal Hard Thresholding** — Gavish & Donoho (2014)
Determines the rank cutoff `k` analytically under a noise model, avoiding manual hyperparameter tuning.

---

## Architecture

```
Raw Text (ArXiv abstracts)
        │
        ▼
  Python Layer  (crawling, tokenization, stop-word removal)
        │
        ▼
  TF-IDF Vectorization  ──►  Sparse CSR Matrix  (words × docs)
        │
        ▼
  Randomized SVD  ──►  U  (k left singular vectors)
                   ──►  Σ  (k singular values)
                   ──►  Vᵀ (k right singular vectors)
        │
        ▼
  Dense Latent Embeddings  (k-dimensional, one per document)
        │
        ▼
  Query → Fold-in Projection → Top-k Retrieval
```

---

## Project Structure

```
semantic_search_cpp/
├── computation/          # C++ core libraries
│   ├── csr.hpp
│   ├── dense.hpp
│   ├── randomized_svd.hpp
│   └── tfidf.hpp
├── bindings/             # Python bindings (pybind11)
├── weights/              # Saved LSA matrices
├── results/              # Evaluation plots & sample queries
├── notebooks/            # Data cleaning & exploration
├── tests/                # Unit tests
└── README.md

```

---

## Core Modules

### `csr.hpp` — Sparse Matrix Storage
Custom CSR (Compressed Sparse Row) implementation supporting sparse matrix-vector and matrix-matrix products without any external BLAS dependency. Built from scratch to understand the memory layout and avoid linking overhead.

### `dense.hpp` — Dense Linear Algebra
Dense matrix class with Householder QR factorization stored in-place (reflectors packed into the lower triangle). Used as the orthogonalization step inside the randomized SVD.

### `tfidf.hpp` — TF-IDF Vectorization
Reads a tokenized corpus and produces a sparse CSR term-document matrix with TF-IDF weights. Output orientation: words × docs, so each column is a document vector.

### `randomized_svd.hpp` — Randomized SVD
Implements Algorithm 4.4 from HMT 2011 with configurable power iteration. Stages:
1. Draw a Gaussian random matrix Ω
2. Form Y = (AAᵀ)^q · AΩ via power iteration (improves singular value gap)
3. Orthogonalize Y via QR → Q
4. Project: B = QᵀA
5. SVD of small matrix B → recover approximate singular triplets of A

---

## Build & Usage

### Prerequisites
- C++17 compiler (GCC ≥ 11 or Clang ≥ 14)
- Python ≥ 3.10
- `pybind11`, `numpy`, `matplotlib`

### Build Python Bindings
```bash
make
```

### Train on a Corpus
```bash
python train.py --corpus data/abstracts/ --k 100 --power-iter 1
```

### Query the Index
```bash
python demo.py --query "Mittag-Leffler stability fractional differential equations"
```

## Dataset

Download dataset from:
https://www.kaggle.com/datasets/Cornell-University/arxiv

Place it in:
processing

---

## What I Learned

- Implementing one-sided Jacobi SVD and Householder QR from scratch — deriving the geometry, not just copying pseudocode
- How power iteration inside randomized SVD sharpens the approximation by amplifying the dominant singular value gap
- The subtleties of pybind11: ownership semantics, buffer protocols for numpy arrays, avoiding copies across the C++/Python boundary
- Why words × docs orientation (not docs × words) simplifies fold-in projection for new queries

---

## Current Limitations & Future Work

The current model (`k=100`, `p=5`, `q=1`) was trained as a proof‑of‑concept.  
While training is extremely fast, retrieval quality on the arXiv dataset is suboptimal due to:

- Insufficient latent dimensions (`k` too small for 25k multidisciplinary abstracts).
- Minimal text preprocessing (no lemmatization, aggressive stopword filtering).
- Short document length (abstracts only).

Sample queries show that the latent space does not yet capture semantic similarity well (see [`results/sample_queries.txt`](results/sample_queries.txt)).

**Planned improvements:**
- Increase `k` to 300.
- Add Porter stemming and custom stopword list.
- Experiment with `q=2` for better randomized SVD accuracy.
- Evaluate on a more focused subset (e.g., only `cs.CL` and `cs.LG`).

Contributions and suggestions are welcome!

## Acknowledgments

Used **Claude (Anthropic)** as a debugging and implementation aid throughout the project — specifically for tracking down non-obvious C++ bugs (an incorrect quadrant in the Jacobi rotation angle derivation, absolute vs. relative convergence threshold issues), writing and structuring the pybind11 bindings, and scaffolding the training pipeline. The mathematical derivations, architecture decisions, and overall design were worked through independently.

---

## References

- Deerwester et al. (1990). *Indexing by Latent Semantic Analysis.* JASIS.
- Halko, Martinsson & Tropp (2011). *Finding Structure with Randomness.* SIAM Review.
  
---
