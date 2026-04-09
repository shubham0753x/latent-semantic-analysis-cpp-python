#pragma once
#include "csr.hpp"
#include "dense.hpp"
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>

struct SVDResult {
    Matrix<double> U;
    std::vector<double> S;
    Matrix<double> V;
    SVDResult(int rows, int cols, int k) : U(rows,k), S(k,0.0), V(cols,k) {}
};

template<typename T>
SVDResult RSVD_PI(const MatrixCSR<T>& A, int k, int p, int q) {
    int m = A.row_size, n = A.col_size;

    // guard: k+p <= min(m,n)
    int kp = std::min(k+p, std::min(m,n));
    if(kp < k) k = kp;
    p = kp - k;

    //  Step 1: Y = A * Ω,  Ω is n × kp
    Matrix<double> omega = getRandomMatrix(n, kp);
    Matrix<double> Y = A * omega;                      // m × kp

    // ── Step 2: power iterations
    auto A_T = A.Transpose();
    for(int i = 0; i < q; i++){
        Matrix<double> Q1 = gram_schmidt(Y);           // m × kp
        Matrix<double> Z  = A_T * Q1;                 // n × kp
        Matrix<double> Q2 = gram_schmidt(Z);           // n × kp
        Y = A * Q2;                                    // m × kp
    }

    // ── Step 3: orthonormal basis Q for range(Y) ──────────────
    Matrix<double> Q = gram_schmidt(Y);                // m × kp

    // ── Step 4: B = Q' A  via SpMV_transpose ─────────────────
    std::vector<std::vector<double>> Z_rows(kp);
    for(int j = 0; j < kp; j++){
        std::vector<double> qj(m);
        for(int i = 0; i < m; i++) qj[i] = Q(i,j);
        Z_rows[j] = A.SpMV_transpose(qj);             // length n
    }

    // ── Step 5: C = B Bᵀ  (kp × kp, symmetric, tiny) ────────
    Matrix<double> C(kp, kp);
    for(int i = 0; i < kp; i++)
        for(int j = i; j < kp; j++){
            double s = 0;
            for(int l = 0; l < n; l++) s += Z_rows[i][l] * Z_rows[j][l];
            C(i,j) = s;
            C(j,i) = s;
        }

    // ── Step 6: eigen-decompose C = Û Λ Ûᵀ ──────────────────
    // SVD_jacobi on symmetric C gives eigendecomposition
    // eigenvalues = σᵢ² of B,  eigenvectors = left singular vectors of B
    auto eig = SVD_jacobi(C);                         // kp × kp — tiny
    Matrix<double>& U_cap  = eig.first.first;         // kp × kp
    std::vector<double>& lam = eig.second;            // kp eigenvalues (sorted desc)

    // ── Step 7: σᵢ = sqrt(λᵢ) ────────────────────────────────
    std::vector<double> sigma_b(kp);
    for(int i = 0; i < kp; i++)
        sigma_b[i] = std::sqrt(std::max(0.0, (double)lam[i]));

    // ── Step 8: U = Q * U_cap  (m × kp) ─────────────────────
    Matrix<double> U_full = Q * U_cap;

    // ── Step 9: V = Bᵀ U_cap / σᵢ  (n × kp) ────────────────
    // V[:,i] = Z_rows · U_cap[:,i] / sigma_b[i]
    // = Σ_j Z_rows[j] * U_cap(j,i) / sigma_b[i]
    Matrix<double> V_full(n, kp);
    for(int i = 0; i < kp; i++){
        if(sigma_b[i] < 1e-15) continue;              // skip near-zero singular values
        for(int l = 0; l < n; l++){
            double s = 0;
            for(int j = 0; j < kp; j++) s += Z_rows[j][l] * U_cap(j,i);
            V_full(l,i) = s / sigma_b[i];
        }
    }

    // ── Step 10: truncate to rank k ───────────────────────────
    SVDResult ans(m, n, k);
    for(int i = 0; i < m; i++) for(int j = 0; j < k; j++) ans.U(i,j) = U_full(i,j);
    for(int i = 0; i < n; i++) for(int j = 0; j < k; j++) ans.V(i,j) = V_full(i,j);
    for(int i = 0; i < k; i++) ans.S[i] = sigma_b[i];

    return ans;
}