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

    int kp = std::min(k+p, std::min(m,n));
    if(kp < k) k = kp;
    p = kp - k;

    Matrix<double> omega = getRandomMatrix(n, kp);
    Matrix<double> Y = A * omega;                      // m × kp

    auto A_T = A.Transpose();
    for(int i = 0; i < q; i++){
        Matrix<double> Q1 = gram_schmidt(Y);           // m × kp
        Matrix<double> Z  = A_T * Q1;                 // n × kp
        Matrix<double> Q2 = gram_schmidt(Z);           // n × kp
        Y = A * Q2;                                    // m × kp
    }

    Matrix<double> Q = gram_schmidt(Y);                // m × kp


    std::vector<std::vector<double>> Z_rows(kp);
    for(int j = 0; j < kp; j++){
        std::vector<double> qj(m);
        for(int i = 0; i < m; i++) qj[i] = Q(i,j);
        Z_rows[j] = A.SpMV_transpose(qj);             // length n
    }

    Matrix<double> C(kp, kp);
    for(int i = 0; i < kp; i++)
        for(int j = i; j < kp; j++){
            double s = 0;
            for(int l = 0; l < n; l++) s += Z_rows[i][l] * Z_rows[j][l];
            C(i,j) = s;
            C(j,i) = s;
        }

    
    auto eig = SVD_jacobi(C);                         
    Matrix<double>& U_cap  = eig.first.first;         // kp × kp
    std::vector<double>& lam = eig.second;            

    std::vector<double> sigma_b(kp);
    for(int i = 0; i < kp; i++)
        sigma_b[i] = std::sqrt(std::max(0.0, (double)lam[i]));

    //U = Q * U_cap  (m × kp)
    Matrix<double> U_full = Q * U_cap;


    Matrix<double> V_full(n, kp);
    for(int i = 0; i < kp; i++){
        if(sigma_b[i] < 1e-15) continue;              // skip near-zero singular values
        for(int l = 0; l < n; l++){
            double s = 0;
            for(int j = 0; j < kp; j++) s += Z_rows[j][l] * U_cap(j,i);
            V_full(l,i) = s / sigma_b[i];
        }
    }

    SVDResult ans(m, n, k);
    for(int i = 0; i < m; i++) for(int j = 0; j < k; j++) ans.U(i,j) = U_full(i,j);
    for(int i = 0; i < n; i++) for(int j = 0; j < k; j++) ans.V(i,j) = V_full(i,j);
    for(int i = 0; i < k; i++) ans.S[i] = sigma_b[i];

    return ans;
}