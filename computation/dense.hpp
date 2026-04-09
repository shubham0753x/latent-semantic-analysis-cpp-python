#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>

template <typename T>
struct Matrix{
    int rows;
    int cols;
    std::vector<T> data;
    Matrix(){ rows=0; cols=0; }
    Matrix(int rows,int cols):rows(rows),cols(cols),data(rows*cols,T{}){}
    Matrix(int size_):rows(size_),cols(size_),data(size_*size_,T{}){
        for(int i=0;i<size_;i++) (*this)(i,i)=T{1};
    }
    Matrix(const std::vector<std::vector<T>>&A):rows(A.size()),cols(A[0].size()),data(rows*cols){
        int p=0;
        for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) data[p++]=A[i][j];
    }
    T& operator()(int i,int j){
        if(i<0||i>=rows||j<0||j>=cols) throw std::out_of_range("Index out of bounds");
        return data[i*cols+j];
    }
    const T& operator()(int i,int j)const{
        if(i<0||i>=rows||j<0||j>=cols) throw std::out_of_range("Index out of bounds");
        return data[i*cols+j];
    }
    template<typename U>
    auto operator*(const Matrix<U>&B)const->Matrix<decltype(T{}*U{})>{
        if(cols!=B.rows) throw std::invalid_argument("dim mismatch");
        using R=decltype(T{}*U{});
        Matrix<R> C(rows,B.cols);
        for(int i=0;i<rows;i++) for(int k=0;k<cols;k++) for(int j=0;j<B.cols;j++)
            C(i,j)+=(*this)(i,k)*B(k,j);
        return C;
    }
    template<typename U>
    auto operator*(const std::vector<U>&V)const->std::vector<decltype(T{}*U{})>{
        if((int)V.size()!=cols) throw std::invalid_argument("dim mismatch");
        std::vector<decltype(T{}*U{})> ans(rows,0);
        for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) ans[i]+=(*this)(i,j)*V[j];
        return ans;
    }
    template<typename U>
    auto operator+(const Matrix<U>&B)const->Matrix<decltype(T{}+U{})>{
        if(B.rows!=rows||B.cols!=cols) throw std::invalid_argument("dim mismatch");
        Matrix<decltype(T{}+U{})> C(rows,cols);
        for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) C(i,j)=(*this)(i,j)+B(i,j);
        return C;
    }
    template<typename U>
    auto operator-(const Matrix<U>&B)const->Matrix<decltype(T{}+U{})>{
        if(B.rows!=rows||B.cols!=cols) throw std::invalid_argument("dim mismatch");
        Matrix<decltype(T{}+U{})> C(rows,cols);
        for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) C(i,j)=(*this)(i,j)-B(i,j);
        return C;
    }
    template<typename U>
    auto MatTranspseVec(const std::vector<U>&V)const->std::vector<decltype(T{}*U{})>{
        if((int)V.size()!=rows) throw std::invalid_argument("dim mismatch");
        std::vector<decltype(T{}*U{})> ans(cols,0);
        for(int i=0;i<cols;i++) for(int j=0;j<rows;j++) ans[i]+=(*this)(j,i)*V[j];
        return ans;
    }
    Matrix<T> transpose()const{
        Matrix<T> C(cols,rows);
        for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) C(j,i)=(*this)(i,j);
        return C;
    }
    // returns full column j starting from row start_row
    std::vector<T> getVector(int start_row,int j)const{
        if(j<0||j>=cols||start_row<0||start_row>rows)
            throw std::out_of_range("Index out of bounds");
        std::vector<T> a(rows-start_row);
        for(int k=start_row;k<rows;k++) a[k-start_row]=(*this)(k,j);
        return a;
    }
};

template<typename T>
T dot(const std::vector<T>&a,const std::vector<T>&b){
    if(a.size()!=b.size()) throw std::invalid_argument("size mismatch");
    T prod=T{};
    for(int i=0;i<(int)a.size();i++) prod+=a[i]*b[i];
    return prod;
}

template<typename T>
double norm(const std::vector<T>&a){ return std::sqrt((double)dot(a,a)); }

// Modified Gram-Schmidt — numerically stable, allocates only m×n
template<typename T>
Matrix<T> gram_schmidt(const Matrix<T>&A){
    int m=A.rows,n=A.cols;
    Matrix<T> Q(m,n);
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++) Q(i,j)=A(i,j);
        // modified GS: subtract projections one at a time
        for(int k=0;k<j;k++){
            double d=0;
            for(int i=0;i<m;i++) d+=Q(i,k)*Q(i,j);
            for(int i=0;i<m;i++) Q(i,j)-=d*Q(i,k);
        }
        double nm=0;
        for(int i=0;i<m;i++) nm+=Q(i,j)*Q(i,j);
        nm=std::sqrt(nm);
        if(nm>1e-12) for(int i=0;i<m;i++) Q(i,j)/=nm;
    }
    return Q;
}

// One-sided Jacobi SVD on matrix A (m×n, m>=n)
// Returns {{U, V}, sigma} where A = U * diag(sigma) * V^T
template<typename T>
std::pair<std::pair<Matrix<T>,Matrix<T>>,std::vector<T>> SVD_jacobi(Matrix<T> A){
    int m=A.rows,n=A.cols;
    Matrix<T> V(n,n);
    for(int i=0;i<n;i++) V(i,i)=T{1};
    double epsilon=1e-12;
    bool changed=true;
    while(changed){
        changed=false;
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                // compute alpha=col_i.col_j, beta=||col_i||^2-||col_j||^2
                double alpha=0,bii=0,bjj=0;
                for(int k=0;k<m;k++){
                    alpha+=A(k,i)*A(k,j);
                    bii+=A(k,i)*A(k,i);
                    bjj+=A(k,j)*A(k,j);
                }
                if(std::abs(alpha)<epsilon) continue;
                changed=true;
                double beta=bii-bjj;
                double c,s;
                if(std::abs(beta)<epsilon){
                    c=s=1.0/std::sqrt(2.0);
                } else {
                    // tan(2t) = 2*alpha/beta  => tau=-beta/(2*alpha) for zeroing
                    double tau=-beta/(2.0*alpha);
                    int sgn=(tau>=0?1:-1);
                    double t=(double)sgn/(std::abs(tau)+std::sqrt(1.0+tau*tau));
                    c=1.0/std::sqrt(1.0+t*t);
                    s=c*t;
                }
                // apply rotation to columns i and j of A
                for(int k=0;k<m;k++){
                    double aik=A(k,i),ajk=A(k,j);
                    A(k,i)=c*aik-s*ajk;
                    A(k,j)=s*aik+c*ajk;
                }
                // accumulate rotation in V
                for(int k=0;k<n;k++){
                    double vik=V(k,i),vjk=V(k,j);
                    V(k,i)=c*vik-s*vjk;
                    V(k,j)=s*vik+c*vjk;
                }
            }
        }
    }
    // extract singular values and normalize U columns
    std::vector<T> sigma(n);
    for(int i=0;i<n;i++){
        double nm=0;
        for(int k=0;k<m;k++) nm+=A(k,i)*A(k,i);
        nm=std::sqrt(nm);
        sigma[i]=(T)nm;
        if(nm>1e-15) for(int k=0;k<m;k++) A(k,i)/=nm;
    }
    // sort by descending singular value
    std::vector<int> idx(n);
    for(int i=0;i<n;i++) idx[i]=i;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){ return sigma[a]>sigma[b]; });
    Matrix<T> U_sorted(m,n);
    Matrix<T> V_sorted(n,n);
    std::vector<T> sig_sorted(n);
    for(int j=0;j<n;j++){
        sig_sorted[j]=sigma[idx[j]];
        for(int i=0;i<m;i++) U_sorted(i,j)=A(i,idx[j]);
        for(int i=0;i<n;i++) V_sorted(i,j)=V(i,idx[j]);
    }
    return {{U_sorted,V_sorted},sig_sorted};
}

Matrix<double> getRandomMatrix(int rows,int cols){
    Matrix<double> X(rows,cols);
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0,1.0);
    for(int i=0;i<(int)X.data.size();i++) X.data[i]=dist(gen);
    return X;
}