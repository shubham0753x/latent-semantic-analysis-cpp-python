#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include "dense.hpp"

template<typename T>
struct MatrixCSR{
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<T> vals;
    int row_size,col_size;

    MatrixCSR(){}

    MatrixCSR(std::vector<std::pair<std::pair<int,int>,T>>&A_coo,int row_size,int col_size,bool isSorted=false){
        if(!isSorted) std::sort(A_coo.begin(),A_coo.end());
        this->row_size=row_size; this->col_size=col_size;
        cols.resize(A_coo.size()); vals.resize(A_coo.size()); rows.resize(row_size,0);
        for(int i=0;i<(int)A_coo.size();i++){
            cols[i]=A_coo[i].first.second;
            vals[i]=A_coo[i].second;
        }
        for(int i=0;i<(int)A_coo.size();i++) rows[A_coo[i].first.first]++;
        for(int i=1;i<row_size;i++) rows[i]+=rows[i-1];
    }

    MatrixCSR operator*(const MatrixCSR&B)const{
        std::vector<T> acc(B.col_size,T{});
        std::vector<bool> touched(B.col_size,false);
        std::vector<int> touched_cols;
        std::vector<std::pair<std::pair<int,int>,T>> coo;
        for(int i=0;i<row_size;i++){
            int k=i==0?0:rows[i-1];
            for(;k<rows[i];k++){
                int k1=cols[k];
                int l=k1==0?0:B.rows[k1-1];
                for(;l<B.rows[k1];l++){
                    int j=B.cols[l];
                    acc[j]+=vals[k]*B.vals[l];
                    if(!touched[j]){touched[j]=true;touched_cols.push_back(j);}
                }
            }
            std::sort(touched_cols.begin(),touched_cols.end());
            for(int j:touched_cols){
                if(acc[j]!=T{}) coo.push_back({{i,j},acc[j]});
                touched[j]=false; acc[j]=T{};
            }
            touched_cols.clear();
        }
        return MatrixCSR(coo,row_size,B.col_size,true);
    }

    template<typename U>
    auto operator*(const Matrix<U>&B)const->Matrix<decltype(T{}*U{})>{
        Matrix<decltype(T{}*U{})> C(row_size,B.cols);
        for(int i=0;i<row_size;i++){
            int start=i==0?0:rows[i-1];
            for(int k=start;k<rows[i];k++)
                for(int j=0;j<B.cols;j++)
                    C(i,j)+=vals[k]*B(cols[k],j);
        }
        return C;
    }

    std::vector<T> operator*(const std::vector<T>&X)const{
        std::vector<T> Y(row_size,T{});
        for(int i=0;i<row_size;i++){
            int start=i==0?0:rows[i-1];
            for(int j=start;j<rows[i];j++) Y[i]+=vals[j]*X[cols[j]];
        }
        return Y;
    }

    std::vector<T> SpMV_transpose(const std::vector<T>&X)const{
        std::vector<T> Y(col_size,T{});
        for(int i=0;i<row_size;i++){
            int start=i==0?0:rows[i-1];
            for(int j=start;j<rows[i];j++) Y[cols[j]]+=vals[j]*X[i];
        }
        return Y;
    }

    MatrixCSR Transpose()const{
        MatrixCSR B;
        B.row_size=col_size; B.col_size=row_size;
        B.rows.resize(B.row_size,0);
        B.cols.resize(cols.size()); B.vals.resize(vals.size());
        for(int j:cols) B.rows[j]++;
        for(int i=1;i<B.row_size;i++) B.rows[i]+=B.rows[i-1];
        int zero_done=0;
        for(int i=0;i<row_size;i++){
            int start=i==0?0:rows[i-1];
            for(int j=start;j<rows[i];j++){
                int re=cols[j];
                int pos=re!=0?B.rows[re-1]++:zero_done++;
                B.cols[pos]=i; B.vals[pos]=vals[j];
            }
        }
        std::fill(B.rows.begin(),B.rows.end(),0);
        for(int j:cols) B.rows[j]++;
        for(int i=1;i<B.row_size;i++) B.rows[i]+=B.rows[i-1];
        return B;
    }
};