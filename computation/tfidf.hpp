#pragma once
#include <vector>
#include <cmath>
#include "csr.hpp"

template<typename T>
std::pair<MatrixCSR<T>,std::vector<double>> TFIDF(
    const std::vector<std::vector<int>>&docs, int vocab_size)
{
    std::vector<std::pair<std::pair<int,int>,T>> coo;
    coo.reserve(docs.size()*70);
    std::vector<double> idf(vocab_size,0.0);
    std::vector<int> freq(vocab_size,0);
    std::vector<int> active;
    active.reserve(200);

    for(int i=0;i<(int)docs.size();i++){
        active.clear();
        for(int tok:docs[i]){
            if(freq[tok]==0) active.push_back(tok);
            freq[tok]++;
        }
        double doc_len=(double)docs[i].size();
        for(int tok:active){
            coo.push_back({{tok,i},(T)(freq[tok]/doc_len)});
            idf[tok]++;
            freq[tok]=0;
        }
    }

    int n=(int)docs.size();
    for(int i=0;i<vocab_size;i++)
        idf[i]=std::log((1.0+n)/(1.0+idf[i]))+1.0;

    for(auto&e:coo) e.second*=(T)idf[e.first.first];

    MatrixCSR<T> mat(coo,vocab_size,n);
    return {mat,idf};
}