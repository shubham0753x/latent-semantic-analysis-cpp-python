#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tfidf.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tfidf, m){
    m.doc()="TF-IDF Engine Module";

    m.def("tfidf_double",
        [](const std::vector<std::vector<int>>&docs, int vocab_size){
            return TFIDF<double>(docs, vocab_size);
        },
        py::arg("docs"), py::arg("vocab_size"),
        "TF-IDF (64-bit). Returns (MatrixCSR_double, idf_weights).");

    m.def("tfidf_float",
        [](const std::vector<std::vector<int>>&docs, int vocab_size){
            return TFIDF<float>(docs, vocab_size);
        },
        py::arg("docs"), py::arg("vocab_size"),
        "TF-IDF (32-bit). Returns (MatrixCSR_float, idf_weights).");
}