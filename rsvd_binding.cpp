#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dense.hpp"
#include "csr.hpp"
#include "randomized_svd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(decomposition, m){
    m.doc()="Matrix Decomposition Module";

    // Register MatrixCSR<double> so pybind knows the type when passed from Python
    // py::class_<MatrixCSR<double>>(m,"MatrixCSR_double_ref");

    m.def("randomized_svd",
        [](const MatrixCSR<double>&A, int k, int p, int q){
            SVDResult res=RSVD_PI<double>(A,k,p,q);

            // return U, sigma, V as numpy arrays — no cross-module type dependency
            py::array_t<double> U({res.U.rows, res.U.cols});
            py::array_t<double> V({res.V.rows, res.V.cols});
            auto pu=U.mutable_unchecked<2>();
            auto pv=V.mutable_unchecked<2>();
            for(int i=0;i<res.U.rows;i++)
                for(int j=0;j<res.U.cols;j++)
                    pu(i,j)=res.U(i,j);
            for(int i=0;i<res.V.rows;i++)
                for(int j=0;j<res.V.cols;j++)
                    pv(i,j)=res.V(i,j);

            return py::make_tuple(U, res.S, V);
        },
        py::arg("A"), py::arg("k"),
        py::arg("p")=5, py::arg("q")=2,
        "Randomized SVD. Returns (U, sigma, V) as numpy arrays.");
}