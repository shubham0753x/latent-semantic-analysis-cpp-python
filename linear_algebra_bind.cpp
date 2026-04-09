#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dense.hpp"
#include "csr.hpp"

namespace py = pybind11;

template<typename T>
py::array_t<T> matrix_to_numpy(const Matrix<T>&M){
    py::array_t<T> result({M.rows, M.cols});
    auto ptr = result.template mutable_unchecked<2>();
    for(int i=0;i<M.rows;i++)
        for(int j=0;j<M.cols;j++)
            ptr(i,j)=M(i,j);
    return result;
}

template<typename T>
void bind_dense_components(py::module_&m, const std::string&typestr){
    std::string class_name="Matrix"+typestr;

    py::class_<Matrix<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def(py::init<int,int>())
        .def(py::init<int>())
        .def(py::init<const std::vector<std::vector<T>>&>())

        // construct from numpy array
        .def(py::init([](py::array_t<T> arr){
            auto buf=arr.request();
            if(buf.ndim!=2) throw std::runtime_error("Input must be 2D");
            T* ptr=static_cast<T*>(buf.ptr);
            int rows=buf.shape[0], cols=buf.shape[1];
            Matrix<T> M(rows,cols);
            for(int i=0;i<rows;i++)
                for(int j=0;j<cols;j++)
                    M(i,j)=ptr[i*cols+j];
            return M;
        }), py::arg("numpy_array"))

        .def_static("from_numpy",[](py::array_t<T> arr){
            auto buf=arr.request();
            if(buf.ndim!=2) throw std::runtime_error("Input must be 2D");
            T* ptr=static_cast<T*>(buf.ptr);
            int rows=buf.shape[0], cols=buf.shape[1];
            Matrix<T> M(rows,cols);
            for(int i=0;i<rows;i++)
                for(int j=0;j<cols;j++)
                    M(i,j)=ptr[i*cols+j];
            return M;
        })

        .def_readwrite("rows",  &Matrix<T>::rows)
        .def_readwrite("cols",  &Matrix<T>::cols)
        .def_readwrite("data",  &Matrix<T>::data)

        .def("__getitem__",[](const Matrix<T>&mat, std::pair<int,int> idx){
            return mat(idx.first, idx.second);
        })
        .def("__setitem__",[](Matrix<T>&mat, std::pair<int,int> idx, T val){
            mat(idx.first, idx.second)=val;
        })

        .def("__add__",[](const Matrix<T>&a,const Matrix<T>&b){ return a+b; })
        .def("__sub__",[](const Matrix<T>&a,const Matrix<T>&b){ return a-b; })
        .def("__mul__",[](const Matrix<T>&a,const Matrix<T>&b){ return a*b; })
        .def("__mul__",[](const Matrix<T>&a,const std::vector<T>&v){ return a*v; })

        .def("mat_transpose_vec",[](const Matrix<T>&mat,const std::vector<T>&v){
            return mat.MatTranspseVec(v);
        })
        .def("transpose",   &Matrix<T>::transpose)
        .def("get_vector",  &Matrix<T>::getVector)

        .def("to_numpy",[](const Matrix<T>&m){ return matrix_to_numpy(m); });

    m.def(("dot_"+typestr).c_str(),          &dot<T>);
    m.def(("norm_"+typestr).c_str(),         &norm<T>);
    m.def(("gram_schmidt_"+typestr).c_str(), &gram_schmidt<T>);
}

template<typename T>
void bind_csr_components(py::module_&m, const std::string&typestr){
    std::string class_name="MatrixCSR_"+typestr;

    py::class_<MatrixCSR<T>>(m, class_name.c_str())
        .def(py::init<>())

        // construct from COO list
        .def(py::init([](std::vector<std::pair<std::pair<int,int>,T>> coo,
                         int r, int c, bool sorted){
            return MatrixCSR<T>(coo, r, c, sorted);
        }), py::arg("A_coo"), py::arg("row_size"), py::arg("col_size"),
            py::arg("isSorted")=false)

        // construct from numpy array
        .def(py::init([](py::array_t<T> arr){
            auto buf=arr.request();
            if(buf.ndim!=2) throw std::runtime_error("Input must be 2D");
            T* ptr=static_cast<T*>(buf.ptr);
            int rows=buf.shape[0], cols=buf.shape[1];
            std::vector<std::pair<std::pair<int,int>,T>> coo;
            for(int i=0;i<rows;i++)
                for(int j=0;j<cols;j++)
                    if(ptr[i*cols+j]!=T{})
                        coo.push_back({{i,j},ptr[i*cols+j]});
            return MatrixCSR<T>(coo,rows,cols,false);
        }), py::arg("numpy_array"))

        .def_static("from_numpy",[](py::array_t<T> arr){
            auto buf=arr.request();
            if(buf.ndim!=2) throw std::runtime_error("Input must be 2D");
            T* ptr=static_cast<T*>(buf.ptr);
            int rows=buf.shape[0], cols=buf.shape[1];
            std::vector<std::pair<std::pair<int,int>,T>> coo;
            for(int i=0;i<rows;i++)
                for(int j=0;j<cols;j++)
                    if(ptr[i*cols+j]!=T{})
                        coo.push_back({{i,j},ptr[i*cols+j]});
            return MatrixCSR<T>(coo,rows,cols,false);
        })

        .def_readwrite("rows",     &MatrixCSR<T>::rows)
        .def_readwrite("cols",     &MatrixCSR<T>::cols)
        .def_readwrite("vals",     &MatrixCSR<T>::vals)
        .def_readwrite("row_size", &MatrixCSR<T>::row_size)
        .def_readwrite("col_size", &MatrixCSR<T>::col_size)

        .def("__mul__",[](const MatrixCSR<T>&a,const MatrixCSR<T>&b){ return a*b; })
        .def("__mul__",[](const MatrixCSR<T>&a,const Matrix<T>&b)   { return a*b; })
        .def("__mul__",[](const MatrixCSR<T>&a,const std::vector<T>&x){ return a*x; })
        
        // FIXED: Explicitly convert the 2D vector to a dense Matrix<T> before multiplying
        .def("__mul__",[](const MatrixCSR<T>&a, const std::vector<std::vector<T>>&b){ 
            return a * Matrix<T>(b); 
        })

        .def("spmv_transpose", &MatrixCSR<T>::SpMV_transpose)
        .def("transpose",      &MatrixCSR<T>::Transpose)

        // fixed to_numpy — uses i==0?0:rows[i-1] pattern, no sentinel
        .def("to_numpy",[](const MatrixCSR<T>&mat){
            py::array_t<T> result({mat.row_size, mat.col_size});
            auto ptr=result.template mutable_unchecked<2>();
            for(int i=0;i<mat.row_size;i++)
                for(int j=0;j<mat.col_size;j++)
                    ptr(i,j)=T{};
            for(int i=0;i<mat.row_size;i++){
                int start=i==0?0:mat.rows[i-1];
                int end=mat.rows[i];
                for(int j=start;j<end;j++)
                    ptr(i,mat.cols[j])=mat.vals[j];
            }
            return result;
        });
}

PYBIND11_MODULE(linear_algebra, m){
    m.doc()="Custom Standalone Linear Algebra Engine";

    py::module_ m_dense=m.def_submodule("dense","Dense matrix operations");
    bind_dense_components<double>(m_dense,"double");
    bind_dense_components<float> (m_dense,"float");
    m_dense.def("svd_jacobi",      [](Matrix<double> A){ return SVD_jacobi(A); });
    m_dense.def("get_random_matrix",&getRandomMatrix);

    py::module_ m_csr=m.def_submodule("csr","Sparse matrix operations");
    bind_csr_components<double>(m_csr,"double");
    bind_csr_components<float> (m_csr,"float");
}